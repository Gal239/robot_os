"""
Async Writer - General purpose background I/O worker

OFFENSIVE: Makes blocking I/O operations async using background thread
USE CASE: Timeline saving, disk writes, network ops, anything that blocks!

Pattern:
    writer = AsyncWriter(maxsize=100)
    writer.submit(lambda: slow_disk_write())  # Returns immediately!
    writer.close()  # Waits for queue to drain

Safety:
    - Bounded queue prevents memory explosion
    - Graceful shutdown with timeout
    - Error callback for monitoring failures
    - Thread-safe (uses queue.Queue)
"""

import queue
import threading
import traceback
from typing import Callable, Optional
import time


class AsyncWriter:
    """Background worker for async I/O operations

    OFFENSIVE: Crashes on worker thread errors if no error_callback provided
    SIMPLE: Just submit callables, they run in background
    SAFE: Bounded queue, graceful shutdown, error handling

    Example:
        writer = AsyncWriter()

        # Submit blocking operations - returns immediately!
        writer.submit(lambda: file.write(data))
        writer.submit(lambda: video_writer.write(frame))

        # Graceful shutdown - waits for all tasks
        writer.close(timeout=10)
    """

    def __init__(self, maxsize: int = 100, error_callback: Optional[Callable] = None, name: str = "AsyncWriter"):
        """Initialize async writer with background thread

        Args:
            maxsize: Max queue size (blocks submit() if full - prevents memory explosion!)
            error_callback: Called on worker errors: callback(exception, task)
            name: Thread name for debugging
        """
        self.name = name
        self.error_callback = error_callback

        # Bounded queue - CRITICAL for preventing memory explosion!
        # If simulation runs at 200 fps but disk writes at 30 fps, queue would grow forever
        self.queue = queue.Queue(maxsize=maxsize)

        # Worker thread state
        self.worker = threading.Thread(target=self._worker_loop, name=name, daemon=True)
        self.running = True
        self.worker_error = None  # Last error from worker thread

        # DEBUG: Track who's calling submit() to find performance bottlenecks
        self.caller_counts = {}  # caller_info -> count
        self.total_submits = 0

        # Start background worker
        self.worker.start()

    def submit(self, task: Callable, timeout: Optional[float] = None) -> bool:
        """Submit task to background queue - RETURNS IMMEDIATELY (unless queue full)

        Args:
            task: Callable to execute in background (no args)
            timeout: Max seconds to wait if queue full (None = block forever)

        Returns:
            True if submitted, False if queue full and timeout expired

        Raises:
            RuntimeError: If worker thread died
        """
        # DEBUG: Track caller to find performance bottlenecks
        # NOTE: traceback.extract_stack() is SLOW! Only enable when debugging!
        if False:  # DISABLED for performance (adds ~40% overhead!)
            caller = traceback.extract_stack()[-2]
            caller_key = f"{caller.filename.split('/')[-1]}:{caller.lineno} in {caller.name}"
            self.caller_counts[caller_key] = self.caller_counts.get(caller_key, 0) + 1
            self.total_submits += 1

        # Check worker health - OFFENSIVE!
        if not self.worker.is_alive():
            raise RuntimeError(
                f"❌ AsyncWriter '{self.name}' worker thread died!\n"
                f"   Last error: {self.worker_error}\n"
                f"   FIX: Check error_callback for what killed the worker"
            )

        try:
            # Put task in queue - blocks if full (prevents memory explosion)
            self.queue.put(task, block=True, timeout=timeout)
            return True
        except queue.Full:
            return False  # Queue full, couldn't submit

    def _worker_loop(self):
        """Background worker - runs tasks from queue

        OFFENSIVE: Calls error_callback on failures, crashes if no callback
        """
        while self.running:
            task = None
            try:
                # Get next task (blocks until available or timeout)
                task = self.queue.get(timeout=0.1)

                if task is None:
                    # Poison pill - shutdown signal
                    self.queue.task_done()  # Mark poison pill as done
                    break

                # Execute task in background
                task()

                # Mark task done
                self.queue.task_done()

            except queue.Empty:
                # No tasks available, continue waiting
                continue

            except Exception as e:
                # Task execution failed!
                # CRITICAL: Mark task as done even on failure to prevent queue.join() hanging!
                if task is not None:
                    self.queue.task_done()

                error_msg = f"AsyncWriter '{self.name}' task failed: {e}"
                self.worker_error = error_msg

                if self.error_callback:
                    # Report error to callback
                    try:
                        self.error_callback(e, task)
                    except Exception as callback_error:
                        # Callback itself failed - OFFENSIVE!
                        print(f"❌ Error callback failed: {callback_error}")
                        traceback.print_exc()
                else:
                    # No callback - print error but DON'T raise (prevents worker death)
                    print(f"❌ {error_msg}")
                    traceback.print_exc()
                    # Continue processing other tasks instead of dying

    def close(self, timeout: float = 10.0) -> bool:
        """Close writer and wait for all tasks to complete - GRACEFUL SHUTDOWN

        Args:
            timeout: Max seconds to wait for queue to drain

        Returns:
            True if all tasks completed, False if timeout

        Raises:
            RuntimeError: If worker thread died with error
        """
        # Stop accepting new tasks
        self.running = False

        # Send poison pill to worker
        try:
            self.queue.put(None, block=False)
        except queue.Full:
            pass  # Queue full, worker will exit on next task

        # Wait for all tasks to complete
        try:
            # Wait for worker thread to exit (this ensures all tasks processed)
            # Don't use queue.join() - it has NO timeout and can hang forever!
            self.worker.join(timeout=timeout)

            if self.worker.is_alive():
                # Worker didn't exit in time - WARN but don't crash
                print(f"⚠️  AsyncWriter '{self.name}' worker didn't exit in {timeout}s")
                return False

            # Check for worker errors
            if self.worker_error:
                raise RuntimeError(
                    f"❌ AsyncWriter '{self.name}' had errors:\n{self.worker_error}"
                )

            return True

        except Exception as e:
            print(f"❌ Error during AsyncWriter '{self.name}' shutdown: {e}")
            raise

    def qsize(self) -> int:
        """Get approximate queue size - useful for monitoring"""
        return self.queue.qsize()

    def is_alive(self) -> bool:
        """Check if worker thread is running"""
        return self.worker.is_alive()

    def print_caller_stats(self):
        """Print statistics about who's calling submit() - DEBUG TOOL"""
        print(f"\n{'='*80}")
        print(f"AsyncWriter '{self.name}' Caller Statistics")
        print(f"{'='*80}")
        print(f"Total submit() calls: {self.total_submits}")
        print(f"\nBreakdown by caller:")
        print(f"{'-'*80}")

        # Sort by count descending
        sorted_callers = sorted(self.caller_counts.items(), key=lambda x: x[1], reverse=True)

        for caller, count in sorted_callers:
            pct = (count / self.total_submits * 100) if self.total_submits > 0 else 0
            print(f"  {count:4d} calls ({pct:5.1f}%)  {caller}")

        print(f"{'='*80}\n")


# Example usage for testing
if __name__ == "__main__":
    import time

    print("Testing AsyncWriter...")

    # Create writer
    writer = AsyncWriter(maxsize=10, name="TestWriter")

    # Submit some slow tasks
    results = []

    def slow_task(i):
        time.sleep(0.1)
        results.append(i)
        print(f"  Task {i} completed")

    print("\n1. Submitting 5 tasks...")
    for i in range(5):
        writer.submit(lambda i=i: slow_task(i))
        print(f"  Submitted task {i} (returned immediately!)")

    print("\n2. Continuing while tasks run in background...")
    time.sleep(0.05)
    print(f"  Queue size: {writer.qsize()}")

    print("\n3. Closing (waits for all tasks)...")
    writer.close(timeout=5)

    print(f"\n✅ All tasks completed: {results}")
    print(f"   Expected: [0, 1, 2, 3, 4]")
    print(f"   Success: {results == list(range(5))}")
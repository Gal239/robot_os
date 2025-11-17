"""
PROCESS TRACKER - Track background processes properly
NOT the broken Claude Code background tracker!

SELF-DESCRIPTIVE: Process tells you its real state (running/finished/crashed)
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


class ProcessTracker:
    """Track background processes with REAL status updates"""

    def __init__(self, log_dir: str = "/tmp"):
        self.processes: Dict[str, dict] = {}
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

    def start(self, name: str, command: str, cwd: Optional[str] = None) -> int:
        """Start a background process and track it

        Args:
            name: Process name (for tracking)
            command: Shell command to run
            cwd: Working directory (optional)

        Returns:
            PID of started process
        """
        # Create log file
        log_file = self.log_dir / f"{name}.log"

        # Start process
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                command,
                shell=True,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=cwd,
                start_new_session=True  # Detach from parent
            )

        # Track it
        self.processes[name] = {
            'pid': proc.pid,
            'command': command,
            'log_file': str(log_file),
            'start_time': time.time(),
            'proc': proc
        }

        return proc.pid

    def get_status(self, name: str) -> Dict[str, any]:
        """Get REAL status of process - SELF-DESCRIPTIVE!

        Returns:
            {
                'state': 'running' | 'finished' | 'crashed' | 'unknown',
                'pid': int,
                'exit_code': int | None,
                'runtime': float,
                'cpu_percent': float | None,
                'output_lines': int
            }
        """
        if name not in self.processes:
            return {'state': 'unknown', 'error': f"Process '{name}' not tracked"}

        info = self.processes[name]
        pid = info['pid']

        # Check if process is still running
        try:
            # poll() returns None if running, exit code if finished
            exit_code = info['proc'].poll()

            if exit_code is None:
                # Still running - get CPU usage
                try:
                    ps_output = subprocess.check_output(
                        f"ps -p {pid} -o %cpu --no-headers",
                        shell=True,
                        text=True
                    ).strip()
                    cpu = float(ps_output)
                except:
                    cpu = None

                state = 'running'
            else:
                # Finished
                state = 'finished' if exit_code == 0 else 'crashed'
                cpu = None
        except:
            state = 'unknown'
            exit_code = None
            cpu = None

        # Count output lines
        log_file = Path(info['log_file'])
        if log_file.exists():
            output_lines = len(log_file.read_text().splitlines())
        else:
            output_lines = 0

        # Calculate runtime
        runtime = time.time() - info['start_time']

        return {
            'state': state,
            'pid': pid,
            'exit_code': exit_code,
            'runtime': runtime,
            'cpu_percent': cpu,
            'output_lines': output_lines,
            'log_file': info['log_file']
        }

    def get_output(self, name: str, lines: int = 50) -> str:
        """Get last N lines of output"""
        if name not in self.processes:
            return f"Process '{name}' not tracked"

        log_file = Path(self.processes[name]['log_file'])
        if not log_file.exists():
            return "No output yet"

        output = log_file.read_text().splitlines()
        return '\n'.join(output[-lines:])

    def kill(self, name: str) -> bool:
        """Kill a process"""
        if name not in self.processes:
            return False

        info = self.processes[name]
        try:
            info['proc'].kill()
            info['proc'].wait(timeout=5)
            return True
        except:
            # Try harder
            try:
                subprocess.run(f"kill -9 {info['pid']}", shell=True)
                return True
            except:
                return False

    def list_all(self) -> Dict[str, Dict]:
        """List all tracked processes with their status"""
        result = {}
        for name in self.processes:
            result[name] = self.get_status(name)
        return result

    def cleanup_finished(self):
        """Remove finished processes from tracking"""
        to_remove = []
        for name in self.processes:
            status = self.get_status(name)
            if status['state'] in ['finished', 'crashed']:
                to_remove.append(name)

        for name in to_remove:
            del self.processes[name]

        return len(to_remove)


# Global tracker instance
tracker = ProcessTracker()


if __name__ == "__main__":
    """Example usage"""

    # Start a test process
    print("Starting test process...")
    pid = tracker.start(
        name="test_sleep",
        command="sleep 5 && echo 'Done!'",
        cwd="/tmp"
    )
    print(f"Started PID: {pid}")

    # Check status a few times
    for i in range(7):
        time.sleep(1)
        status = tracker.get_status("test_sleep")
        print(f"\n[{i+1}s] Status: {status['state']}")
        print(f"    Runtime: {status['runtime']:.1f}s")
        print(f"    CPU: {status['cpu_percent']}%")
        print(f"    Output lines: {status['output_lines']}")

    # Get final output
    print("\nFinal output:")
    print(tracker.get_output("test_sleep"))

    # List all
    print("\nAll processes:")
    for name, status in tracker.list_all().items():
        print(f"  {name}: {status['state']} (PID {status['pid']})")
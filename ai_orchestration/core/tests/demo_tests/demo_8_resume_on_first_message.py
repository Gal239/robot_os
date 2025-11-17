#!/usr/bin/env python3
"""
DEMO 8: RESUME ON FIRST MESSAGE
Tests that Orchestrator.resume() works even when logs don't exist yet
(i.e., resuming after the first message before any logs are written)
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 8: RESUME ON FIRST MESSAGE")
    print("="*60)

    # PART 1: Create a new session and run first task
    print("\n[PART 1] Creating new session with first task...")
    ops = Orchestrator()

    # Single agent, no delegation
    ops.agent.create(
        id="solo",
        describe="General assistant",
        tools=["handoff"]
    )

    print("[OK] Agent: solo")
    session_id = ops.graph_ops.modal.session_id
    print(f"[OK] Session ID: {session_id}")

    # First task
    print("\n[TURN 1] What is 2 + 2?")
    result1 = await ops.start_root_task(
        task="What is 2 + 2?",
        main_agent="solo"
    )

    print(f"[TURN 1] Answer: {result1.get('result', 'N/A')}")

    # PART 2: Resume the session (this should work even if logs don't exist yet)
    print("\n" + "="*60)
    print("[PART 2] Resuming session...")
    print("="*60)

    try:
        # This is the critical test - resume() should handle missing logs gracefully
        ops_resumed = Orchestrator.resume(session_id)
        print(f"[SUCCESS] Resume worked! Loaded session: {session_id}")
        print(f"[OK] Tasks loaded: {len(ops_resumed.graph_ops.modal.nodes)}")
        print(f"[OK] Events in log: {len(ops_resumed.log_ops.modal.generate_master_log(ops_resumed.graph_ops.modal))}")

        # Re-register agent in resumed session
        ops_resumed.agent.create(
            id="solo",
            describe="General assistant",
            tools=["handoff"]
        )

        # Continue with another task
        print("\n[TURN 2] What is 3 + 3?")
        result2 = await ops_resumed.start_root_task(
            task="What is 3 + 3?",
            main_agent="solo"
        )

        print(f"[TURN 2] Answer: {result2.get('result', 'N/A')}")

        # Verify both tasks are in the graph
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        nodes = ops_resumed.graph_ops.modal.nodes
        print(f"\n[OK] Total tasks after resume: {len(nodes)}")
        print("[OK] Expected: 2 root tasks")

        # Check both are root tasks
        root_tasks = [n for n in nodes.values() if n.parent_task_id is None]
        print(f"\n[OK] Root tasks found: {len(root_tasks)}")

        for i, task in enumerate(root_tasks, 1):
            print(f"\n  Task {i}: {task.task_id}")
            print(f"    Payload: {task.task_payload[:50]}..." if len(task.task_payload) > 50 else f"    Payload: {task.task_payload}")
            print(f"    Status: {task.status}")

        print("\n[SUCCESS] Resume() works correctly with missing logs!")
        print(f"[SUCCESS] Session preserved across resume: {session_id}")

    except FileNotFoundError as e:
        print(f"[FAIL] Resume crashed with FileNotFoundError: {e}")
        print("[FAIL] This means the fix didn't work!")
        raise
    except Exception as e:
        print(f"[FAIL] Resume failed with unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())

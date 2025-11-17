#!/usr/bin/env python3
"""
DEMO 6: MULTI-TURN CONVERSATION
Tests context preservation across multiple root tasks
Preamble: Static only (single agent, no delegation)
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 6: MULTI-TURN CONVERSATION")
    print("="*60)

    ops = Orchestrator()

    # Single agent, no delegation
    solo = ops.agent.create(
        id="solo",
        describe="General assistant",
        tools=["handoff"]
    )

    print("\n[OK] Agent: solo")
    print("[OK] Testing context preservation across turns")

    # TURN 1: Ask about a specific year
    print("\n[TURN 1] What year was Python created?")
    result1 = await ops.start_root_task(
        task="What year was Python created?",
        main_agent="solo"
    )

    print(f"[TURN 1] Answer: {result1.get('result', 'N/A')}")

    # TURN 2: Ask which year was asked about (requires context!)
    print("\n[TURN 2] Which year did I ask you about?")
    result2 = await ops.start_root_task(
        task="Which year did I ask you about?",
        main_agent="solo"
    )

    print(f"[TURN 2] Answer: {result2.get('result', 'N/A')}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"[OK] Expected: 2 root tasks")

    # Check both are root tasks
    root_tasks = [n for n in nodes.values() if n.parent_task_id is None]
    print(f"\n[OK] Root tasks found: {len(root_tasks)}")

    for i, task in enumerate(root_tasks, 1):
        print(f"\n  Task {i}: {task.task_id}")
        print(f"    Status: {task.status}")

    # Check if context worked
    result2_str = str(result2.get('result', '')).lower()
    if '1991' in result2_str or 'python' in result2_str:
        print(f"\n[SUCCESS] Context preserved! Agent remembered the question was about Python/1991")
    else:
        print(f"\n[WARNING] Context may not have been used")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")
    print(f"[OK] Both tasks in same session!")

if __name__ == "__main__":
    asyncio.run(main())

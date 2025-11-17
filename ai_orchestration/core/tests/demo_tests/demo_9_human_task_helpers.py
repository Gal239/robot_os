#!/usr/bin/env python3
"""
DEMO 9: HUMAN TASK HELPER METHODS
Shows agent_orc_web app workflow for human ask_master:
1. Human starts task
2. Agent asks human (ask_master)
3. Web detects pending question
4. Human answers
5. Task completes
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator


async def main():
    print("="*60)
    print("DEMO 9: HUMAN TASK HELPER METHODS")
    print("="*60)

    ops = Orchestrator()

    # Create agent that asks questions
    assistant = ops.agent.create(
        id="assistant",
        describe="Assistant that asks user when it doesn't know",
        tools=["handoff", "ask_master"],
        instructions="""You are a helpful assistant.

If you don't know something, use ask_master to ask the user.
After getting information, provide answer using handoff."""
    )

    print("\n[OK] Agent: assistant")
    print("[OK] Testing agent_orc_web workflow: detect question → answer → resume")

    # 1. Human starts task
    print("\n[STEP 1] Human starts task: 'What is my favorite color?'")
    root_id = ops.start_human_root_task(
        task="What is my favorite color?",
        main_agent="assistant"
    )

    # 2. Execute until agent asks human
    print("[STEP 2] Agent processing...")
    while True:
        ready = ops.graph_ops.get_ready_tasks()
        agent_ready = [t for t in ready if ops.graph_ops.modal.nodes[t].agent_id != "human"]
        if not agent_ready:
            break
        for task_id in agent_ready:
            ops.graph_ops.update_status(task_id, "running")
            await ops.run_single_task(task_id, ops.graph_ops.modal.nodes[task_id].agent_id)

    # 3. Web detects pending question (MOP: use helper method!)
    print("\n[STEP 3] Web app detects pending question...")
    pending = ops.graph_ops.modal.get_pending_human_task()
    if pending:
        human_task_id, human_node = pending
        print(f"✓ Question detected: '{human_node.task_payload}'")
        print(f"✓ Task ID: {human_task_id}")
        print("→ Web returns this to UI (non-blocking)")

        # 4. Human answers (next HTTP request)
        print("\n[STEP 4] User answers: 'Blue'")
        await ops.resume_from_human_answer(human_task_id, {"answer": "Blue"})

    # 5. Verify completion
    result = ops.graph_ops.modal.nodes[root_id].result
    print(f"\n[STEP 5] Task completed!")
    print(f"✓ Result: {result}")

    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes
    print(f"\n[OK] Total tasks: {len(nodes)}")

    # Count task types
    root_tasks = [n for n in nodes.values() if n.tool_type.value == "root"]
    ask_master_tasks = [n for n in nodes.values() if n.tool_type.value == "ask_master"]

    print(f"[OK] Root tasks: {len(root_tasks)}")
    print(f"[OK] Ask_master tasks: {len(ask_master_tasks)}")

    # Verify all completed
    incomplete = [n for n in nodes.values() if n.status != "completed"]
    if not incomplete:
        print(f"\n[✓] All tasks completed!")
    else:
        print(f"\n[WARNING] {len(incomplete)} incomplete tasks")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")
    print(f"[OK] Graph saved to: ai_orchestration/databases/runs/{ops.graph_ops.modal.session_id}/")

    print("\n" + "="*60)
    print("✅ DEMO 9 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
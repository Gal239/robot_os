#!/usr/bin/env python3
"""
DEMO 7: ASK_MASTER + MULTI-TURN CONVERSATION
Tests ask_master usage + context preservation across turns
Agent should use ask_master when it doesn't know something
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 7: ASK_MASTER + MULTI-TURN CONVERSATION")
    print("="*60)

    ops = Orchestrator()

    # Single agent with ask_master capability
    assistant = ops.agent.create(
        id="assistant",
        describe="Helpful assistant that asks user when it doesn't know something",
        tools=["handoff", "ask_master"],
        instructions="""You are a helpful assistant.

IMPORTANT: If you don't know something, use ask_master to ask the user.
For example, if the user asks about their name and you don't know it,
use ask_master to ask them "What is your name?".

After getting information, provide a complete answer using handoff."""
    )

    print("\n[OK] Agent: assistant")
    print("[OK] Testing ask_master + multi-turn context")

    # TURN 1: Ask about user's name (agent doesn't know, should ask_master)
    print("\n[TURN 1] What is my name?")
    print("[NOTE] Agent should use ask_master to ask you")

    # Human initiates task (creates delegation)
    root_id_1 = ops.start_human_root_task(
        task="What is my name?",
        main_agent="assistant"
    )

    # Execute until ask_master is ready
    while True:
        ready = ops.graph_ops.get_ready_tasks()
        agent_ready = [t for t in ready if ops.graph_ops.modal.nodes[t].agent_id != "human"]
        if not agent_ready:
            break
        for task_id in agent_ready:
            ops.graph_ops.update_status(task_id, "running")
            await ops.run_single_task(task_id, ops.graph_ops.modal.nodes[task_id].agent_id)

    # Detect pending human task
    pending = ops.graph_ops.modal.get_pending_human_task()
    if pending:
        human_task_id, human_node = pending
        print(f"[ASK_MASTER] Agent asks: '{human_node.task_payload}'")

        # Human answers
        print(f"[HUMAN] Answer: 'Alice'")
        await ops.resume_from_human_answer(human_task_id, {"answer": "Alice"})

    # Get result
    result1 = ops.graph_ops.modal.nodes[root_id_1].result
    print(f"[TURN 1] Answer: {result1.get('result', 'N/A')}")

    # TURN 2: Ask what the agent didn't know (requires context from Turn 1!)
    print("\n[TURN 2] What didn't you know, that you had to ask me?")
    print("[NOTE] Agent should remember it asked about your name using ask_master")

    # Human initiates second task
    root_id_2 = ops.start_human_root_task(
        task="What didn't you know, that you had to ask me?",
        main_agent="assistant"
    )

    # Execute task (no ask_master expected this time)
    while True:
        ready = ops.graph_ops.get_ready_tasks()
        agent_ready = [t for t in ready if ops.graph_ops.modal.nodes[t].agent_id != "human"]
        if not agent_ready:
            break
        for task_id in agent_ready:
            ops.graph_ops.update_status(task_id, "running")
            await ops.run_single_task(task_id, ops.graph_ops.modal.nodes[task_id].agent_id)

    # Get result
    result2 = ops.graph_ops.modal.nodes[root_id_2].result
    print(f"[TURN 2] Answer: {result2.get('result', 'N/A')}")

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"[OK] Expected: 2 root tasks + ask_master subtasks")

    # Check root tasks (parent_task_id="human" for human-initiated tasks)
    root_tasks = [n for n in nodes.values() if n.tool_type.value == "root"]
    print(f"\n[OK] Root tasks found: {len(root_tasks)}")

    for i, task in enumerate(root_tasks, 1):
        print(f"\n  Task {i}: {task.task_id}")
        print(f"    Status: {task.status}")

        # Check if used ask_master
        ask_master_calls = [e for e in task.tool_timeline if e.get("tool") == "ask_master"]
        if ask_master_calls:
            print(f"    Used ask_master: YES ({len(ask_master_calls)} times)")
        else:
            print(f"    Used ask_master: NO")

    # Check if Turn 2 context worked
    result2_str = str(result2.get('result', '')).lower()
    if 'name' in result2_str or 'ask' in result2_str:
        print(f"\n[SUCCESS] Context preserved! Agent remembered using ask_master for name")
    else:
        print(f"\n[WARNING] Context may not have been used")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")
    print(f"[OK] All tasks in same session with full context!")

if __name__ == "__main__":
    asyncio.run(main())

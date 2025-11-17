#!/usr/bin/env python3
"""
DEMO 10: COMPREHENSIVE HUMAN ORCHESTRATION TEST
Tests all human interaction patterns at different depths:
- Human delegation to agent (start_human_root_task)
- Agent ask_master at depth 1
- Agent→Agent→Human ask_master at depth 2
- Human handoff (answer_human_task)
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*80)
    print("DEMO 10: COMPREHENSIVE HUMAN ORCHESTRATION TEST")
    print("="*80)

    ops = Orchestrator()

    # Create agents
    coordinator = ops.agent.create(
        id="coordinator",
        describe="Coordinator that delegates to specialist and asks human when needed",
        tools=["handoff", "ask_master", "route_to_specialist"],
        instructions="""You coordinate tasks. You can:
- Ask the human for clarification using ask_master
- Delegate to specialist agent using route_to_specialist
- Complete tasks with handoff"""
    )

    specialist = ops.agent.create(
        id="specialist",
        describe="Specialist that processes requests and may ask coordinator for help",
        tools=["handoff", "ask_master"],
        instructions="""You are a specialist. You can:
- Ask your coordinator (parent) for help using ask_master
- Complete tasks with handoff"""
    )

    print("\n[OK] Agents created:")
    print("  - coordinator (can delegate to specialist, ask human)")
    print("  - specialist (can ask coordinator)")

    # ========================================================================
    # TEST 1: Human → Agent (depth 1 ask_master)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 1: DEPTH 1 - Human → Coordinator → Human")
    print("="*80)

    print("\n[HUMAN] Starting task: 'What is my favorite color?'")

    # Human delegation
    root_id_1 = ops.start_human_root_task(
        task="What is my favorite color?",
        main_agent="coordinator"
    )

    # Execute until ask_master
    while True:
        ready = ops.graph_ops.get_ready_tasks()
        agent_ready = [t for t in ready if ops.graph_ops.modal.nodes[t].agent_id != "human"]
        if not agent_ready:
            break
        for task_id in agent_ready:
            ops.graph_ops.update_status(task_id, "running")
            await ops.run_single_task(task_id, ops.graph_ops.modal.nodes[task_id].agent_id)

    # Check for ask_master
    pending = ops.graph_ops.modal.get_pending_human_task()
    if pending:
        human_task_id, human_node = pending
        print(f"\n[COORDINATOR → HUMAN] '{human_node.task_payload}'")
        print(f"[HUMAN] Answering: 'Blue'")

        # Human answers and resumes
        await ops.resume_from_human_answer(human_task_id, {"answer": "Blue"})

    # Get result
    result1 = ops.graph_ops.modal.nodes[root_id_1].result
    print(f"\n[✓] Test 1 Result: {result1}")

    # ========================================================================
    # TEST 2: Human → Agent → Agent → Human (depth 2 ask_master)
    # ========================================================================
    print("\n" + "="*80)
    print("TEST 2: DEPTH 2 - Human → Coordinator → Specialist → Coordinator → Human")
    print("="*80)

    print("\n[HUMAN] Starting task: 'Process special request X'")

    # Human delegation
    root_id_2 = ops.start_human_root_task(
        task="Process special request X that requires specialist help",
        main_agent="coordinator"
    )

    # Execute coordinator (should delegate to specialist)
    while True:
        ready = ops.graph_ops.get_ready_tasks()
        agent_ready = [t for t in ready if ops.graph_ops.modal.nodes[t].agent_id != "human"]
        if not agent_ready:
            break
        for task_id in agent_ready:
            node = ops.graph_ops.modal.nodes[task_id]
            if node.agent_id == "coordinator" and node.status == "ready":
                print(f"\n[COORDINATOR] Processing task...")
                ops.graph_ops.update_status(task_id, "running")
                await ops.run_single_task(task_id, node.agent_id)
                break
            elif node.agent_id == "specialist" and node.status == "ready":
                print(f"\n[SPECIALIST] Processing delegated task...")
                ops.graph_ops.update_status(task_id, "running")
                await ops.run_single_task(task_id, node.agent_id)
                break
        else:
            break

    # Check for ask_master (specialist might ask coordinator, who asks human)
    pending = ops.graph_ops.modal.get_pending_human_task()
    if pending:
        human_task_id, human_node = pending
        print(f"\n[AGENT → HUMAN] '{human_node.task_payload}'")
        print(f"[HUMAN] Answering: 'Approved'")

        # Human answers and resumes
        await ops.resume_from_human_answer(human_task_id, {"answer": "Approved"})

    # Get result
    result2 = ops.graph_ops.modal.nodes[root_id_2].result
    print(f"\n[✓] Test 2 Result: {result2}")

    # ========================================================================
    # VALIDATION
    # ========================================================================
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)

    nodes = ops.graph_ops.modal.nodes

    print(f"\n[OK] Total tasks: {len(nodes)}")

    # Count by type
    root_tasks = [n for n in nodes.values() if n.tool_type.value == "root"]
    ask_master_tasks = [n for n in nodes.values() if n.tool_type.value == "ask_master"]
    delegation_tasks = [n for n in nodes.values() if n.tool_type.value == "agent_as_tool"]

    print(f"\n[OK] Root tasks: {len(root_tasks)}")
    print(f"[OK] Ask_master tasks: {len(ask_master_tasks)}")
    print(f"[OK] Delegation tasks: {len(delegation_tasks)}")

    # Verify all tasks completed
    incomplete = [n for n in nodes.values() if n.status != "completed"]
    if incomplete:
        print(f"\n[WARNING] {len(incomplete)} incomplete tasks:")
        for task in incomplete:
            print(f"  - {task.task_id}: {task.agent_id} ({task.status})")
    else:
        print(f"\n[✓] All tasks completed successfully!")

    # Verify edges exist (delegation, handoff, ask_master)
    edge_count = {"delegation": 0, "handoff": 0, "ask_master": 0}

    for node in nodes.values():
        # Count timeline events
        for event in node.tool_timeline:
            event_type = event.get("type")
            if event_type == "agent_as_tool":
                edge_count["delegation"] += 1
            elif event_type == "handoff":
                edge_count["handoff"] += 1
            elif event_type == "ask_master":
                edge_count["ask_master"] += 1

    print(f"\n[OK] Edge events:")
    print(f"  - Delegation: {edge_count['delegation']}")
    print(f"  - Handoff: {edge_count['handoff']}")
    print(f"  - Ask_master: {edge_count['ask_master']}")

    # Session info
    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")
    print(f"[OK] All tasks in same session with full orchestration!")

    print("\n" + "="*80)
    print("✅ DEMO 10 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
DEMO 1: NORMAL TEAM
3 different agents with different roles
Preamble: Static + Delegation (no recursive)
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 1: NORMAL TEAM")
    print("="*60)

    ops = Orchestrator()

    # Create specialized team
    coordinator = ops.agent.create(
        id="coordinator",
        describe="Project coordinator",
        tools=["route_to_researcher", "route_to_writer", "handoff"]
    )

    researcher = ops.agent.create(
        id="researcher",
        describe="Researches topics",
        tools=["search_web", "write_file", "handoff"]
    )

    writer = ops.agent.create(
        id="writer",
        describe="Creates documents",
        tools=["write_file", "ask_data", "handoff"]
    )

    print("\n[OK] Agents: coordinator, researcher, writer")

    # Generic task - NO hand-holding!
    result = await ops.start_root_task(
        task="Research quantum computing trends and create a summary report",
        main_agent="coordinator"
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes
    agents_used = {}
    for node in nodes.values():
        agents_used[node.agent_id] = agents_used.get(node.agent_id, 0) + 1

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"\n[DISTRIBUTION] Agent distribution:")
    for agent_id, count in agents_used.items():
        print(f"  {agent_id}: {count} task(s)")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")

if __name__ == "__main__":
    asyncio.run(main())

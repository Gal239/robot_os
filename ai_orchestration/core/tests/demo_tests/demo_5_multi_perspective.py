#!/usr/bin/env python3
"""
DEMO 5: MULTI-PERSPECTIVE ANALYSIS
Agent analyzes from multiple perspectives by calling itself
Preamble: Static + Delegation + Recursive
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 5: MULTI-PERSPECTIVE ANALYSIS")
    print("="*60)

    ops = Orchestrator()

    # Agent with route_to_self
    analyst = ops.agent.create(
        id="analyst",
        describe="Analyzes topics from multiple viewpoints",
        tools=["route_to_analyst", "write_file", "handoff"]  # Can call itself!
    )

    print("\n[OK] Agent: analyst")
    print("[OK] Has route_to_analyst (can analyze from multiple angles!)")

    # Generic task - agent should discover multi-perspective pattern
    result = await ops.start_root_task(
        task="Analyze the impact of remote work on: productivity, mental health, company culture",
        main_agent="analyst"
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes
    analyst_tasks = [n for n in nodes.values() if n.agent_id == "analyst"]

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"[OK] Analyst instances: {len(analyst_tasks)}")

    if len(analyst_tasks) > 1:
        print(f"\n[SUCCESS] MULTI-PERSPECTIVE DETECTED! Analyst spawned {len(analyst_tasks)} instances")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")

if __name__ == "__main__":
    asyncio.run(main())

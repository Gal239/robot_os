#!/usr/bin/env python3
"""
DEMO 4: SELF-REVIEWING CODE AGENT
Agent writes code then reviews its own code
Preamble: Static + Delegation + Recursive
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 4: SELF-REVIEWING CODE")
    print("="*60)

    ops = Orchestrator()

    # Agent with route_to_self
    developer = ops.agent.create(
        id="developer",
        describe="Writes and reviews code",
        tools=["route_to_developer", "write_file", "ask_data", "handoff"]  # Can call itself!
    )

    print("\n[OK] Agent: developer")
    print("[OK] Has route_to_developer (can review own code!)")

    # Generic task - agent should discover write then self-review pattern
    result = await ops.start_root_task(
        task="Implement binary search in Python with proper error handling",
        main_agent="developer"
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes
    dev_tasks = [n for n in nodes.values() if n.agent_id == "developer"]

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"[OK] Developer instances: {len(dev_tasks)}")

    if len(dev_tasks) > 1:
        print(f"\n[SUCCESS] SELF-REVIEW DETECTED! Developer reviewed own code")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")

if __name__ == "__main__":
    asyncio.run(main())

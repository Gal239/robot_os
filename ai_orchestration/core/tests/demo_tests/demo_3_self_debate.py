#!/usr/bin/env python3
"""
DEMO 3: SELF-DEBATING AGENT
Agent can route to itself for debates
Preamble: Static + Delegation + Recursive
"""


import asyncio
from ai_orchestration.core.orchestrator import Orchestrator

async def main():
    print("="*60)
    print("DEMO 3: SELF-DEBATING AGENT")
    print("="*60)

    ops = Orchestrator()

    # Agent with route_to_self
    debater = ops.agent.create(
        id="debater",
        describe="Debates topics from multiple angles",
        tools=["route_to_debater", "handoff"]  # Can call itself!
    )

    print("\n[OK] Agent: debater")
    print("[OK] Has route_to_debater (can call itself!)")

    # Generic task - agent should discover it can debate with itself
    result = await ops.start_root_task(
        task="Analyze: are cat better than dogs? Consider all perspectives.-use backand-forth delegation to debate youself results and expand idea and do deslf convstion and the goal is to debate to find the best answer and not just delegate and forget it a debae so you ned to agfre disagfger and use youself to make the debate better likje you dont must tell him waht do do you can like say you thignk taht delgeat to see his thgouht tahn he can hadfod you disn if you have anotehr oint you want to debner if yu agferte on rpeoves point nad so on fell free to be hearded abd curese and getg mad when need dont be offical this is a undegroaud debarel be exreme and peronal dont be polite but be funny mad and sacrastin but like curse the shit outof him youa re brither it fine one is a supre act lover one is dog lover both are throgin knife at eahc other use asj amster to give your asnwer if yiuy dekega to adn use deaogea if you want toe epxan so use delagea ask msater adn ahdfoo coorelty to make the debate better use hsort jabs ans arguemnet like twitter thread argument debater it imoroente to use ask_msagtet!!! and not just hadnfoff and dealgate so use hadnfofo ask msater and deakget to debate stargiclly",
        main_agent="debater"
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    nodes = ops.graph_ops.modal.nodes
    debater_tasks = [n for n in nodes.values() if n.agent_id == "debater"]

    print(f"\n[OK] Total tasks: {len(nodes)}")
    print(f"[OK] Debater instances: {len(debater_tasks)}")

    if len(debater_tasks) > 1:
        print(f"\n[SUCCESS] RECURSIVE DETECTED! Debater called itself {len(debater_tasks)} times")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")

if __name__ == "__main__":
    asyncio.run(main())

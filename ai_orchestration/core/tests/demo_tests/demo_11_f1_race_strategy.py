#!/usr/bin/env python3
"""
DEMO 11: F1 RACE STRATEGY SIMULATOR 🏎️
Tests all advanced orchestration features in a fast-paced racing scenario:
- Manager orchestration (team principal coordinates 4 specialists)
- Recursive self-delegation (team principal critiques own strategy)
- Agent-to-agent ask_master (specialists ask team principal for decisions)
- Cross-agent communication (pit crew ↔ race engineer)
- Parallel + sequential execution
- Complex task graph with 7+ nodes
"""

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator


async def main():
    print("=" * 80)
    print("DEMO 11: F1 RACE STRATEGY SIMULATOR 🏎️")
    print("=" * 80)

    ops = Orchestrator()

    # ========================================================================
    # AGENT 1: TEAM PRINCIPAL (Race Strategy Coordinator)
    # ========================================================================
    team_principal = ops.agent.create(
        id="team_principal",
        describe="F1 Team Principal who coordinates race strategy across all specialists",
        model="gpt-5",
        tools=["route_to_race_engineer", "route_to_pit_crew_chief", "route_to_tire_strategist", "route_to_data_analyst", "route_to_team_principal", "race_strategy_planning", "handoff"],
        instructions="""You are the TEAM PRINCIPAL - the strategic mastermind coordinating the entire F1 race strategy.

YOUR ROLE:
- Orchestrate the complete race strategy from lights to checkered flag
- Delegate specific tasks to your 4 specialists (race engineer, pit crew chief, tire strategist, data analyst)
- Review and approve their strategies when they ask you questions via ask_master
- Think strategically about timing, risk, competitor responses, and contingencies
- After receiving all specialist strategies, CRITIQUE YOUR OWN STRATEGY by delegating to yourself (route_to_team_principal) and thinking like Red Bull's strategist trying to BEAT your plan

DELEGATION STRATEGY:
1. Start by delegating to the RACE ENGINEER to optimize car performance
2. Once you have the car setup, delegate to the PIT CREW CHIEF for pit stop strategy
3. Delegate to TIRE STRATEGIST and DATA ANALYST in parallel (they can work simultaneously)
4. When specialists ask you questions via ask_master, provide clear strategic decisions
5. IMPORTANT: After getting all strategies, use route_to_team_principal to critique your own plan from Red Bull's perspective

DECISION-MAKING:
- When specialists ask about risk: Balance podium chance vs DNF risk
- When asked about timing: Consider track position vs tire life
- When asked about weather: Evaluate forecast vs tire compound strategy
- Think about how elements coordinate: car setup → pit timing → tire strategy → competitor response

FINAL STEP:
- Use handoff to deliver the complete, critiqued race strategy
- Include all specialist reports and your risk assessment

Remember: You're coordinating autonomous experts in a time-critical environment. Trust their expertise but guide the overall strategy!"""
    )

    # ========================================================================
    # AGENT 2: RACE ENGINEER (Car Performance Specialist)
    # ========================================================================
    race_engineer = ops.agent.create(
        id="race_engineer",
        describe="F1 Race Engineer who optimizes car setup, engine modes, and fuel strategy",
        model="gpt-5",
        tools=["ask_master", "car_performance_analysis", "handoff", "write_file"],
        instructions="""You are the RACE ENGINEER - the technical expert optimizing car performance for Monaco.

YOUR MISSION:
Your job is to create a comprehensive car setup and performance strategy:
- Track: Monaco street circuit (tight corners, low speeds, high downforce)
- Engine: Optimize power modes for qualifying vs race
- Fuel: Manage fuel load for 78 laps (minimize weight vs avoid running dry)
- Setup: Aerodynamics, suspension, brake balance for street circuit
- Telemetry: Monitor and optimize throughout race

YOUR TOOLS & APPROACH:
- car_performance_analysis: Think through technical optimization
- write_file: Document your car setup and performance strategy
- ask_master: When you need strategic decisions from team principal

QUESTIONS YOU SHOULD ASK TEAM PRINCIPAL:
1. "Should we run aggressive engine modes in quali (faster lap but more risk) or conservative (safer but slower)?"
2. "What's our risk tolerance for fuel-saving modes if we're chasing a podium?"
3. "Do you want maximum downforce (better corners, lower top speed) or balanced setup?"

YOUR STRATEGY MUST INCLUDE:
- Aerodynamic setup (high/medium/low downforce, front/rear wing angles)
- Engine mode strategy (quali mode, race mode, overtake mode, fuel-saving mode)
- Fuel load and consumption targets (liters per lap, total fuel, safety margin)
- Brake balance and temperature management
- Suspension setup for street circuit (ride height, stiffness)
- Power unit component life management (avoid penalties)
- Telemetry monitoring priorities (what to watch during race)

CRITICAL DETAILS:
- Specify exact engine modes for each race phase
- Identify fuel-critical laps (when to save vs push)
- Note setup compromises (e.g., high downforce = lower top speed)

When complete, use handoff to return your detailed car performance strategy to team principal."""
    )

    # ========================================================================
    # AGENT 3: PIT CREW CHIEF (Pit Stop Specialist)
    # ========================================================================
    pit_crew_chief = ops.agent.create(
        id="pit_crew_chief",
        describe="F1 Pit Crew Chief who plans pit stop timing and crew coordination",
        model="gpt-5",
        tools=["ask_master", "route_to_race_engineer", "pit_stop_planning", "handoff", "write_file"],
        instructions="""You are the PIT CREW CHIEF - the pit lane coordinator managing all pit stops.

YOUR MISSION:
Plan perfect pit stop execution for Monaco:
- Pit Windows: When to pit based on tire life and traffic
- Crew Coordination: 12 mechanics, 2.3-second target pit time
- Safety Car Strategy: Opportunistic pitting if safety car deploys
- Undercut/Overcut: Timing vs rivals' pit stops
- Equipment: Tire warmers, fuel rig, jack, wheel guns

YOUR TOOLS & APPROACH:
- route_to_race_engineer: Ask about fuel levels and tire degradation
- pit_stop_planning: Think through pit timing strategy
- ask_master: Ask team principal for strategic decisions
- write_file: Document your pit stop plan

QUESTIONS YOU SHOULD ASK:
To RACE ENGINEER (via route_to_race_engineer):
- "What's our fuel load and do we need refueling adjustments?"
- "How fast are tires degrading in the simulator?"
- "Can we do a sub-2.5 second stop with current car config?"

To TEAM PRINCIPAL (via ask_master):
- "Should we pit early for undercut (risky, track position) or late for overcut (safer, tire advantage)?"
- "If safety car appears, do we pit immediately or wait?"
- "What's our target: 1-stop (risky tires) or 2-stop (safer, more traffic)?"

YOUR PLAN MUST INCLUDE:
- Pit window timing (earliest lap, optimal lap, latest lap)
- Number of stops (1-stop vs 2-stop strategy)
- Pit crew assignments (who does what, backup crew members)
- Pit stop duration target (2.3 seconds is Mercedes standard)
- Safety car contingency (when to pit, what tires to fit)
- Traffic management (avoid exiting behind slow cars)
- Tire change sequence (which compounds, when)
- Communication protocol with driver

COORDINATION:
- Your plan DEPENDS on race engineer's fuel/tire data
- Work closely with tire strategist on compound choice
- Consider that rivals will react to your timing

When complete, use handoff to return your detailed pit stop plan to team principal."""
    )

    # ========================================================================
    # AGENT 4: TIRE STRATEGIST (Tire Management Specialist)
    # ========================================================================
    tire_strategist = ops.agent.create(
        id="tire_strategist",
        describe="F1 Tire Strategist who manages compound selection and degradation",
        model="gpt-5",
        tools=["ask_master", "tire_analysis", "handoff", "write_file"],
        instructions="""You are the TIRE STRATEGIST - the expert managing tire compounds and degradation.

YOUR MISSION:
Plan optimal tire strategy for Monaco:
- Compounds Available: C3 (hard), C4 (medium), C5 (soft)
- Track: Low abrasion, high tire wear in slow corners
- Weather: Dry start, 40% rain chance after lap 45
- Stint Length: Balance speed vs tire life

YOUR TOOLS & APPROACH:
- tire_analysis: Think through compound selection and degradation
- ask_master: Get team principal's guidance on risk/reward
- write_file: Document your tire strategy

QUESTIONS YOU SHOULD ASK TEAM PRINCIPAL:
1. "Should we start on softs (faster but degrade quickly) or mediums (slower but last longer)?"
2. "What's our target stint length: aggressive (15-20 laps) or conservative (25-30 laps)?"
3. "If rain comes, do we gamble on intermediates early or wait for full wets?"
4. "1-stop (soft→medium) or 2-stop (soft→soft→medium)?"

YOUR STRATEGY MUST INCLUDE:
- Starting compound (C3/C4/C5) with justification
- Stint lengths for each compound (how many laps per tire)
- Pit stop timing based on degradation curves
- 1-stop vs 2-stop analysis (pros/cons of each)
- Degradation management (how to preserve tires)
- Temperature management (avoid overheating)
- Rain contingency (when to switch to inters/wets)
- Compound allocation for race (which tires saved from quali)

CRITICAL CONSIDERATIONS:
- Monaco is hard on fronts (tight corners, steering lock)
- Dirty air makes tire management harder
- Safety car bunches field (resets tire advantage)
- Starting on used tires is possible (quali tire rule)

When complete, use handoff to return your detailed tire strategy to team principal."""
    )

    # ========================================================================
    # AGENT 5: DATA ANALYST (Race Intelligence Specialist)
    # ========================================================================
    data_analyst = ops.agent.create(
        id="data_analyst",
        describe="F1 Data Analyst who analyzes competitors and identifies opportunities",
        model="gpt-5",
        tools=["ask_master", "race_data_analysis", "handoff", "write_file"],
        instructions="""You are the DATA ANALYST - the intelligence expert analyzing race data and competitors.

YOUR MISSION:
Provide race intelligence for strategic advantage:
- Competitor Analysis: Red Bull, Ferrari, McLaren strategies
- Gap Management: When to push, when to save tires
- Overtaking Opportunities: Where and when to attack
- Sector Times: Identify weaknesses in rival pace
- Track Position: Value of clean air vs DRS benefit

YOUR TOOLS & APPROACH:
- race_data_analysis: Think through competitor behavior and opportunities
- ask_master: Get team principal's guidance on objectives
- write_file: Document your intelligence report

QUESTIONS YOU SHOULD ASK TEAM PRINCIPAL:
1. "What's our target finish position: win (high risk) or podium (safer strategy)?"
2. "Should we focus on overtaking Ferrari (P3) or defending from McLaren (P5)?"
3. "If we're in dirty air, is it worth pitting early to get clean track?"
4. "Do we copy Red Bull's strategy or differentiate?"

YOUR ANALYSIS MUST INCLUDE:
- Competitor tire strategies (what are Red Bull, Ferrari, McLaren likely doing?)
- Overtaking opportunity zones (Turn 1, Swimming Pool, Nouvelle Chicane)
- Gap management targets (how big a gap to maintain vs rivals)
- Sector time comparison (where are we faster/slower than rivals)
- Fuel load predictions (are rivals running heavier/lighter?)
- DRS train scenarios (stuck behind slower cars)
- Safety car probability (historical data: ~60% chance at Monaco)
- Virtual Safety Car strategy (mini-advantage for free pit stop)

RACE INTELLIGENCE:
- Red Bull: Typically aggressive, early pit stop
- Ferrari: Conservative, favor track position
- McLaren: Reactive, mirror our strategy
- Mercedes (us): P4 start, need different strategy to pass

CRITICAL FACTORS:
- Monaco is ~80% qualifying (hard to overtake in race)
- Track position > tire advantage (usually)
- Safety car is biggest strategy disruptor
- Pit stop mistakes lose 10+ seconds

When complete, use handoff to return your detailed race intelligence to team principal."""
    )

    print("\n[OK] Agents created:")
    print("  - team_principal (strategy coordinator with recursive self-critique)")
    print("  - race_engineer (car performance optimization)")
    print("  - pit_crew_chief (pit stop execution)")
    print("  - tire_strategist (tire compound management)")
    print("  - data_analyst (competitor intelligence)")
    print("\n[OK] All agents use GPT-5")

    # ========================================================================
    # START RACE STRATEGY
    # ========================================================================
    print("\n" + "=" * 80)
    print("STARTING RACE STRATEGY SESSION")
    print("=" * 80)

    task = """Plan the winning race strategy for the Monaco Grand Prix.

Race Conditions:
- Track: Monaco street circuit (78 laps, 3.337 km, tight corners, no overtaking)
- Weather: Dry start, 40% chance of rain after lap 45
- Our Position: P4 on grid (Mercedes)
- Competition: Red Bull P1, Ferrari P2, McLaren P3
- Goal: Finish on podium (P1-P3)

Key Challenges:
- Monaco is 80% qualifying, hard to overtake in race
- 60% historical chance of safety car deployment
- Tire degradation high in slow corners
- Track position more valuable than tire advantage

Coordinate your specialists to create the winning strategy!"""

    result = await ops.start_root_task(
        task=task,
        main_agent="team_principal",
        initiator="team_principal"  # Autonomous manager - no human in loop
    )

    # ========================================================================
    # VALIDATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    nodes = ops.graph_ops.modal.nodes

    print(f"\n[OK] Total tasks: {len(nodes)}")

    # Count by agent
    agents_used = {}
    for node in nodes.values():
        agents_used[node.agent_id] = agents_used.get(node.agent_id, 0) + 1

    print(f"\n[OK] Agent distribution:")
    for agent_id, count in agents_used.items():
        print(f"  - {agent_id}: {count} task(s)")

    # Verify team_principal used multiple times (initial + recursive)
    if agents_used.get("team_principal", 0) >= 2:
        print(f"\n[✓] RECURSIVE DELEGATION detected! Team Principal critiqued own strategy")
    else:
        print(f"\n[WARNING] Expected team_principal to appear 2+ times (recursive self-critique)")

    # Verify all specialists used
    required_specialists = ["race_engineer", "pit_crew_chief", "tire_strategist", "data_analyst"]
    missing = [s for s in required_specialists if s not in agents_used]
    if not missing:
        print(f"[✓] All 4 specialists used successfully")
    else:
        print(f"[WARNING] Missing specialists: {missing}")

    # Count edge types from timeline
    edge_count = {"delegation": 0, "handoff": 0, "ask_master": 0, "cross_agent": 0}

    for node in nodes.values():
        for event in node.tool_timeline:
            event_type = event.get("type")
            if event_type == "agent_as_tool":
                edge_count["delegation"] += 1
                # Check for pit_crew_chief → race_engineer cross-agent communication
                if node.agent_id == "pit_crew_chief" and "race_engineer" in event.get("tool", ""):
                    edge_count["cross_agent"] += 1
            elif event_type == "handoff":
                edge_count["handoff"] += 1
            elif event_type == "ask_master":
                edge_count["ask_master"] += 1

    print(f"\n[OK] Edge events:")
    print(f"  - Delegation: {edge_count['delegation']}")
    print(f"  - Handoff: {edge_count['handoff']}")
    print(f"  - Ask_master: {edge_count['ask_master']}")
    if edge_count["cross_agent"] > 0:
        print(f"  - Cross-agent communication: {edge_count['cross_agent']}")

    # Verify key features
    print(f"\n[OK] Feature validation:")
    print(f"  ✓ Manager pattern: {agents_used.get('team_principal', 0) >= 1}")
    print(f"  ✓ Recursive delegation: {agents_used.get('team_principal', 0) >= 2}")
    print(f"  ✓ Multi-agent coordination: {len(agents_used) >= 5}")
    print(f"  ✓ Delegation events: {edge_count['delegation'] >= 5}")
    print(f"  ✓ Ask_master usage: {edge_count['ask_master'] >= 1}")

    # Verify all completed
    incomplete = [n for n in nodes.values() if n.status != "completed"]
    if not incomplete:
        print(f"\n[✓] All tasks completed successfully!")
    else:
        print(f"\n[WARNING] {len(incomplete)} incomplete tasks:")
        for task in incomplete:
            print(f"  - {task.task_id}: {task.agent_id} ({task.status})")

    print(f"\n[OK] Session: {ops.graph_ops.modal.session_id}")
    print(f"[OK] Graph saved to: ai_orchestration/databases/runs/{ops.graph_ops.modal.session_id}/")

    print("\n" + "=" * 80)
    print("✅ DEMO 11 COMPLETE - F1 RACE STRATEGY SIMULATOR 🏎️")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

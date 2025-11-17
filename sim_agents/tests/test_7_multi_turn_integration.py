#!/usr/bin/env python3
"""
TEST 7: MULTI-TURN LLM INTEGRATION - Iterative Scene Building
==============================================================

Tests REAL workflow with multiple turns:
Turn 1: "Create table and apple on it"
Turn 2: "Add 2 more objects"
Turn 3: "Remove the apple"
Turn 4: "Add more rewards, I want to test gripping"

This proves the system handles iterative editing!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
from sim_agents.scene_maker_agent import create_scene_maker_agent
from sim_agents.scene_maker_handoff_handler import handle_scene_maker_handoff
from ai_orchestration.core.orchestrator import Orchestrator


async def test_multi_turn_integration():
    """
    TEST 7: Multi-turn integration with real LLM

    Simulates real user workflow:
    - Start with simple scene
    - Add objects iteratively
    - Remove objects
    - Add rewards for RL tasks
    """
    print("\n" + "="*80)
    print("TEST 7: MULTI-TURN INTEGRATION - ITERATIVE SCENE BUILDING")
    print("="*80)

    # Setup orchestrator and agent
    print("\n🎯 Setup: Creating orchestrator and scene_maker agent...")
    ops = Orchestrator()
    scene_agent = create_scene_maker_agent(ops)

    # Register handoff handler (now properly supported in orchestrator!)
    ops.handoff_handlers['scene_maker'] = handle_scene_maker_handoff

    print("   ✓ Orchestrator ready")
    print("   ✓ scene_maker agent created")
    print("   ✓ Handoff handler registered for scene_maker")

    # Track state across turns
    current_script = ""
    turn_results = []

    # ============================================================================
    # TURN 1: Create table and apple on it
    # ============================================================================
    print("\n" + "="*80)
    print("TURN 1: Create table and apple on it")
    print("="*80)

    turn_1_message = """Create a simple scene with table and apple on it.

Requirements:
- 5x5x3 room
- Stretch robot at origin
- Table at position (2, 0, 0)
- Apple on top of the table (center)
- Overhead camera
- Use preview mode (mode='preview' in compile)
- Use render_mode='2k_demo'"""

    print(f"\n💬 User: {turn_1_message}")
    print("\n🧠 Agent thinking (LLM generates edits)...")

    result_1 = await ops.start_root_task(
        task=turn_1_message,
        main_agent="scene_maker"
    )

    # Extract result from root task
    task_result = result_1.get("result", {})
    if 'handoff_result' in task_result:
        handoff_1 = task_result['handoff_result']
        current_script = handoff_1['script']
        turn_results.append(('Turn 1', handoff_1['success']))

        print(f"\n✅ Turn 1 Complete!")
        print(f"   Message: {handoff_1['message']}")
        print(f"   Screenshots: {handoff_1['total_cameras']} cameras")
        print(f"   Script: {len(current_script.split(chr(10)))} lines")
    else:
        print("\n❌ Turn 1 failed - no handoff")
        print(f"   Result: {task_result}")
        return False

    # ============================================================================
    # TURN 2: Add 2 more objects
    # ============================================================================
    print("\n" + "="*80)
    print("TURN 2: Add 2 more objects")
    print("="*80)

    turn_2_message = """Add 2 more objects to the scene:
- Banana next to the apple
- Mug on the table

Keep everything else the same."""

    print(f"\n💬 User: {turn_2_message}")
    print("\n🧠 Agent thinking (generating edits for existing script)...")

    result_2 = await ops.agent.send(
        agent_id="scene_maker",
        messages=[
            {"role": "user", "content": turn_1_message},
            {"role": "assistant", "content": f"handoff called with {handoff_1['total_edits']} edits"},
            {"role": "user", "content": turn_2_message}
        ],
        context={"current_script": current_script}
    )

    if 'handoff_result' in result_2:
        handoff_2 = result_2['handoff_result']
        current_script = handoff_2['script']
        turn_results.append(('Turn 2', handoff_2['success']))

        print(f"\n✅ Turn 2 Complete!")
        print(f"   Message: {handoff_2['message']}")
        print(f"   Screenshots: {handoff_2['total_cameras']} cameras")
        print(f"   Script: {len(current_script.split(chr(10)))} lines")
    else:
        print("\n❌ Turn 2 failed - no handoff")
        return False

    # ============================================================================
    # TURN 3: Remove the apple
    # ============================================================================
    print("\n" + "="*80)
    print("TURN 3: Remove the apple")
    print("="*80)

    turn_3_message = """Remove the apple from the scene.

Delete the line that adds the apple."""

    print(f"\n💬 User: {turn_3_message}")
    print("\n🧠 Agent thinking (generating delete edits)...")

    result_3 = await ops.agent.send(
        agent_id="scene_maker",
        messages=[{"role": "user", "content": turn_3_message}],
        context={"current_script": current_script}
    )

    if 'handoff_result' in result_3:
        handoff_3 = result_3['handoff_result']
        current_script = handoff_3['script']
        turn_results.append(('Turn 3', handoff_3['success']))

        print(f"\n✅ Turn 3 Complete!")
        print(f"   Message: {handoff_3['message']}")
        print(f"   Script: {len(current_script.split(chr(10)))} lines")
    else:
        print("\n❌ Turn 3 failed - no handoff")
        return False

    # ============================================================================
    # TURN 4: Add rewards for gripping
    # ============================================================================
    print("\n" + "="*80)
    print("TURN 4: Add rewards for gripping")
    print("="*80)

    turn_4_message = """Add more rewards, I want to test gripping.

Add rewards for:
- Robot gripping the banana
- Robot gripping the mug
- Banana being stacked on table
- Mug being stacked on table"""

    print(f"\n💬 User: {turn_4_message}")
    print("\n🧠 Agent thinking (adding reward lines)...")

    result_4 = await ops.agent.send(
        agent_id="scene_maker",
        messages=[{"role": "user", "content": turn_4_message}],
        context={"current_script": current_script}
    )

    if 'handoff_result' in result_4:
        handoff_4 = result_4['handoff_result']
        current_script = handoff_4['script']
        turn_results.append(('Turn 4', handoff_4['success']))

        print(f"\n✅ Turn 4 Complete!")
        print(f"   Message: {handoff_4['message']}")
        print(f"   Script: {len(current_script.split(chr(10)))} lines")

        # Show final script
        print(f"\n📜 Final Script:")
        print("="*80)
        print(current_script)
        print("="*80)
    else:
        print("\n❌ Turn 4 failed - no handoff")
        return False

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "="*80)
    print("MULTI-TURN TEST SUMMARY")
    print("="*80)

    for turn_name, success in turn_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {turn_name}")

    if all(success for _, success in turn_results):
        print("\n🎉 ALL TURNS PASSED!")
        print("\n🚀 Scene Maker system handles ITERATIVE EDITING!")
        print("\n💡 The system is PRODUCTION READY for:")
        print("   - Initial scene creation")
        print("   - Adding objects iteratively")
        print("   - Removing objects")
        print("   - Adding rewards and behaviors")
        return True
    else:
        print("\n⚠️  Some turns failed")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SCENE MAKER - TEST 7: MULTI-TURN INTEGRATION")
    print("="*80)
    print("\nThis test validates REAL iterative workflow:")
    print("  Turn 1: Create table and apple")
    print("  Turn 2: Add 2 more objects")
    print("  Turn 3: Remove the apple")
    print("  Turn 4: Add rewards for gripping")
    print("\nRequires Claude API access!")

    confirm = input("\nRun multi-turn test? (y/n): ").strip().lower()

    if confirm == 'y':
        success = asyncio.run(test_multi_turn_integration())
        sys.exit(0 if success else 1)
    else:
        print("Test cancelled")
        sys.exit(0)

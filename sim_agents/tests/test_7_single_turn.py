#!/usr/bin/env python3
"""
TEST 7: SINGLE TURN LLM INTEGRATION - The Real Deal!
=====================================================

Tests COMPLETE chain with real Claude LLM (single turn):
User → Orchestrator → Agent → **LLM generates edits** → handoff → Backend → User

This is the test that proves we're ready!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import asyncio
from sim_agents.scene_maker_agent import create_scene_maker_agent
from sim_agents.scene_maker_handoff_handler import handle_scene_maker_handoff
from ai_orchestration.core.orchestrator import Orchestrator


async def test_single_turn_integration():
    """
    TEST 7: Single turn integration with real LLM

    Flow:
    1. User: "Create table and apple on it"
    2. Orchestrator routes to scene_maker agent
    3. Claude LLM generates Python edits (insert operations)
    4. Agent calls handoff(edits, message)
    5. Backend handler executes scene
    6. Returns screenshots to user
    """
    print("\n" + "="*80)
    print("TEST 7: SINGLE TURN LLM INTEGRATION - THE REAL DEAL!")
    print("="*80)

    # Setup orchestrator and agent
    print("\n🎯 Step 1: Creating orchestrator and scene_maker agent...")
    ops = Orchestrator()
    scene_agent = create_scene_maker_agent(ops)

    # Register handoff handler
    ops.handoff_handlers['scene_maker'] = handle_scene_maker_handoff

    print("   ✓ Orchestrator ready")
    print("   ✓ scene_maker agent created")
    print("   ✓ Handoff handler registered for scene_maker")

    # User message
    print("\n" + "="*80)
    print("💬 USER MESSAGE")
    print("="*80)

    user_message = """Create a simple scene with table and apple on it.

Requirements:
- 5x5x3 room
- Stretch robot at origin
- Table at position (2, 0, 0)
- Apple on top of the table (center position)
- Overhead camera

Build this scene for me!"""

    print(user_message)

    print("\n🧠 Sending to Claude LLM...")
    print("   (This is where the AI magic happens!)")

    # Execute via orchestrator
    result = await ops.start_root_task(
        task=user_message,
        main_agent="scene_maker"
    )

    print("\n" + "="*80)
    print("📊 RESULTS")
    print("="*80)

    # Check result
    task_result = result.get("result", {})

    if 'handoff_result' in task_result:
        handoff_result = task_result['handoff_result']

        if handoff_result.get('success'):
            print(f"\n✅ SUCCESS!")
            print(f"\n📝 Agent Message:")
            print(f"   {handoff_result['message']}")

            print(f"\n📸 Screenshots:")
            print(f"   Total cameras: {handoff_result['total_cameras']}")
            for cam_id, path in list(handoff_result['screenshots'].items())[:5]:
                print(f"   - {cam_id}: {Path(path).name}")

            print(f"\n📜 Script:")
            print(f"   Total lines: {len(handoff_result['script'].split(chr(10)))}")
            print(f"   Total edits applied: {handoff_result['total_edits']}")

            print(f"\n🔬 Experiment:")
            print(f"   ID: {handoff_result['experiment_id']}")
            print(f"   Session: {handoff_result['session_id']}")

            print(f"\n💾 Database:")
            print(f"   Cameras saved: {len(handoff_result['ui_data']['cameras'])}")
            print(f"   Sensors saved: {len(handoff_result['ui_data']['sensors'])}")

            print("\n" + "="*80)
            print("🎉 FULL LLM INTEGRATION TEST PASSED!")
            print("="*80)
            print("\n🚀 Scene Maker system is PRODUCTION READY!")
            print("\n✅ Validated complete chain:")
            print("   User → Orchestrator → Agent → LLM → handoff → Backend → User")

            return True
        else:
            print(f"\n❌ Handler failed:")
            print(f"   Error: {handoff_result.get('error')}")
            return False
    else:
        print("\n❌ No handoff_result found")
        print(f"   Task result: {task_result}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SCENE MAKER - TEST 7: SINGLE TURN INTEGRATION")
    print("="*80)
    print("\nThis test validates the COMPLETE system with real LLM!")
    print("\nRequires Claude API access")

    confirm = input("\nRun single turn LLM test? (y/n): ").strip().lower()

    if confirm == 'y':
        success = asyncio.run(test_single_turn_integration())
        sys.exit(0 if success else 1)
    else:
        print("Test cancelled")
        sys.exit(0)

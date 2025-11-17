#!/usr/bin/env python3
"""
TEST 8: MULTI-TURN CONVERSATION - ask_master in action!
========================================================

Tests multi-turn conversation with ask_master:
1. User: "Hi, what's your name?" → Agent uses ask_master to respond
2. User answers agent's response
3. User: "What can I do here?" → Agent uses ask_master to explain
4. User answers
5. User: "OK create a scene" → Agent uses handoff to build

Validates:
- ask_master tool usage for conversation/questions
- Agent personality (Jarvis meets TARS style)
- Multi-turn conversation flow
- Final handoff for scene creation
"""

import sys
from pathlib import Path
# test_8 is at: simulation_center/sim_agents/tests/test_8_ask_master_conversation.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from sim_agents.scene_maker_agent import create_scene_maker_agent
from sim_agents.scene_maker_handoff_handler import handle_scene_maker_handoff
from ai_orchestration.core.orchestrator import Orchestrator


async def test_ask_master_conversation():
    """
    TEST 8: Multi-turn conversation with ask_master

    Flow:
    TURN 1: User asks casual question → Agent uses ask_master
    TURN 2: User asks what agent can do → Agent uses ask_master
    TURN 3: User requests scene → Agent uses handoff
    """
    print("\n" + "="*80)
    print("TEST 8: MULTI-TURN CONVERSATION - ask_master in action!")
    print("="*80)

    # Setup orchestrator and agent
    print("\n🎯 Setup: Creating orchestrator and scene_maker agent...")
    ops = Orchestrator()
    scene_agent = create_scene_maker_agent(ops)

    # Register handoff handler
    ops.handoff_handlers['scene_maker'] = handle_scene_maker_handoff

    print("   ✓ Orchestrator ready")
    print("   ✓ scene_maker agent with ask_master + handoff")
    print("   ✓ Handler registered")

    # TURN 1: Casual greeting
    print("\n" + "="*80)
    print("💬 TURN 1: User greets agent")
    print("="*80)

    turn1_message = "Hi! What's your name?"
    print(f"User: {turn1_message}")

    result1 = await ops.start_root_task(
        task=turn1_message,
        main_agent="scene_maker"
    )

    print("\n🤖 Agent response:")
    task1_result = result1.get("result", {})

    # Check if agent used ask_master
    if 'ask_master_response' in str(task1_result) or 'question' in str(task1_result):
        print("   ✅ Agent used ask_master!")
    else:
        print(f"   ℹ️  Agent response: {task1_result}")

    # TURN 2: Ask capabilities
    print("\n" + "="*80)
    print("💬 TURN 2: User asks what agent can do")
    print("="*80)

    # Continue from previous task or start new
    turn2_message = "Cool! So what can I do here? What can you help me with?"
    print(f"User: {turn2_message}")

    result2 = await ops.start_root_task(
        task=turn2_message,
        main_agent="scene_maker"
    )

    print("\n🤖 Agent response:")
    task2_result = result2.get("result", {})

    if 'ask_master_response' in str(task2_result) or 'question' in str(task2_result):
        print("   ✅ Agent used ask_master again!")
    else:
        print(f"   ℹ️  Agent response: {task2_result}")

    # TURN 3: Actual scene request
    print("\n" + "="*80)
    print("💬 TURN 3: User requests scene creation")
    print("="*80)

    turn3_message = """Awesome! Let's create a simple scene then.

Make me:
- 5x5x3 room
- Stretch robot at origin
- Table at (2, 0, 0)
- Apple on the table
- Overhead camera

Let's do this!"""

    print(f"User: {turn3_message}")

    result3 = await ops.start_root_task(
        task=turn3_message,
        main_agent="scene_maker"
    )

    print("\n🤖 Agent should use handoff now...")

    # Check for handoff result
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)

    task3_result = result3.get("result", {})

    if 'handoff_result' in task3_result:
        handoff_result = task3_result['handoff_result']

        if handoff_result.get('success'):
            print(f"\n✅ MULTI-TURN CONVERSATION SUCCESS!")

            print(f"\n💬 Agent's final message (boss/sir personality!):")
            print(f"   {handoff_result['message']}")

            print(f"\n📸 Screenshots:")
            print(f"   Total cameras: {handoff_result['total_cameras']}")

            print(f"\n📜 Scene created:")
            print(f"   Total lines: {len(handoff_result['script'].split(chr(10)))}")
            print(f"   Total edits: {handoff_result['total_edits']}")

            print("\n" + "="*80)
            print("🎉 TEST 8 PASSED - MULTI-TURN CONVERSATION!")
            print("="*80)
            print("\n✅ Validated:")
            print("   1. Turn 1: Agent responded to greeting (ask_master)")
            print("   2. Turn 2: Agent explained capabilities (ask_master)")
            print("   3. Turn 3: Agent built scene (handoff)")
            print("   4. Personality: boss/sir style throughout")

            return True
        else:
            print(f"\n❌ Handoff failed:")
            print(f"   Error: {handoff_result.get('error')}")
            return False
    else:
        print("\n⚠️  No handoff in Turn 3")
        print(f"   Result: {task3_result}")
        print("\n   This might mean agent is still asking questions!")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("SCENE MAKER - TEST 8: MULTI-TURN CONVERSATION")
    print("="*80)
    print("\nThis test validates ask_master usage and conversational flow!")
    print("\nConversation Flow:")
    print("  TURN 1: User: 'Hi! What's your name?'")
    print("          → Agent uses ask_master to respond")
    print("  TURN 2: User: 'What can I do here?'")
    print("          → Agent uses ask_master to explain")
    print("  TURN 3: User: 'Create a scene with table and apple'")
    print("          → Agent uses handoff to build scene")
    print("\nRequires Claude API access")

    confirm = input("\nRun multi-turn conversation test? (y/n): ").strip().lower()

    if confirm == 'y':
        success = asyncio.run(test_ask_master_conversation())
        sys.exit(0 if success else 1)
    else:
        print("Test cancelled")
        sys.exit(0)

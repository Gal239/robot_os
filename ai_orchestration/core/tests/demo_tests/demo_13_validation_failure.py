#!/usr/bin/env python3
"""
DEMO: Test Validation FAILURE with Tool Override

This demo intentionally causes validation failure to test:
1. What error message the LLM receives
2. Whether the hint system works correctly
3. Whether LLM can recover after seeing the error
4. Task status when validation fails

Flow:
1. Create agent with custom handoff requiring 'summary'
2. Give instructions that will make LLM use WRONG format (default 'answer')
3. Observe validation failure
4. Give correct instructions
5. Observe successful retry
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from ai_orchestration.core.orchestrator import Orchestrator



async def demo_intentional_failure():
    """
    DEMO: Force validation failure by giving bad instructions
    """
    print("=" * 70)
    print("DEMO: Intentional Validation Failure Test")
    print("=" * 70)

    ops = Orchestrator()

    # Create agent with custom handoff (requires 'summary')
    # BUT give instructions that will make LLM use default format!
    print("\n[1] Creating agent with custom handoff schema...")
    agent = ops.agent.create(
        id="confused_bot",
        describe="Bot that will initially fail validation",
        instructions="""You are a helpful bot.
When you're done, use the handoff tool to complete the task.

INTENTIONALLY BAD INSTRUCTION (to test validation):
Return your answer using the 'answer' field and 'documents' field.
""",  # This will FAIL because override requires 'summary'!
        tools=["handoff"],
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": "Complete task with summary",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Your response summary"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    )

    print(f"✓ Agent created: {agent.agent_id}")
    print(f"  Override requires: ['summary']")
    print(f"  Instructions tell it to use: ['answer', 'documents']")
    print(f"  🎯 This SHOULD cause validation failure!")

    # Start task
    print("\n[2] Starting task...")
    task_description = "What is 10 + 5? Use handoff when done."

    print(f"  Task: {task_description}")
    print("  Calling LLM...\n")

    await ops.start_root_task(
        task=task_description,
        main_agent="confused_bot"
    )

    # Find the root task
    root_tasks = [tid for tid, node in ops.graph_ops.modal.nodes.items()
                  if node.tool_type.value == "root"]

    if not root_tasks:
        print("❌ No root task found!")
        return

    task_id = root_tasks[0]
    node = ops.graph_ops.modal.nodes[task_id]

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nTask ID: {task_id}")
    print(f"Status: {node.status.value}")

    # Analyze timeline
    print(f"\n[3] Tool Timeline:")
    timeline = node.tool_timeline

    validation_failures = []
    handoff_successes = []

    for i, event in enumerate(timeline, 1):
        tool = event.get('tool', '?')
        event_type = event.get('type', '?')

        if "input_validation_hint" in tool:
            validation_failures.append(event)
            result = event.get('result', {})
            print(f"\n  {i}. ⚠️  VALIDATION FAILED!")
            print(f"      Tool: {tool}")
            print(f"      Error: {result.get('error', 'unknown')}")
            print(f"      Provided fields: {result.get('provided_fields', [])}")
            print(f"      Required fields: {result.get('required_fields', [])}")
            print(f"      Missing fields: {result.get('missing_fields', [])}")

            # Show the hint that was sent back to LLM
            if 'hint' in result:
                print(f"      Hint sent to LLM: {result['hint']}")

        elif tool == 'handoff' and event_type == 'handoff':
            handoff_successes.append(event)
            print(f"\n  {i}. ✅ HANDOFF SUCCESSFUL")
            result = event.get('result', {})
            if 'summary' in result:
                print(f"      Summary: {result['summary']}")

        else:
            print(f"\n  {i}. {tool} ({event_type})")

    # Summary
    print(f"\n[4] Summary:")
    print(f"  Validation failures: {len(validation_failures)}")
    print(f"  Successful handoffs: {len(handoff_successes)}")
    print(f"  Final status: {node.status.value}")

    if len(validation_failures) > 0:
        print("\n✅ GOOD! Validation correctly rejected invalid input!")

        if len(handoff_successes) > 0:
            print("✅ EXCELLENT! LLM recovered and used correct format after seeing error!")
        else:
            print("⚠️  LLM did not recover - task may have failed or hit retry limit")
    else:
        print("\n⚠️  Unexpected: No validation failures occurred!")
        print("     LLM may have ignored bad instructions and used correct format anyway")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    return {
        "task_id": task_id,
        "status": node.status.value,
        "validation_failures": len(validation_failures),
        "handoff_successes": len(handoff_successes)
    }


async def demo_direct_validation_check():
    """
    DEMO: Test validation layer directly (no LLM)
    Shows exactly what happens when wrong fields are provided
    """
    print("\n\n" + "=" * 70)
    print("DEMO: Direct Validation Layer Test (No LLM)")
    print("=" * 70)

    ops = Orchestrator()

    # Create agent with override
    agent = ops.agent.create(
        id="test_agent",
        describe="Test agent",
        instructions="Test",
        tools=["handoff"],
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": "Complete with summary",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"}
                    },
                    "required": ["summary"]
                }
            }
        }
    )

    print("\n[1] Testing BAD input (using default 'answer' + 'documents')...")

    # Build tools to get the override schema
    tools = ops.tool_ops.build_tools_for_agent(agent)
    handoff_tool = next(t for t in tools if t["name"] == "handoff")

    # Register the override in tool modal so validation can find it
    from ai_orchestration.core.modals import ToolEntry
    ops.tool_ops.modal.register_tool(ToolEntry(
        name="handoff",
        type="handoff",
        input_schema=handoff_tool["input_schema"],
        output_schema={}
    ))

    # Test with WRONG format
    bad_input = {
        "answer": {"result": "15"},
        "documents": []
    }

    print(f"  Input: {bad_input}")

    error = ops.tool_ops.modal.validate_input("handoff", bad_input)

    if error:
        print(f"  ✅ Validation REJECTED (as expected!)")
        print(f"  Error: {error.get('error', 'unknown')}")
        print(f"  Missing fields: {error.get('missing_fields', [])}")
    else:
        print(f"  ❌ Validation PASSED (unexpected!)")

    print("\n[2] Testing GOOD input (using 'summary')...")

    good_input = {
        "summary": "10 + 5 = 15"
    }

    print(f"  Input: {good_input}")

    error = ops.tool_ops.modal.validate_input("handoff", good_input)

    if error:
        print(f"  ❌ Validation REJECTED (unexpected!)")
        print(f"  Error: {error}")
    else:
        print(f"  ✅ Validation PASSED (as expected!)")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


async def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  VALIDATION FAILURE TEST".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n⚠️  This test intentionally causes validation failures!")
    print("We want to see:")
    print("  1. What error message LLM receives")
    print("  2. Whether LLM can recover")
    print("  3. Task behavior when validation fails")

    print("\n⚠️  WARNING: Demo 2 calls the LLM (uses API credits)")
    print("Press Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
    except EOFError:
        print("\n(Running in non-interactive mode, proceeding...)")

    try:
        # Demo 1: Direct validation (fast, no LLM)
        await demo_direct_validation_check()

        # Demo 2: Full run with intentional failure (uses LLM)
        result = await demo_intentional_failure()

        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "  TEST COMPLETE!".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")

        print("\n📊 Final Summary:")
        print(f"  Task status: {result['status']}")
        print(f"  Validation failures: {result['validation_failures']}")
        print(f"  Successful handoffs: {result['handoff_successes']}")

        if result['validation_failures'] > 0 and result['handoff_successes'] > 0:
            print("\n✅ Perfect! System:")
            print("  - Rejected invalid input")
            print("  - Sent helpful error to LLM")
            print("  - LLM recovered with correct format")
            print("  - Task completed successfully")
        elif result['validation_failures'] > 0:
            print("\n⚠️  System rejected invalid input but LLM didn't recover")
        else:
            print("\n⚠️  No validation failures - LLM used correct format immediately")

    except KeyboardInterrupt:
        print("\n\n⏸️  Demo cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
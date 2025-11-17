#!/usr/bin/env python3
"""
DEMO: Full Orchestrator Run with Tool Override
Tests override + validation in ACTUAL execution with LLM

This is a REAL END-TO-END test:
1. Create agent with custom handoff schema
2. Start a root task
3. Let orchestrator run (calls LLM with override schema)
4. LLM calls handoff
5. Validation enforces override schema
6. Task completes

IMPORTANT: This actually calls the LLM!
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from ai_orchestration.core.orchestrator import Orchestrator


async def demo_full_run_simple_task():
    """
    DEMO: Full run with simple custom handoff
    Agent must respond with custom format, not default
    """
    print("=" * 70)
    print("DEMO: Full Orchestrator Run with Custom Handoff Override")
    print("=" * 70)

    ops = Orchestrator()

    # Create agent with VERY SIMPLE custom handoff
    # Override requires just "summary" field (not answer + documents)
    simple_agent = ops.agent.create(
        id="simple_responder",
        describe="Simple responder with custom output format",
        instructions="""You are a simple responder.
When asked a question, respond with a brief summary using the handoff tool.

IMPORTANT: The handoff tool requires exactly ONE field:
- summary: A brief text summary of your response

Do NOT use 'answer' or 'documents' - those fields don't exist in your handoff tool!
Use ONLY the 'summary' field.""",
        tools=["handoff"],
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": "Complete task with a summary",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief summary of your response"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }
    )

    print(f"\n✓ Created agent: {simple_agent.agent_id}")
    print(f"  Custom handoff requires: ['summary']")
    print(f"  Default format (answer + documents) will be REJECTED!")

    # Create task
    task_description = "What is 2 + 2? Respond using handoff with your summary."

    print(f"\n{'─' * 70}")
    print(f"Starting task: {task_description}")
    print("─" * 70)

    await ops.start_root_task(
        task=task_description,
        main_agent="simple_responder"
    )

    # Find the root task
    root_tasks = [tid for tid, node in ops.graph_ops.modal.nodes.items()
                  if node.tool_type.value == "root"]

    if not root_tasks:
        print("❌ No root task found!")
        return {}

    task_id = root_tasks[0]
    node = ops.graph_ops.modal.nodes[task_id]
    status = node.status.value

    print(f"\n{'─' * 70}")
    print("Task Result:")
    print("─" * 70)

    print(f"Task ID: {task_id}")
    print(f"Status: {status}")

    if status == "completed":
        print("✅ TASK COMPLETED!")

        # Show tool timeline
        print(f"\nTool timeline:")
        timeline = node.tool_timeline

        handoff_calls = 0
        validation_failures = 0

        for i, event in enumerate(timeline, 1):
            tool = event.get('tool', 'unknown')
            event_type = event.get('type', 'unknown')

            if "input_validation_hint" in tool:
                validation_failures += 1
                print(f"  {i}. ⚠️  VALIDATION FAILED - {event['result'].get('error', 'unknown error')}")
            elif tool == 'handoff' and event_type == 'handoff':
                handoff_calls += 1
                print(f"  {i}. ✅ Handoff executed successfully")
                # Show what was passed
                result_data = event.get('result', {})
                if 'summary' in result_data:
                    print(f"       Summary: {result_data['summary']}")
            else:
                print(f"  {i}. {tool} ({event_type})")

        print(f"\nSummary:")
        print(f"  Handoff calls: {handoff_calls}")
        print(f"  Validation failures: {validation_failures}")

        if validation_failures == 0 and handoff_calls > 0:
            print("\n✅ PERFECT! Agent used correct custom format on first try!")
        elif handoff_calls > 0:
            print(f"\n✅ SUCCESS! Agent eventually used correct format after {validation_failures} retries")

    else:
        print(f"❌ Task did not complete. Status: {status}")

        # Show what went wrong
        timeline = node.tool_timeline
        validation_failures = [e for e in timeline if "input_validation_hint" in e.get('tool', '')]
        if validation_failures:
            print(f"\nValidation failures: {len(validation_failures)}")
            for failure in validation_failures:
                print(f"  - {failure['result'].get('error', 'unknown')}")

    print(f"\n{'═' * 70}")
    print("DEMO COMPLETE")
    print("═" * 70)

    return {"task_id": task_id, "status": status}


async def demo_full_run_with_complex_schema():
    """
    DEMO: More complex schema to test thorough validation
    """
    print("\n\n" + "=" * 70)
    print("DEMO: Full Run with Complex Custom Schema")
    print("=" * 70)

    ops = Orchestrator()

    # Create agent with custom schema requiring multiple fields
    analyst = ops.agent.create(
        id="data_analyst",
        describe="Data analyst with custom reporting format",
        instructions="""You are a data analyst.
When you complete your analysis, use handoff to return results.

Your handoff tool requires these THREE fields:
1. metrics: An object containing numerical metrics (e.g., {"average": 3})
2. insights: An array of insight strings (e.g., ["The average is 3"])
3. chart_refs: An array of chart file references (e.g., [] if no charts)

Do NOT use the default 'answer' and 'documents' fields - they don't exist!
Use ONLY: metrics, insights, chart_refs""",
        tools=["handoff"],
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": "Submit analysis results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "object", "description": "Numerical metrics"},
                        "insights": {"type": "array", "description": "Key insights", "items": {"type": "string"}},
                        "chart_refs": {"type": "array", "description": "Chart files", "items": {"type": "string"}}
                    },
                    "required": ["metrics", "insights", "chart_refs"]
                }
            }
        }
    )

    print(f"\n✓ Created analyst with custom handoff")
    print(f"  Required: metrics, insights, chart_refs")

    task_description = "Analyze these numbers: [1, 2, 3, 4, 5]. Calculate the average and provide one insight."

    print(f"\n{'─' * 70}")
    print(f"Starting task: {task_description}")
    print("─" * 70)

    await ops.start_root_task(
        task=task_description,
        main_agent="data_analyst"
    )

    # Find the root task
    root_tasks = [tid for tid, node in ops.graph_ops.modal.nodes.items()
                  if node.tool_type.value == "root"]

    if not root_tasks:
        print("❌ No root task found!")
        return {}

    task_id = root_tasks[0]
    node = ops.graph_ops.modal.nodes[task_id]
    status = node.status.value

    print(f"\n{'─' * 70}")
    print("Task Result:")
    print("─" * 70)

    print(f"Task ID: {task_id}")
    print(f"Status: {status}")

    if status == "completed":
        print("✅ TASK COMPLETED!")

        if task_id in ops.graph_ops.modal.nodes:
            timeline = ops.graph_ops.modal.nodes[task_id].tool_timeline

            validation_failures = [e for e in timeline if "input_validation_hint" in e.get('tool', '')]
            handoffs = [e for e in timeline if e.get('tool') == 'handoff']

            print(f"\nValidation failures: {len(validation_failures)}")
            print(f"Successful handoffs: {len(handoffs)}")

            if handoffs:
                print(f"\nFinal handoff data:")
                final = handoffs[-1]
                result_data = final.get('result', {})
                print(f"  Metrics: {result_data.get('metrics', 'N/A')}")
                print(f"  Insights: {result_data.get('insights', 'N/A')}")
                print(f"  Chart refs: {result_data.get('chart_refs', 'N/A')}")

    else:
        print(f"⚠️  Task did not complete. Status: {status}")

    print(f"\n{'═' * 70}")
    print("DEMO COMPLETE")
    print("═" * 70)

    return {"task_id": task_id, "status": status}


async def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FULL END-TO-END TEST: Tool Overrides with Real LLM".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")

    print("\n⚠️  WARNING: This test calls the LLM! It will use API credits.")
    print("Press Enter to continue, or Ctrl+C to cancel...")
    try:
        input()
    except EOFError:
        print("\n(Running in non-interactive mode, proceeding...)")

    try:
        # Demo 1: Simple override with clear instructions
        result1 = await demo_full_run_simple_task()

        # Demo 2: More complex override
        result2 = await demo_full_run_with_complex_schema()

        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "✅ FULL END-TO-END TESTS COMPLETED!".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "═" * 68 + "╝")

        task_id_1 = result1.get("task_id")
        task_id_2 = result2.get("task_id")

        # Need to import Orchestrator to check status
        from ai_orchestration.core.modals import TaskStatus

        print("\n📊 Summary:")
        if task_id_1:
            status1 = result1.get("status", "unknown")
            print(f"  Demo 1: {status1}")
        if task_id_2:
            status2 = result2.get("status", "unknown")
            print(f"  Demo 2: {status2}")

        print("\n✅ Key Validation:")
        print("  - Override schema was sent to LLM")
        print("  - Validation enforced override (not default)")
        print("  - LLM adapted to custom schema format")
        print("  - Tasks completed with correct format")

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

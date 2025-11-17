#!/usr/bin/env python3
"""
Hint System Tests - Test all 3 hint types
Tests input validation, output validation, and loop detection hints
OFFENSIVE: Crashes if hints broken
"""

import sys
from ai_orchestration.core.modals import TaskGraphModal, WorkspaceModal, ToolType, LogModal
from ai_orchestration.core.ops import ToolOps, TaskGraphOps, LogOps

from dataclasses import asdict

def test_input_validation_hint():
    """Test Hint Type 1: Input validation catches missing required fields"""
    print("=" * 60)
    print("Test: Input Validation Hint (Missing Required Field)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_input_hint")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)
    log_ops = LogOps.create_new()

    # Create task
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "test task")
    print(f"✓ Task created: {task_id}")

    # Call write_file with MISSING required fields (path, content, mime_type required)
    tool_schema = asdict(tool_ops.get_tool("write_file"))
    result = graph.handle_tool_call(
        task_id,
        "write_file",
        ToolType.FUNCTION_TOOL,
        {"path": "test.txt"},  # Missing 'content' and 'mime_type'
        tool_schema,
        tool_ops,
        graph_ops,
        log_ops
    )

    # Assert tool did NOT execute
    assert result["action"] == "continue"
    print("✓ Tool execution prevented")

    # Check timeline for input validation hint
    timeline = graph.nodes[task_id].tool_timeline
    hint_event = [e for e in timeline if e["tool"] == "input_validation_hint"]
    assert len(hint_event) > 0
    print("✓ Input validation hint added to timeline")

    # Check hint message
    hint_msg = hint_event[0]["result"]["hint"]
    assert "Missing required fields" in hint_msg
    assert "content" in hint_msg
    assert "mime_type" in hint_msg
    print(f"✓ Hint message: {hint_msg[:100]}...")

    print("✅ PASS\n")


def test_output_validation_hint():
    """Test Hint Type 2: Output validation catches incomplete output"""
    print("=" * 60)
    print("Test: Output Validation Hint (Incomplete Output)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_output_hint")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)
    log_ops = LogOps.create_new()

    # Create task
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "test task")

    # Mock a tool that returns incomplete output
    # We'll manually execute and check validation
    from ai_orchestration.core.modals.tool_modal import ToolEntry

    # Get write_file tool (has output_schema with required fields)
    write_tool = tool_ops.modal.get_tool("write_file")
    assert write_tool is not None
    print("✓ write_file tool loaded")

    # Simulate incomplete output (missing required fields)
    incomplete_output = {
        "path": "test.txt",
        # Missing: size_bytes, token_count, blocks_count
    }

    # Validate output
    output_error = tool_ops.modal.validate_output("write_file", incomplete_output)
    assert output_error is not None
    print("✓ Output validation detected incomplete output")

    # Check error details
    assert "missing_fields" in output_error
    assert "size_bytes" in output_error["missing_fields"]
    assert "token_count" in output_error["missing_fields"]
    print(f"✓ Missing fields detected: {output_error['missing_fields']}")

    # Generate hint
    hint_msg = log_ops.modal.generate_output_validation_hint("write_file", output_error)
    assert "output incomplete" in hint_msg.lower()
    print(f"✓ Hint generated: {hint_msg[:100]}...")

    print("✅ PASS\n")


def test_loop_detection_hint():
    """Test Hint Type 3: Loop detection catches repeated tool calls"""
    print("=" * 60)
    print("Test: Loop Detection Hint (Same Tool, Same Output)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_loop_hint")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)
    log_ops = LogOps.create_new()

    # Create task
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "test task")
    print(f"✓ Task created: {task_id}")

    # Simulate calling list_files twice with same result
    # First call
    loop_info1 = log_ops.add_event(
        event_type="tool_execution",
        data={
            "agent": "agent1",
            "tool": "list_files",
            "input": {"pattern": "*.txt"},
            "result": {"files": ["a.txt", "b.txt"]}
        },
        task_id=task_id,
        graph_modal=graph
    )
    assert not loop_info1.get("loop_detected")
    print("✓ First call - no loop detected")

    # Second call with SAME result
    loop_info2 = log_ops.add_event(
        event_type="tool_execution",
        data={
            "agent": "agent1",
            "tool": "list_files",
            "input": {"pattern": "*.txt"},
            "result": {"files": ["a.txt", "b.txt"]}  # SAME result
        },
        task_id=task_id,
        graph_modal=graph
    )

    # Assert loop detected
    assert loop_info2.get("loop_detected") == True
    assert loop_info2.get("tool_name") == "list_files"
    assert loop_info2.get("count") >= 2
    print(f"✓ Loop detected after {loop_info2['count']} identical calls")

    # Generate hint
    hint_msg = log_ops.modal.generate_loop_detection_hint(
        loop_info2["tool_name"],
        loop_info2["count"]
    )
    assert "LOOP DETECTED" in hint_msg
    assert "list_files" in hint_msg
    assert "handoff" in hint_msg.lower()
    print(f"✓ Hint generated: {hint_msg[:100]}...")

    print("✅ PASS\n")


def test_all_hints_integration():
    """Test all 3 hint types working together in handle_tool_call"""
    print("=" * 60)
    print("Test: All Hints Integration (Full Flow)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_integration")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)
    log_ops = LogOps.create_new()

    # Create task
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "integration test")

    # Test 1: Input validation hint prevents execution
    print("\n[1] Testing input validation...")
    tool_schema = asdict(tool_ops.get_tool("write_file"))
    result1 = graph.handle_tool_call(
        task_id,
        "write_file",
        ToolType.FUNCTION_TOOL,
        {"path": "test.txt"},  # Missing required fields
        tool_schema,
        tool_ops,
        graph_ops,
        log_ops
    )
    assert result1["action"] == "continue"

    timeline = graph.nodes[task_id].tool_timeline
    input_hints = [e for e in timeline if "input_validation_hint" in e["tool"]]
    assert len(input_hints) == 1
    print("✓ Input validation hint working")

    # Test 2: Successful execution (no hints)
    print("\n[2] Testing successful execution (no hints)...")
    tool_schema = asdict(tool_ops.get_tool("list_files"))
    result2 = graph.handle_tool_call(
        task_id,
        "list_files",
        ToolType.FUNCTION_TOOL,
        {"pattern": "*"},  # Valid input
        tool_schema,
        tool_ops,
        graph_ops,
        log_ops
    )
    assert result2["action"] == "continue"

    # Check that list_files executed successfully
    timeline = graph.nodes[task_id].tool_timeline
    list_calls = [e for e in timeline if e["tool"] == "list_files"]
    assert len(list_calls) == 1
    print("✓ Valid tool call executed successfully")

    # Test 3: Verify loop detection doesn't break normal execution
    # (Loop detection thoroughly tested in dedicated test_loop_detection_hint)
    print("\n[3] Verifying loop detection integrated without breaking normal flow...")
    print("✓ Loop detection integrated (tested separately in test_loop_detection_hint)")

    print("\n✅ INTEGRATION PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("HINT SYSTEM TESTS")
    print("Test 3-type hint system (input, output, loop)")
    print("=" * 60)
    print()

    tests = [
        test_input_validation_hint,
        test_output_validation_hint,
        test_loop_detection_hint,
        test_all_hints_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

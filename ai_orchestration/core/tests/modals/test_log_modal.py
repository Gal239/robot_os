#!/usr/bin/env python3
"""
LogModal Tests - Event logging, 3-level logs, log formatting
Tests event routing, orchestration detection, log formatting with graph state
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import LogModal, TaskGraphModal, ToolType


def test_log_modal_structure():
    """LogModal should have 3-level logs"""
    print("=" * 60)
    print("Test: LogModal Structure")
    print("=" * 60)

    log_modal = LogModal()

    # Has 3 log levels
    assert hasattr(log_modal, "master_log")
    assert hasattr(log_modal, "meta_log")
    assert hasattr(log_modal, "task_logs")
    print("✓ Has 3-level logs")

    # All empty initially
    assert len(log_modal.master_log) == 0
    assert len(log_modal.meta_log) == 0
    assert len(log_modal.task_logs) == 0
    print("✓ Logs empty initially")

    # Has session_id
    assert log_modal.session_id is not None
    print("✓ Has session_id")

    print("✅ PASS\n")


def test_add_event_to_master_log():
    """add_event should ALWAYS add to master_log"""
    print("=" * 60)
    print("Test: Add Event to Master Log")
    print("=" * 60)

    log_modal = LogModal()

    # Add event
    log_modal.add_event("test_event", {"key": "value"})

    assert len(log_modal.master_log) == 1
    print("✓ Event added to master_log")

    # Check event structure
    event = log_modal.master_log[0]
    assert event["type"] == "test_event"
    assert event["data"] == {"key": "value"}
    assert "timestamp" in event
    print("✓ Event has correct structure")

    print("✅ PASS\n")


def test_add_event_to_meta_log_orchestration():
    """Orchestration events should go to meta_log"""
    print("=" * 60)
    print("Test: Add Orchestration Events to Meta Log")
    print("=" * 60)

    log_modal = LogModal()

    # Add orchestration events
    orchestration_events = [
        "task_created",
        "task_started",
        "task_completed",
        "state_change",
        "task_unblocked",
        "system_hint"
    ]

    for event_type in orchestration_events:
        log_modal.add_event(event_type, {})

    # All should be in master
    assert len(log_modal.master_log) == len(orchestration_events)
    print(f"✓ {len(orchestration_events)} events in master_log")

    # All should be in meta
    assert len(log_modal.meta_log) == len(orchestration_events)
    print(f"✓ {len(orchestration_events)} events in meta_log")

    print("✅ PASS\n")


def test_add_event_to_task_logs():
    """Events with task_id should go to task_logs"""
    print("=" * 60)
    print("Test: Add Events to Task Logs")
    print("=" * 60)

    log_modal = LogModal()

    # Add task-specific events
    log_modal.add_event("tool_execution", {"tool": "write_file"}, task_id="task_0")
    log_modal.add_event("tool_execution", {"tool": "ask_data"}, task_id="task_0")
    log_modal.add_event("tool_execution", {"tool": "load_to_context"}, task_id="task_1")

    # Check master
    assert len(log_modal.master_log) == 3
    print("✓ All events in master_log")

    # Check task logs
    assert "task_0" in log_modal.task_logs
    assert "task_1" in log_modal.task_logs
    assert len(log_modal.task_logs["task_0"]) == 2
    assert len(log_modal.task_logs["task_1"]) == 1
    print("✓ Events routed to task_logs correctly")

    print("✅ PASS\n")


def test_add_event_orchestration_tools_to_meta():
    """Orchestration tools (route_to_, handoff, ask_master) should go to meta_log"""
    print("=" * 60)
    print("Test: Orchestration Tools to Meta Log")
    print("=" * 60)

    log_modal = LogModal()

    # Add orchestration tool events
    log_modal.add_event("tool_execution", {"tool": "route_to_researcher"})
    log_modal.add_event("tool_execution", {"tool": "handoff"})
    log_modal.add_event("tool_execution", {"tool": "ask_master"})

    # All should be in master
    assert len(log_modal.master_log) == 3
    print("✓ 3 events in master_log")

    # All should be in meta (orchestration tools)
    assert len(log_modal.meta_log) == 3
    print("✓ 3 orchestration tools in meta_log")

    # Regular tools should NOT go to meta
    log_modal.add_event("tool_execution", {"tool": "write_file"})
    assert len(log_modal.master_log) == 4
    assert len(log_modal.meta_log) == 3  # Still 3 (write_file not added)
    print("✓ Regular tools NOT in meta_log")

    print("✅ PASS\n")


def test_add_hint():
    """add_hint should create system_hint event"""
    print("=" * 60)
    print("Test: Add Hint")
    print("=" * 60)

    log_modal = LogModal()

    # Add hint
    log_modal.add_hint("pattern_detected", {"pattern": "delegation", "suggestion": "Use route_to_X"})

    # Check master
    assert len(log_modal.master_log) == 1
    event = log_modal.master_log[0]
    assert event["type"] == "system_hint"
    assert event["data"]["hint_type"] == "pattern_detected"
    print("✓ Hint added to master_log")

    # Check meta (system_hint is orchestration event)
    assert len(log_modal.meta_log) == 1
    print("✓ Hint in meta_log")

    print("✅ PASS\n")


def test_render_for_display():
    """render_for_display should create human-readable log"""
    print("=" * 60)
    print("Test: Render for Display")
    print("=" * 60)

    log_modal = LogModal(session_id="test_session")

    # Add some events
    log_modal.add_event("task_created", {"task_id": "task_0"})
    log_modal.add_event("task_completed", {"task_id": "task_0"})

    # Render
    display = log_modal.render_for_display()

    assert "test_session" in display
    assert "Total events:" in display
    assert "Orchestration events:" in display
    print("✓ Display contains session and event counts")

    print("✅ PASS\n")


def test_render_for_json():
    """render_for_json should serialize all logs"""
    print("=" * 60)
    print("Test: Render for JSON")
    print("=" * 60)

    log_modal = LogModal(session_id="test_session")

    # Add events
    log_modal.add_event("task_created", {"task_id": "task_0"}, task_id="task_0")

    # Render
    data = log_modal.render_for_json()

    assert data["session_id"] == "test_session"
    assert "master_log" in data
    assert "meta_log" in data
    assert "task_logs" in data
    assert len(data["master_log"]) == 1
    assert len(data["meta_log"]) == 1
    assert "task_0" in data["task_logs"]
    print("✓ JSON has all log levels")

    print("✅ PASS\n")


def test_format_meta_log_reads_graph():
    """format_meta_log should read graph state"""
    print("=" * 60)
    print("Test: Format Meta Log (reads graph)")
    print("=" * 60)

    log_modal = LogModal()
    graph_modal = TaskGraphModal()

    # Create tasks
    task1 = graph_modal.create_node("agent1", None, ToolType.ROOT, "Task 1", None)
    task2 = graph_modal.create_node("agent2", task1, ToolType.AGENT_AS_TOOL, "Task 2", "agent1")

    # Complete one
    graph_modal.mark_node_completed(task2, {})

    # Format meta log
    meta_log = log_modal.format_meta_log(graph_modal)

    assert "Task Graph:" in meta_log
    assert "2 tasks" in meta_log
    assert "1 completed" in meta_log
    print("✓ Meta log shows graph state")

    print("✅ PASS\n")


def test_format_task_log_reads_task_state():
    """format_task_log should read task state from graph"""
    print("=" * 60)
    print("Test: Format Task Log (reads task state)")
    print("=" * 60)

    log_modal = LogModal()
    graph_modal = TaskGraphModal()

    # Create task
    task_id = graph_modal.create_node("agent", None, ToolType.ROOT, "Test task", None)

    # Add timeline event
    graph_modal.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, "write_file", {"path": "test.txt"}, {"path": "test.txt"})

    # Format task log
    task_log = log_modal.format_task_log(graph_modal, task_id)

    assert "Current Task ID:" in task_log
    assert "Tool Type:" in task_log
    assert "Test task" in task_log
    assert "Tool Calls:" in task_log
    assert "write_file" in task_log
    print("✓ Task log shows task state and timeline")

    print("✅ PASS\n")


def test_build_log_context():
    """build_log_context should combine meta + task logs"""
    print("=" * 60)
    print("Test: Build Log Context")
    print("=" * 60)

    log_modal = LogModal()
    graph_modal = TaskGraphModal()

    # Create task
    task_id = graph_modal.create_node("agent", None, ToolType.ROOT, "Task", None)

    # Build context
    context = log_modal.build_log_context(graph_modal, task_id)

    assert "=== META LOG ===" in context
    assert "=== TASK LOG ===" in context
    print("✓ Context combines meta + task logs")

    print("✅ PASS\n")


def test_from_json_deserialization():
    """from_json should reconstruct LogModal"""
    print("=" * 60)
    print("Test: From JSON Deserialization")
    print("=" * 60)

    # Create original
    log_modal = LogModal(session_id="test")
    log_modal.add_event("task_created", {"task_id": "task_0"}, task_id="task_0")

    # Serialize
    data = log_modal.render_for_json()

    # Deserialize
    restored = LogModal.from_json(data)

    assert restored.session_id == "test"
    assert len(restored.master_log) == 1
    assert len(restored.meta_log) == 1
    assert "task_0" in restored.task_logs
    print("✓ LogModal restored from JSON")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("LOG MODAL TESTS")
    print("=" * 60)
    print()

    tests = [
        test_log_modal_structure,
        test_add_event_to_master_log,
        test_add_event_to_meta_log_orchestration,
        test_add_event_to_task_logs,
        test_add_event_orchestration_tools_to_meta,
        test_add_hint,
        test_render_for_display,
        test_render_for_json,
        test_format_meta_log_reads_graph,
        test_format_task_log_reads_task_state,
        test_build_log_context,
        test_from_json_deserialization
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

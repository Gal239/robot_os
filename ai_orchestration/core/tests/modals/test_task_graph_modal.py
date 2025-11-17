#!/usr/bin/env python3
"""
TaskGraphModal Tests - Timeline, blocking, state machine
Tests typed events, auto-blocking/unblocking, render_for_llm, child injection
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import TaskGraphModal, TaskNode, TaskStatus, ToolType


def test_timeline_typed_events():
    """Timeline should store typed events with ToolType enum"""
    print("=" * 60)
    print("Test: Timeline Typed Events")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create task
    task_id = modal.create_node(
        agent_id="test_agent",
        parent_task_id=None,
        tool_type=ToolType.ROOT,
        task_payload="Test task",
        master_agent_id=None
    )

    # Add typed events
    modal.add_to_timeline(
        task_id=task_id,
        event_type=ToolType.FUNCTION_TOOL,
        tool="write_file",
        input={"path": "test.txt", "content": "data"},
        result={"path": "test.txt"}
    )

    modal.add_to_timeline(
        task_id=task_id,
        event_type=ToolType.FUNCTION_TOOL,
        tool="load_to_context",
        input={"path": "test.txt"},
        result={"type": "text", "text": "data"}
    )

    # Check timeline
    node = modal.nodes[task_id]
    assert len(node.tool_timeline) == 2
    print(f"✓ Timeline has {len(node.tool_timeline)} events")

    # Check event types are ToolType enum values
    assert node.tool_timeline[0]["type"] == ToolType.FUNCTION_TOOL.value
    assert node.tool_timeline[1]["type"] == ToolType.FUNCTION_TOOL.value
    print("✓ Events have ToolType enum values")

    # Check event structure
    event = node.tool_timeline[0]
    assert "type" in event
    assert "tool" in event
    assert "input" in event
    assert "result" in event
    assert "timestamp" in event
    print("✓ Events have required fields")

    print("✅ PASS\n")


def test_auto_blocking_agent_as_tool():
    """Creating agent_as_tool node should auto-block parent"""
    print("=" * 60)
    print("Test: Auto-Blocking (agent_as_tool)")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create parent
    parent_id = modal.create_node(
        agent_id="parent",
        parent_task_id=None,
        tool_type=ToolType.ROOT,
        task_payload="Parent task",
        master_agent_id=None
    )

    # Create child (agent_as_tool should auto-block parent)
    child_id = modal.create_node(
        agent_id="child",
        parent_task_id=parent_id,
        tool_type=ToolType.AGENT_AS_TOOL,
        task_payload="Child task",
        master_agent_id="parent"
    )

    # Check parent is blocked
    parent = modal.nodes[parent_id]
    assert child_id in parent.blockers
    assert parent.status == TaskStatus.WAITING
    print("✓ Parent auto-blocked by child")

    # Check child is ready
    child = modal.nodes[child_id]
    assert len(child.blockers) == 0
    assert child.status == TaskStatus.READY
    print("✓ Child is ready")

    print("✅ PASS\n")


def test_auto_blocking_ask_master():
    """Creating ask_master node should auto-block parent"""
    print("=" * 60)
    print("Test: Auto-Blocking (ask_master)")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create parent
    parent_id = modal.create_node(
        agent_id="parent",
        parent_task_id=None,
        tool_type=ToolType.ROOT,
        task_payload="Parent task",
        master_agent_id=None
    )

    # Create ask_master (should auto-block parent)
    question_id = modal.create_node(
        agent_id="parent",  # Ask master, so same agent
        parent_task_id=parent_id,
        tool_type=ToolType.ASK_MASTER,
        task_payload="Question?",
        master_agent_id="parent"
    )

    # Check parent is blocked
    parent = modal.nodes[parent_id]
    assert question_id in parent.blockers
    assert parent.status == TaskStatus.WAITING
    print("✓ Parent auto-blocked by ask_master")

    print("✅ PASS\n")


def test_auto_unblocking_on_completion():
    """Completing child should auto-unblock parent"""
    print("=" * 60)
    print("Test: Auto-Unblocking on Completion")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create parent and child
    parent_id = modal.create_node("parent", None, ToolType.ROOT, "Task", None)
    child_id = modal.create_node("child", parent_id, ToolType.AGENT_AS_TOOL, "Sub", "parent")

    # Parent should be blocked
    assert child_id in modal.nodes[parent_id].blockers
    print("✓ Parent blocked initially")

    # Complete child
    modal.mark_node_completed(child_id, {"result": "done"})

    # Check parent is unblocked
    parent = modal.nodes[parent_id]
    assert child_id not in parent.blockers
    assert parent.status == TaskStatus.READY
    print("✓ Parent unblocked after child completion")

    # Check child is completed
    child = modal.nodes[child_id]
    assert child.status == TaskStatus.COMPLETED
    assert child.result == {"result": "done"}
    print("✓ Child marked as completed")

    print("✅ PASS\n")


def test_child_injection_with_original_type():
    """Child result should inject to parent with original ToolType"""
    print("=" * 60)
    print("Test: Child Injection with Original ToolType")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create parent and child (agent_as_tool)
    parent_id = modal.create_node("parent", None, ToolType.ROOT, "Task", None)
    child_id = modal.create_node("child", parent_id, ToolType.AGENT_AS_TOOL, "Sub", "parent")

    # Complete child
    modal.mark_node_completed(child_id, {"answer": "42"})

    # Check injection in parent timeline
    parent = modal.nodes[parent_id]
    assert len(parent.tool_timeline) > 0
    print(f"✓ Parent timeline has {len(parent.tool_timeline)} events")

    # Find injection event (last one)
    injection = parent.tool_timeline[-1]
    assert injection["type"] == ToolType.AGENT_AS_TOOL.value, f"Expected agent_as_tool, got {injection['type']}"
    assert "← result from child" in injection["tool"]
    assert injection["result"] == {"answer": "42"}
    print("✓ Child result injected with original ToolType (agent_as_tool)")

    print("✅ PASS\n")


def test_ask_master_injection():
    """Ask master result should inject with ask_master type"""
    print("=" * 60)
    print("Test: Ask Master Injection")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create parent and ask_master child
    parent_id = modal.create_node("parent", None, ToolType.ROOT, "Task", None)
    ask_id = modal.create_node("parent", parent_id, ToolType.ASK_MASTER, "Question?", "parent")

    # Complete ask_master
    modal.mark_node_completed(ask_id, {"answer": "Yes"})

    # Check injection
    parent = modal.nodes[parent_id]
    injection = parent.tool_timeline[-1]
    assert injection["type"] == ToolType.ASK_MASTER.value
    assert "← answer from parent" in injection["tool"]
    assert injection["result"] == {"answer": "Yes"}
    print("✓ Ask master injected with ask_master type")

    print("✅ PASS\n")


def test_render_for_llm_extracts_timeline():
    """render_for_llm should extract load_to_context from timeline"""
    print("=" * 60)
    print("Test: render_for_llm Extracts from Timeline")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create task
    task_id = modal.create_node("agent", None, ToolType.ROOT, "Task", None)

    # Add load_to_context event
    content_block = {
        "type": "text",
        "text": "def foo():\n    return 42"
    }
    modal.add_to_timeline(
        task_id=task_id,
        event_type=ToolType.FUNCTION_TOOL,
        tool="load_to_context",
        input={"path": "code.py"},
        result=content_block
    )

    # Render for LLM
    agent_config = {"instructions": "Test agent"}
    messages = modal.render_for_llm(
        task_id=task_id,
        agent_config=agent_config,
        orchestration_preamble="System preamble",
        log_context="Log context"
    )

    # Check structure
    assert len(messages) == 2  # [system, user]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    print("✓ Messages: [system, user]")

    # Check user content has blocks
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 2  # log + loaded file
    print(f"✓ User content: {len(user_content)} blocks")

    # Check first block is log
    assert user_content[0]["type"] == "text"
    assert user_content[0]["text"] == "Log context"
    print("✓ First block is log context")

    # Check second block is loaded content from timeline
    assert user_content[1]["type"] == "text"
    assert "def foo():" in user_content[1]["text"]
    print("✓ Second block is loaded content from timeline")

    print("✅ PASS\n")


def test_timeline_order_preserved():
    """Timeline should preserve event order"""
    print("=" * 60)
    print("Test: Timeline Order Preserved")
    print("=" * 60)

    modal = TaskGraphModal()
    task_id = modal.create_node("agent", None, ToolType.ROOT, "Task", None)

    # Add events in order
    modal.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, "tool1", {}, "result1")
    modal.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, "tool2", {}, "result2")
    modal.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, "tool3", {}, "result3")

    # Check order
    timeline = modal.nodes[task_id].tool_timeline
    assert timeline[0]["tool"] == "tool1"
    assert timeline[1]["tool"] == "tool2"
    assert timeline[2]["tool"] == "tool3"
    print("✓ Timeline order preserved")

    print("✅ PASS\n")


def test_get_ready_tasks():
    """get_ready_tasks should return only ready tasks"""
    print("=" * 60)
    print("Test: Get Ready Tasks")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create ready task
    ready_id = modal.create_node("agent1", None, ToolType.ROOT, "Ready", None)

    # Create parent and blocked child
    parent_id = modal.create_node("agent2", None, ToolType.ROOT, "Parent", None)
    child_id = modal.create_node("agent3", parent_id, ToolType.AGENT_AS_TOOL, "Child", "agent2")

    # Get ready tasks
    ready = modal.get_ready_tasks()

    # Should have ready_id and child_id (child has no blockers)
    # parent_id is WAITING (blocked by child)
    assert ready_id in ready
    assert child_id in ready
    assert parent_id not in ready  # Blocked
    print(f"✓ Ready tasks: {len(ready)}")

    # Complete child
    modal.mark_node_completed(child_id, {})

    # Now parent should be ready
    ready = modal.get_ready_tasks()
    assert parent_id in ready
    assert child_id not in ready  # Completed
    print("✓ Parent ready after child completes")

    print("✅ PASS\n")


def test_task_node_is_ready():
    """TaskNode.is_ready() should check blockers and status"""
    print("=" * 60)
    print("Test: TaskNode.is_ready()")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create task
    task_id = modal.create_node("agent", None, ToolType.ROOT, "Task", None)
    node = modal.nodes[task_id]

    # Ready initially
    assert node.is_ready()
    print("✓ Node ready initially")

    # Add blocker
    node.add_blocker("other_task")
    assert not node.is_ready()
    print("✓ Node not ready when blocked")

    # Remove blocker
    node.remove_blocker("other_task")
    assert node.is_ready()
    print("✓ Node ready when unblocked")

    # Complete
    node.mark_completed({})
    assert not node.is_ready()
    print("✓ Completed node not ready")

    print("✅ PASS\n")


def test_get_status_summary():
    """get_status_summary should aggregate task states"""
    print("=" * 60)
    print("Test: Get Status Summary")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create tasks in different states
    ready_id = modal.create_node("agent1", None, ToolType.ROOT, "Ready", None)

    parent_id = modal.create_node("agent2", None, ToolType.ROOT, "Parent", None)
    child_id = modal.create_node("agent3", parent_id, ToolType.AGENT_AS_TOOL, "Child", "agent2")

    # Complete one
    modal.mark_node_completed(child_id, {})

    # Get summary
    summary = modal.get_status_summary()

    assert summary["total"] == 3
    assert len(summary["completed"]) == 1
    assert len(summary["ready"]) >= 1  # ready_id
    print(f"✓ Summary: {summary['total']} total, {len(summary['completed'])} completed")

    print("✅ PASS\n")


def test_get_graph_for_viz():
    """get_graph_for_viz should format for visualization"""
    print("=" * 60)
    print("Test: Get Graph for Visualization")
    print("=" * 60)

    modal = TaskGraphModal()

    # Create simple graph
    parent_id = modal.create_node("parent", None, ToolType.ROOT, "Parent", None)
    child_id = modal.create_node("child", parent_id, ToolType.AGENT_AS_TOOL, "Child", "parent")

    # Get viz data
    viz = modal.get_graph_for_viz()

    assert "nodes" in viz
    assert "edges" in viz
    assert "session_id" in viz
    assert len(viz["nodes"]) == 2
    assert len(viz["edges"]) >= 1  # Parent → child edge
    print(f"✓ Viz: {len(viz['nodes'])} nodes, {len(viz['edges'])} edges")

    # Check node has required fields
    node = viz["nodes"][0]
    assert "id" in node
    assert "label" in node
    assert "status" in node
    assert "color" in node
    print("✓ Nodes have required fields")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK GRAPH MODAL TESTS")
    print("=" * 60)
    print()

    tests = [
        test_timeline_typed_events,
        test_auto_blocking_agent_as_tool,
        test_auto_blocking_ask_master,
        test_auto_unblocking_on_completion,
        test_child_injection_with_original_type,
        test_ask_master_injection,
        test_render_for_llm_extracts_timeline,
        test_timeline_order_preserved,
        test_get_ready_tasks,
        test_task_node_is_ready,
        test_get_status_summary,
        test_get_graph_for_viz
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
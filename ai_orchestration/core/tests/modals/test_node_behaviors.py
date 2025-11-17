#!/usr/bin/env python3
"""
Node Behavior Tests - Test all 6 ToolTypes in task graph context
Tests each ToolType behavior (blocking, documents, execution)
OFFENSIVE: Crashes if behavior wrong
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import TaskGraphModal, WorkspaceModal, TaskStatus, ToolType
from dataclasses import asdict
from ai_orchestration.core.ops import ToolOps, TaskGraphOps
from ai_orchestration.core.tools.file_tools import write_file


# ========== FUNCTION_TOOL BEHAVIOR TESTS ==========

def test_function_tool_behavior():
    """FUNCTION_TOOL should execute, log, and continue (not block)"""
    print("=" * 60)
    print("Test: FUNCTION_TOOL Behavior")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Create node
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "test task")
    print(f"✓ Created task: {task_id}")

    # Call FUNCTION_TOOL via handle_tool_call
    tool_schema = asdict(tool_ops.get_tool("write_file"))
    result = graph.handle_tool_call(
        task_id, "write_file", ToolType.FUNCTION_TOOL,
        {"path": "test.md", "content": "# Test", "mime_type": "text/markdown"},
        tool_schema, tool_ops, graph_ops
    )

    # Assert behavior
    assert result["action"] == "continue"  # Does NOT block or stop
    print("✓ Action: continue (no block)")

    assert len(graph.nodes[task_id].tool_timeline) == 1  # Logged to timeline
    assert graph.nodes[task_id].tool_timeline[0]["type"] == ToolType.FUNCTION_TOOL.value
    print("✓ Logged to timeline")

    assert graph.nodes[task_id].status == TaskStatus.READY  # Still ready
    print("✓ Status: READY (not blocked)")

    assert workspace.get_document("test.md") is not None  # Executed
    print("✓ Function executed (file created)")

    print("✅ PASS\n")


# ========== NON_FUNCTION_TOOL BEHAVIOR TESTS ==========

def test_non_function_tool_behavior():
    """NON_FUNCTION_TOOL should log input only, instant continue (no execution)"""
    print("=" * 60)
    print("Test: NON_FUNCTION_TOOL Behavior")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")
    tool_ops = ToolOps.create_new()
    graph_ops = TaskGraphOps(graph)

    # Create node
    task_id = graph.create_node("agent1", None, ToolType.ROOT, "test task")

    # Call NON_FUNCTION_TOOL
    tool_schema = asdict(tool_ops.get_tool("stop_and_think"))
    result = graph.handle_tool_call(
        task_id, "stop_and_think", ToolType.NON_FUNCTION_TOOL,
        {"thoughts": "Analyzing the problem..."},
        tool_schema, tool_ops, graph_ops
    )

    # Assert behavior
    assert result["action"] == "continue"  # Instant continue
    print("✓ Action: continue (instant)")

    assert len(graph.nodes[task_id].tool_timeline) == 1  # Logged input
    event = graph.nodes[task_id].tool_timeline[0]
    assert event["type"] == ToolType.NON_FUNCTION_TOOL.value
    assert event["tool"] == "stop_and_think"
    assert event["result"] == {"logged": True}
    print("✓ Logged input only (no execution)")

    print("✅ PASS\n")


# ========== HANDOFF BEHAVIOR TESTS ==========

def test_handoff_behavior():
    """HANDOFF should complete task, stop agent, unblock parent"""
    print("=" * 60)
    print("Test: HANDOFF Behavior (without documents)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    tool_ops = ToolOps.create_new()
    graph_ops = TaskGraphOps(graph)

    # Create parent + child
    parent_id = graph.create_node("agent1", None, ToolType.ROOT, "parent task")
    child_id = graph.create_node("agent2", parent_id, ToolType.AGENT_AS_TOOL, "child task")

    # Parent should be blocked by child
    assert parent_id in graph.nodes[child_id].blockers or child_id in graph.nodes[parent_id].blockers
    print("✓ Parent blocked by child")

    # Child calls handoff
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    result = graph.handle_tool_call(
        child_id, "handoff", ToolType.HANDOFF,
        {"result": {"status": "done", "count": 5}},
        tool_schema, tool_ops, graph_ops
    )

    # Assert behavior
    assert result["action"] == "return"  # STOPS agent
    print("✓ Action: return (stops agent)")

    assert graph.nodes[child_id].status == TaskStatus.COMPLETED  # Marked complete
    print("✓ Child status: COMPLETED")

    assert graph.nodes[parent_id].is_ready()  # Parent unblocked
    print("✓ Parent unblocked (ready)")

    # Result injected to parent
    assert len(graph.nodes[parent_id].tool_timeline) > 0
    parent_event = graph.nodes[parent_id].tool_timeline[-1]
    assert "result from" in parent_event["tool"]
    print("✓ Result injected to parent")

    print("✅ PASS\n")


def test_handoff_with_documents():
    """HANDOFF with documents should store and inject documents to parent"""
    print("=" * 60)
    print("Test: HANDOFF with Documents")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Create parent + child
    parent_id = graph.create_node("agent1", None, ToolType.ROOT, "parent task")
    child_id = graph.create_node("agent2", parent_id, ToolType.AGENT_AS_TOOL, "child task")

    # Child creates documents
    write_file(workspace, child_id, "report.md", "# Report", "text/markdown")
    write_file(workspace, child_id, "data.csv", "a,b\n1,2", "text/csv")
    print("✓ Child created 2 documents")

    # Child handoffs with documents
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    result = graph.handle_tool_call(
        child_id, "handoff", ToolType.HANDOFF,
        {"result": {"status": "done"}, "documents": ["report.md", "data.csv"]},
        tool_schema, tool_ops, graph_ops
    )

    # Assert documents stored in child
    assert "report.md" in graph.nodes[child_id].documents
    assert "data.csv" in graph.nodes[child_id].documents
    print("✓ Documents stored in child task")

    # Assert documents injected to parent
    parent_event = graph.nodes[parent_id].tool_timeline[-1]
    parent_result = parent_event["result"]
    assert "child_documents" in parent_result
    assert "report.md" in parent_result["child_documents"]
    assert "data.csv" in parent_result["child_documents"]
    print("✓ Documents injected to parent")

    print("✅ PASS\n")


# ========== AGENT_AS_TOOL BEHAVIOR TESTS ==========

def test_agent_as_tool_with_documents():
    """AGENT_AS_TOOL should create child with documents, block parent"""
    print("=" * 60)
    print("Test: AGENT_AS_TOOL with Documents")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    graph_ops = TaskGraphOps(graph)

    # Parent creates document
    parent_id = graph.create_node("agent1", None, ToolType.ROOT, "parent task")
    write_file(workspace, parent_id, "spec.md", "# Requirements", "text/markdown")
    print("✓ Parent created spec.md")

    # Delegate with documents
    tool_schema = asdict(tool_ops.get_tool("route_to_developer"))
    result = graph.handle_tool_call(
        parent_id, "route_to_developer", ToolType.AGENT_AS_TOOL,
        {"request": "Build this feature", "documents": ["spec.md"]},
        tool_schema, tool_ops, graph_ops
    )

    # Assert behavior
    assert result["action"] == "return"  # Parent BLOCKS
    print("✓ Action: return (blocks parent)")

    # Find child task
    child_id = None
    for task_id, node in graph.nodes.items():
        if node.tool_type == ToolType.AGENT_AS_TOOL and task_id != parent_id:
            child_id = task_id
            break

    assert child_id is not None
    print(f"✓ Child created: {child_id}")

    # Assert child receives documents
    assert "spec.md" in graph.nodes[child_id].documents
    print("✓ Child received spec.md")

    # Assert parent blocked
    assert not graph.nodes[parent_id].is_ready()
    print("✓ Parent blocked (waiting for child)")

    print("✅ PASS\n")


# ========== ASK_MASTER BEHAVIOR TESTS ==========

def test_ask_master_with_documents():
    """ASK_MASTER should create master task with documents, block self"""
    print("=" * 60)
    print("Test: ASK_MASTER with Documents")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    graph_ops = TaskGraphOps(graph)

    # Create parent + child (child has master_agent_id)
    parent_id = graph.create_node("agent1", None, ToolType.ROOT, "parent task")
    child_id = graph.create_node(
        "agent2", parent_id, ToolType.AGENT_AS_TOOL,
        "child task", master_agent_id="agent1"
    )

    # Child creates draft
    write_file(workspace, child_id, "draft.md", "# Draft Report", "text/markdown")
    print("✓ Child created draft.md")

    # Ask master with documents
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["ask_master"]
    result = graph.handle_tool_call(
        child_id, "ask_master", ToolType.ASK_MASTER,
        {"question": "Is this analysis correct?", "documents": ["draft.md"]},
        tool_schema, tool_ops, graph_ops
    )

    # Assert behavior
    assert result["action"] == "return"  # Child BLOCKS
    print("✓ Action: return (blocks child)")

    # Find master task (created for parent agent)
    master_task_id = None
    for task_id, node in graph.nodes.items():
        if node.tool_type == ToolType.ASK_MASTER and task_id not in [parent_id, child_id]:
            master_task_id = task_id
            break

    assert master_task_id is not None
    print(f"✓ Master task created: {master_task_id}")

    # Assert master receives documents
    assert "draft.md" in graph.nodes[master_task_id].documents
    print("✓ Master received draft.md")

    # Assert child blocked
    assert not graph.nodes[child_id].is_ready()
    print("✓ Child blocked (waiting for answer)")

    print("✅ PASS\n")


# ========== RENDER_FOR_LLM DOCUMENT INJECTION TESTS ==========

def test_render_for_llm_injects_documents():
    """render_for_llm should inject documents as content blocks"""
    print("=" * 60)
    print("Test: render_for_llm Document Injection")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_behavior")

    # Create task with documents
    task_id = graph.create_node(
        "agent1", None, ToolType.ROOT, "test task",
        documents=["file1.md", "file2.md"]
    )

    # Create documents in workspace
    write_file(workspace, task_id, "file1.md", "# File 1 Content", "text/markdown")
    write_file(workspace, task_id, "file2.md", "# File 2 Content", "text/markdown")
    print("✓ Created 2 documents in workspace")

    # Render messages
    messages = graph.render_for_llm(
        task_id,
        agent_config={"instructions": "You are a test agent"},
        log_context="Test context",
        workspace=workspace
    )

    # Assert message structure
    assert len(messages) == 2  # [system, user]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    print("✓ Message structure correct")

    # Assert content blocks
    content_blocks = messages[1]["content"]
    assert len(content_blocks) >= 3  # text + 2 documents
    print(f"✓ Content blocks: {len(content_blocks)} (text + documents)")

    # Assert document blocks exist
    doc_blocks = [b for b in content_blocks if b.get("type") == "document"]
    assert len(doc_blocks) >= 2
    print(f"✓ Document content blocks: {len(doc_blocks)}")

    # Assert document block format
    assert doc_blocks[0]["source"]["type"] == "base64"
    assert "markdown" in doc_blocks[0]["source"]["media_type"]
    print("✓ Document block format correct")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("NODE BEHAVIOR TESTS - All 6 ToolTypes")
    print("=" * 60)
    print()

    tests = [
        test_function_tool_behavior,
        test_non_function_tool_behavior,
        test_handoff_behavior,
        test_handoff_with_documents,
        test_agent_as_tool_with_documents,
        test_ask_master_with_documents,
        test_render_for_llm_injects_documents,
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

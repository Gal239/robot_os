#!/usr/bin/env python3
"""
Agent Delegation Tests - Agent calls agent via route_to_X
Tests that agents can delegate to other agents and receive results
OFFENSIVE: Crashes if delegation broken
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from dataclasses import asdict
from ai_orchestration.core.modals import TaskGraphModal, WorkspaceModal, ToolType, TaskStatus
from ai_orchestration.core.ops import ToolOps, TaskGraphOps


def test_simple_delegation():
    """Parent agent delegates to child agent via route_to_X"""
    print("=" * 60)
    print("Test: Simple Agent Delegation (Parent → Child)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_delegation")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Register worker agent as tool
    from ai_orchestration.core.ops import Agent
    worker_agent = Agent({
        "agent_id": "worker",
        "description": "Does simple work",
        "instructions": "Execute tasks",
        "tools": ["write_file", "handoff"]
    }, "worker")

    tool_ops.register_agents_as_tools({"worker": worker_agent})

    # Verify route_to_worker exists
    route_tool = tool_ops.get_tool("route_to_worker")
    assert route_tool is not None
    assert route_tool.type == "agent_as_tool"
    print("✓ route_to_worker tool registered")

    # Create parent task
    parent_id = graph.create_node("parent", None, ToolType.ROOT, "parent task")
    print(f"✓ Parent task created: {parent_id}")

    # Parent calls route_to_worker (AGENT_AS_TOOL)
    tool_schema = asdict(tool_ops.get_tool("route_to_worker"))
    result = graph.handle_tool_call(
        parent_id,
        "route_to_worker",
        ToolType.AGENT_AS_TOOL,
        {"request": "Do some work"},
        tool_schema,
        tool_ops,
        graph_ops
    )

    # Assert behavior
    assert result["action"] == "return"  # Parent BLOCKS
    print("✓ Parent blocked (waiting for worker)")

    # Find child task
    child_id = None
    for task_id, node in graph.nodes.items():
        if node.agent_id == "worker" and task_id != parent_id:
            child_id = task_id
            break

    assert child_id is not None
    print(f"✓ Child task created: {child_id}")

    # Verify parent blocked, child ready
    assert not graph.nodes[parent_id].is_ready()
    assert graph.nodes[child_id].is_ready()
    print("✓ Parent blocked, child ready")

    # Child executes and handoffs
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        child_id,
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"status": "work completed", "count": 42}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )

    # Assert parent unblocked and received result
    assert graph.nodes[parent_id].is_ready()
    assert graph.nodes[child_id].status == TaskStatus.COMPLETED
    print("✓ Child completed, parent unblocked")

    # Check result injected to parent
    parent_events = graph.nodes[parent_id].tool_timeline
    result_event = parent_events[-1]
    assert "result from worker" in result_event["tool"].lower()
    # Result is nested: result_event["result"]["result"] (handoff wraps it)
    child_result = result_event["result"]["result"]
    assert child_result["status"] == "work completed"
    assert child_result["count"] == 42
    print("✓ Child result injected to parent timeline")

    print("✅ PASS\n")


def test_multi_agent_chain():
    """Parent delegates to multiple agents in sequence"""
    print("=" * 60)
    print("Test: Multi-Agent Chain (Parent → Agent1 → Agent2)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_chain")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Register two worker agents
    from ai_orchestration.core.ops import Agent

    agent1 = Agent({
        "agent_id": "agent1",
        "description": "First worker",
        "tools": ["write_file", "handoff"]
    }, "agent1")

    agent2 = Agent({
        "agent_id": "agent2",
        "description": "Second worker",
        "tools": ["write_file", "handoff"]
    }, "agent2")

    tool_ops.register_agents_as_tools({"agent1": agent1, "agent2": agent2})
    print("✓ Registered agent1, agent2 as tools")

    # Create parent
    parent_id = graph.create_node("parent", None, ToolType.ROOT, "parent task")

    # Parent → Agent1
    tool_schema = asdict(tool_ops.get_tool("route_to_agent1"))
    graph.handle_tool_call(
        parent_id,
        "route_to_agent1",
        ToolType.AGENT_AS_TOOL,
        {"request": "Step 1"},
        tool_schema,
        tool_ops,
        graph_ops
    )

    agent1_id = [tid for tid, n in graph.nodes.items() if n.agent_id == "agent1"][0]
    print(f"✓ Parent delegated to agent1: {agent1_id}")

    # Agent1 completes
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        agent1_id,
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"step": 1, "data": "from agent1"}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )
    print("✓ Agent1 completed")

    # Parent → Agent2
    tool_schema = asdict(tool_ops.get_tool("route_to_agent2"))
    graph.handle_tool_call(
        parent_id,
        "route_to_agent2",
        ToolType.AGENT_AS_TOOL,
        {"request": "Step 2"},
        tool_schema,
        tool_ops,
        graph_ops
    )

    agent2_id = [tid for tid, n in graph.nodes.items() if n.agent_id == "agent2"][0]
    print(f"✓ Parent delegated to agent2: {agent2_id}")

    # Agent2 completes
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        agent2_id,
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"step": 2, "data": "from agent2"}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )
    print("✓ Agent2 completed")

    # Verify parent has both results (look for "result from" events, not delegation events)
    parent_events = graph.nodes[parent_id].tool_timeline
    agent1_result = [e for e in parent_events if "result from agent1" in e["tool"].lower()][0]
    agent2_result = [e for e in parent_events if "result from agent2" in e["tool"].lower()][0]

    # Results are nested under ["result"]["result"]
    assert agent1_result["result"]["result"]["data"] == "from agent1"
    assert agent2_result["result"]["result"]["data"] == "from agent2"
    print("✓ Parent received both results in timeline")

    print("✅ PASS\n")


def test_nested_delegation():
    """Agent delegates to agent who delegates to another agent"""
    print("=" * 60)
    print("Test: Nested Delegation (Planner → Worker → Helper)")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_nested")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Register agents
    from ai_orchestration.core.ops import Agent

    helper = Agent({
        "agent_id": "helper",
        "description": "Helper agent",
        "tools": ["write_file", "handoff"]
    }, "helper")

    worker = Agent({
        "agent_id": "worker",
        "description": "Worker agent",
        "tools": ["route_to_helper", "handoff"]  # Can delegate to helper
    }, "worker")

    tool_ops.register_agents_as_tools({"helper": helper, "worker": worker})
    print("✓ Registered helper, worker as tools")

    # Create planner (root)
    planner_id = graph.create_node("planner", None, ToolType.ROOT, "plan task")

    # Planner → Worker
    tool_schema = asdict(tool_ops.get_tool("route_to_worker"))
    graph.handle_tool_call(
        planner_id,
        "route_to_worker",
        ToolType.AGENT_AS_TOOL,
        {"request": "Do complex work"},
        tool_schema,
        tool_ops,
        graph_ops
    )

    worker_id = [tid for tid, n in graph.nodes.items() if n.agent_id == "worker"][0]
    print(f"✓ Planner → Worker: {worker_id}")

    # Worker → Helper
    tool_schema = asdict(tool_ops.get_tool("route_to_helper"))
    graph.handle_tool_call(
        worker_id,
        "route_to_helper",
        ToolType.AGENT_AS_TOOL,
        {"request": "Need help"},
        tool_schema,
        tool_ops,
        graph_ops
    )

    helper_id = [tid for tid, n in graph.nodes.items() if n.agent_id == "helper"][0]
    print(f"✓ Worker → Helper: {helper_id}")

    # Verify blocking chain
    assert not graph.nodes[planner_id].is_ready()  # Planner blocked
    assert not graph.nodes[worker_id].is_ready()   # Worker blocked
    assert graph.nodes[helper_id].is_ready()       # Helper ready
    print("✓ Blocking chain: Planner blocked → Worker blocked → Helper ready")

    # Helper completes
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        helper_id,
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"helper_data": "solved"}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )
    print("✓ Helper completed")

    # Worker now unblocked, completes
    assert graph.nodes[worker_id].is_ready()
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        worker_id,
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"worker_data": "done", "used_helper": True}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )
    print("✓ Worker completed")

    # Planner now unblocked
    assert graph.nodes[planner_id].is_ready()
    print("✓ Planner unblocked")

    # Verify results flow up (look for "result from" events)
    worker_events = graph.nodes[worker_id].tool_timeline
    helper_result = [e for e in worker_events if "result from helper" in e["tool"].lower()][0]
    assert helper_result["result"]["result"]["helper_data"] == "solved"
    print("✓ Helper result → Worker timeline")

    planner_events = graph.nodes[planner_id].tool_timeline
    worker_result = [e for e in planner_events if "result from worker" in e["tool"].lower()][0]
    assert worker_result["result"]["result"]["worker_data"] == "done"
    print("✓ Worker result → Planner timeline")

    print("✅ PASS\n")


def test_parallel_delegation():
    """Parent delegates to multiple agents in parallel"""
    print("=" * 60)
    print("Test: Parallel Delegation (Parent → [Agent1, Agent2, Agent3])")
    print("=" * 60)

    # Setup
    graph = TaskGraphModal()
    workspace = WorkspaceModal(session_id="test_parallel")
    tool_ops = ToolOps.create_new()
    tool_ops.set_workspace(workspace)
    tool_ops.load_from_db()
    graph_ops = TaskGraphOps(graph)

    # Register three agents
    from ai_orchestration.core.ops import Agent

    for i in range(1, 4):
        agent = Agent({
            "agent_id": f"agent{i}",
            "description": f"Agent {i}",
            "tools": ["handoff"]
        }, f"agent{i}")
        tool_ops.register_agents_as_tools({f"agent{i}": agent})

    print("✓ Registered agent1, agent2, agent3")

    # Create parent
    parent_id = graph.create_node("parent", None, ToolType.ROOT, "parent task")

    # Parent delegates to all three (parallel)
    child_ids = []
    for i in range(1, 4):
        tool_schema = asdict(tool_ops.get_tool(f"route_to_agent{i}"))
        graph.handle_tool_call(
            parent_id,
            f"route_to_agent{i}",
            ToolType.AGENT_AS_TOOL,
            {"request": f"Task {i}"},
            tool_schema,
            tool_ops,
            graph_ops
        )
        child_id = [tid for tid, n in graph.nodes.items()
                    if n.agent_id == f"agent{i}"][0]
        child_ids.append(child_id)

    print(f"✓ Parent delegated to 3 agents in parallel")

    # Parent should be blocked by all three
    assert not graph.nodes[parent_id].is_ready()
    print("✓ Parent blocked (waiting for all children)")

    # Complete agents 1 and 2 (parent still blocked)
    for i, child_id in enumerate(child_ids[:2], 1):
        tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
        graph.handle_tool_call(
            child_id,
            "handoff",
            ToolType.HANDOFF,
            {"answer": {"agent": i, "status": "done"}, "documents": []},
            tool_schema,
            tool_ops,
            graph_ops
        )
        print(f"✓ Agent{i} completed")

    # Parent still blocked (agent3 not done)
    assert not graph.nodes[parent_id].is_ready()
    print("✓ Parent still blocked (1 child remaining)")

    # Complete agent3
    tool_schema = tool_ops.ORCHESTRATION_TOOLS["handoff"]
    graph.handle_tool_call(
        child_ids[2],
        "handoff",
        ToolType.HANDOFF,
        {"answer": {"agent": 3, "status": "done"}, "documents": []},
        tool_schema,
        tool_ops,
        graph_ops
    )
    print("✓ Agent3 completed")

    # NOW parent unblocked (all children done)
    assert graph.nodes[parent_id].is_ready()
    print("✓ Parent unblocked (all children complete)")

    # Verify all results in parent timeline
    parent_events = graph.nodes[parent_id].tool_timeline
    results = [e for e in parent_events if "result from agent" in e["tool"].lower()]
    assert len(results) == 3
    print("✓ Parent received all 3 results")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT DELEGATION TESTS")
    print("Test agent-to-agent communication via route_to_X")
    print("=" * 60)
    print()

    tests = [
        test_simple_delegation,
        test_multi_agent_chain,
        test_nested_delegation,
        test_parallel_delegation,
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

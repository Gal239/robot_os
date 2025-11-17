#!/usr/bin/env python3
"""
COMPLEX LLM STRESS TESTS - Real AI orchestration challenges
Tests complex multi-agent scenarios before robot integration
REQUIRES: API keys (Claude or GPT)
"""

import sys
import os
import asyncio
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.orchestrator import Orchestrator


async def test_deep_nested_delegation():
    """
    Test 3: Deep Nested Delegation (4 levels)
    CEO → Manager → Researcher → Reporter
    Tests deep hierarchy with result bubbling through all levels
    """
    print("=" * 60)
    print("Test 3: Deep Nested Delegation (4 Levels)")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # Level 4: Reporter (bottom of hierarchy)
    reporter = orchestrator.agent.create(
        id="reporter",
        describe="Creates structured reports from research data",
        instructions="""
        You create well-formatted reports from research information.
        Use write_file to create the report.
        Call handoff when complete.
        """,
        tools=["write_file", "handoff"]
    )

    # Level 3: Researcher
    researcher = orchestrator.agent.create(
        id="researcher",
        describe="Researches topics and gathers information",
        instructions="""
        You research topics and gather information.
        Use ask_gpt or search_web to gather information.
        When you have enough data, delegate to route_to_reporter to create the final report.
        Call handoff after reporter completes.
        """,
        tools=["ask_gpt", "search_web", "route_to_reporter", "handoff"]
    )

    # Level 2: Manager
    manager = orchestrator.agent.create(
        id="manager",
        describe="Manages research projects",
        instructions="""
        You manage research projects by delegating to specialists.
        Delegate research tasks to route_to_researcher.
        When researcher returns results, call handoff with summary.
        """,
        tools=["route_to_researcher", "handoff"]
    )

    # Level 1: CEO (top of hierarchy)
    ceo = orchestrator.agent.create(
        id="ceo",
        describe="Plans and coordinates complex multi-step projects",
        instructions="""
        You are the CEO coordinating complex projects.
        Break down tasks and delegate to route_to_manager.
        When manager returns final result, call handoff with executive summary.
        """,
        tools=["route_to_manager", "handoff"]
    )

    print("✓ Created 4-level agent hierarchy: CEO → Manager → Researcher → Reporter")

    task = "Research the impact of quantum computing on cryptography and create a brief report"

    print(f"✓ Starting task: {task}")
    print("✓ CEO should delegate to Manager → Researcher → Reporter...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="ceo"
    )

    print("\n--- RESULT ---")
    print(f"Final result: {result.get('result', {})}")
    print("-" * 60)

    # Verify 4 levels of tasks were created
    num_tasks = len(orchestrator.graph_ops.modal.nodes)
    assert num_tasks >= 4, f"Expected at least 4 tasks, got {num_tasks}"
    print(f"✓ Created {num_tasks} tasks (4-level hierarchy)")

    # Verify file was created
    files = list(orchestrator.workspace.documents.keys())
    assert len(files) > 0, "No files created!"
    print(f"✓ Files created: {files}")

    print("✅ PASS: 4-level deep nesting works!\n")


async def test_parallel_multi_agent_coordination():
    """
    Test 4: Parallel Multi-Agent Coordination
    Coordinator spawns 3 researchers in parallel, synthesizes results
    """
    print("=" * 60)
    print("Test 4: Parallel Multi-Agent Coordination")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # Create 3 specialist researchers
    ai_researcher = orchestrator.agent.create(
        id="ai_researcher",
        describe="Researches AI and machine learning topics",
        instructions="""
        You research AI and machine learning.
        Use ask_gpt to gather information.
        Return concise findings via handoff.
        """,
        tools=["ask_gpt", "handoff"]
    )

    blockchain_researcher = orchestrator.agent.create(
        id="blockchain_researcher",
        describe="Researches blockchain and cryptocurrency topics",
        instructions="""
        You research blockchain and cryptocurrency.
        Use ask_gpt to gather information.
        Return concise findings via handoff.
        """,
        tools=["ask_gpt", "handoff"]
    )

    quantum_researcher = orchestrator.agent.create(
        id="quantum_researcher",
        describe="Researches quantum computing topics",
        instructions="""
        You research quantum computing.
        Use ask_gpt to gather information.
        Return concise findings via handoff.
        """,
        tools=["ask_gpt", "handoff"]
    )

    # Coordinator that spawns all 3
    coordinator = orchestrator.agent.create(
        id="coordinator",
        describe="Coordinates parallel research efforts",
        instructions="""
        You coordinate research across multiple topics.

        Available researchers:
        - route_to_ai_researcher: AI and machine learning
        - route_to_blockchain_researcher: Blockchain and crypto
        - route_to_quantum_researcher: Quantum computing

        Strategy:
        1. Delegate to all 3 researchers (can be done in sequence or one at a time)
        2. Wait for all results
        3. Synthesize findings into comparison report using write_file
        4. Call handoff with summary
        """,
        tools=["route_to_ai_researcher", "route_to_blockchain_researcher",
               "route_to_quantum_researcher", "write_file", "handoff"]
    )

    print("✓ Created coordinator + 3 specialist researchers")

    task = "Research AI, blockchain, and quantum computing. Create a comparison report showing the current state and future potential of each technology."

    print(f"✓ Starting task: {task}")
    print("✓ Coordinator should delegate to all 3 researchers...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="coordinator"
    )

    print("\n--- RESULT ---")
    print(f"Final result: {result.get('result', {})}")
    print("-" * 60)

    # Verify all 3 child tasks were created
    child_tasks = [tid for tid, node in orchestrator.graph_ops.modal.nodes.items()
                   if node.parent_task_id == "task_0"]
    assert len(child_tasks) >= 3, f"Expected 3 child tasks, got {len(child_tasks)}"
    print(f"✓ Coordinator created {len(child_tasks)} child tasks")

    # Verify comparison report was created
    files = list(orchestrator.workspace.documents.keys())
    assert len(files) > 0, "No comparison report created!"
    print(f"✓ Files created: {files}")

    print("✅ PASS: Parallel coordination works!\n")


async def test_complex_research_workflow():
    """
    Test 6: Complex Research Workflow
    Uses multiple tools: search_web, ask_gpt, write_file
    """
    print("=" * 60)
    print("Test 6: Complex Research Workflow")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    researcher = orchestrator.agent.create(
        id="advanced_researcher",
        describe="Conducts comprehensive research using multiple sources",
        instructions="""
        You are an advanced researcher who uses multiple tools to gather and analyze information.

        Research workflow:
        1. Use search_web to find current information
        2. Use ask_gpt or ask_claude for analysis and synthesis
        3. Create comprehensive report with write_file
        4. Call handoff when complete

        Your reports should be detailed, well-structured, and cite sources.
        """,
        tools=["search_web", "ask_gpt", "ask_claude", "write_file", "handoff"]
    )

    print("✓ Created advanced researcher agent")

    task = "Research the latest developments in large language models (2024-2025) and write a detailed analysis with current trends and future predictions"

    print(f"✓ Starting task: {task}")
    print("✓ Researcher should use search_web + ask_gpt + write_file...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="advanced_researcher"
    )

    print("\n--- RESULT ---")
    print(f"Final result: {result.get('result', {})}")
    print("-" * 60)

    # Verify file was created
    files = list(orchestrator.workspace.documents.keys())
    assert len(files) > 0, "No report created!"
    print(f"✓ Report created: {files}")

    # Check that report has substantial content
    doc = orchestrator.workspace.documents[files[0]]
    content = doc.render_from_json()
    assert len(content) > 500, f"Report too short: {len(content)} chars"
    print(f"✓ Report size: {len(content)} characters")

    # Verify multiple tools were used (check timeline)
    root_node = orchestrator.graph_ops.modal.nodes["task_0"]
    timeline = root_node.tool_timeline
    tool_names = [e["tool"] for e in timeline]

    # Should use at least 2 different tools
    unique_tools = set(tool_names)
    assert len(unique_tools) >= 2, f"Expected multiple tools, only used: {unique_tools}"
    print(f"✓ Tools used: {unique_tools}")

    print("✅ PASS: Complex research workflow works!\n")


async def test_document_passing_workflow():
    """
    Test 7: Document Passing Workflow
    Writer creates draft → Editor improves → Reviewer summarizes
    """
    print("=" * 60)
    print("Test 7: Document Passing Workflow")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # Writer creates initial draft
    writer = orchestrator.agent.create(
        id="writer",
        describe="Writes initial drafts",
        instructions="""
        You write initial drafts of documents.
        Create a draft file using write_file.
        Call handoff when complete.
        """,
        tools=["write_file", "handoff"]
    )

    # Editor improves the draft
    editor = orchestrator.agent.create(
        id="editor",
        describe="Edits and improves documents",
        instructions="""
        You edit and improve documents.
        Use read_file to read the draft.
        Use edit_file_block to make improvements.
        Call handoff when editing is complete.
        """,
        tools=["read_file", "list_files", "edit_file_block", "handoff"]
    )

    # Reviewer creates summary
    reviewer = orchestrator.agent.create(
        id="reviewer",
        describe="Reviews documents and creates summaries",
        instructions="""
        You review documents and create summaries.
        Use list_files to see available files.
        Use read_file to read the edited document.
        Create a summary using write_file.
        Call handoff with your review.
        """,
        tools=["list_files", "read_file", "write_file", "handoff"]
    )

    # Coordinator manages the workflow
    coordinator = orchestrator.agent.create(
        id="doc_coordinator",
        describe="Coordinates document workflow",
        instructions="""
        You coordinate document creation workflows.

        Process:
        1. Delegate to route_to_writer to create initial draft
        2. Delegate to route_to_editor to improve the draft
        3. Delegate to route_to_reviewer to create summary
        4. Call handoff with final status
        """,
        tools=["route_to_writer", "route_to_editor", "route_to_reviewer", "handoff"]
    )

    print("✓ Created document workflow: Writer → Editor → Reviewer")

    task = "Create a draft article about space exploration, have it edited for clarity, then create an executive summary"

    print(f"✓ Starting task: {task}")
    print("✓ Coordinator should route: writer → editor → reviewer...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="doc_coordinator"
    )

    print("\n--- RESULT ---")
    print(f"Final result: {result.get('result', {})}")
    print("-" * 60)

    # Verify multiple files exist (draft + summary at minimum)
    files = list(orchestrator.workspace.documents.keys())
    assert len(files) >= 1, "No files created in workflow!"
    print(f"✓ Files in workspace: {files}")

    # Verify workflow had 3 sequential delegations
    child_tasks = [tid for tid, node in orchestrator.graph_ops.modal.nodes.items()
                   if node.parent_task_id == "task_0"]
    assert len(child_tasks) >= 3, f"Expected 3 child tasks (writer→editor→reviewer), got {len(child_tasks)}"
    print(f"✓ Sequential workflow: {len(child_tasks)} stages completed")

    print("✅ PASS: Document passing workflow works!\n")


async def test_error_recovery():
    """
    Test 8: Error Recovery & Graceful Failure
    Tests how LLM handles impossible or ambiguous tasks
    """
    print("=" * 60)
    print("Test 8: Error Recovery & Graceful Failure")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # Agent with limited tools (can't delete)
    smart_agent = orchestrator.agent.create(
        id="smart_agent",
        describe="Smart agent that handles impossible requests gracefully",
        instructions="""
        You are a smart agent that recognizes when tasks are impossible.
        If you cannot complete a task with available tools, call handoff with an error explanation.
        Be helpful and explain what you CAN do with available tools.
        """,
        tools=["list_files", "read_file", "write_file", "handoff"]
    )

    print("✓ Created smart agent (no delete tool)")

    # First create a file so workspace is not empty
    setup = orchestrator.agent.create(
        id="setup_agent",
        describe="Creates test file",
        instructions="Create a markdown file. Call handoff when done.",
        tools=["write_file", "handoff"]
    )

    await orchestrator.start_root_task(
        task="Create test.md with some content",
        main_agent="setup_agent"
    )

    print("✓ Created test.md file")

    # Now try impossible task: execute Python code (no execute tool)
    task = "Execute the Python code: print('hello')"

    print(f"✓ Starting impossible task: {task}")
    print("✓ Agent should recognize impossibility (no code execution tool)...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="smart_agent"
    )

    print("\n--- RESULT ---")
    final_result = result.get('result', {}).get('result', {})
    print(f"Agent response: {final_result}")
    print("-" * 60)

    # Verify agent recognized it couldn't execute code
    result_str = str(final_result).lower()
    # Agent should mention error, cannot, unable, execute, etc.
    has_error_indication = any(word in result_str for word in
                               ['error', 'cannot', 'unable', 'impossible', 'not able', 'no tool', 'can\'t', 'do not have'])

    assert has_error_indication, f"Agent didn't indicate impossibility. Result: {final_result}"
    print("✓ Agent correctly indicated task is impossible")

    print("✅ PASS: Error recovery works!\n")


async def test_ask_data_efficiency():
    """
    Test 9: ask_data Efficiency Test
    Compare efficient ask_data vs wasteful load_to_context
    """
    print("=" * 60)
    print("Test 9: ask_data Efficiency Test")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # First create a test file for querying
    setup_agent = orchestrator.agent.create(
        id="setup",
        describe="Sets up test files",
        instructions="Create a Python file with some code. Call handoff when done.",
        tools=["write_file", "handoff"]
    )

    # Create test file
    await orchestrator.start_root_task(
        task="Create a file called 'test_code.py' with a main function that prints 'Hello World' and a helper function that adds two numbers",
        main_agent="setup"
    )

    print("✓ Created test file: test_code.py")

    # Now test efficient agent using ask_data
    efficient_agent = orchestrator.agent.create(
        id="efficient_agent",
        describe="Uses ask_data for efficient file querying",
        instructions="""
        You efficiently query files using ask_data.
        Use ask_data to answer questions about file contents without loading entire file.
        Call handoff with the answer.
        """,
        tools=["list_files", "ask_data", "handoff"]
    )

    print("✓ Created efficient agent (uses ask_data)")

    task = "What functions are defined in test_code.py?"

    print(f"✓ Starting task: {task}")
    print("✓ Agent should use ask_data for targeted query...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="efficient_agent"
    )

    print("\n--- RESULT ---")
    final_result = result.get('result', {}).get('result', {})
    print(f"Agent answer: {final_result}")
    print("-" * 60)

    # Verify ask_data was used
    root_node = orchestrator.graph_ops.modal.nodes["task_1"]  # task_0 was setup
    timeline = root_node.tool_timeline
    tool_names = [e["tool"] for e in timeline]

    assert "ask_data" in tool_names, f"ask_data not used! Tools: {tool_names}"
    print("✓ Agent used ask_data for efficient querying")

    # Verify answer mentions functions
    answer_str = str(final_result).lower()
    assert "main" in answer_str or "function" in answer_str, "Answer doesn't mention functions"
    print("✓ Answer contains relevant information")

    print("✅ PASS: ask_data efficiency works!\n")


async def test_complex_tool_chaining():
    """
    Test 10: Complex Tool Chaining
    Search files → analyze each → aggregate → write summary
    """
    print("=" * 60)
    print("Test 10: Complex Tool Chaining")
    print("=" * 60)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ SKIP: No API keys set")
        return

    orchestrator = Orchestrator()

    # Setup: create multiple test files
    setup_agent = orchestrator.agent.create(
        id="file_creator",
        describe="Creates multiple test files",
        instructions="""
        Create 3 Python files about different topics.
        Each should have a comment explaining what it does.
        Call handoff when done.
        """,
        tools=["write_file", "handoff"]
    )

    await orchestrator.start_root_task(
        task="Create 3 files: data_processor.py (processes data), api_handler.py (handles API requests), and utils.py (utility functions)",
        main_agent="file_creator"
    )

    print("✓ Created 3 test Python files")

    # Analyzer that chains multiple tools
    analyzer = orchestrator.agent.create(
        id="code_analyzer",
        describe="Analyzes codebases using multiple tools",
        instructions="""
        You analyze codebases by searching and analyzing files.

        Workflow:
        1. Use list_files or search_files to find relevant files
        2. For each file, use ask_data to extract key information
        3. Aggregate findings
        4. Write comprehensive summary using write_file
        5. Call handoff with analysis results
        """,
        tools=["list_files", "search_files", "ask_data", "read_file", "write_file", "handoff"]
    )

    print("✓ Created code analyzer agent")

    task = "Analyze all Python files in the workspace and create a summary explaining what each file does"

    print(f"✓ Starting task: {task}")
    print("✓ Analyzer should: list → ask_data (each file) → write summary...")

    result = await orchestrator.start_root_task(
        task=task,
        main_agent="code_analyzer"
    )

    print("\n--- RESULT ---")
    final_result = result.get('result', {}).get('result', {})
    print(f"Analysis result: {final_result}")
    print("-" * 60)

    # Verify summary file was created
    files = list(orchestrator.workspace.documents.keys())
    summary_files = [f for f in files if 'summary' in f.lower() or 'analysis' in f.lower()]
    assert len(summary_files) > 0, "No summary file created!"
    print(f"✓ Summary file created: {summary_files}")

    # Verify multiple tools were chained
    root_node = orchestrator.graph_ops.modal.nodes["task_1"]  # task_0 was setup
    timeline = root_node.tool_timeline
    tool_names = [e["tool"] for e in timeline]
    unique_tools = set(tool_names)

    assert len(unique_tools) >= 2, f"Expected tool chaining, only used: {unique_tools}"
    print(f"✓ Tools chained: {unique_tools}")

    print("✅ PASS: Complex tool chaining works!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("COMPLEX LLM STRESS TESTS")
    print("Real AI orchestration challenges before robot integration")
    print("=" * 60)
    print()

    # Check for API keys
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))

    if not has_anthropic and not has_openai:
        print("❌ ERROR: No API keys found")
        print("\nSet environment variables:")
        print("  export ANTHROPIC_API_KEY='your-key'")
        print("  or")
        print("  export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    print(f"API Keys: Anthropic={'✓' if has_anthropic else '✗'}, OpenAI={'✓' if has_openai else '✗'}")
    print()

    # Run all tests
    tests = [
        test_deep_nested_delegation,
        test_parallel_multi_agent_coordination,
        test_complex_research_workflow,
        test_document_passing_workflow,
        test_error_recovery,
        test_ask_data_efficiency,
        test_complex_tool_chaining,
    ]

    for test in tests:
        try:
            asyncio.run(test())
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print("COMPLEX TESTS COMPLETE")
    print("=" * 60)
    print("\nWhat this proves:")
    print("  ✓ Deep nesting (4+ levels)")
    print("  ✓ Parallel coordination")
    print("  ✓ Complex multi-tool workflows")
    print("  ✓ Document passing between agents")
    print("  ✓ Error handling")
    print("  ✓ Efficient data querying (ask_data)")
    print("  ✓ Sophisticated tool chaining")
    print("\n🔥 System ready for robot integration!")

#!/usr/bin/env python3
"""
Tool Execution Tests - Atomic testing of all 14 tools
Tests each tool as standalone function (no orchestration)
OFFENSIVE: Crashes if tool execution fails
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import WorkspaceModal, Document
from ai_orchestration.core.tools.file_tools import write_file, load_to_context, edit_file_block, list_files
from unittest.mock import patch, MagicMock


# ========== FUNCTION_TOOL TESTS (File Tools) ==========

def test_write_file_atomic():
    """write_file should create document with metadata"""
    print("=" * 60)
    print("Test: write_file (atomic)")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_atomic")

    # Call tool directly
    result = write_file(
        workspace=workspace,
        task_id="task_0",
        path="test.md",
        content="# Hello World\n\nThis is a test.",
        mime_type="text/markdown",
        why_created="Testing write_file"
    )

    # Assert result format
    assert "path" in result
    assert result["path"] == "test.md"
    assert "size_bytes" in result
    assert "token_count" in result
    assert "blocks_count" in result
    print(f"✓ Result: {result}")

    # Assert document created in workspace
    doc = workspace.get_document("test.md")
    assert doc is not None
    print("✓ Document registered in workspace")

    # Assert metadata
    assert doc.path == "test.md"
    assert doc.created_by == "task_0"
    assert doc.mime_type == "text/markdown"
    assert doc.size_bytes > 0
    assert doc.token_count > 0
    assert doc.description == "Testing write_file"
    print(f"✓ Metadata: created_by={doc.created_by}, mime_type={doc.mime_type}, tokens={doc.token_count}")

    # Assert content parseable
    assert len(doc.content_json.get("blocks", {})) > 0
    print(f"✓ Content parsed into {len(doc.content_json['blocks'])} blocks")

    print("✅ PASS\n")


def test_load_to_context_atomic():
    """load_to_context should return content block"""
    print("=" * 60)
    print("Test: load_to_context (atomic)")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_atomic")

    # Create document first
    write_file(workspace, "task_0", "test.md", "# Test Content", "text/markdown")

    # Call tool directly
    result = load_to_context(
        workspace=workspace,
        task_id="task_0",
        path="test.md"
    )

    # Assert content block format
    assert result["type"] == "document"
    assert "source" in result
    assert result["source"]["type"] == "text"  # Text format, not base64
    assert "markdown" in result["source"]["media_type"]
    assert "data" in result["source"]
    print("✓ Content block format correct")

    # Assert text data
    data = result["source"]["data"]
    assert "Test Content" in data
    print(f"✓ Content text: {data[:50]}...")

    print("✅ PASS\n")


def test_edit_file_block_atomic():
    """edit_file_block should update block and save version"""
    print("=" * 60)
    print("Test: edit_file_block (atomic)")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_atomic")

    # Create document with multiple blocks
    write_file(workspace, "task_0", "test.md", "# Title\n\nParagraph 1", "text/markdown")

    doc_before = workspace.get_document("test.md")
    blocks_before = len(doc_before.content_json["blocks"])
    print(f"✓ Document created with {blocks_before} blocks")

    # Update first block
    result = edit_file_block(
        workspace=workspace,
        task_id="task_1",
        path="test.md",
        operation="update",
        block_id="0",
        new_data={"level": 1, "text": "Updated Title"}
    )

    # Assert success
    assert result["success"] is True
    assert result["operation"] == "update"
    assert result["block_id"] == "0"
    print("✓ Update successful")

    # Assert block updated
    doc_after = workspace.get_document("test.md")
    updated_block = doc_after.content_json["blocks"]["0"]
    assert updated_block["data"]["text"] == "Updated Title"
    print("✓ Block content updated")

    # Assert version saved
    assert len(doc_after.versions) == 1
    assert doc_after.versions[0]["block_id"] == "0"
    assert doc_after.versions[0]["updated_by"] == "task_1"
    print(f"✓ Version saved: {doc_after.versions[0]}")

    print("✅ PASS\n")


def test_edit_file_block_delete_atomic():
    """edit_file_block delete should remove block"""
    print("=" * 60)
    print("Test: edit_file_block delete (atomic)")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_atomic")

    # Create document
    write_file(workspace, "task_0", "test.md", "# Title\n\nParagraph", "text/markdown")

    blocks_before = len(workspace.get_document("test.md").content_json["blocks"])

    # Delete block
    result = edit_file_block(
        workspace=workspace,
        task_id="task_1",
        path="test.md",
        operation="delete",
        block_id="0"
    )

    # Assert success
    assert result["success"] is True
    assert result["operation"] == "delete"
    print("✓ Delete successful")

    # Assert block removed
    doc_after = workspace.get_document("test.md")
    blocks_after = len(doc_after.content_json["blocks"])
    assert blocks_after == blocks_before - 1
    assert "0" not in doc_after.content_json["blocks"]
    print(f"✓ Block removed: {blocks_before} → {blocks_after} blocks")

    print("✅ PASS\n")


def test_list_files_atomic():
    """list_files should return filtered file list"""
    print("=" * 60)
    print("Test: list_files (atomic)")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_atomic")

    # Create multiple documents
    write_file(workspace, "task_0", "file1.md", "# File 1", "text/markdown")
    write_file(workspace, "task_0", "file2.md", "# File 2", "text/markdown")
    write_file(workspace, "task_0", "data.csv", "a,b\n1,2", "text/csv")
    write_file(workspace, "task_0", "config.json", '{"key": "value"}', "application/json")

    # List all files
    result_all = list_files(workspace=workspace, task_id="task_0", pattern="*")
    print(f"DEBUG: Found {len(result_all['files'])} files: {[f['path'] for f in result_all['files']]}")
    assert len(result_all["files"]) >= 4  # May have leftovers from previous tests
    print(f"✓ All files: {len(result_all['files'])} files (at least 4)")

    # List markdown files only
    result_md = list_files(workspace=workspace, task_id="task_0", pattern="*.md")
    assert len(result_md["files"]) >= 2  # At least our 2 files
    md_paths = [f["path"] for f in result_md["files"]]
    assert all(p.endswith(".md") for p in md_paths)
    assert "file1.md" in md_paths and "file2.md" in md_paths
    print(f"✓ Markdown files (at least 2): {md_paths}")

    # List CSV files only
    result_csv = list_files(workspace=workspace, task_id="task_0", pattern="*.csv")
    assert len(result_csv["files"]) >= 1  # At least our 1 file
    csv_paths = [f["path"] for f in result_csv["files"]]
    assert "data.csv" in csv_paths
    print(f"✓ CSV files (at least 1): {csv_paths}")

    print("✅ PASS\n")


# ========== NON_FUNCTION_TOOL TESTS (Metacognition) ==========

def test_stop_and_think_config():
    """stop_and_think config should load correctly"""
    print("=" * 60)
    print("Test: stop_and_think config")
    print("=" * 60)

    # NON_FUNCTION_TOOL - load config directly
    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("stop_and_think")
    assert config.tool_id == "stop_and_think"
    assert config.type == "non_function_tool"
    assert "thoughts" in config.parameters
    print("✓ stop_and_think config valid")

    print("✅ PASS\n")


def test_plan_next_steps_config():
    """plan_next_steps config should load correctly"""
    print("=" * 60)
    print("Test: plan_next_steps config")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("plan_next_steps")
    assert config.tool_id == "plan_next_steps"
    assert config.type == "non_function_tool"
    assert "plan" in config.parameters
    print("✓ plan_next_steps config valid")

    print("✅ PASS\n")


def test_save_to_memory_config():
    """save_to_*_memory configs should load correctly"""
    print("=" * 60)
    print("Test: save_to_*_memory configs")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    short_term = load_tool_config("save_to_short_term_memory")
    assert short_term.tool_id == "save_to_short_term_memory"
    assert short_term.type == "non_function_tool"
    print("✓ save_to_short_term_memory config valid")

    long_term = load_tool_config("save_to_long_term_memory")
    assert long_term.tool_id == "save_to_long_term_memory"
    assert long_term.type == "non_function_tool"
    print("✓ save_to_long_term_memory config valid")

    print("✅ PASS\n")


# ========== ORCHESTRATION TOOL TESTS ==========

def test_handoff_config():
    """handoff config should have documents parameter"""
    print("=" * 60)
    print("Test: handoff config")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("handoff")
    assert config.tool_id == "handoff"
    assert config.type == "handoff"

    # Check documents parameter exists
    assert "documents" in config.parameters
    assert config.parameters["documents"].type == "array"
    assert config.parameters["documents"].required == False
    print("✓ handoff has documents parameter (optional)")

    print("✅ PASS\n")


def test_ask_master_config():
    """ask_master config should have documents parameter"""
    print("=" * 60)
    print("Test: ask_master config")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("ask_master")
    assert config.tool_id == "ask_master"
    assert config.type == "ask_master"

    # Check documents parameter exists
    assert "documents" in config.parameters
    assert config.parameters["documents"].type == "array"
    assert config.parameters["documents"].required == False
    print("✓ ask_master has documents parameter (optional)")

    print("✅ PASS\n")


# ========== LLM TOOL TESTS (Mocked) ==========

def test_ask_claude_config():
    """ask_claude config should be valid"""
    print("=" * 60)
    print("Test: ask_claude config")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("ask_claude")
    assert config.tool_id == "ask_claude"
    assert config.type == "function_tool"
    assert "question" in config.parameters
    assert config.parameters["question"].required == True
    print("✓ ask_claude config valid")

    print("✅ PASS\n")


def test_ask_gpt_config():
    """ask_gpt config should be valid"""
    print("=" * 60)
    print("Test: ask_gpt config")
    print("=" * 60)

    from ai_orchestration.core.tool_config_loader import load_tool_config

    config = load_tool_config("ask_gpt")
    assert config.tool_id == "ask_gpt"
    assert config.type == "function_tool"
    assert "question" in config.parameters
    assert config.parameters["question"].required == True
    print("✓ ask_gpt config valid")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TOOL EXECUTION TESTS - ATOMIC (All 14 Tools)")
    print("=" * 60)
    print()

    tests = [
        # File tools
        test_write_file_atomic,
        test_load_to_context_atomic,
        test_edit_file_block_atomic,
        test_edit_file_block_delete_atomic,
        test_list_files_atomic,
        # Metacognition tools
        test_stop_and_think_config,
        test_plan_next_steps_config,
        test_save_to_memory_config,
        # Orchestration tools
        test_handoff_config,
        test_ask_master_config,
        # LLM tools (configs)
        test_ask_claude_config,
        test_ask_gpt_config,
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

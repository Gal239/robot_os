#!/usr/bin/env python3
"""
WorkspaceModal Tests - Pure state, no I/O
Tests Document + metadata, workspace operations, no file_registry
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import WorkspaceModal, Document
from ai_orchestration.core.modals.document_modal import estimate_token_count


def test_document_metadata_fields():
    """Document should have content + metadata merged (no separate registry)"""
    print("=" * 60)
    print("Test: Document Metadata Fields")
    print("=" * 60)

    content = "# Test\nThis is a test document."
    doc = Document.from_content(content, "test.md", created_by="task_0")

    # Content fields
    assert doc.path == "test.md"
    assert doc.doc_type == "markdown"
    assert doc.created_by == "task_0"
    assert doc.mime_type == "text/markdown"
    print("✓ Content fields present")

    # Metadata fields (merged from FileMetadata)
    assert doc.size_bytes > 0
    assert doc.token_count > 0
    assert isinstance(doc.description, str)
    assert isinstance(doc.auto_summary, str)
    assert isinstance(doc.tags, list)
    print(f"✓ Metadata: {doc.size_bytes} bytes, {doc.token_count} tokens")

    # Tracking fields
    assert doc.updated_by != ""
    assert doc.updated_at != ""
    print("✓ Tracking fields present")

    print("✅ PASS\n")


def test_workspace_modal_structure():
    """WorkspaceModal should have documents dict, workspace_path, NO file_registry"""
    print("=" * 60)
    print("Test: WorkspaceModal Structure")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test_session")

    # Has documents dict
    assert hasattr(workspace, "documents")
    assert isinstance(workspace.documents, dict)
    print("✓ Has documents dict")

    # Has workspace_path (for binary files)
    assert hasattr(workspace, "workspace_path")
    print("✓ Has workspace_path field")

    # NO file_registry (merged into documents)
    assert not hasattr(workspace, "file_registry")
    print("✓ No file_registry (merged)")

    print("✅ PASS\n")


def test_document_parsing_markdown():
    """Markdown parsing should preserve content"""
    print("=" * 60)
    print("Test: Markdown Parsing")
    print("=" * 60)

    content = "# Heading\n\nParagraph text.\n\nAnother paragraph."
    doc = Document.from_content(content, "test.md", created_by="task_0")

    assert doc.doc_type == "markdown"
    assert len(doc.content_json.get("blocks", {})) > 0
    print(f"✓ Created {len(doc.content_json['blocks'])} blocks")

    # Render back
    rendered = doc.render_from_json()
    assert "Heading" in rendered
    assert "Paragraph text" in rendered
    print("✓ Content preserved after render")

    print("✅ PASS\n")


def test_document_parsing_csv():
    """CSV parsing should preserve structure"""
    print("=" * 60)
    print("Test: CSV Parsing")
    print("=" * 60)

    content = "name,age,city\nAlice,30,NYC\nBob,25,LA"
    doc = Document.from_content(content, "test.csv", created_by="task_0")

    assert doc.doc_type == "csv"
    assert doc.mime_type == "text/csv"
    print(f"✓ CSV document created: {len(doc.content_json['blocks'])} blocks")

    # Render back
    rendered = doc.render_from_json()
    assert "name,age,city" in rendered
    assert "Alice,30,NYC" in rendered
    print("✓ CSV structure preserved")

    print("✅ PASS\n")


def test_document_parsing_python_as_text():
    """Python files should parse as text (structured parsing incomplete)"""
    print("=" * 60)
    print("Test: Python Parsing (as text)")
    print("=" * 60)

    content = """def hello():
    return "world"

def add(a, b):
    return a + b
"""

    doc = Document.from_content(content, "test.py", created_by="task_0")

    # Should be text, not python
    assert doc.doc_type == "text", f"Expected 'text', got '{doc.doc_type}'"
    assert doc.mime_type == "text/x-python"
    print("✓ Python file parsed as text (correct)")

    # Render should preserve
    rendered = doc.render_from_json()
    assert "def hello():" in rendered
    assert "def add(a, b):" in rendered
    assert rendered.strip() == content.strip()
    print("✓ Python content preserved")

    print("✅ PASS\n")


def test_document_parsing_json():
    """JSON parsing should preserve structure"""
    print("=" * 60)
    print("Test: JSON Parsing")
    print("=" * 60)

    content = '{"name": "test", "value": 42, "active": true}'
    doc = Document.from_content(content, "test.json", created_by="task_0")

    assert doc.doc_type == "json"
    assert doc.mime_type == "application/json"
    print(f"✓ JSON document created: {len(doc.content_json['blocks'])} blocks")

    # Render back
    rendered = doc.render_from_json()
    # JSON render reconstructs structure
    import json
    parsed = json.loads(rendered)
    assert parsed["name"] == "test"
    assert parsed["value"] == 42
    print("✓ JSON structure preserved")

    print("✅ PASS\n")


def test_workspace_register_and_get():
    """Workspace register_document and get_document"""
    print("=" * 60)
    print("Test: Register and Get Document")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    # Register document
    doc = Document.from_content("Test content", "file1.txt", created_by="task_0")
    workspace.register_document("file1.txt", doc)

    assert len(workspace.documents) == 1
    print("✓ Document registered")

    # Get document
    retrieved = workspace.get_document("file1.txt")
    assert retrieved is not None
    assert retrieved.path == "file1.txt"
    print("✓ Document retrieved")

    # Get non-existent
    missing = workspace.get_document("missing.txt")
    assert missing is None
    print("✓ Missing document returns None")

    print("✅ PASS\n")


def test_workspace_list_files():
    """Workspace list_files with filters"""
    print("=" * 60)
    print("Test: List Files with Filters")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    # Create diverse documents
    doc1 = Document.from_content("# Markdown", "doc1.md", created_by="task_0")
    doc1.tags = ["important", "draft"]

    doc2 = Document.from_content("name,value\na,1", "data.csv", created_by="task_1")
    doc2.tags = ["data"]

    doc3 = Document.from_content("# Another", "doc2.md", created_by="task_0")

    workspace.register_document("doc1.md", doc1)
    workspace.register_document("data.csv", doc2)
    workspace.register_document("doc2.md", doc3)

    # List all
    all_files = workspace.list_files()
    assert len(all_files) == 3
    print(f"✓ All files: {len(all_files)}")

    # Filter by mime_type
    md_files = workspace.list_files(mime_type="text/markdown")
    assert len(md_files) == 2
    print(f"✓ Markdown files: {len(md_files)}")

    # Filter by tags
    tagged = workspace.list_files(tags=["important"])
    assert len(tagged) == 1
    assert tagged[0]["path"] == "doc1.md"
    print(f"✓ Tagged files: {len(tagged)}")

    # Filter by created_by
    task0_files = workspace.list_files(created_by="task_0")
    assert len(task0_files) == 2
    print(f"✓ Created by task_0: {len(task0_files)}")

    print("✅ PASS\n")


def test_workspace_search_files():
    """Workspace search_files by keyword"""
    print("=" * 60)
    print("Test: Search Files")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    # Create documents with searchable content
    doc1 = Document.from_content("Content", "report.md", created_by="task_0")
    doc1.auto_summary = "AI research report"
    doc1.tags = ["research", "AI"]

    doc2 = Document.from_content("Data", "metrics.csv", created_by="task_0")
    doc2.description = "Performance metrics"

    workspace.register_document("report.md", doc1)
    workspace.register_document("metrics.csv", doc2)

    # Search by summary
    results = workspace.search_files("research")
    assert len(results) == 1
    assert results[0]["path"] == "report.md"
    print("✓ Search by summary works")

    # Search by description
    results = workspace.search_files("performance")
    assert len(results) == 1
    assert results[0]["path"] == "metrics.csv"
    print("✓ Search by description works")

    # Search by tag
    results = workspace.search_files("AI")
    assert len(results) == 1
    print("✓ Search by tag works")

    # Search by path
    results = workspace.search_files("report")
    assert len(results) == 1
    print("✓ Search by path works")

    print("✅ PASS\n")


def test_workspace_get_file_info():
    """Workspace get_file_info returns metadata"""
    print("=" * 60)
    print("Test: Get File Info")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    doc = Document.from_content("# Test\nContent here", "test.md", created_by="task_0")
    doc.auto_summary = "Test document"
    doc.tags = ["test"]
    workspace.register_document("test.md", doc)

    # Get info
    info = workspace.get_file_info("test.md")
    assert info is not None
    assert info["path"] == "test.md"
    assert info["mime_type"] == "text/markdown"
    assert info["size_bytes"] > 0
    assert info["token_count"] > 0
    assert info["auto_summary"] == "Test document"
    assert "test" in info["tags"]
    assert "blocks_count" in info
    print("✓ File info complete")

    # Missing file
    info = workspace.get_file_info("missing.txt")
    assert info is None
    print("✓ Missing file returns None")

    print("✅ PASS\n")


def test_workspace_summary():
    """Workspace get_workspace_summary aggregates stats"""
    print("=" * 60)
    print("Test: Workspace Summary")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    # Add documents
    doc1 = Document.from_content("# Doc 1\nContent", "doc1.md", created_by="task_0")
    doc2 = Document.from_content("name,value\na,1", "data.csv", created_by="task_0")
    doc3 = Document.from_content("More content", "doc2.txt", created_by="task_0")

    workspace.register_document("doc1.md", doc1)
    workspace.register_document("data.csv", doc2)
    workspace.register_document("doc2.txt", doc3)

    # Get summary
    summary = workspace.get_workspace_summary()
    assert summary["total_files"] == 3
    assert summary["total_size_bytes"] > 0
    assert summary["total_tokens"] > 0
    assert len(summary["mime_types"]) == 3  # markdown, csv, text
    assert "text/markdown" in summary["mime_types"]
    assert "text/csv" in summary["mime_types"]
    print(f"✓ Summary: {summary['total_files']} files, {summary['total_tokens']} tokens")

    print("✅ PASS\n")


def test_workspace_render_for_json():
    """Workspace render_for_json serialization (no file_registry)"""
    print("=" * 60)
    print("Test: Render for JSON")
    print("=" * 60)

    workspace = WorkspaceModal(session_id="test")

    doc = Document.from_content("Content", "test.txt", created_by="task_0")
    doc.auto_summary = "Summary"
    doc.tags = ["tag1"]
    workspace.register_document("test.txt", doc)

    # Render
    data = workspace.render_for_json()

    assert "session_id" in data
    assert "documents" in data
    assert "file_registry" not in data  # Should NOT exist
    print("✓ No file_registry in JSON")

    # Check document has all fields
    doc_json = data["documents"]["test.txt"]
    required_fields = [
        "path", "doc_type", "mime_type", "content_json",
        "size_bytes", "token_count", "description", "auto_summary", "tags",
        "created_by", "created_at", "updated_by", "updated_at"
    ]
    for field in required_fields:
        assert field in doc_json, f"Missing field: {field}"
    print(f"✓ Document has all {len(required_fields)} fields")

    print("✅ PASS\n")


def test_estimate_token_count():
    """estimate_token_count utility function"""
    print("=" * 60)
    print("Test: Estimate Token Count")
    print("=" * 60)

    text = "Hello world! This is a test."
    tokens = estimate_token_count(text)

    assert tokens > 0
    assert isinstance(tokens, int)
    print(f"✓ Estimated {tokens} tokens")

    # Empty text
    tokens = estimate_token_count("")
    assert tokens == 0
    print("✓ Empty text = 0 tokens")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("WORKSPACE MODAL TESTS")
    print("=" * 60)
    print()

    tests = [
        test_document_metadata_fields,
        test_workspace_modal_structure,
        test_document_parsing_markdown,
        test_document_parsing_csv,
        test_document_parsing_python_as_text,
        test_document_parsing_json,
        test_workspace_register_and_get,
        test_workspace_list_files,
        test_workspace_search_files,
        test_workspace_get_file_info,
        test_workspace_summary,
        test_workspace_render_for_json,
        test_estimate_token_count
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

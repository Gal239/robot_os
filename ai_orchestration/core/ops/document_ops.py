"""
DOCUMENT OPS - Business logic for document editing
Operates on WorkspaceModal (contains Document objects)
Handles document editing + persistence
"""

from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path

# Import centralized database
from ai_orchestration.utils.global_config import agent_engine_db

from ..modals.document_modal import WorkspaceModal, Document


class DocumentOps:
    """
    Business logic and persistence for Document editing
    Modal doesn't know DB - Ops handles save/load
    Ops handles edit operations with change tracking
    """

    def __init__(self, workspace: WorkspaceModal):
        self.workspace = workspace
        # Track edit operations for streaming to UI
        self.edit_stream: List[Dict] = []

    # ========== DOCUMENT OPERATIONS ==========

    def create_document(self, path: str, content: str = "", created_by: str = "system") -> Document:
        """
        Create new document in workspace

        Args:
            path: Document path (e.g., "scene_script.py")
            content: Initial content (default: empty)
            created_by: Creator ID (task_id or agent_id)

        Returns:
            Document object
        """
        # Create Document from content
        doc = Document.from_content(content, path, created_by)

        # Register in workspace
        self.workspace.register_document(path, doc)

        # Track creation as edit event
        self.edit_stream.append({
            "event": "create",
            "path": path,
            "timestamp": datetime.now().isoformat(),
            "created_by": created_by
        })

        return doc

    def get_document(self, path: str) -> Optional[Document]:
        """Get document from workspace"""
        return self.workspace.get_document(path)

    def edit_line(self, path: str, line_num: int, new_code: str, updated_by: str = "system"):
        """
        Edit specific line in document

        Args:
            path: Document path
            line_num: Line number (1-indexed)
            new_code: New code for line
            updated_by: Editor ID
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        # Document stores lines as blocks with block_id = str(line_num - 1)
        # Line 1 → block_id "0", Line 2 → block_id "1", etc.
        block_id = str(line_num - 1)

        # Edit block
        doc.edit_block(block_id, {"text": new_code}, updated_by)

        # Track edit event
        self.edit_stream.append({
            "event": "edit",
            "path": path,
            "op": "replace",
            "line": line_num,
            "code": new_code,
            "timestamp": datetime.now().isoformat(),
            "updated_by": updated_by
        })

    def insert_line(self, path: str, after_line: int, code: str, updated_by: str = "system"):
        """
        Insert new line after specified line

        Args:
            path: Document path
            after_line: Line number to insert after (0 = start of file)
            code: Code to insert
            updated_by: Editor ID
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        # Insert block (Document handles block ID assignment)
        after_block_id = str(after_line) if after_line > 0 else None
        doc.insert_block(after_block_id, "line", {"text": code}, updated_by)

        # Track edit event
        self.edit_stream.append({
            "event": "edit",
            "path": path,
            "op": "insert",
            "after_line": after_line,
            "code": code,
            "timestamp": datetime.now().isoformat(),
            "updated_by": updated_by
        })

    def delete_line(self, path: str, line_num: int, updated_by: str = "system"):
        """
        Delete line from document

        Args:
            path: Document path
            line_num: Line number to delete (1-indexed)
            updated_by: Editor ID
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        # Delete block
        block_id = str(line_num - 1)
        doc.delete_block(block_id, updated_by)

        # Track edit event
        self.edit_stream.append({
            "event": "edit",
            "path": path,
            "op": "delete",
            "line": line_num,
            "timestamp": datetime.now().isoformat(),
            "updated_by": updated_by
        })

    def get_content(self, path: str) -> str:
        """
        Get rendered content from document

        Args:
            path: Document path

        Returns:
            Rendered file content as string
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        return doc.render_from_json()

    def get_edit_stream(self) -> List[Dict]:
        """Get all edit operations (for streaming to UI)"""
        return self.edit_stream

    def clear_edit_stream(self):
        """Clear edit stream (after UI has consumed it)"""
        self.edit_stream = []

    # ========== PERSISTENCE ==========

    def save(self):
        """Save workspace to runs/session_id/workspace/"""
        # Serialize workspace
        workspace_data = self.workspace.render_for_json()

        # Save to database
        agent_engine_db.save_workspace(self.workspace.session_id, workspace_data)

        # Save each document's content to disk
        for path, doc in self.workspace.documents.items():
            content = doc.render_from_json()
            agent_engine_db.save_workspace_file(
                self.workspace.session_id,
                path,
                content
            )

    def load(self, session_id: str) -> WorkspaceModal:
        """
        Load workspace from runs/session_id/workspace/

        Args:
            session_id: Session ID to load

        Returns:
            Loaded WorkspaceModal
        """
        # Load workspace data from database
        workspace_data = agent_engine_db.load_workspace(session_id)

        if not workspace_data:
            # Create new workspace if doesn't exist
            return WorkspaceModal(session_id=session_id)

        # Reconstruct workspace from data
        workspace = WorkspaceModal(session_id=session_id)

        # Reconstruct documents
        for path, doc_data in workspace_data.get("documents", {}).items():
            doc = Document(
                path=doc_data["path"],
                doc_type=doc_data["doc_type"],
                content_json=doc_data["content_json"],
                created_by=doc_data["created_by"],
                created_at=doc_data["created_at"],
                size_bytes=doc_data.get("size_bytes", 0),
                token_count=doc_data.get("token_count", 0),
                description=doc_data.get("description", ""),
                auto_summary=doc_data.get("auto_summary", ""),
                tags=doc_data.get("tags", []),
                mime_type=doc_data.get("mime_type", ""),
                updated_by=doc_data.get("updated_by", ""),
                updated_at=doc_data.get("updated_at", ""),
                versions=doc_data.get("versions", [])
            )
            workspace.register_document(path, doc)

        return workspace

    # ========== BULK OPERATIONS ==========

    def apply_edits(self, path: str, edits: List[Dict], updated_by: str = "system"):
        """
        Apply multiple edits to document (for scene_maker handoff)

        Args:
            path: Document path
            edits: List of edit operations from agent
            updated_by: Editor ID

        Edit format:
            {"op": "insert", "after_line": 5, "code": "..."}
            {"op": "delete", "line": 3}
            {"op": "replace", "line": 4, "code": "..."}
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        # Apply each edit
        for edit in edits:
            op = edit["op"]

            if op == "insert":
                self.insert_line(path, edit["after_line"], edit["code"], updated_by)
            elif op == "delete":
                self.delete_line(path, edit["line"], updated_by)
            elif op == "replace":
                self.edit_line(path, edit["line"], edit["code"], updated_by)
            else:
                raise ValueError(f"Unknown edit operation: {op}")

    def apply_block_edits(self, path: str, edits: List[Dict], updated_by: str = "system"):
        """
        Apply block-based edits directly to document (for scene_maker handoff)

        Agent sees blocks, edits blocks - direct mapping to Document methods!

        Args:
            path: Document path
            edits: List of block-based edit operations
            updated_by: Editor ID

        Edit format:
            {"op": "insert", "after_block": "1", "code": "..."}  # Insert after block 1
            {"op": "insert", "after_block": null, "code": "..."}  # Insert at start (appends with next ID)
            {"op": "delete", "block": "2"}  # Delete block 2
            {"op": "replace", "block": "1", "code": "..."}  # Replace block 1 content
        """
        doc = self.workspace.get_document(path)
        if not doc:
            raise ValueError(f"Document not found: {path}")

        # Apply each edit using Document's block methods
        for edit in edits:
            op = edit["op"]

            if op == "insert":
                after_block = edit.get("after_block")  # None or block_id string
                code = edit["code"]

                # Insert block (Document handles ID assignment)
                doc.insert_block(after_block, "line", {"text": code}, updated_by)

                # Track for live UI streaming
                self.edit_stream.append({
                    "event": "edit",
                    "path": path,
                    "op": "insert",
                    "after_block": after_block,
                    "code": code,
                    "timestamp": datetime.now().isoformat(),
                    "updated_by": updated_by
                })

            elif op == "delete":
                block_id = edit["block"]

                # Delete block
                doc.delete_block(block_id, updated_by)

                # Track for UI
                self.edit_stream.append({
                    "event": "edit",
                    "path": path,
                    "op": "delete",
                    "block": block_id,
                    "timestamp": datetime.now().isoformat(),
                    "updated_by": updated_by
                })

            elif op == "replace":
                block_id = edit["block"]
                code = edit["code"]

                # Replace block content
                doc.edit_block(block_id, {"text": code}, updated_by)

                # Track for UI
                self.edit_stream.append({
                    "event": "edit",
                    "path": path,
                    "op": "replace",
                    "block": block_id,
                    "code": code,
                    "timestamp": datetime.now().isoformat(),
                    "updated_by": updated_by
                })
            else:
                raise ValueError(f"Unknown edit operation: {op}")

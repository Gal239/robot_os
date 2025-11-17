"""
DOCUMENT MODAL - Universal document storage
All document types use same JSON structure (blocks)
Content + metadata in one place - single source of truth
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import csv
import io
import tiktoken


def estimate_token_count(text: str, model: str = "gpt-4.5") -> int:
    """Estimate token count for text using tiktoken"""
    if not text:
        return 0

    try:
        # Get encoding for model (cl100k_base for GPT-4, Claude similar)
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception:
        # Fallback to heuristic if tiktoken fails
        return int((len(text) // 4) * 1.1)


@dataclass
class Document:
    """
    Single document with universal JSON structure + metadata
    Content + metadata in one place - single source of truth

    All document types stored as blocks:
    {
      "blocks": {
        "0": {"type": "...", "data": {...}},
        "1": {"type": "...", "data": {...}}
      }
    }
    """
    path: str
    doc_type: str  # markdown, csv, json, python, text
    content_json: Dict  # Universal: {"blocks": {"0": {}, "1": {}}}
    created_by: str
    created_at: str

    # Metadata (merged from FileMetadata)
    size_bytes: int = 0
    token_count: int = 0
    description: str = ""
    auto_summary: str = ""
    tags: List[str] = field(default_factory=list)
    mime_type: str = ""  # e.g., text/markdown, text/csv

    # Tracking
    updated_by: str = ""
    updated_at: str = ""
    versions: List[Dict] = field(default_factory=list)

    def render_from_json(self) -> str:
        """Convert universal JSON structure → actual file content"""
        if self.doc_type == "markdown":
            return self._render_markdown()
        elif self.doc_type == "csv":
            return self._render_csv()
        elif self.doc_type == "json":
            return self._render_json()
        elif self.doc_type == "python":
            return self._render_python()
        elif self.doc_type == "text":
            return self._render_text()
        else:
            return ""

    def _render_markdown(self) -> str:
        """Render markdown blocks → markdown text"""
        lines = []
        blocks = self.content_json.get("blocks", {})

        for block_id in sorted(blocks.keys(), key=int):
            block = blocks[block_id]
            block_type = block.get("type")
            data = block.get("data", {})

            if block_type == "heading":
                level = data.get("level", 1)
                text = data.get("text", "")
                lines.append("#" * level + " " + text)
            elif block_type == "paragraph":
                lines.append(data.get("text", ""))
            elif block_type == "list":
                ordered = data.get("ordered", False)
                items = data.get("items", {})
                for i in sorted(items.keys(), key=int):
                    prefix = f"{int(i)+1}." if ordered else "-"
                    lines.append(f"{prefix} {items[i]}")
            elif block_type == "code":
                lang = data.get("lang", "")
                code = data.get("code", "")
                lines.append(f"```{lang}")
                lines.append(code)
                lines.append("```")

        return "\n".join(lines)

    def _render_csv(self) -> str:
        """Render CSV blocks → CSV text"""
        blocks = self.content_json.get("blocks", {})
        output = io.StringIO()
        writer = None
        headers = None

        for block_id in sorted(blocks.keys(), key=int):
            block = blocks[block_id]
            block_type = block.get("type")
            data = block.get("data", {})

            if block_type == "header":
                columns = data.get("columns", {})
                headers = [columns[str(i)] for i in sorted([int(k) for k in columns.keys()])]
                writer = csv.writer(output)
                writer.writerow(headers)
            elif block_type == "row" and writer:
                values = data.get("values", {})
                row = [values[str(i)] for i in sorted([int(k) for k in values.keys()])]
                writer.writerow(row)

        return output.getvalue()

    def _render_json(self) -> str:
        """Render JSON blocks → JSON text"""
        # Reconstruct original JSON structure from blocks
        result = {}
        blocks = self.content_json.get("blocks", {})

        for block_id in sorted(blocks.keys(), key=int):
            block = blocks[block_id]
            data = block.get("data", {})
            key = data.get("key")
            value = data.get("value")

            if key:
                result[key] = value

        return json.dumps(result, indent=2)

    def _render_python(self) -> str:
        """Render Python blocks → Python code"""
        lines = []
        blocks = self.content_json.get("blocks", {})

        for block_id in sorted(blocks.keys(), key=int):
            block = blocks[block_id]
            block_type = block.get("type")
            data = block.get("data", {})

            if block_type == "import":
                module = data.get("module", "")
                lines.append(f"import {module}")
            elif block_type == "function":
                func_lines = data.get("lines", {})
                for line_id in sorted(func_lines.keys(), key=int):
                    lines.append(func_lines[line_id])
            elif block_type == "class":
                class_lines = data.get("lines", {})
                for line_id in sorted(class_lines.keys(), key=int):
                    lines.append(class_lines[line_id])

        return "\n".join(lines)

    def _render_text(self) -> str:
        """Render text blocks → plain text"""
        lines = []
        blocks = self.content_json.get("blocks", {})

        for block_id in sorted(blocks.keys(), key=int):
            block = blocks[block_id]
            data = block.get("data", {})
            lines.append(data.get("text", ""))

        return "\n".join(lines)

    # ========== UTILITY METHODS FOR EDITING ==========

    @staticmethod
    def from_content(content: str, path: str, created_by: str = "system") -> 'Document':
        """Create Document from file content (stateless utility)"""
        # Detect doc_type from file extension
        suffix = Path(path).suffix.lower()
        doc_type_map = {
            ".md": "markdown",
            ".csv": "csv",
            ".json": "json",
            ".py": "text",  # Python as text (structured parsing incomplete)
            ".txt": "text"
        }
        doc_type = doc_type_map.get(suffix, "text")

        # MIME type map
        mime_type_map = {
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".json": "application/json",
            ".py": "text/x-python",
            ".txt": "text/plain"
        }
        mime_type = mime_type_map.get(suffix, "text/plain")

        # Parse content to JSON blocks
        content_json = Document.parse_to_json(content, doc_type)

        # Calculate metadata
        size_bytes = len(content.encode('utf-8'))
        token_count = estimate_token_count(content)

        # Create Document object with metadata
        now = datetime.now().isoformat()
        return Document(
            path=path,
            doc_type=doc_type,
            content_json=content_json,
            created_by=created_by,
            created_at=now,
            size_bytes=size_bytes,
            token_count=token_count,
            mime_type=mime_type,
            updated_by=created_by,
            updated_at=now
        )

    def edit_block(self, block_id: str, new_data: Dict, updated_by: str = "system"):
        """Edit block in-place"""
        if block_id in self.content_json.get("blocks", {}):
            # Save version before editing
            self.versions.append({
                "timestamp": datetime.now().isoformat(),
                "updated_by": updated_by,
                "block_id": block_id,
                "old_data": self.content_json["blocks"][block_id].copy()
            })

            # Update block data
            self.content_json["blocks"][block_id]["data"] = new_data
            self.updated_by = updated_by
            self.updated_at = datetime.now().isoformat()

    def delete_block(self, block_id: str, updated_by: str = "system"):
        """Delete block in-place"""
        if block_id in self.content_json.get("blocks", {}):
            # Save version
            self.versions.append({
                "timestamp": datetime.now().isoformat(),
                "updated_by": updated_by,
                "block_id": block_id,
                "action": "delete",
                "old_data": self.content_json["blocks"][block_id].copy()
            })

            # Delete block
            del self.content_json["blocks"][block_id]
            self.updated_by = updated_by
            self.updated_at = datetime.now().isoformat()

    def insert_block(self, after_block_id: str, block_type: str, data: Dict, updated_by: str = "system"):
        """Insert new block after specified block"""
        blocks = self.content_json.get("blocks", {})

        # Find max block ID
        max_id = max([int(bid) for bid in blocks.keys()]) if blocks else -1
        new_block_id = str(max_id + 1)

        # Insert block
        blocks[new_block_id] = {
            "type": block_type,
            "data": data
        }

        self.updated_by = updated_by
        self.updated_at = datetime.now().isoformat()

    # ========== END UTILITY METHODS ==========

    @staticmethod
    def parse_to_json(raw_content: str, doc_type: str) -> Dict:
        """Convert actual file content → universal JSON structure"""
        if doc_type == "markdown":
            return Document._parse_markdown(raw_content)
        elif doc_type == "csv":
            return Document._parse_csv(raw_content)
        elif doc_type == "json":
            return Document._parse_json(raw_content)
        elif doc_type == "python":
            return Document._parse_python(raw_content)
        elif doc_type == "text":
            return Document._parse_text(raw_content)
        else:
            return {"blocks": {}}

    @staticmethod
    def _parse_markdown(content: str) -> Dict:
        """Parse markdown → blocks"""
        blocks = {}
        lines = content.split("\n")
        block_id = 0

        for line in lines:
            if line.startswith("#"):
                # Heading
                level = len(line) - len(line.lstrip("#"))
                text = line.lstrip("#").strip()
                blocks[str(block_id)] = {
                    "type": "heading",
                    "data": {"level": level, "text": text}
                }
                block_id += 1
            elif line.strip():
                # Paragraph
                blocks[str(block_id)] = {
                    "type": "paragraph",
                    "data": {"text": line}
                }
                block_id += 1

        return {"blocks": blocks}

    @staticmethod
    def _parse_csv(content: str) -> Dict:
        """Parse CSV → blocks"""
        blocks = {}
        reader = csv.reader(io.StringIO(content))
        block_id = 0

        for i, row in enumerate(reader):
            if i == 0:
                # Header
                columns = {str(j): col for j, col in enumerate(row)}
                blocks[str(block_id)] = {
                    "type": "header",
                    "data": {"columns": columns}
                }
            else:
                # Row
                values = {str(j): val for j, val in enumerate(row)}
                blocks[str(block_id)] = {
                    "type": "row",
                    "data": {"values": values}
                }
            block_id += 1

        return {"blocks": blocks}

    @staticmethod
    def _parse_json(content: str) -> Dict:
        """Parse JSON → blocks"""
        blocks = {}
        try:
            data = json.loads(content)
            block_id = 0

            for key, value in data.items():
                blocks[str(block_id)] = {
                    "type": "object" if isinstance(value, dict) else "array" if isinstance(value, list) else "value",
                    "data": {"key": key, "value": value}
                }
                block_id += 1
        except:
            pass

        return {"blocks": blocks}

    @staticmethod
    def _parse_python(content: str) -> Dict:
        """Parse Python → blocks"""
        blocks = {}
        lines = content.split("\n")
        block_id = 0

        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                # Import
                module = line.replace("import ", "").replace("from ", "").split()[0]
                blocks[str(block_id)] = {
                    "type": "import",
                    "data": {"module": module}
                }
                block_id += 1
            elif line.strip():
                # Other code (simplified)
                blocks[str(block_id)] = {
                    "type": "line",
                    "data": {"text": line}
                }
                block_id += 1

        return {"blocks": blocks}

    @staticmethod
    def _parse_text(content: str) -> Dict:
        """Parse text → blocks"""
        blocks = {}
        lines = content.split("\n")

        for i, line in enumerate(lines):
            blocks[str(i)] = {
                "type": "line",
                "data": {"text": line}
            }

        return {"blocks": blocks}


@dataclass
class WorkspaceModal:
    """
    Workspace modal - single source of truth for documents
    Documents contain both content + metadata (no separate registry)
    NO disk I/O - all file operations via auto_db
    """
    session_id: str
    documents: Dict[str, Document] = field(default_factory=dict)  # path → Document (content + metadata)
    workspace_path: Optional[Path] = None  # Optional physical path for binary files (PDFs, images, Excel)

    def register_document(self, path: str, doc: Document):
        """Add document to workspace"""
        self.documents[path] = doc

    def get_document(self, path: str) -> Optional[Document]:
        """Get document from workspace"""
        return self.documents.get(path)

    def render_for_display(self) -> str:
        """Human-readable workspace view"""
        lines = [f"# Workspace: {self.session_id}\n"]
        lines.append(f"Files: {len(self.documents)}\n")

        for path, doc in self.documents.items():
            blocks_count = len(doc.content_json.get("blocks", {}))
            lines.append(f"  {path} ({doc.doc_type}) - {blocks_count} blocks")

        return "\n".join(lines)

    # Workspace Query Methods (operate on self.documents directly)
    def list_files(self, mime_type: str = None, tags: List[str] = None, created_by: str = None) -> List[Dict]:
        """
        List all files in workspace with metadata

        Args:
            mime_type: Optional filter by MIME type (e.g., text/markdown)
            tags: Optional filter by tags (returns files with ANY of these tags)
            created_by: Optional filter by task_id that created the file

        Returns:
            List of file metadata dicts
        """
        results = list(self.documents.values())

        if mime_type:
            results = [doc for doc in results if doc.mime_type == mime_type]

        if tags:
            results = [doc for doc in results if any(tag in doc.tags for tag in tags)]

        if created_by:
            results = [doc for doc in results if doc.created_by == created_by]

        return [{
            "path": doc.path,
            "mime_type": doc.mime_type,
            "doc_type": doc.doc_type,
            "size_bytes": doc.size_bytes,
            "token_count": doc.token_count,
            "description": doc.description,
            "auto_summary": doc.auto_summary,
            "tags": doc.tags,
            "created_by": doc.created_by,
            "created_at": doc.created_at,
            "updated_by": doc.updated_by,
            "updated_at": doc.updated_at
        } for doc in results]

    def search_files(self, query: str) -> List[Dict]:
        """
        Search files by keyword (searches path, description, summary, tags)

        Args:
            query: Search query string

        Returns:
            List of matching file metadata dicts
        """
        query_lower = query.lower()
        results = []

        for doc in self.documents.values():
            # Check if query appears in any text field
            searchable = (
                doc.path.lower() +
                doc.description.lower() +
                doc.auto_summary.lower() +
                " ".join(doc.tags).lower()
            )

            if query_lower in searchable:
                results.append({
                    "path": doc.path,
                    "mime_type": doc.mime_type,
                    "doc_type": doc.doc_type,
                    "size_bytes": doc.size_bytes,
                    "token_count": doc.token_count,
                    "description": doc.description,
                    "auto_summary": doc.auto_summary,
                    "tags": doc.tags,
                    "created_by": doc.created_by,
                    "created_at": doc.created_at
                })

        return results

    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get detailed metadata for a specific file

        Args:
            file_path: Path to file in workspace

        Returns:
            File metadata dict or None
        """
        doc = self.documents.get(file_path)
        if not doc:
            return None

        return {
            "path": doc.path,
            "mime_type": doc.mime_type,
            "doc_type": doc.doc_type,
            "size_bytes": doc.size_bytes,
            "token_count": doc.token_count,
            "description": doc.description,
            "auto_summary": doc.auto_summary,
            "tags": doc.tags,
            "created_by": doc.created_by,
            "created_at": doc.created_at,
            "updated_by": doc.updated_by,
            "updated_at": doc.updated_at,
            "blocks_count": len(doc.content_json.get("blocks", {}))
        }

    def get_workspace_summary(self) -> Dict:
        """
        Get summary statistics about workspace files

        Returns:
            Dict with total files, size, breakdown by type, etc.
        """
        by_mime_type = {}
        total_size = 0
        total_tokens = 0

        for doc in self.documents.values():
            by_mime_type[doc.mime_type] = by_mime_type.get(doc.mime_type, 0) + 1
            total_size += doc.size_bytes
            total_tokens += doc.token_count

        return {
            "total_files": len(self.documents),
            "total_size_bytes": total_size,
            "total_tokens": total_tokens,
            "files_by_type": by_mime_type,
            "mime_types": list(by_mime_type.keys())
        }

    def render_for_json(self) -> Dict:
        """Full serialization - documents contain content + metadata"""
        return {
            "session_id": self.session_id,
            "documents": {
                path: {
                    "path": doc.path,
                    "doc_type": doc.doc_type,
                    "mime_type": doc.mime_type,
                    "content_json": doc.content_json,
                    "size_bytes": doc.size_bytes,
                    "token_count": doc.token_count,
                    "description": doc.description,
                    "auto_summary": doc.auto_summary,
                    "tags": doc.tags,
                    "created_by": doc.created_by,
                    "created_at": doc.created_at,
                    "updated_by": doc.updated_by,
                    "updated_at": doc.updated_at,
                    "versions": doc.versions
                }
                for path, doc in self.documents.items()
            }
        }

    def load_from_disk(self):
        """
        Load all workspace files from disk on resume
        Scans runs/session_id/workspace/ and creates Document objects
        """
        from ai_orchestration.utils.global_config import agent_engine_db

        # Get all workspace files
        file_paths = agent_engine_db.list_workspace_files(self.session_id)

        for rel_path in file_paths:
            try:
                # Load file content
                content = agent_engine_db.load_workspace_file(self.session_id, rel_path)

                # Create Document from content
                doc = Document.from_content(content, rel_path, created_by="system")

                # Register in workspace
                self.documents[rel_path] = doc
            except Exception as e:
                print(f"[WARN] Failed to load workspace file {rel_path}: {e}")

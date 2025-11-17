"""
File Tools - Actual Python functions for file operations
Uses @tool decorator for auto-registration
All file I/O via auto_db
"""

from typing import Dict
from pathlib import Path
import sys

from ..modals import WorkspaceModal, Document
from ..tool_decorator import tool

# Import auto_db for all file I/O
from ai_orchestration.utils.global_config import agent_engine_db

# Import ask_llm for auto-summarization
from ai_orchestration.third_pary_llms.ask_llm import ask_llm


# load_to_context REMOVED - no config file exists


@tool(execution_type="function")
def write_file(workspace: WorkspaceModal, task_id: str, agent_model: str, path: str, content: str, mime_type: str, why_created: str = "") -> Dict:
    """write_file"""
    # MOP: agent_model injected from agent.force_model
    # 1. Create Document from content (includes metadata: size_bytes, token_count)
    doc = Document.from_content(content, path, created_by=task_id)

    # Set description if provided
    if why_created:
        doc.description = why_created

    # 2. Save to auto_db
    agent_engine_db.save_workspace_file(workspace.session_id, path, content)

    # 3. Register in workspace
    workspace.register_document(path, doc)

    # 4. Auto-generate summary for text-based files
    summary = ""
    if mime_type.startswith("text/") or mime_type == "application/json":
        # Generate summary using Claude
        messages = [{
            "role": "user",
            "content": f"Summarize this file in 1-2 sentences:\n\n{content[:2000]}"  # Use first 2000 chars
        }]

        # MOP: Use agent's model instead of hardcoded value
        response = ask_llm(
            messages=messages,
            model=agent_model
        )

        # Extract summary from response
        if "error" not in response:
            content_blocks = response.get("content", [])
            for block in content_blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    summary = block.get("text", "")
                    break
                elif isinstance(block, str):
                    summary = block
                    break

        # Store summary in document metadata
        if summary:
            doc.auto_summary = summary

    return {
        "path": doc.path,
        "size_bytes": doc.size_bytes,
        "token_count": doc.token_count,
        "blocks_count": len(doc.content_json.get("blocks", {})),
        "summary": summary
    }


@tool(execution_type="function")
def edit_file_block(workspace: WorkspaceModal, task_id: str, path: str, operation: str, block_id: str = None, block_type: str = None, new_data: Dict = None) -> Dict:
    """edit_file_block"""
    # 1. Load document from registry or auto_db
    doc = workspace.get_document(path)
    if not doc:
        try:
            content = agent_engine_db.load_workspace_file(workspace.session_id, path)
            doc = Document.from_content(content, path, created_by=task_id)
            workspace.register_document(path, doc)
        except FileNotFoundError:
            return {"success": False, "error": "File not found"}

    # 2. Perform operation
    if operation == "update":
        # Update existing block data
        if not block_id or new_data is None:
            return {"success": False, "error": "block_id and new_data required for update"}

        if block_id not in doc.content_json.get("blocks", {}):
            return {"success": False, "error": "Block not found"}

        doc.edit_block(block_id, new_data, updated_by=task_id)

        result = {
            "success": True,
            "operation": "update",
            "block_id": block_id,
            "updated_block": doc.content_json["blocks"].get(block_id, {})
        }

    elif operation == "add":
        # Add new block
        if not block_type or new_data is None:
            return {"success": False, "error": "block_type and new_data required for add"}

        doc.insert_block(None, block_type, new_data, updated_by=task_id)

        # Find the new block ID (last one)
        blocks = doc.content_json.get("blocks", {})
        new_block_id = max([int(bid) for bid in blocks.keys()])

        result = {
            "success": True,
            "operation": "add",
            "block_id": str(new_block_id),
            "new_block": blocks[str(new_block_id)]
        }

    elif operation == "replace":
        # Replace entire block (type + data)
        if not block_id or not block_type or new_data is None:
            return {"success": False, "error": "block_id, block_type, and new_data required for replace"}

        if block_id not in doc.content_json.get("blocks", {}):
            return {"success": False, "error": "Block not found"}

        # Save version
        from datetime import datetime
        doc.versions.append({
            "timestamp": datetime.now().isoformat(),
            "updated_by": task_id,
            "block_id": block_id,
            "action": "replace",
            "old_data": doc.content_json["blocks"][block_id].copy()
        })

        # Replace entire block
        doc.content_json["blocks"][block_id] = {
            "type": block_type,
            "data": new_data
        }
        doc.updated_by = task_id
        doc.updated_at = datetime.now().isoformat()

        result = {
            "success": True,
            "operation": "replace",
            "block_id": block_id,
            "replaced_block": doc.content_json["blocks"][block_id]
        }

    elif operation == "delete":
        # Delete block
        if not block_id:
            return {"success": False, "error": "block_id required for delete"}

        if block_id not in doc.content_json.get("blocks", {}):
            return {"success": False, "error": "Block not found"}

        doc.delete_block(block_id, updated_by=task_id)

        result = {
            "success": True,
            "operation": "delete",
            "block_id": block_id,
            "blocks_remaining": len(doc.content_json.get("blocks", {}))
        }

    else:
        return {
            "success": False,
            "error": f"Unknown operation: {operation}. Use: update, add, replace, delete"
        }

    # 3. Save back to auto_db
    content = doc.render_from_json()
    agent_engine_db.save_workspace_file(workspace.session_id, path, content)

    return result


@tool(execution_type="function")
def list_files(workspace: WorkspaceModal, task_id: str, pattern: str = "*") -> Dict:
    """list_files"""
    # Get all files from auto_db
    all_files = agent_engine_db.list_workspace_files(workspace.session_id)

    # Filter by pattern (simple implementation)
    import fnmatch
    matching_files = [f for f in all_files if fnmatch.fnmatch(f, pattern)]

    # Build result with file info
    files_info = []
    for file_path in matching_files:
        # Check registry first
        doc = workspace.get_document(file_path)

        # If not in registry, load from auto_db
        if not doc:
            try:
                content = agent_engine_db.load_workspace_file(workspace.session_id, file_path)
                doc = Document.from_content(content, file_path, created_by=task_id)
                workspace.register_document(file_path, doc)
            except:
                continue  # Skip files that can't be loaded

        files_info.append({
            "path": doc.path,
            "doc_type": doc.doc_type,
            "blocks_count": len(doc.content_json.get("blocks", {}))
        })

    return {"files": files_info}



"""
Scene Maker Handoff Handler - The Execution Engine

Receives handoff(edits, message) from scene_maker agent and:
1. Applies edits to create script
2. Executes script (compile + run)
3. Captures ALL cameras (10+)
4. Saves to Scene Maker database
5. Returns message + screenshots to user

PURE MOP:
- Agent = Text editor (provides edits)
- Handler = Execution engine (compiles + runs + captures)
- Database = Complete record (scene + simulation data)
"""

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sim_agents.scene_maker_db import SceneMakerDB


def render_script_with_imports(document_ops, path: str = "scene_script.py") -> str:
    """
    Render script blocks as executable Python with auto-prepended imports

    Agent is BLIND to imports - we add them!

    Args:
        document_ops: DocumentOps instance with workspace
        path: Document path

    Returns:
        Complete executable script with imports
    """
    try:
        # Get raw content from blocks (no imports)
        raw_content = document_ops.get_content(path)
    except:
        return ""

    if not raw_content.strip():
        return ""

    # Prepend imports if not present
    if "import sys" not in raw_content:
        imports = '''import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from core.main.experiment_ops_unified import ExperimentOps

'''
        return imports + raw_content

    return raw_content


def format_script_as_blocks(document_ops, path: str = "scene_script.py") -> str:
    """
    Format script as blocks for agent prompt

    Agent sees:
    Block 0: ops = ExperimentOps(...)
    Block 1: ops.create_scene(...)
    Block 2: ops.add_robot(...)

    Args:
        document_ops: DocumentOps instance with workspace
        path: Document path

    Returns:
        Block-formatted view for agent context
    """
    doc = document_ops.get_document(path)
    if not doc:
        return "No script yet - create initial scene!"

    blocks = doc.content_json.get("blocks", {})
    if not blocks:
        return "No script yet - create initial scene!"

    lines = []
    for block_id in sorted(blocks.keys(), key=int):
        block_data = blocks[block_id]
        code = block_data.get("data", {}).get("text", "")
        lines.append(f"Block {block_id}: {code}")

    return "\n".join(lines)


def apply_edits_to_script(edits: List[Dict[str, Any]], current_script: str = "") -> str:
    """
    Apply line-based edits to script

    Args:
        edits: List of edit operations
        current_script: Current script content (empty string for new scene)

    Returns:
        Updated script content

    Example:
        edits = [
            {"op": "insert", "after_line": 0, "code": "ops = ExperimentOps()"},
            {"op": "insert", "after_line": 1, "code": "ops.create_scene()"},
        ]
        script = apply_edits_to_script(edits, "")
    """
    # Split into lines
    lines = current_script.split('\n') if current_script.strip() else []

    # Sort edits in FORWARD order (build script line by line)
    sorted_edits = sorted(
        edits,
        key=lambda e: e.get('line', e.get('after_line', 0)),
        reverse=False  # Forward order!
    )

    # Apply each edit
    for edit in sorted_edits:
        op = edit['op']

        if op == 'insert':
            after_line = edit['after_line']
            code = edit['code']
            # Insert AT position after_line (0 = start, 1 = after first line, etc.)
            # For empty file, after_line=0 means insert at position 0
            lines.insert(after_line, code)

        elif op == 'delete':
            line_num = edit['line']
            # Delete line (1-indexed in user's view, 0-indexed in array)
            if 1 <= line_num <= len(lines):
                del lines[line_num - 1]

        elif op == 'replace':
            line_num = edit['line']
            code = edit['code']
            # Replace line (1-indexed in user's view, 0-indexed in array)
            if 1 <= line_num <= len(lines):
                lines[line_num - 1] = code

    # Rebuild script
    return '\n'.join(lines)


def execute_scene_script(script: str, scene_name: str = "scene") -> Dict[str, Any]:
    """
    Execute scene script and capture results

    Args:
        script: Python script content (NO imports needed - auto-added!)
        scene_name: Scene name for identification

    Returns:
        {
            "success": bool,
            "ops": ExperimentOps instance,
            "screenshots": Dict[camera_id -> path],
            "experiment_id": str,
            "error": str (if failed)
        }
    """
    script_path = None
    try:
        # AUTO-PREPEND IMPORTS (backend responsibility, not LLM!)
        if "import sys" not in script:
            imports = """import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from core.main.experiment_ops_unified import ExperimentOps

"""
            script = imports + script
            print("   ℹ️  Auto-added imports to script")

        # Create temporary file for script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name

        # Execute script in isolated namespace with __file__ defined
        namespace = {
            '__file__': script_path,
            '__name__': '__main__'
        }
        exec(compile(script, script_path, 'exec'), namespace)

        # Get ops instance from namespace
        if 'ops' not in namespace:
            return {
                "success": False,
                "error": "Script did not create 'ops' variable"
            }

        ops = namespace['ops']

        # Capture ALL camera screenshots
        # Note: Use frame=0 explicitly since preview mode disables timeline
        screenshots = ops.save_all_screenshots(
            frame=0,  # Current frame
            subdir=f"scene_maker_{scene_name}",
            print_summary=False
        )

        return {
            "success": True,
            "ops": ops,
            "screenshots": screenshots,
            "experiment_id": ops.experiment_id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        # Clean up temp file
        if script_path:
            try:
                Path(script_path).unlink()
            except:
                pass


def handle_scene_maker_handoff(
    handoff_type: str,
    edits: List[Dict[str, Any]],
    message: str,
    scene_name: str = None,
    document_ops = None  # DocumentOps instance for edit tracking
) -> Dict[str, Any]:
    """
    Main handoff handler for scene_maker agent

    This is called by orchestrator when scene_maker calls handoff.

    Flow:
    1. Apply edits to Document (using DocumentOps)
    2. Execute script (compile + run)
    3. Capture ALL cameras
    4. Save to Scene Maker database
    5. Return message + screenshots to user

    Args:
        handoff_type: "normal_answer" (chat) or "script_edit" (build scene)
        edits: List of edit operations from agent
        message: Agent's message to user
        scene_name: Scene name (default: auto-generate from timestamp)
        document_ops: DocumentOps instance for edit tracking

    Returns:
        {
            "success": bool,
            "message": str,  # Agent's message to user
            "screenshots": Dict[camera_id -> path],
            "script": str,  # Final script content
            "experiment_id": str,
            "session_id": str,  # Scene Maker DB session
            "ui_data": Dict,  # Complete export for UI
            "error": str (if failed)
        }
    """
    print(f"\n🎬 Scene Maker Handoff Handler")
    print(f"   Type: {handoff_type}")
    print(f"   Edits: {len(edits)}")
    print(f"   Message: {message}")

    # EXPLICIT TYPE CHECK - agent declares intent!
    if handoff_type == "normal_answer":
        print(f"\n💬 normal_answer handoff - skip execution, just return message")
        # Get current script from Document
        current_script = ""
        if document_ops:
            try:
                current_script = document_ops.get_content("scene_script.py")
            except:
                current_script = ""

        return {
            "success": True,
            "message": message,
            "screenshots": {},  # No scene yet
            "script": current_script,  # Keep current script
            "ui_data": {},  # No new data
            "total_cameras": 0,
            "total_edits": 0
        }
    elif handoff_type != "script_edit":
        # OFFENSIVE: Crash if invalid type!
        raise ValueError(f"Invalid handoff_type: {handoff_type}. Must be 'normal_answer' or 'script_edit'")

    # 1. Apply edits to Document (script_edit mode)
    print(f"\n📝 Applying {len(edits)} block edits to Document...")
    print(f"   Edits received:")
    for i, edit in enumerate(edits[:3], 1):  # Show first 3
        print(f"     {i}. {edit['op']}: {edit.get('code', '')[:60]}...")

    # Apply block-based edits using DocumentOps (tracks each edit for UI streaming)
    if not document_ops:
        raise ValueError("document_ops required for script_edit mode!")

    document_ops.apply_block_edits("scene_script.py", edits, updated_by="scene_maker")

    # Get final script from Document WITH auto-prepended imports
    script = render_script_with_imports(document_ops)
    raw_script = document_ops.get_content("scene_script.py")
    print(f"   ✓ Script: {len(raw_script.split(chr(10)))} blocks")
    print(f"\n   Generated script preview (blocks):")
    block_view = format_script_as_blocks(document_ops)
    for line in block_view.split('\n')[:5]:
        print(f"     {line}")

    # Persist workspace to disk (creates workspace/scene_script.py WITH IMPORTS!)
    document_ops.save()
    # Overwrite scene_script.py with version that HAS imports (for UI display and execution)
    from ai_orchestration.utils.global_config import agent_engine_db
    agent_engine_db.save_workspace_file(
        document_ops.workspace.session_id,
        "scene_script.py",
        script  # Script WITH imports!
    )
    print(f"   ✓ Workspace saved to database/runs/{document_ops.workspace.session_id}/workspace/")
    print(f"   ✓ scene_script.py saved WITH imports ({len(script.split(chr(10)))} lines)")

    # 2. Execute script
    print(f"\n⚙️  Executing scene script...")
    result = execute_scene_script(script, scene_name or f"scene_{datetime.now().strftime('%H%M%S')}")

    if not result["success"]:
        return {
            "success": False,
            "message": f"❌ Scene execution failed: {result['error']}",
            "error": result["error"]
        }

    ops = result["ops"]
    screenshots = result["screenshots"]
    experiment_id = result["experiment_id"]

    print(f"   ✓ Scene executed successfully")
    print(f"   ✓ Experiment ID: {experiment_id}")
    print(f"   ✓ Screenshots: {len(screenshots)} cameras")

    # 3. Save to Scene Maker database
    print(f"\n💾 Saving to Scene Maker database...")
    db = SceneMakerDB()
    session_id = db.create_session(scene_name or "scene", ops)

    # Save conversation turn
    db.save_conversation_turn({
        "turn_number": 1,  # TODO: Track turn number
        "timestamp": datetime.now().isoformat(),
        "user_message": "",  # TODO: Pass user message
        "agent_message": message,
        "agent_action": "handoff",
        "edits_applied": edits,
        "all_cameras": screenshots  # ALL cameras!
    })

    # Save edits to history (block-based)
    for i, edit in enumerate(edits):
        db.save_edit({
            "edit_number": i + 1,
            "turn_number": 1,  # TODO: Track turn number
            "timestamp": datetime.now().isoformat(),
            "operation": edit["op"],
            "block": edit.get("block", edit.get("after_block")),  # Block ID
            "code": edit.get("code", ""),
            "success": True
        })

    # Generate knowledge base
    db.generate_knowledge_base()

    print(f"   ✓ Session ID: {session_id}")

    # 4. Export complete data for UI
    print(f"\n📦 Exporting data for UI...")
    ui_data = db.export_for_ui()
    print(f"   ✓ UI data exported")
    print(f"      - Cameras: {len(ui_data['cameras'])}")
    print(f"      - Sensors: {len(ui_data['sensors'])}")
    print(f"      - Assets: {len(ui_data['assets'])}")

    # 5. Close ops
    ops.close()

    # 6. Return to user
    return {
        "success": True,
        "message": message,  # Agent's message
        "screenshots": screenshots,  # Dict[camera_id -> path]
        "script": script,  # Final script
        "experiment_id": experiment_id,
        "session_id": session_id,
        "ui_data": ui_data,  # Complete export
        "total_cameras": len(screenshots),
        "total_edits": len(edits)
    }


if __name__ == "__main__":
    # Test the handler
    print("\n" + "="*80)
    print("SCENE MAKER HANDOFF HANDLER - TEST")
    print("="*80)

    # Test 1: Apply edits
    print("\n📝 Test 1: Apply Edits")
    edits = [
        {"op": "insert", "after_line": 0, "code": "# Line 1"},
        {"op": "insert", "after_line": 1, "code": "# Line 2"},
        {"op": "insert", "after_line": 2, "code": "# Line 3"},
    ]
    script = apply_edits_to_script(edits, "")
    print(f"   ✓ Script created: {len(script.split(chr(10)))} lines")
    print(f"   Content:\n{script}")

    # Test 2: Full handler (with real scene)
    print("\n🎬 Test 2: Full Handler Execution")
    print("   (Requires ExperimentOps - skipping in unit test)")
    print("   Use test_scene_maker_complete.py for full integration test")

    print("\n✅ Handler tests complete!")
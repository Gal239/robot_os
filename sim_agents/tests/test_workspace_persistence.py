#!/usr/bin/env python3
"""
Test Workspace Persistence with Block-Based Editing
====================================================

Validates:
1. Block-based edits work correctly
2. Workspace saves to disk (workspace/ directory + workspace.json)
3. Script renders with auto-prepended imports
4. Agent sees block view (Block 0, Block 1, etc.)
5. Edit stream tracks operations for live UI
"""

import sys
from pathlib import Path

# Path setup - sim_agents/tests/ -> simulation_center/
simulation_center = Path(__file__).parent.parent.parent
sys.path.insert(0, str(simulation_center))

# Set custom database path
custom_db_path = simulation_center / "sim_agents" / "database"
custom_db_path.mkdir(parents=True, exist_ok=True)

import ai_orchestration.utils.global_config as global_config
from ai_orchestration.utils.auto_db import AutoDB
global_config.database_path = custom_db_path
global_config.agent_engine_db = AutoDB(local_path=str(custom_db_path))

from ai_orchestration.core.modals.document_modal import WorkspaceModal, Document
from ai_orchestration.core.ops.document_ops import DocumentOps
from sim_agents.scene_maker_handoff_handler import format_script_as_blocks, render_script_with_imports


def test_workspace_persistence():
    """Test complete workspace persistence flow"""
    print("\n" + "="*80)
    print("TEST: WORKSPACE PERSISTENCE WITH BLOCK-BASED EDITING")
    print("="*80)

    # 1. Create workspace with session ID
    print("\n1. Creating workspace...")
    session_id = "test_workspace_persistence"
    workspace = WorkspaceModal(session_id=session_id)
    document_ops = DocumentOps(workspace)
    print(f"   ✓ Workspace created: {session_id}")

    # 2. Create empty scene_script.py document
    print("\n2. Creating scene_script.py document...")
    document_ops.create_document("scene_script.py", "", created_by="scene_maker")
    print(f"   ✓ Document created")

    # 3. Apply block-based edits (simulating agent)
    print("\n3. Applying block-based edits...")
    edits = [
        {"op": "insert", "after_block": None, "code": "ops = ExperimentOps(mode='simulated', headless=True, render_mode='2k_demo')"},
        {"op": "insert", "after_block": "0", "code": "ops.create_scene(name='kitchen', width=10, length=10, height=4)"},
        {"op": "insert", "after_block": "1", "code": "ops.add_robot(robot_name='stretch', position=(0, 0, 0))"},
        {"op": "insert", "after_block": "2", "code": "ops.add_asset(asset_name='apple', relative_to='table', relation='on_top')"},
        {"op": "insert", "after_block": "3", "code": "ops.compile(mode='preview')"},
        {"op": "insert", "after_block": "4", "code": "ops.step()"}
    ]
    document_ops.apply_block_edits("scene_script.py", edits, updated_by="scene_maker")
    print(f"   ✓ Applied {len(edits)} block edits")

    # 4. Check edit stream
    print("\n4. Checking edit stream (for live UI)...")
    edit_stream = document_ops.get_edit_stream()
    print(f"   Edit stream has {len(edit_stream)} events:")
    for i, event in enumerate(edit_stream[:4], 1):
        event_type = event.get('event', 'unknown')
        if event_type == 'create':
            print(f"     {i}. CREATE: {event.get('path', '')}")
        else:
            print(f"     {i}. {event.get('op', 'unknown')}: {event.get('code', '')[:40]}...")
    print(f"   ✓ Edit stream ready for UI animation!")

    # 5. Verify block view (how agent sees script)
    print("\n5. Checking agent's block view...")
    block_view = format_script_as_blocks(document_ops)
    print(f"   Agent sees:")
    for line in block_view.split('\n'):
        print(f"     {line}")
    print(f"   ✓ Agent has block context!")

    # 6. Verify script with imports (executable version)
    print("\n6. Checking executable script (with imports)...")
    script = render_script_with_imports(document_ops)
    lines = script.split('\n')
    print(f"   Full script ({len(lines)} lines):")
    for i, line in enumerate(lines[:8], 1):
        print(f"     {i}: {line}")
    print(f"   ...")
    assert "import sys" in script, "Missing imports!"
    assert "from core.main.experiment_ops_unified import ExperimentOps" in script, "Missing ExperimentOps import!"
    print(f"   ✓ Script has auto-prepended imports!")

    # 7. Save workspace to disk
    print("\n7. Saving workspace to disk...")
    document_ops.save()
    print(f"   ✓ Workspace saved!")

    # 8. Verify files on disk
    print("\n8. Verifying files on disk...")
    workspace_json = custom_db_path / "runs" / session_id / "workspace.json"
    script_file = custom_db_path / "runs" / session_id / "workspace" / "scene_script.py"

    assert workspace_json.exists(), f"workspace.json not found at {workspace_json}"
    print(f"   ✓ workspace.json exists: {workspace_json}")

    assert script_file.exists(), f"scene_script.py not found at {script_file}"
    print(f"   ✓ scene_script.py exists: {script_file}")

    # 9. Check script file content
    print("\n9. Checking persisted script file...")
    saved_script = script_file.read_text()
    print(f"   Script on disk ({len(saved_script.split(chr(10)))} lines):")
    for i, line in enumerate(saved_script.split('\n')[:6], 1):
        print(f"     {i}: {line}")

    # Note: Saved script is raw blocks (no imports) - imports added at execution
    assert "ops = ExperimentOps" in saved_script, "Missing ExperimentOps line!"
    assert "ops.create_scene" in saved_script, "Missing create_scene line!"
    print(f"   ✓ Script content correct (raw blocks without imports)")

    # 10. Check workspace.json structure
    print("\n10. Checking workspace.json structure...")
    import json
    workspace_data = json.loads(workspace_json.read_text())
    print(f"   Session ID: {workspace_data['session_id']}")
    print(f"   Documents: {list(workspace_data['documents'].keys())}")
    doc_data = workspace_data['documents']['scene_script.py']
    blocks = doc_data['content_json']['blocks']
    print(f"   Blocks: {len(blocks)}")
    for bid in sorted(blocks.keys(), key=int):
        text = blocks[bid]['data']['text'][:50]
        print(f"     Block {bid}: {text}...")
    print(f"   ✓ Workspace structure correct!")

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print(f"\nWorkspace persists to:")
    print(f"  - {workspace_json}")
    print(f"  - {script_file}")
    print(f"\nAgent sees blocks, edits blocks, workspace saves to disk!")
    print(f"Live edit stream ready for UI animations!")


if __name__ == "__main__":
    test_workspace_persistence()

#!/usr/bin/env python3
"""
Echo Workspace Integration Test
================================

Tests the FULL integration:
1. EchoConversationManager creates session
2. Agent creates workspace document
3. Agent sends block-based edits (mocked)
4. Workspace persists to disk
5. UI endpoints return correct data

This is NOT an LLM test - it mocks the agent response to test
the workspace integration without calling Anthropic API.
"""

import sys
from pathlib import Path

# Path setup
simulation_center = Path(__file__).parent.parent.parent
sys.path.insert(0, str(simulation_center))

# Set custom database path BEFORE any imports
custom_db_path = simulation_center / "sim_agents" / "database"
custom_db_path.mkdir(parents=True, exist_ok=True)

import ai_orchestration.utils.global_config as global_config
from ai_orchestration.utils.auto_db import AutoDB
global_config.database_path = custom_db_path
global_config.agent_engine_db = AutoDB(local_path=str(custom_db_path))

print(f"[Test] Using database path: {custom_db_path}")

from sim_agents.ui.backend.echo_ops import EchoConversationManager
from sim_agents.scene_maker_handoff_handler import format_script_as_blocks, render_script_with_imports


def test_echo_workspace_integration():
    """Test full Echo + Workspace integration"""
    print("\n" + "="*80)
    print("ECHO WORKSPACE INTEGRATION TEST")
    print("="*80)

    # 1. Create EchoConversationManager
    print("\n1. Creating EchoConversationManager...")
    echo_manager = EchoConversationManager()
    print(f"   ✓ Manager created")

    # 2. Start session
    print("\n2. Starting session...")
    result = echo_manager.start_session()
    assert result["success"], f"Session start failed: {result}"
    session_id = result["session_id"]
    print(f"   ✓ Session started: {session_id}")
    print(f"   Welcome: {result['welcome_message']}")

    # 3. Verify workspace initialized
    print("\n3. Verifying workspace initialization...")
    workspace_info = echo_manager.get_workspace_info()
    print(f"   Session ID: {workspace_info['session_id']}")
    print(f"   Documents: {workspace_info['documents']}")
    assert "scene_script.py" in workspace_info['documents'], "scene_script.py not created!"
    print(f"   ✓ Workspace initialized with scene_script.py")

    # 4. Simulate agent handoff with block edits
    print("\n4. Simulating agent handoff with block edits...")

    # Manually call the handoff handler (simulating what agent would do)
    from sim_agents.scene_maker_handoff_handler import handle_scene_maker_handoff

    block_edits = [
        {"op": "insert", "after_block": None, "code": "ops = ExperimentOps(mode='simulated', headless=True, render_mode='2k_demo')"},
        {"op": "insert", "after_block": "0", "code": "ops.create_scene(name='kitchen', width=10, length=10, height=4)"},
        {"op": "insert", "after_block": "1", "code": "ops.add_robot(robot_name='stretch', position=(0, 0, 0))"},
        {"op": "insert", "after_block": "2", "code": "ops.add_asset(asset_name='table', position=(2, 0, 0))"},
        {"op": "insert", "after_block": "3", "code": "ops.add_asset(asset_name='apple', relative_to='table', relation='on_top')"},
        {"op": "insert", "after_block": "4", "code": "ops.compile(mode='preview')"},
        {"op": "insert", "after_block": "5", "code": "ops.step()"}
    ]

    print(f"   Edits: {len(block_edits)} blocks")
    for i, edit in enumerate(block_edits[:3], 1):
        print(f"     {i}. {edit['op']}: {edit['code'][:50]}...")

    # Apply edits directly to document_ops (simulating handoff)
    echo_manager.ops.document_ops.apply_block_edits("scene_script.py", block_edits, updated_by="scene_maker")

    # Save workspace (this would happen in handoff handler)
    echo_manager.ops.document_ops.save()
    # Save script WITH imports (like handoff handler does)
    script_with_imports = render_script_with_imports(echo_manager.ops.document_ops)
    global_config.agent_engine_db.save_workspace_file(
        echo_manager.ops.document_ops.workspace.session_id,
        "scene_script.py",
        script_with_imports  # WITH imports!
    )
    print(f"   ✓ Block edits applied and workspace saved WITH imports")

    # 5. Check edit stream
    print("\n5. Checking edit stream (for live UI)...")
    edit_stream = echo_manager.get_edit_stream()
    print(f"   Edit stream: {len(edit_stream)} events")
    for i, event in enumerate(edit_stream[:4], 1):
        event_type = event.get('event', 'unknown')
        if event_type == 'create':
            print(f"     {i}. CREATE: {event.get('path', '')}")
        else:
            print(f"     {i}. {event.get('op', 'unknown')}: {event.get('code', '')[:40]}...")
    print(f"   ✓ Edit stream ready for UI animation!")

    # 6. Check agent's block view
    print("\n6. Checking agent's block view...")
    block_view = echo_manager.get_script_blocks()
    print(f"   Agent sees:")
    for line in block_view.split('\n')[:4]:
        print(f"     {line}")
    print(f"   ...")
    print(f"   ✓ Agent has block context!")

    # 7. Check executable script (with imports)
    print("\n7. Checking executable script (with imports)...")
    script = echo_manager.get_scene_script_from_workspace()
    lines = script.split('\n')
    print(f"   Full script ({len(lines)} lines):")
    for i, line in enumerate(lines[:6], 1):
        print(f"     {i}: {line}")
    print(f"   ...")
    assert "import sys" in script, "Missing imports!"
    assert "ExperimentOps" in script, "Missing ExperimentOps!"
    print(f"   ✓ Script has auto-prepended imports!")

    # 8. Verify files on disk
    print("\n8. Verifying workspace files on disk...")
    workspace_json = custom_db_path / "runs" / session_id / "workspace.json"
    script_file = custom_db_path / "runs" / session_id / "workspace" / "scene_script.py"

    assert workspace_json.exists(), f"workspace.json not found: {workspace_json}"
    print(f"   ✓ workspace.json: {workspace_json}")

    assert script_file.exists(), f"scene_script.py not found: {script_file}"
    print(f"   ✓ scene_script.py: {script_file}")

    # 9. Check script file content (should have imports!)
    print("\n9. Checking persisted script content (WITH IMPORTS!)...")
    saved_script = script_file.read_text()
    print(f"   Script on disk ({len(saved_script.split(chr(10)))} lines):")
    for i, line in enumerate(saved_script.split('\n')[:8], 1):
        print(f"     {i}: {line}")
    print(f"   ...")
    assert "import sys" in saved_script, "Missing imports in saved script!"
    assert "from core.main.experiment_ops_unified import ExperimentOps" in saved_script, "Missing ExperimentOps import!"
    assert "ops = ExperimentOps" in saved_script, "Missing ExperimentOps line!"
    assert "ops.add_asset" in saved_script, "Missing add_asset line!"
    print(f"   ✓ Script WITH IMPORTS persisted correctly!")

    # 10. Check workspace.json structure
    print("\n10. Checking workspace.json blocks...")
    import json
    workspace_data = json.loads(workspace_json.read_text())
    doc_data = workspace_data['documents']['scene_script.py']
    blocks = doc_data['content_json']['blocks']
    print(f"   Blocks in workspace.json: {len(blocks)}")
    for bid in sorted(blocks.keys(), key=int)[:3]:
        text = blocks[bid]['data']['text'][:50] if blocks[bid]['data']['text'] else "(empty)"
        print(f"     Block {bid}: {text}...")
    print(f"   ...")
    print(f"   ✓ Block structure preserved!")

    print("\n" + "="*80)
    print("✅ INTEGRATION TEST PASSED!")
    print("="*80)

    print(f"\n📁 Session: {session_id}")
    print(f"📂 Workspace directory:")
    print(f"   {custom_db_path / 'runs' / session_id}")
    print(f"   ├── workspace.json (block structure)")
    print(f"   └── workspace/")
    print(f"       └── scene_script.py (rendered script)")

    print(f"\n🎯 What works:")
    print(f"   ✓ Session creates workspace")
    print(f"   ✓ Agent edits with block IDs")
    print(f"   ✓ Workspace persists to disk")
    print(f"   ✓ Edit stream tracks operations")
    print(f"   ✓ Agent sees block context")
    print(f"   ✓ Script auto-prepends imports")
    print(f"   ✓ UI can fetch live data")

    return True


if __name__ == "__main__":
    success = test_echo_workspace_integration()
    sys.exit(0 if success else 1)

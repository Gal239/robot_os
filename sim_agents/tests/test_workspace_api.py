#!/usr/bin/env python3
"""
Test Workspace API Endpoints
=============================

Tests the Flask API endpoints for workspace functionality.
Starts a test server and makes requests to verify:
- /api/workspace/script
- /api/workspace/blocks
- /api/workspace/info
- /api/edits
"""

import sys
from pathlib import Path

# Path setup
simulation_center = Path(__file__).parent.parent.parent
sys.path.insert(0, str(simulation_center))
sys.path.insert(0, str(simulation_center / "sim_agents" / "ui" / "backend"))

# Set custom database path BEFORE any imports
custom_db_path = simulation_center / "sim_agents" / "database"
custom_db_path.mkdir(parents=True, exist_ok=True)

import ai_orchestration.utils.global_config as global_config
from ai_orchestration.utils.auto_db import AutoDB
global_config.database_path = custom_db_path
global_config.agent_engine_db = AutoDB(local_path=str(custom_db_path))

from api import app, echo_manager
from echo_ops import EchoConversationManager
import api


def test_workspace_api():
    """Test workspace API endpoints"""
    print("\n" + "="*80)
    print("WORKSPACE API ENDPOINT TEST")
    print("="*80)

    # 1. Create test client
    print("\n1. Creating Flask test client...")
    app.config['TESTING'] = True
    client = app.test_client()
    print(f"   ✓ Test client created")

    # 2. Start session
    print("\n2. Starting session via POST /api/start...")
    response = client.post('/api/start')
    data = response.get_json()
    assert data['success'], f"Start failed: {data}"
    session_id = data['session_id']
    print(f"   ✓ Session started: {session_id}")
    print(f"   Welcome: {data.get('welcome_message', 'N/A')}")

    # 3. Check workspace info (empty at start)
    print("\n3. Testing GET /api/workspace/info...")
    response = client.get('/api/workspace/info')
    data = response.get_json()
    assert data['success'], f"Workspace info failed: {data}"
    print(f"   Session ID: {data['workspace']['session_id']}")
    print(f"   Documents: {data['workspace']['documents']}")
    assert "scene_script.py" in data['workspace']['documents']
    print(f"   ✓ Workspace info correct!")

    # 4. Check blocks (empty initially)
    print("\n4. Testing GET /api/workspace/blocks (before edits)...")
    response = client.get('/api/workspace/blocks')
    data = response.get_json()
    assert data['success'], f"Blocks failed: {data}"
    print(f"   Block view: {data['blocks'][:50]}...")
    print(f"   Block count: {data['block_count']}")
    print(f"   ✓ Block endpoint works!")

    # 5. Simulate adding edits (directly to document_ops)
    print("\n5. Adding block edits to workspace...")
    block_edits = [
        {"op": "insert", "after_block": None, "code": "ops = ExperimentOps(mode='simulated')"},
        {"op": "insert", "after_block": "0", "code": "ops.create_scene(name='test')"},
        {"op": "insert", "after_block": "1", "code": "ops.compile()"},
    ]
    api.echo_manager.ops.document_ops.apply_block_edits("scene_script.py", block_edits, updated_by="test")
    api.echo_manager.ops.document_ops.save()
    print(f"   ✓ Applied {len(block_edits)} block edits")

    # 6. Check blocks after edits
    print("\n6. Testing GET /api/workspace/blocks (after edits)...")
    response = client.get('/api/workspace/blocks')
    data = response.get_json()
    print(f"   Block view:")
    for line in data['blocks'].split('\n')[:3]:
        print(f"     {line}")
    print(f"   Block count: {data['block_count']}")
    assert data['block_count'] > 0, "No blocks after edits!"
    print(f"   ✓ Blocks updated correctly!")

    # 7. Check script with imports
    print("\n7. Testing GET /api/workspace/script...")
    response = client.get('/api/workspace/script')
    data = response.get_json()
    assert data['success'], f"Script failed: {data}"
    print(f"   Script ({data['line_count']} lines):")
    for i, line in enumerate(data['script'].split('\n')[:6], 1):
        print(f"     {i}: {line}")
    print(f"   Has imports: {data['has_imports']}")
    assert data['has_imports'], "Script missing imports!"
    print(f"   ✓ Script with imports correct!")

    # 8. Check edit stream
    print("\n8. Testing GET /api/edits...")
    response = client.get('/api/edits')
    data = response.get_json()
    assert data['success'], f"Edits failed: {data}"
    print(f"   Edit stream: {len(data['edits'])} events")
    for i, event in enumerate(data['edits'][:4], 1):
        event_type = event.get('event', 'unknown')
        if event_type == 'create':
            print(f"     {i}. CREATE: {event.get('path', '')}")
        else:
            print(f"     {i}. {event.get('op', 'unknown')}: {event.get('code', '')[:30]}...")
    print(f"   ✓ Edit stream accessible!")

    # 9. Clear edit stream
    print("\n9. Testing POST /api/workspace/edits/clear...")
    response = client.post('/api/workspace/edits/clear')
    data = response.get_json()
    assert data['success'], f"Clear failed: {data}"
    print(f"   ✓ Edit stream cleared")

    # Verify cleared
    response = client.get('/api/edits')
    data = response.get_json()
    print(f"   Edit stream after clear: {len(data['edits'])} events")
    assert len(data['edits']) == 0, "Edit stream not cleared!"
    print(f"   ✓ Edit stream empty!")

    # 10. Verify files on disk
    print("\n10. Verifying workspace files on disk...")
    workspace_json = custom_db_path / "runs" / session_id / "workspace.json"
    script_file = custom_db_path / "runs" / session_id / "workspace" / "scene_script.py"
    assert workspace_json.exists(), f"workspace.json not found!"
    assert script_file.exists(), f"scene_script.py not found!"
    print(f"   ✓ workspace.json exists")
    print(f"   ✓ scene_script.py exists")

    print("\n" + "="*80)
    print("✅ ALL API TESTS PASSED!")
    print("="*80)

    print(f"\n🎯 Working endpoints:")
    print(f"   GET  /api/workspace/script   - Script with imports")
    print(f"   GET  /api/workspace/blocks   - Agent's block view")
    print(f"   GET  /api/workspace/info     - Workspace metadata")
    print(f"   GET  /api/edits              - Live edit stream")
    print(f"   POST /api/workspace/edits/clear - Clear edit stream")

    return True


if __name__ == "__main__":
    success = test_workspace_api()
    sys.exit(0 if success else 1)

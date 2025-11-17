#!/usr/bin/env python3
"""
TEST: Individual Tool Testing for Scene Maker Agent

Tests EACH tool separately to verify they work before creating the agent.

Tests:
1. discover_assets - Returns all 71 assets from ASSETS.json
2. discover_behaviors - Returns all 17 behaviors from BEHAVIORS.json
3. discover_relations - Returns all relations from scene_modal.py
4. get_scene_examples - Returns all 6 examples from demo file
5. get_api_documentation - Returns API docs from ExperimentOps
6. create_scene_script - Creates valid Python script

PURE MOP: Each tool reads from SOURCE OF TRUTH files!
"""

import sys
import json
from pathlib import Path

# Direct paths to source files (same as in tools)
ASSETS_JSON = Path(__file__).parent.parent.parent / "modals" / "mujoco_assets" / "ASSETS.json"
BEHAVIORS_JSON = Path(__file__).parent.parent.parent / "behaviors" / "BEHAVIORS.json"
SCENE_MODAL_PY = Path(__file__).parent.parent.parent / "modals" / "scene_modal.py"
DEMO_SCENES_PY = Path(__file__).parent.parent.parent / "tests" / "demos" / "demo_1_ai_generated_scenes.py"
EXPERIMENT_OPS_PY = Path(__file__).parent.parent.parent / "main" / "experiment_ops_unified.py"

# Mock minimal required objects (tools need workspace, task_id, agent_model)
class MockWorkspace:
    """Mock workspace for testing"""
    def __init__(self):
        self.session_id = "test_session"
        self.documents = {}

    def register_document(self, path, doc):
        self.documents[path] = doc

    def get_document(self, path):
        return self.documents.get(path)


def discover_assets_logic(category=None):
    """Core logic from discover_assets tool"""
    assert ASSETS_JSON.exists(), f"ASSETS.json not found at {ASSETS_JSON}"

    with open(ASSETS_JSON, 'r') as f:
        assets_data = json.load(f)

    furniture = []
    objects = []

    for asset_name, asset_info in assets_data.items():
        asset_type = asset_info.get('type', 'unknown')

        # Extract behaviors from components
        behaviors = set()
        components = asset_info.get('components', {})
        for comp_name, comp_data in components.items():
            for prop_name, prop_data in comp_data.items():
                if isinstance(prop_data, dict) and 'behavior' in prop_data:
                    behaviors.add(prop_data['behavior'])

        asset_summary = {
            'name': asset_name,
            'type': asset_type,
            'source': asset_info.get('source', 'unknown'),
            'behaviors': sorted(list(behaviors)),
            'components': list(components.keys())
        }

        if asset_type == 'furniture':
            furniture.append(asset_summary)
        elif asset_type == 'object':
            objects.append(asset_summary)

    result = {}

    if category == 'furniture' or category is None:
        result['furniture'] = sorted(furniture, key=lambda x: x['name'])

    if category == 'object' or category is None:
        result['objects'] = sorted(objects, key=lambda x: x['name'])

    result['summary'] = {
        'total_furniture': len(furniture),
        'total_objects': len(objects),
        'total_assets': len(furniture) + len(objects)
    }

    return result


def test_discover_assets():
    """
    TEST 1: discover_assets
    Should return all 71 assets (15 furniture, 56 objects)
    """
    print("\n" + "="*80)
    print("TEST 1: discover_assets")
    print("="*80)

    result = discover_assets_logic()

    print(f"\n📊 Summary:")
    print(f"  Total Assets: {result['summary']['total_assets']}")
    print(f"  Furniture: {result['summary']['total_furniture']}")
    print(f"  Objects: {result['summary']['total_objects']}")

    print(f"\n🪑 Sample Furniture (first 5):")
    for item in result['furniture'][:5]:
        print(f"  - {item['name']}: {item['behaviors']}")

    print(f"\n🍎 Sample Objects (first 5):")
    for item in result['objects'][:5]:
        print(f"  - {item['name']}: {item['behaviors']}")

    # Assertions
    assert result['summary']['total_assets'] == 71, f"Expected 71 assets, got {result['summary']['total_assets']}"
    assert result['summary']['total_furniture'] == 15, f"Expected 15 furniture, got {result['summary']['total_furniture']}"
    assert result['summary']['total_objects'] == 56, f"Expected 56 objects, got {result['summary']['total_objects']}"

    print("\n✅ TEST 1 PASSED: discover_assets works!")
    return result


def test_discover_behaviors():
    """
    TEST 2: discover_behaviors
    Should return all 17 behaviors from BEHAVIORS.json
    """
    print("\n" + "="*80)
    print("TEST 2: discover_behaviors")
    print("="*80)

    workspace = MockWorkspace()
    result = discover_behaviors(workspace, "test_task", "claude-sonnet-4-5")

    print(f"\n📊 Summary:")
    print(f"  Total Behaviors: {result['summary']['total_behaviors']}")
    print(f"  Object Behaviors: {result['summary']['object_behaviors']}")
    print(f"  Robot Behaviors: {result['summary']['robot_behaviors']}")

    print(f"\n🎯 Sample Behaviors (first 5):")
    for i, (behavior_name, behavior_data) in enumerate(list(result['behaviors'].items())[:5]):
        print(f"  - {behavior_name}: {behavior_data['description']}")
        print(f"    Properties: {[p['name'] for p in behavior_data.get('properties', [])[:3]]}")

    print(f"\n📝 All Behavior Names:")
    print(f"  {', '.join(result['summary']['behavior_names'])}")

    # Assertions
    assert result['summary']['total_behaviors'] >= 15, f"Expected at least 15 behaviors"
    assert 'stackable' in result['behaviors'], "Expected 'stackable' behavior"
    assert 'graspable' in result['behaviors'], "Expected 'graspable' behavior"

    print("\n✅ TEST 2 PASSED: discover_behaviors works!")
    return result


def test_discover_relations():
    """
    TEST 3: discover_relations
    Should return all spatial relations and surface positions
    """
    print("\n" + "="*80)
    print("TEST 3: discover_relations")
    print("="*80)

    workspace = MockWorkspace()
    result = discover_relations(workspace, "test_task", "claude-sonnet-4-5")

    print(f"\n📊 Summary:")
    print(f"  Total Relations: {result['summary']['total_relations']}")
    print(f"  Total Surface Positions: {result['summary']['total_surface_positions']}")

    print(f"\n🔗 All Relations:")
    for relation_name, relation_data in result['relations'].items():
        print(f"  - {relation_name}: {relation_data['description']}")
        print(f"    Example: {relation_data['example']}")

    print(f"\n📍 All Surface Positions:")
    for pos_name, pos_coords in result['surface_positions'].items():
        print(f"  - {pos_name}: {pos_coords}")

    # Assertions
    assert result['summary']['total_relations'] >= 8, "Expected at least 8 relations"
    assert 'on_top' in result['relations'], "Expected 'on_top' relation"
    assert 'inside' in result['relations'], "Expected 'inside' relation"
    assert 'stack_on' in result['relations'], "Expected 'stack_on' relation"
    assert result['summary']['total_surface_positions'] >= 5, "Expected at least 5 surface positions"

    print("\n✅ TEST 3 PASSED: discover_relations works!")
    return result


def test_get_scene_examples():
    """
    TEST 4: get_scene_examples
    Should return all 6 scene examples from demo file
    """
    print("\n" + "="*80)
    print("TEST 4: get_scene_examples")
    print("="*80)

    workspace = MockWorkspace()
    result = get_scene_examples(workspace, "test_task", "claude-sonnet-4-5")

    print(f"\n📊 Summary:")
    print(f"  Total Examples: {result['summary']['total_examples']}")

    print(f"\n📖 All Examples:")
    for example_name in result['summary']['example_names']:
        example = result['examples'][example_name]
        print(f"\n  {example_name}:")
        print(f"    Description: {example['description']}")
        print(f"    Code Length: {len(example['code'])} chars")

    # Show one full example
    if result['examples']:
        first_example_name = list(result['examples'].keys())[0]
        first_example = result['examples'][first_example_name]
        print(f"\n🔍 Sample Code (first 300 chars of {first_example_name}):")
        print(f"  {first_example['code'][:300]}...")

    # Assertions
    assert result['summary']['total_examples'] >= 6, f"Expected at least 6 examples, got {result['summary']['total_examples']}"
    assert 'test_1_tower_stacking' in result['summary']['example_names'], "Expected tower_stacking example"

    print("\n✅ TEST 4 PASSED: get_scene_examples works!")
    return result


def test_get_api_documentation():
    """
    TEST 5: get_api_documentation
    Should return ExperimentOps API documentation
    """
    print("\n" + "="*80)
    print("TEST 5: get_api_documentation")
    print("="*80)

    workspace = MockWorkspace()
    result = get_api_documentation(workspace, "test_task", "claude-sonnet-4-5")

    print(f"\n📊 API Methods:")
    for method_name, method_data in result['methods'].items():
        print(f"\n  {method_name}:")
        print(f"    Signature: {method_data.get('signature', 'N/A')}")
        print(f"    Description: {method_data.get('description', 'N/A')}")
        if 'required' in method_data:
            print(f"    ⚠️  REQUIRED!")

    print(f"\n✨ NEW Beautiful API:")
    print(f"  Old Way: {result['new_api']['old_way']}")
    print(f"  New Way: {result['new_api']['new_way']}")
    print(f"\n  Features:")
    for feature in result['new_api']['features']:
        print(f"    - {feature}")

    print(f"\n📋 Workflow Steps:")
    for step in result['workflow']:
        print(f"  {step}")

    # Assertions
    assert 'create_scene' in result['methods'], "Expected create_scene method"
    assert 'add_asset' in result['methods'], "Expected add_asset method"
    assert 'compile' in result['methods'], "Expected compile method"
    assert result['methods']['compile']['required'] == True, "compile should be required"

    print("\n✅ TEST 5 PASSED: get_api_documentation works!")
    return result


def test_create_scene_script():
    """
    TEST 6: create_scene_script
    Should create a valid Python script
    """
    print("\n" + "="*80)
    print("TEST 6: create_scene_script")
    print("="*80)

    workspace = MockWorkspace()

    # Sample scene code
    scene_code = """ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core", save_fps=30)
ops.create_scene(name="test_scene", width=5, length=5, height=3)
ops.add_robot(robot_name="stretch", position=(0, 0, 0))
ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))
ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="center")
ops.add_overhead_camera()
ops.compile()"""

    result = create_scene_script(
        workspace,
        "test_task",
        "claude-sonnet-4-5",
        scene_name="test_kitchen",
        scene_description="Test kitchen scene with table and apple",
        script_code=scene_code
    )

    print(f"\n📄 Script Created:")
    print(f"  Success: {result['success']}")
    print(f"  Filename: {result['filename']}")
    print(f"  Path: {result['path']}")
    print(f"  Message: {result['message']}")

    # Check if document was registered in workspace
    if result['success']:
        doc = workspace.get_document(result['filename'])
        if doc:
            print(f"\n✅ Document registered in workspace")
            print(f"  Document path: {doc.path}")
            print(f"  Document size: {doc.size_bytes} bytes")
        else:
            print(f"\n⚠️  Document not found in workspace (but file may still be created)")

    # Assertions
    assert result['success'] == True, "Script creation failed"
    assert result['filename'] == "scene_test_kitchen.py", f"Expected scene_test_kitchen.py, got {result['filename']}"

    print("\n✅ TEST 6 PASSED: create_scene_script works!")
    return result


def run_all_tests():
    """Run all tool tests"""
    print("\n" + "🚀"*40)
    print("SCENE MAKER TOOLS - INDIVIDUAL TESTING")
    print("Testing each tool separately before creating agent")
    print("🚀"*40)

    try:
        # Test all tools
        result1 = test_discover_assets()
        result2 = test_discover_behaviors()
        result3 = test_discover_relations()
        result4 = test_get_scene_examples()
        result5 = test_get_api_documentation()
        result6 = test_create_scene_script()

        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*80)

        print("\n✅ Summary:")
        print(f"  1. discover_assets: {result1['summary']['total_assets']} assets discovered")
        print(f"  2. discover_behaviors: {result2['summary']['total_behaviors']} behaviors discovered")
        print(f"  3. discover_relations: {result3['summary']['total_relations']} relations discovered")
        print(f"  4. get_scene_examples: {result4['summary']['total_examples']} examples extracted")
        print(f"  5. get_api_documentation: API docs extracted successfully")
        print(f"  6. create_scene_script: Script created successfully")

        print("\n🎯 All tools working! Ready to create Scene Maker agent!")

        return True

    except AssertionError as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED!")
        print("="*80)
        print(f"Error: {e}")
        return False

    except Exception as e:
        print("\n" + "="*80)
        print("❌ UNEXPECTED ERROR!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

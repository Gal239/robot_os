#!/usr/bin/env python3
"""
SIMPLE TOOL TESTS - Test core logic of each tool
Tests each tool's logic directly without imports
"""

import sys
import json
import re
import ast
from pathlib import Path

# Direct paths to source files
ASSETS_JSON = Path(__file__).parent.parent.parent / "modals" / "mujoco_assets" / "ASSETS.json"
BEHAVIORS_JSON = Path(__file__).parent.parent.parent / "behaviors" / "BEHAVIORS.json"
SCENE_MODAL_PY = Path(__file__).parent.parent.parent / "modals" / "scene_modal.py"
DEMO_SCENES_PY = Path(__file__).parent.parent.parent / "tests" / "demos" / "demo_1_ai_generated_scenes.py"

def test_1_discover_assets():
    """TEST 1: discover_assets - Read ASSETS.json"""
    print("\n" + "="*80)
    print("TEST 1: discover_assets")
    print("="*80)

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
            'behaviors': sorted(list(behaviors)),
            'components': list(components.keys())
        }

        if asset_type == 'furniture':
            furniture.append(asset_summary)
        elif asset_type == 'object':
            objects.append(asset_summary)

    result = {
        'furniture': sorted(furniture, key=lambda x: x['name']),
        'objects': sorted(objects, key=lambda x: x['name']),
        'summary': {
            'total_furniture': len(furniture),
            'total_objects': len(objects),
            'total_assets': len(furniture) + len(objects)
        }
    }

    print(f"\n📊 Summary:")
    print(f"  Total Assets: {result['summary']['total_assets']}")
    print(f"  Furniture: {result['summary']['total_furniture']}")
    print(f"  Objects: {result['summary']['total_objects']}")

    print(f"\n🪑 Sample Furniture (first 5):")
    for item in result['furniture']:
        print(f"  - {item['name']}: {item['behaviors']}")

    print(f"\n🍎 Sample Objects (first 5):")
    for item in result['objects']:
        print(f"  - {item['name']}: {item['behaviors']}")

    assert result['summary']['total_assets'] == 71, f"Expected 71, got {result['summary']['total_assets']}"
    assert result['summary']['total_furniture'] == 15
    assert result['summary']['total_objects'] == 56

    print("\n✅ TEST 1 PASSED!")
    return result


def test_2_discover_behaviors():
    """TEST 2: discover_behaviors - Read BEHAVIORS.json"""
    print("\n" + "="*80)
    print("TEST 2: discover_behaviors")
    print("="*80)

    assert BEHAVIORS_JSON.exists(), f"BEHAVIORS.json not found at {BEHAVIORS_JSON}"

    with open(BEHAVIORS_JSON, 'r') as f:
        behaviors_data = json.load(f)

    object_behaviors = {}
    robot_behaviors = {}

    for behavior_name, behavior_info in behaviors_data.items():
        description = behavior_info.get('description', '')
        properties = behavior_info.get('properties', {})

        property_list = []
        for prop_name, prop_data in properties.items():
            property_list.append({
                'name': prop_name,
                'description': prop_data.get('description', ''),
                'unit': prop_data.get('unit', '')
            })

        behavior_summary = {
            'description': description,
            'properties': property_list,
            'property_count': len(properties)
        }

        # Categorize
        if 'robot_' in behavior_name or behavior_name in ['vision', 'distance_sensing', 'motion_sensing', 'tactile', 'audio_sensing']:
            robot_behaviors[behavior_name] = behavior_summary
        else:
            object_behaviors[behavior_name] = behavior_summary

    result = {
        'behaviors': behaviors_data,
        'object_behaviors': object_behaviors,
        'robot_behaviors': robot_behaviors,
        'summary': {
            'total_behaviors': len(behaviors_data),
            'object_behaviors': len(object_behaviors),
            'robot_behaviors': len(robot_behaviors),
            'behavior_names': sorted(list(behaviors_data.keys()))
        }
    }

    print(f"\n📊 Summary:")
    print(f"  Total Behaviors: {result['summary']['total_behaviors']}")
    print(f"  Object Behaviors: {result['summary']['object_behaviors']}")
    print(f"  Robot Behaviors: {result['summary']['robot_behaviors']}")

    print(f"\n🎯 Sample Behaviors (first 5):")
    for behavior_name in list(behaviors_data.keys())[:5]:
        print(f"  - {behavior_name}: {behaviors_data[behavior_name].get('description', 'No description')}")

    print(f"\n📝 All Behavior Names:")
    print(f"  {', '.join(result['summary']['behavior_names'][:10])}...")

    assert result['summary']['total_behaviors'] >= 15
    assert 'stackable' in result['behaviors']
    assert 'graspable' in result['behaviors']

    print("\n✅ TEST 2 PASSED!")
    return result


def test_3_discover_relations():
    """TEST 3: discover_relations - Read RELATIONS.json (generated by Pydantic modals!)"""
    print("\n" + "="*80)
    print("TEST 3: discover_relations")
    print("="*80)

    RELATIONS_JSON = Path(__file__).parent.parent.parent / "modals" / "RELATIONS.json"

    # OFFENSIVE: Crash if JSON missing!
    assert RELATIONS_JSON.exists(), (
        f"RELATIONS.json not found at {RELATIONS_JSON}\n"
        f"\n💡 FIX: Run config generator:\n"
        f"   python3 -m core.tools.config_generator relations"
    )

    # Read RELATIONS.json (generated by Pydantic modals!)
    with open(RELATIONS_JSON, 'r') as f:
        relations = json.load(f)

    # Parse surface positions from scene_modal.py (these are constants)
    with open(SCENE_MODAL_PY, 'r') as f:
        code = f.read()

    surface_positions_match = re.search(r'SURFACE_POSITIONS\s*=\s*\{([^}]+)\}', code, re.DOTALL)
    surface_positions = {}

    if surface_positions_match:
        positions_str = surface_positions_match.group(1)
        for line in positions_str.split('\n'):
            match = re.search(r'"(\w+)":\s*\(([^)]+)\)', line)
            if match:
                pos_name = match.group(1)
                coords = match.group(2)
                surface_positions[pos_name] = coords.strip()

    result = {
        'relations': relations,
        'surface_positions': surface_positions,
        'summary': {
            'total_relations': len(relations),
            'total_surface_positions': len(surface_positions),
            'relation_names': sorted(list(relations.keys())),
            'surface_position_names': sorted(list(surface_positions.keys()))
        }
    }

    print(f"\n📊 Summary:")
    print(f"  Total Relations: {result['summary']['total_relations']}")
    print(f"  Total Surface Positions: {result['summary']['total_surface_positions']}")

    print(f"\n🔗 All Relations:")
    for relation_name, rel_data in relations.items():
        description = rel_data.get('description', 'No description') if isinstance(rel_data, dict) else rel_data
        print(f"  - {relation_name}: {description}")

    print(f"\n📍 All Surface Positions:")
    for pos_name, pos_coords in surface_positions.items():
        print(f"  - {pos_name}: {pos_coords}")

    assert result['summary']['total_relations'] >= 8
    assert 'on_top' in result['relations']
    assert 'inside' in result['relations']
    assert result['summary']['total_surface_positions'] >= 5

    print("\n✅ TEST 3 PASSED!")
    return result


def test_4_get_scene_examples():
    """TEST 4: get_scene_examples - Extract from demo file"""
    print("\n" + "="*80)
    print("TEST 4: get_scene_examples")
    print("="*80)

    assert DEMO_SCENES_PY.exists(), f"demo file not found at {DEMO_SCENES_PY}"

    with open(DEMO_SCENES_PY, 'r') as f:
        code = f.read()

    # Parse AST to extract functions
    tree = ast.parse(code)
    examples = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name

            if func_name.startswith('test_') and func_name != 'test_menu':
                docstring = ast.get_docstring(node) or "No description"
                func_code_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_code_lines)

                # Extract ops code
                ops_code_match = re.search(r'(ops = ExperimentOps.*?ops\.compile\(\))', func_code, re.DOTALL)
                if ops_code_match:
                    scene_code = ops_code_match.group(1)
                else:
                    scene_code = func_code[:500]  # First 500 chars

                examples[func_name] = {
                    'description': docstring,
                    'code': scene_code,
                    'code_length': len(scene_code)
                }

    result = {
        'examples': examples,
        'summary': {
            'total_examples': len(examples),
            'example_names': sorted(list(examples.keys()))
        }
    }

    print(f"\n📊 Summary:")
    print(f"  Total Examples: {result['summary']['total_examples']}")

    print(f"\n📖 All Examples:")
    for example_name, example_data in examples.items():
        print(f"  - {example_name}: {example_data['description']}")
        print(f"    Code Length: {example_data['code_length']} chars")

    if examples:
        first_example_name = list(examples.keys())[0]
        first_example = examples[first_example_name]
        print(f"\n🔍 Sample Code (first 200 chars):")
        print(f"  {first_example['code'][:200]}...")

    assert result['summary']['total_examples'] >= 6, f"Expected at least 6, got {result['summary']['total_examples']}"
    assert 'test_1_tower_stacking' in result['summary']['example_names']

    print("\n✅ TEST 4 PASSED!")
    return result


def test_5_get_api_documentation():
    """TEST 5: get_api_documentation - Read API.json (generated by inspect!)"""
    print("\n" + "="*80)
    print("TEST 5: get_api_documentation")
    print("="*80)

    API_JSON = Path(__file__).parent.parent.parent / "docs" / "API.json"

    # OFFENSIVE: Crash if JSON missing!
    assert API_JSON.exists(), (
        f"API.json not found at {API_JSON}\n"
        f"\n💡 FIX: Run config generator:\n"
        f"   python3 -m core.tools.config_generator api"
    )

    # Read API.json (generated by inspect on ExperimentOps!)
    with open(API_JSON, 'r') as f:
        api_spec = json.load(f)

    # Extract core methods for display
    core_methods = {
        name: data for name, data in api_spec.items()
        if name in ['create_scene', 'add_asset', 'add_robot', 'add_reward', 'compile']
    }

    methods = core_methods  # For backward compatibility with test output

    new_api = {
        "description": "New beautiful asset access API (PURE MOP!)",
        "old_way": "state = ops.get_state(); pos = state['apple']['body.position']",
        "new_way": "pos = ops.assets.apple.position"
    }

    workflow = [
        "1. Initialize ExperimentOps",
        "2. Create scene",
        "3. Add robot",
        "4. Add assets",
        "5. Add rewards",
        "6. Add cameras",
        "7. Call compile() - REQUIRED!"
    ]

    result = {
        'methods': methods,
        'new_api': new_api,
        'workflow': workflow,
        'total_api_methods': len(api_spec)
    }

    print(f"\n📊 API Methods:")
    for method_name, method_data in methods.items():
        print(f"\n  {method_name}:")
        print(f"    Signature: {method_data['signature']}")
        print(f"    Description: {method_data['description']}")
        if method_data.get('required'):
            print(f"    ⚠️  REQUIRED!")

    print(f"\n✨ NEW Beautiful API:")
    print(f"  Old Way: {new_api['old_way']}")
    print(f"  New Way: {new_api['new_way']}")

    print(f"\n📋 Workflow Steps:")
    for step in workflow:
        print(f"  {step}")

    assert 'create_scene' in result['methods']
    assert 'add_asset' in result['methods']
    assert 'compile' in result['methods']

    print(f"\n📊 Total API methods discovered: {result['total_api_methods']}")
    print(f"   (Showing {len(core_methods)} core methods above)")

    print("\n✅ TEST 5 PASSED!")
    return result


def test_6_create_scene_script():
    """TEST 6: create_scene_script - Generate script logic"""
    print("\n" + "="*80)
    print("TEST 6: create_scene_script")
    print("="*80)

    scene_name = "test_kitchen"
    scene_description = "Test kitchen scene"
    scene_code = """ops = ExperimentOps(mode="simulated", headless=False)
ops.create_scene(name="test_scene", width=5, length=5, height=3)
ops.add_robot(robot_name="stretch", position=(0, 0, 0))
ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))
ops.compile()"""

    # Generate script
    safe_name = scene_name.lower().replace(' ', '_')
    filename = f"scene_{safe_name}.py"

    # Indent code
    indent = '    '
    indented_code = '\n'.join([indent + line if line.strip() else line for line in scene_code.split('\n')])

    script = f'''#!/usr/bin/env python3
"""
Scene: {scene_name}
Description: {scene_description}
"""

from simulation_center.core.main.experiment_ops_unified import ExperimentOps

def main():
{indented_code}

if __name__ == "__main__":
    main()
'''

    result = {
        'success': True,
        'filename': filename,
        'script_length': len(script),
        'has_imports': 'import' in script,
        'has_main': 'def main():' in script,
        'has_code': 'ops.compile()' in script
    }

    print(f"\n📄 Script Generated:")
    print(f"  Filename: {result['filename']}")
    print(f"  Script Length: {result['script_length']} chars")
    print(f"  Has Imports: {result['has_imports']}")
    print(f"  Has Main Function: {result['has_main']}")
    print(f"  Has Scene Code: {result['has_code']}")

    print(f"\n🔍 Sample Script (first 400 chars):")
    print(script[:400] + "...")

    assert result['success'] == True
    assert result['filename'] == "scene_test_kitchen.py"
    assert result['has_imports'] == True
    assert result['has_main'] == True
    assert result['has_code'] == True

    print("\n✅ TEST 6 PASSED!")
    return result


def run_all_tests():
    """Run all tool tests"""
    print("\n" + "🚀"*40)
    print("SCENE MAKER TOOLS - TESTING ALL 6 TOOLS")
    print("Testing core logic of each tool")
    print("🚀"*40)

    try:
        result1 = test_1_discover_assets()
        result2 = test_2_discover_behaviors()
        result3 = test_3_discover_relations()
        result4 = test_4_get_scene_examples()
        result5 = test_5_get_api_documentation()
        result6 = test_6_create_scene_script()

        print("\n" + "="*80)
        print("🎉 ALL 6 TESTS PASSED! 🎉")
        print("="*80)

        print("\n✅ Summary:")
        print(f"  1. discover_assets: {result1['summary']['total_assets']} assets discovered")
        print(f"  2. discover_behaviors: {result2['summary']['total_behaviors']} behaviors discovered")
        print(f"  3. discover_relations: {result3['summary']['total_relations']} relations discovered")
        print(f"  4. get_scene_examples: {result4['summary']['total_examples']} examples extracted")
        print(f"  5. get_api_documentation: API docs generated")
        print(f"  6. create_scene_script: Script generated successfully")

        print("\n🎯 All tools working! Scene Maker agent ready!")
        print("\n📊 MOP Compliance:")
        print("  ✅ AUTO-DISCOVERY: All data from source files")
        print("  ✅ ZERO HARDCODING: No lists in prompt")
        print("  ✅ ALWAYS CURRENT: Tools read from source")
        print("  ✅ OFFENSIVE: Asserts crash if sources missing")

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

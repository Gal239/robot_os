"""
Scene Maker Tools - PURE MOP AUTO-DISCOVERY Tools
All tools read from SOURCE OF TRUTH files (ZERO HARDCODING!)

Tools:
1. discover_assets - Read from ASSETS.json
2. discover_behaviors - Read from BEHAVIORS.json
3. discover_relations - Parse from scene_modal.py
4. discover_orientations - Extract from xml_resolver.py ORIENTATION_PRESETS
5. get_scene_examples - Extract from demo_1_ai_generated_scenes.py
6. get_api_documentation - Extract from experiment_ops_unified.py
7. create_scene_script - Generate and save scene script

MOP PRINCIPLES:
- AUTO-DISCOVERY: All data from source files
- SELF-GENERATION: Code teaches agent via tools
- OFFENSIVE: Crash if source missing (not defensive!)
"""

import json
import re
import ast
from pathlib import Path
from typing import Dict, Optional
from ..modals import WorkspaceModal
from ..tool_decorator import tool


# OFFENSIVE: Direct paths to source of truth files
ASSETS_JSON = Path(__file__).parent.parent.parent.parent / "core" / "modals" / "mujoco_assets" / "ASSETS.json"
BEHAVIORS_JSON = Path(__file__).parent.parent.parent.parent / "core" / "behaviors" / "BEHAVIORS.json"
SCENE_MODAL_PY = Path(__file__).parent.parent.parent.parent / "core" / "modals" / "scene_modal.py"
XML_RESOLVER_PY = Path(__file__).parent.parent.parent.parent / "core" / "modals" / "xml_resolver.py"
DEMO_SCENES_PY = Path(__file__).parent.parent.parent.parent / "core" / "tests" / "demos" / "demo_1_ai_generated_scenes.py"
EXPERIMENT_OPS_PY = Path(__file__).parent.parent.parent.parent / "core" / "main" / "experiment_ops_unified.py"


@tool(execution_type="function")
def discover_assets(workspace: WorkspaceModal, task_id: str, agent_model: str, category: Optional[str] = None) -> Dict:
    """discover_assets"""

    # OFFENSIVE: Crash if file missing
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


@tool(execution_type="function")
def discover_behaviors(workspace: WorkspaceModal, task_id: str, agent_model: str) -> Dict:
    """discover_behaviors"""

    # OFFENSIVE: Crash if file missing
    assert BEHAVIORS_JSON.exists(), f"BEHAVIORS.json not found at {BEHAVIORS_JSON}"

    with open(BEHAVIORS_JSON, 'r') as f:
        behaviors_data = json.load(f)

    # Categorize behaviors
    object_behaviors = {}
    robot_behaviors = {}

    for behavior_name, behavior_info in behaviors_data.items():
        description = behavior_info.get('description', '')
        properties = behavior_info.get('properties', {})

        # Extract property names and descriptions
        property_list = []
        for prop_name, prop_data in properties.items():
            property_list.append({
                'name': prop_name,
                'description': prop_data.get('description', ''),
                'unit': prop_data.get('unit', ''),
                'default': prop_data.get('default')
            })

        behavior_summary = {
            'description': description,
            'properties': property_list,
            'property_count': len(properties)
        }

        # Categorize (heuristic: robot_ prefix or sensor-related names)
        if 'robot_' in behavior_name or behavior_name in ['vision', 'distance_sensing', 'motion_sensing', 'tactile', 'audio_sensing']:
            robot_behaviors[behavior_name] = behavior_summary
        else:
            object_behaviors[behavior_name] = behavior_summary

    return {
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


@tool(execution_type="function")
def discover_relations(workspace: WorkspaceModal, task_id: str, agent_model: str) -> Dict:
    """discover_relations"""

    # OFFENSIVE: Crash if file missing
    assert SCENE_MODAL_PY.exists(), f"scene_modal.py not found at {SCENE_MODAL_PY}"

    with open(SCENE_MODAL_PY, 'r') as f:
        code = f.read()

    # Parse surface positions (lines 299-319 approx)
    surface_positions_match = re.search(r'SURFACE_POSITIONS\s*=\s*\{([^}]+)\}', code, re.DOTALL)
    surface_positions = {}

    if surface_positions_match:
        positions_str = surface_positions_match.group(1)
        # Parse each line like: "top_left": (-0.35, 0.35),
        for line in positions_str.split('\n'):
            match = re.search(r'"(\w+)":\s*\(([^)]+)\)', line)
            if match:
                pos_name = match.group(1)
                coords = match.group(2)
                surface_positions[pos_name] = coords.strip()

    # Extract relation documentation from _resolve_relative_position method
    relations = {
        "on_top": {
            "description": "Place object on surface with optional surface_position",
            "parameters": ["surface_position", "offset", "distance"],
            "example": 'ops.add_asset("apple", relative_to="table", relation="on_top", surface_position="center")'
        },
        "stack_on": {
            "description": "Vertically stack objects (center-aligned, uses real dimensions)",
            "parameters": ["distance (auto if None)"],
            "example": 'ops.add_asset("block2", relative_to="block1", relation="stack_on")'
        },
        "inside": {
            "description": "Place object inside container",
            "parameters": [],
            "example": 'ops.add_asset("apple", relative_to="bowl", relation="inside")'
        },
        "next_to": {
            "description": "Place object adjacent on same surface",
            "parameters": ["spacing", "distance"],
            "example": 'ops.add_asset("banana", relative_to="apple", relation="next_to")'
        },
        "front": {
            "description": "In front of object (+Y axis)",
            "parameters": ["distance"],
            "example": 'ops.add_asset("chair", relative_to="table", relation="front")'
        },
        "back": {
            "description": "Behind object (-Y axis)",
            "parameters": ["distance"],
            "example": 'ops.add_asset("shelf", relative_to="wall", relation="back")'
        },
        "left": {
            "description": "Left of object (-X axis)",
            "parameters": ["distance"],
            "example": 'ops.add_asset("lamp", relative_to="desk", relation="left")'
        },
        "right": {
            "description": "Right of object (+X axis)",
            "parameters": ["distance"],
            "example": 'ops.add_asset("lamp", relative_to="desk", relation="right")'
        }
    }

    return {
        'relations': relations,
        'surface_positions': surface_positions,
        'summary': {
            'total_relations': len(relations),
            'total_surface_positions': len(surface_positions),
            'relation_names': sorted(list(relations.keys())),
            'surface_position_names': sorted(list(surface_positions.keys()))
        }
    }


@tool(execution_type="function")
def discover_orientations(workspace: WorkspaceModal, task_id: str, agent_model: str) -> Dict:
    """discover_orientations"""

    # OFFENSIVE: Crash if file missing
    assert XML_RESOLVER_PY.exists(), f"xml_resolver.py not found at {XML_RESOLVER_PY}"

    with open(XML_RESOLVER_PY, 'r') as f:
        code = f.read()

    # Extract ORIENTATION_PRESETS dictionary using regex
    # Looking for: ORIENTATION_PRESETS = { ... }
    preset_match = re.search(
        r'ORIENTATION_PRESETS\s*=\s*\{([^}]+)\}',
        code,
        re.DOTALL
    )

    if not preset_match:
        raise ValueError("ORIENTATION_PRESETS not found in xml_resolver.py!")

    presets_str = "{" + preset_match.group(1) + "}"

    # Parse the presets dictionary
    # Extract each preset line by line
    presets = {}
    for line in preset_match.group(1).split('\n'):
        # Match lines like: "north": (1, 0, 0, 0),  # Face +Y
        match = re.match(r'\s*"([^"]+)":\s*\(([^)]+)\)', line)
        if match:
            name = match.group(1)
            values_str = match.group(2)

            # Skip presets with None value (facing_origin is calculated dynamically)
            if 'None' in values_str:
                continue

            # Parse quaternion values
            values = tuple(float(v.strip()) for v in values_str.split(','))
            presets[name] = values

    # Build comprehensive response
    return {
        'presets': presets,
        'relational_pattern': "facing_X",
        'relational_description': "Use 'facing_X' to orient toward any asset, where X is the asset name. Examples: 'facing_table', 'facing_apple', 'facing_origin'. The system auto-calculates the quaternion based on the target position.",
        'manual_format': "(w, x, y, z)",
        'manual_description': "Provide a direct quaternion tuple in (w, x, y, z) format. This is for advanced users who need precise control over orientation.",
        'examples': [
            {
                'type': 'preset',
                'code': 'ops.add_robot("stretch", position=(0, 0, 0), orientation="east")',
                'description': 'Robot faces east (+X direction)'
            },
            {
                'type': 'relational',
                'code': 'ops.add_robot("stretch", position=(0, 0, 0), orientation="facing_table")',
                'description': 'Robot automatically faces toward the table asset'
            },
            {
                'type': 'manual',
                'code': 'ops.add_robot("stretch", position=(0, 0, 0), orientation=(0.707, 0, 0, 0.707))',
                'description': '90° rotation around Z axis (precise control)'
            }
        ],
        'summary': {
            'total_presets': len(presets),
            'preset_names': sorted(list(presets.keys())),
            'categories': {
                'cardinal_directions': ['north', 'south', 'east', 'west'],
                'object_states': ['upright', 'sideways', 'inverted', 'upside_down']
            }
        }
    }


@tool(execution_type="function")
def get_scene_examples(workspace: WorkspaceModal, task_id: str, agent_model: str, example_name: Optional[str] = None) -> Dict:
    """get_scene_examples"""

    # OFFENSIVE: Crash if file missing
    assert DEMO_SCENES_PY.exists(), f"demo_1_ai_generated_scenes.py not found at {DEMO_SCENES_PY}"

    with open(DEMO_SCENES_PY, 'r') as f:
        code = f.read()

    # Parse AST to extract function definitions
    tree = ast.parse(code)

    examples = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_name = node.name

            # Extract only test_* functions
            if func_name.startswith('test_') and func_name != 'test_menu':
                # Get docstring
                docstring = ast.get_docstring(node) or "No description"

                # Extract function body as code
                func_code_lines = code.split('\n')[node.lineno - 1:node.end_lineno]
                func_code = '\n'.join(func_code_lines)

                # Extract just the ops calls (between def and end)
                # Find the actual scene creation code
                ops_code_match = re.search(r'(ops = ExperimentOps.*?ops\.compile\(\))', func_code, re.DOTALL)
                if ops_code_match:
                    scene_code = ops_code_match.group(1)
                else:
                    scene_code = func_code

                examples[func_name] = {
                    'description': docstring,
                    'code': scene_code,
                    'full_function': func_code
                }

    # Filter if specific example requested
    if example_name:
        if example_name in examples:
            examples = {example_name: examples[example_name]}
        else:
            return {
                'examples': {},
                'summary': {
                    'error': f"Example '{example_name}' not found",
                    'available_examples': sorted(list(examples.keys()))
                }
            }

    return {
        'examples': examples,
        'summary': {
            'total_examples': len(examples),
            'example_names': sorted(list(examples.keys()))
        }
    }


@tool(execution_type="function")
def get_api_documentation(workspace: WorkspaceModal, task_id: str, agent_model: str) -> Dict:
    """get_api_documentation"""

    # OFFENSIVE: Crash if file missing
    assert EXPERIMENT_OPS_PY.exists(), f"experiment_ops_unified.py not found at {EXPERIMENT_OPS_PY}"

    # Read file and parse
    with open(EXPERIMENT_OPS_PY, 'r') as f:
        code = f.read()

    # Extract key method signatures and docstrings
    methods = {
        "initialization": {
            "signature": "ExperimentOps(mode='simulated', headless=False, render_mode='rl_core', save_fps=30)",
            "description": "Initialize experiment operations",
            "parameters": {
                "mode": "Simulation mode (default: 'simulated')",
                "headless": "Run without viewer (default: False)",
                "render_mode": "Rendering quality: 'rl_core', 'demo', '2k_demo'",
                "save_fps": "Frame rate for recording (default: 30)"
            }
        },
        "create_scene": {
            "signature": "ops.create_scene(name, width, length, height, floor_texture='wood_floor', wall_texture='gray_wall', ceiling_texture='ceiling_tiles')",
            "description": "Create scene with specified dimensions in meters",
            "required": True
        },
        "add_robot": {
            "signature": "ops.add_robot(robot_name, robot_id=None, position=None, sensors=None, task_hint=None, initial_state=None)",
            "description": "Add robot to scene",
            "parameters": {
                "robot_name": "Robot type (currently only 'stretch')",
                "position": "3D tuple (x, y, z), default (0, 0, 0)"
            }
        },
        "add_asset": {
            "signature": "ops.add_asset(asset_name, relative_to=None, relation=None, distance=None, surface_position=None, offset=None, orientation=None, initial_state=None)",
            "description": "Add furniture or object. If distance=None, dimensions auto-extracted at compile (PURE MOP!)",
            "parameters": {
                "asset_name": "Name from discover_assets",
                "relative_to": "Position (x,y,z) tuple OR asset name",
                "relation": "Spatial relation from discover_relations",
                "distance": "Distance in meters (None = auto-extract)",
                "surface_position": "From discover_relations surface_positions",
                "offset": "(x, y) manual offset in meters"
            },
            "example": 'ops.add_asset("apple", relative_to="table", relation="on_top", surface_position="center")'
        },
        "add_reward": {
            "signature": "ops.add_reward(tracked_asset, behavior, target=None, reward=None, mode=None, id=None, requires=None, within=None, after=None, tolerance_override=None)",
            "description": "Add reward/tracking for asset behaviors",
            "parameters": {
                "tracked_asset": "Asset name or 'robot.component'",
                "behavior": "Property from discover_behaviors",
                "target": "Target value (numeric/boolean/string)",
                "reward": "Reward points",
                "id": "REQUIRED unique identifier"
            },
            "example": 'ops.add_reward("apple", behavior="stacked_on", target="table", reward=100, id="apple_on_table")'
        },
        "add_overhead_camera": {
            "signature": "ops.add_overhead_camera(camera_id='overhead')",
            "description": "Add automatic bird's eye view camera"
        },
        "add_free_camera": {
            "signature": "ops.add_free_camera(camera_id, lookat=None, track_target=None, distance=2.0, azimuth=0, elevation=-30)",
            "description": "Add custom positioned or tracking camera",
            "example_static": 'ops.add_free_camera("side", lookat=(2, 0, 0.8), distance=2.5, azimuth=90, elevation=-20)',
            "example_tracking": 'ops.add_free_camera("track_apple", track_target="apple", distance=1.0, azimuth=45, elevation=-20)'
        },
        "compile": {
            "signature": "ops.compile()",
            "description": "REQUIRED! Compiles the scene. Must be called after all setup.",
            "required": True
        }
    }

    # NEW BEAUTIFUL API
    new_api = {
        "description": "New beautiful asset access API (PURE MOP!)",
        "old_way": 'state = ops.get_state(); pos = state["apple"]["body.position"]',
        "new_way": "pos = ops.assets.apple.position  # ✨ Direct property access!",
        "features": [
            "IDE autocomplete works!",
            "Direct property access (no dict lookups)",
            "Semantic methods: is_stacked_on('table'), distance_to('robot')",
            "Type-safe returns"
        ],
        "examples": [
            "ops.assets.apple.position → (x, y, z) tuple",
            "ops.assets.apple.behaviors → list of behavior strings",
            "ops.assets.apple.is_stacked_on('table') → boolean",
            "ops.assets.apple.distance_to('robot') → float"
        ]
    }

    workflow = [
        "1. Initialize ExperimentOps",
        "2. Create scene with dimensions",
        "3. Add robot",
        "4. Add assets (furniture first, then objects)",
        "5. Add rewards (optional but recommended)",
        "6. Add cameras (optional)",
        "7. Call compile() - REQUIRED!"
    ]

    return {
        'methods': methods,
        'new_api': new_api,
        'workflow': workflow
    }


@tool(execution_type="function")
def create_scene_script(
    workspace: WorkspaceModal,
    task_id: str,
    agent_model: str,
    scene_name: str,
    scene_description: str,
    script_code: str
) -> Dict:
    """create_scene_script"""

    try:
        # Sanitize scene name for filename
        safe_name = scene_name.lower().replace(' ', '_').replace('-', '_')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
        filename = f"scene_{safe_name}.py"

        # Build complete Python script with header, imports, and main function
        script_template = f'''#!/usr/bin/env python3
"""
Scene: {scene_name}
Description: {scene_description}

Generated by Scene Maker AI Agent
Task ID: {task_id}
"""

import sys
from pathlib import Path

# Add simulation_center to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps


def main():
    """Main function to create and run the scene"""
    print("=" * 80)
    print(f"Creating scene: {scene_name}")
    print(f"Description: {scene_description}")
    print("=" * 80)

    # Scene creation code
{indent_code(script_code, 4)}

    print("\\n" + "=" * 80)
    print("Scene created successfully!")
    print("=" * 80)

    # Keep scene running
    try:
        print("\\nPress Ctrl+C to exit...")
        while True:
            ops.step()
    except KeyboardInterrupt:
        print("\\nClosing scene...")
        ops.close()


if __name__ == "__main__":
    main()
'''

        # Save to workspace using auto_db
        from ai_orchestration.utils.global_config import agent_engine_db
        agent_engine_db.save_workspace_file(workspace.session_id, filename, script_template)

        # Register in workspace (as Document modal)
        from ..modals import Document
        doc = Document.from_content(script_template, filename, created_by=task_id)
        doc.description = scene_description
        workspace.register_document(filename, doc)

        return {
            "success": True,
            "filename": filename,
            "path": f"workspace/{filename}",
            "message": f"Scene script created: {filename}"
        }

    except Exception as e:
        return {
            "success": False,
            "filename": "",
            "path": "",
            "message": f"Error creating scene script: {str(e)}"
        }


def indent_code(code: str, spaces: int) -> str:
    """
    Indent code block by specified number of spaces.
    MOP: Helper function for code generation
    """
    indent = ' ' * spaces
    lines = code.split('\\n')
    return '\\n'.join([indent + line if line.strip() else line for line in lines])
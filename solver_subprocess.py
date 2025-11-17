"""
Solver subprocess - Runs scene compilation and solver in isolated process
Returns placement data to main process via stdout (JSON)
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Redirect all print statements to stderr so only JSON goes to stdout
import builtins
original_print = builtins.print
builtins.print = lambda *args, **kwargs: original_print(*args, **kwargs, file=sys.stderr)

from core.main.experiment_ops_unified import ExperimentOps
from core.main.robot_ops import create_robot

def run_solver(scene_config, robot_id, task, target_asset):
    """Run solver in subprocess - compile, solve, return placement"""

    # Create scene from config
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene(
        scene_config['name'],
        width=scene_config['width'],
        length=scene_config['length'],
        height=scene_config['height']
    )

    # Add assets from config
    for asset in scene_config['assets']:
        ops.add_asset(
            asset['name'],
            relative_to=tuple(asset['relative_to']) if isinstance(asset['relative_to'], list) else asset['relative_to'],
            relation=asset.get('relation'),
            distance=asset.get('distance')
        )

    # Compile to extract dimensions
    ops.compile()

    # Create temp robot for capability discovery
    temp_robot = create_robot(robot_id, robot_id)

    # Run solver with compiled model
    placement = ops.scene.solve_robot_placement(
        robot=temp_robot,
        task=task,
        target_asset_name=target_asset,
        model=ops.backend.model
    )

    # Return placement as JSON
    return placement

if __name__ == "__main__":
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    # Run solver
    placement = run_solver(
        input_data['scene_config'],
        input_data['robot_id'],
        input_data['task'],
        input_data['target_asset']
    )

    # Output placement as JSON to stdout (use original_print to bypass redirect)
    original_print(json.dumps(placement))

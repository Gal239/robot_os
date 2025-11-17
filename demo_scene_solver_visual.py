"""
VISUAL DEMO: Scene Solver - Auto-calculates robot placement for grasping

This demo shows:
1. Scene with table and apple
2. Robot auto-positioned to grasp the apple
3. Rendered visually (headless=False)

PURE MOP: Robot placement calculated from:
- Robot actuator capabilities (discovered from actuators)
- Apple position and dimensions (extracted from MuJoCo)
- Task requirements (grasp needs comfortable reach)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

print("\n" + "="*80)
print("VISUAL DEMO: Scene Solver - Auto-Placement for Grasping")
print("="*80)

# Create scene
print("\n📦 Creating scene with table and apple...")
ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core")
ops.create_scene("demo_room", width=10, length=10, height=3)

# Add table and apple
ops.add_asset("table", relative_to=(2, 0, 0))
# Use explicit distance so add_robot() doesn't need model for "facing_apple" orientation
ops.add_asset("apple", relative_to="table", relation="on_top", distance=0.796)

# Use scene solver to calculate robot placement IN SUBPROCESS!
print("\n🤖 Using scene solver (subprocess) to calculate robot placement...")
print("   Task: grasp")
print("   Target: apple")

# Build scene config for subprocess
import json
import subprocess

scene_config = {
    'name': 'demo_room',
    'width': 10,
    'length': 10,
    'height': 3,
    'assets': [
        {
            'name': 'table',
            'relative_to': [2, 0, 0],
            'relation': None,
            'distance': None
        },
        {
            'name': 'apple',
            'relative_to': 'table',
            'relation': 'on_top',
            'distance': None
        }
    ]
}

solver_input = {
    'scene_config': scene_config,
    'robot_id': 'stretch',
    'task': 'grasp',
    'target_asset': 'apple'
}

# Run solver in subprocess
print("  🔧 Launching solver subprocess (isolated compile)...")
import os
env = os.environ.copy()
env['PYTHONPATH'] = '/home/gal-labs/PycharmProjects/echo_robot/simulation_center'
env['MUJOCO_GL'] = 'egl'

result = subprocess.run(
    ['python3', 'solver_subprocess.py'],
    input=json.dumps(solver_input),
    capture_output=True,
    text=True,
    cwd='/home/gal-labs/PycharmProjects/echo_robot/simulation_center',
    env=env
)

if result.returncode != 0:
    print(f"❌ Solver subprocess failed:")
    print(result.stderr)
    exit(1)

# Parse placement from subprocess output
placement = json.loads(result.stdout.strip())

print(f"\n✅ Calculated placement (from subprocess):")
print(f"   Position: {placement['position']}")
print(f"   Orientation: {placement['orientation']}")
print(f"   Joint positions:")
for joint_name, value in placement['initial_state'].items():
    print(f"      {joint_name}: {value:.3f}")

# Add robot with calculated placement
print("\n🤖 Adding robot with auto-calculated placement...")
ops.add_robot("stretch", **placement)

# ONE CLEAN COMPILE with robot!
print("🔧 Compiling scene with robot (saving package)...")
ops.compile()

print("\n🎬 Starting visual render...")
print("   Press ESC to close window\n")

# Run simulation with rendering
print("   Running for 2000 steps...")
for i in range(2000):
    obs = ops.step()

    if i == 0:
        print(f"✅ Robot positioned and rendering!")

    if i % 200 == 0:
        print(f"   Step {i}/2000...")

print("\n✅ Demo complete!")
print("\n📦 Checking MuJoCo package contents...")

# Check what was saved
import os
package_dir = os.path.join(ops.experiment_dir, "mujoco_package")
if os.path.exists(package_dir):
    print(f"\n📁 Package directory: {package_dir}")
    print("📄 Files saved:")
    for file in sorted(os.listdir(package_dir)):
        file_path = os.path.join(package_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   - {file} ({size} bytes)")
        else:
            print(f"   - {file}/ (directory)")

ops.close()
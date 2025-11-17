#!/usr/bin/env python3
"""
Simple test: Table + Apple - using MODALS
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
ops.create_scene("test_room", width=10, length=10, height=3)

ops.add_asset("table", relative_to=(2, 0, 0))
ops.add_asset("apple", relative_to="table", relation="on_top")
ops.add_robot("stretch", position=(0, 0, 0))
ops.compile()

# Use modals!
state = ops.get_state()

print("\n" + "="*60)
print("TABLE (table_surface component):")
print("="*60)
table = state["table"]["table_surface"]
for key, val in sorted(table.items()):
    print(f"  {key}: {val}")

print("\n" + "="*60)
print("APPLE (body component):")
print("="*60)
apple = state["apple"]["body"]
for key, val in sorted(apple.items()):
    print(f"  {key}: {val}")

print("\n" + "="*60)
print("ROBOT (stretch - ONE asset with components):")
print("="*60)
robot = state.get("stretch", {})
print(f"Components: {list(robot.keys())}")

if "arm" in robot:
    print(f"\n--- arm ---")
    for key, val in sorted(robot["arm"].items()):
        print(f"  {key}: {val}")

if "gripper" in robot:
    print(f"\n--- gripper ---")
    for key, val in sorted(robot["gripper"].items()):
        print(f"  {key}: {val}")

if "base" in robot:
    print(f"\n--- base ---")
    for key, val in sorted(robot["base"].items()):
        print(f"  {key}: {val}")

ops.close()

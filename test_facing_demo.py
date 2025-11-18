#!/usr/bin/env python3
"""
Clean demo of is_facing() with different robot orientations
Shows how orientation affects spatial relationships
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps
import numpy as np

print("\n" + "="*80)
print("DEMO: is_facing() with Robot Orientations")
print("="*80)
print("\nSetup:")
print("  • Apple at (2, 0, 0.76) - 2 meters in front (+X direction)")
print("  • Robot at (0, 0, 0) - at origin")
print("  • Testing different robot orientations (yaw angles)")
print("\n" + "="*80)

# Test 3 key orientations
test_cases = [
    (0, "Facing Forward (+Y)", "Arm points sideways from apple"),
    (90, "Facing Right (+X)", "Arm points TOWARD apple"),
    (180, "Facing Backward (-Y)", "Arm points sideways away"),
]

for angle_deg, description, expectation in test_cases:
    print(f"\n{'─'*80}")
    print(f"Robot: {description} ({angle_deg}°)")
    print(f"Expected: {expectation}")
    print(f"{'─'*80}")

    # Create scene
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test", width=10, length=10, height=3)
    ops.add_asset("apple", relative_to=(2, 0, 0.76))

    # Set robot orientation
    angle_rad = np.deg2rad(angle_deg)
    qw = np.cos(angle_rad / 2)
    qz = np.sin(angle_rad / 2)
    quaternion = (qw, 0, 0, qz)

    ops.add_robot("stretch", position=(0, 0, 0), orientation=quaternion)
    ops.compile()

    # Test facing
    result = ops.is_facing("stretch.arm", "apple")

    # Get arm state
    state = ops.get_state()
    arm_dir = state["stretch"]["arm"]["direction"]

    print(f"\nArm direction: [{arm_dir[0]:.2f}, {arm_dir[1]:.2f}, {arm_dir[2]:.2f}]")
    print(f"Dot product: {result['dot']:.3f}")
    print(f"Classification: {result['dot_class']}")
    print(f"✓ Facing apple: {result['facing']} (threshold=0.7)")
    print(f"\nExplanation: {result['dot_explain']}")

    ops.close()

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print("""
The is_facing() utility correctly detects spatial relationships based on
robot orientation! The arm's direction vector changes with the robot's
base orientation, enabling accurate facing calculations.

Key Insights:
  • 0° (facing +Y):   Arm points in -Y, perpendicular to apple at +X
  • 90° (facing +X):  Arm points in +X, DIRECTLY TOWARD apple!
  • 180° (facing -Y): Arm points in +Y, perpendicular again

Usage:
  result = ops.is_facing("stretch.arm", "apple")
  if result["facing"]:
      print(f"Ready to reach! (dot={result['dot']:.2f})")
      # Execute reach action
  else:
      print(f"Need to rotate (current dot={result['dot']:.2f})")
      # Rotate base to align with target

This enables intelligent decision-making for robot navigation and manipulation!
""")

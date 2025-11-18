#!/usr/bin/env python3
"""
Test is_facing() with different robot orientations
Predict and verify arm facing direction at different angles
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps
import numpy as np

# Apple at fixed position
APPLE_POS = (2, 0, 0.76)

# Test different robot orientations (in degrees)
test_cases = [
    (0,    "Robot facing +X (toward apple)", "directly_facing or facing"),
    (45,   "Robot facing +X+Y (45° right of apple)", "partially_facing"),
    (90,   "Robot facing +Y (perpendicular to apple)", "perpendicular"),
    (135,  "Robot facing -X+Y (135° away)", "partially_away"),
    (180,  "Robot facing -X (directly away from apple)", "directly_opposite or facing_away"),
    (-90,  "Robot facing -Y (perpendicular, left side)", "perpendicular"),
    (-45,  "Robot facing +X-Y (45° left of apple)", "partially_facing"),
]

print("\n" + "="*90)
print("TESTING ROBOT ORIENTATIONS - ARM FACING APPLE")
print("="*90)
print(f"Apple position: {APPLE_POS}")
print(f"Robot position: (0, 0, 0)")
print("="*90)

for angle_deg, description, expected_class in test_cases:
    print(f"\n{'='*90}")
    print(f"TEST: {description}")
    print(f"Robot yaw: {angle_deg}° (Z-axis rotation)")
    print(f"Expected classification: {expected_class}")
    print("="*90)

    # Create scene with robot at specific orientation
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=10, length=10, height=3)
    ops.add_asset("apple", relative_to=APPLE_POS)

    # Convert degrees to quaternion (rotation around Z-axis)
    angle_rad = np.deg2rad(angle_deg)
    # Quaternion for Z-axis rotation: [w, x, y, z] = [cos(θ/2), 0, 0, sin(θ/2)]
    qw = np.cos(angle_rad / 2)
    qz = np.sin(angle_rad / 2)
    quaternion = (qw, 0, 0, qz)

    ops.add_robot("stretch", position=(0, 0, 0), orientation=quaternion)
    ops.compile()

    # Test arm facing apple
    result = ops.is_facing("stretch.arm", "apple")

    # Get arm state to show direction
    state = ops.get_state()
    arm_dir = state["stretch"]["arm"]["direction"]
    arm_pos = state["stretch"]["arm"]["position"]

    # Calculate angle between arm and apple
    apple_pos = np.array(APPLE_POS)
    arm_pos_np = np.array(arm_pos)
    to_apple = apple_pos - arm_pos_np
    to_apple_norm = to_apple / np.linalg.norm(to_apple)
    arm_dir_np = np.array(arm_dir)

    # Angle in degrees
    angle_to_apple = np.rad2deg(np.arccos(np.clip(result['dot'], -1.0, 1.0)))

    print(f"\nRESULTS:")
    print(f"  Arm position: [{arm_pos[0]:.2f}, {arm_pos[1]:.2f}, {arm_pos[2]:.2f}]")
    print(f"  Arm direction: [{arm_dir[0]:.2f}, {arm_dir[1]:.2f}, {arm_dir[2]:.2f}]")
    print(f"  Vector to apple: [{to_apple_norm[0]:.2f}, {to_apple_norm[1]:.2f}, {to_apple_norm[2]:.2f}]")
    print(f"\n  Dot product: {result['dot']:.3f}")
    print(f"  Angle to apple: {angle_to_apple:.1f}°")
    print(f"  Classification: {result['dot_class']}")
    print(f"  Explanation: {result['dot_explain']}")
    print(f"  Facing (threshold=0.7): {result['facing']}")

    # Check if prediction matches
    expected_match = result['dot_class'] in expected_class
    status = "✓ MATCHES PREDICTION" if expected_match else "⚠️  UNEXPECTED"
    print(f"\n  {status}")

    ops.close()

print("\n" + "="*90)
print("SUMMARY: Orientation affects arm direction, which changes dot product with target!")
print("="*90)
print("""
Key Insights:
  • 0° (facing +X):    Arm points toward apple → high dot product (>0.7)
  • 90° (facing +Y):   Arm points perpendicular → dot product ≈ 0
  • 180° (facing -X):  Arm points away from apple → dot product ≈ -1
  • -90° (facing -Y):  Arm points perpendicular (other side) → dot product ≈ 0

The is_facing() utility uses the arm's direction vector (from orientation/xmat)
to calculate alignment with the target. This is exactly what you need for checking
if the robot is positioned correctly to reach an object!
""")

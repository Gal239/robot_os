#!/usr/bin/env python3
"""
Visual demo: Robot positioned in FRONT of table with arm aligned to grasp apple
headless=False, 1000 steps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

print("\n" + "="*80)
print("DEMO: Robot Positioned to Grasp Apple (Visual, 1000 steps)")
print("="*80)

# Create scene with viewer
ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core")
ops.create_scene("grasp_demo", width=8, length=8, height=3)

# Place table at (2, 0, 0)
ops.add_asset("table", relative_to=(2, 0, 0))

# Place apple on table center
ops.add_asset("apple", relative_to="table", relation="on_top", surface_position="center")

# Place robot IN FRONT of table (at the WIDER SIDE, like sitting at dining table)
# Table wider side is along Y axis (depth dimension)
# Robot at (2, -1.0, 0) - in front of wider side
# Face SOUTH so arm points FORWARD (+Y, toward table/apple)
# Stretch arm extends perpendicular to base - need base facing so arm points toward apple
ops.add_robot("stretch", position=(2, -1.0, 0), orientation="south")

ops.compile()

print("\n✓ Scene compiled!")
print("  Table at (2, 0, 0)")
print("  Apple on center of table")
print("  Robot at (2, -1.0, 0) facing SOUTH - arm points FORWARD toward apple")

# Check if arm is aligned with apple
result = ops.is_facing("stretch.arm", "apple")
state = ops.get_state()
arm = state["stretch"]["arm"]
apple = state["apple"]["body"]
gripper = state["stretch"]["gripper"]

print(f"\n📍 Spatial Alignment:")
print(f"  {result['dot_explain']}")
print(f"  Dot product: {result['dot']:.3f}")
print(f"  Distance: {result['distance']:.2f}m")
print(f"  ✓ Facing: {result['facing']}")

print(f"\n🤖 Robot Components:")
print(f"  Arm position: [{arm['position'][0]:.2f}, {arm['position'][1]:.2f}, {arm['position'][2]:.2f}]")
print(f"  Arm direction: [{arm['direction'][0]:.2f}, {arm['direction'][1]:.2f}, {arm['direction'][2]:.2f}]")
print(f"  Arm extension: {arm['extension']:.3f}m")
print(f"  Gripper position: [{gripper.get('position', [0,0,0])[0] if 'position' in gripper else 'N/A'}]")
print(f"  Gripper aperture: {gripper['aperture']:.4f}m")

print(f"\n🍎 Apple:")
print(f"  Position: [{apple['position'][0]:.2f}, {apple['position'][1]:.2f}, {apple['position'][2]:.2f}]")
print(f"  Height: {apple['height']:.2f}m (Z coordinate)")
print(f"  Stacked on table: {apple.get('stacked_on_table', False)}")

# Calculate if apple is within reaching distance
apple_pos = apple['position']
arm_pos = arm['position']
horizontal_dist = ((apple_pos[0] - arm_pos[0])**2 + (apple_pos[1] - arm_pos[1])**2)**0.5
vertical_dist = abs(apple_pos[2] - arm_pos[2])

print(f"\n📏 Reach Analysis:")
print(f"  Horizontal distance to apple: {horizontal_dist:.2f}m")
print(f"  Vertical distance to apple: {vertical_dist:.2f}m")
print(f"  Arm max extension: ~0.52m")

if horizontal_dist < 0.52 and vertical_dist < 0.3:
    print(f"  ✓ Apple is WITHIN REACH!")
else:
    print(f"  ⚠️  Apple may be out of reach (need to extend arm or move closer)")

print("\n" + "="*80)
print("Running 1000 steps (watch the viewer - robot should be close to table)...")
print("Press Ctrl+C to stop early")
print("="*80)

try:
    for i in range(1000):
        ops.step()

        # Print progress every 200 steps
        if (i + 1) % 200 == 0:
            print(f"  Step {i+1}/1000")

            # Check alignment periodically
            result = ops.is_facing("stretch.arm", "apple")
            state = ops.get_state()
            arm = state["stretch"]["arm"]
            print(f"    Arm still aligned: {result['facing']} (dot={result['dot']:.3f}, dist={result['distance']:.2f}m)")
            print(f"    Arm extension: {arm['extension']:.3f}m")

except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")

print("\n" + "="*80)
print("Demo complete!")
print("="*80)

# Final check
result = ops.is_facing("stretch.arm", "apple")
state = ops.get_state()
apple = state["apple"]["body"]

print(f"\nFinal State:")
print(f"  {result['dot_explain']}")
print(f"  Dot: {result['dot']:.3f}")
print(f"  Distance: {result['distance']:.2f}m")
print(f"  Apple still on table: {apple.get('stacked_on_table', False)}")
print(f"  ✓ Robot positioned for grasping!")

ops.close()
print("\n✓ Scene closed")

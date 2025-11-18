#!/usr/bin/env python3
"""
Visual demo: Robot arm facing apple on table
headless=False, 1000 steps
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

print("\n" + "="*80)
print("DEMO: Arm Facing Apple on Table (Visual, 1000 steps)")
print("="*80)

# Create scene with viewer
ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core")
ops.create_scene("demo", width=8, length=8, height=3)

# Place table
ops.add_asset("table", relative_to=(3, 0, 0))

# Place apple on table
ops.add_asset("apple", relative_to="table", relation="on_top")

# Place robot facing EAST (toward table)
ops.add_robot("stretch", position=(0, 0, 0), orientation="east")

ops.compile()

print("\n✓ Scene compiled!")
print("  Table at (3, 0, 0)")
print("  Apple on top of table")
print("  Robot at origin facing EAST (toward table)")

# Check if arm is facing apple
result = ops.is_facing("stretch.arm", "apple")
print(f"\n📍 Spatial Check:")
print(f"  {result['dot_explain']}")
print(f"  Dot: {result['dot']:.3f}")
print(f"  Distance: {result['distance']:.2f}m")
print(f"  Facing: {result['facing']}")

# Get initial state
state = ops.get_state()
arm = state["stretch"]["arm"]
apple = state["apple"]["body"]

print(f"\n🤖 Arm state:")
print(f"  Position: [{arm['position'][0]:.2f}, {arm['position'][1]:.2f}, {arm['position'][2]:.2f}]")
print(f"  Direction: [{arm['direction'][0]:.2f}, {arm['direction'][1]:.2f}, {arm['direction'][2]:.2f}]")
print(f"  Extension: {arm['extension']:.3f}m")

print(f"\n🍎 Apple state:")
print(f"  Position: [{apple['position'][0]:.2f}, {apple['position'][1]:.2f}, {apple['position'][2]:.2f}]")
print(f"  Stacked on table: {apple.get('stacked_on_table', False)}")

print("\n" + "="*80)
print("Running 1000 steps (watch the viewer window)...")
print("Press Ctrl+C to stop early")
print("="*80)

try:
    for i in range(1000):
        ops.step()

        # Print progress every 100 steps
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/1000")

            # Check if still facing
            if (i + 1) % 200 == 0:
                result = ops.is_facing("stretch.arm", "apple")
                print(f"    Arm still facing apple: {result['facing']} (dot={result['dot']:.3f})")
except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")

print("\n" + "="*80)
print("Demo complete!")
print("="*80)

# Final check
result = ops.is_facing("stretch.arm", "apple")
print(f"\nFinal spatial relationship:")
print(f"  {result['dot_explain']}")
print(f"  Dot: {result['dot']:.3f}")
print(f"  Facing: {result['facing']}")

ops.close()
print("\n✓ Scene closed")

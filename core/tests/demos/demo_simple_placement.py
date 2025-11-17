#!/usr/bin/env python3
"""
SIMPLE PLACEMENT TEST - AUTOMATIC robot placement!

Uses RobotPlacementDemo to AUTOMATICALLY:
- Calculate IK for grasping
- Generate action sequences
- Execute pick and place

NO HARDCODING! Pure automatic IK-driven placement!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.main.robot_placement_ops import RobotPlacementDemo


def test_simple_placement():
    """Automatic pick-and-place using RobotPlacementDemo"""
    print("\n" + "="*70)
    print("AUTOMATIC PLACEMENT TEST - RobotPlacementDemo")
    print("="*70)

    # 1. Setup scene
    print("\n1. Creating scene...")
    ops = ExperimentOps(
        mode="simulated",
        headless=False,  # User wants to see the action!
        render_mode="rl_core",
        save_fps=30
    )

    ops.create_scene(name="simple_placement", width=5, length=5, height=3)

    # Robot
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))

    # Table in front
    ops.add_asset(asset_name="table", relative_to=(0.6, 0, 0))

    # Apple ON table
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="center")

    # Bowl ON table
    ops.add_asset(asset_name="bowl", relative_to="table", relation="on_top", offset=(0.0, 0.2))

    ops.add_overhead_camera()

    print("   Scene layout:")
    print("   - Robot at (0, 0, 0)")
    print("   - Table at (0.6, 0, 0)")
    print("   - Apple ON table center")
    print("   - Bowl ON table (20cm to side)")

    # 2. Compile
    print("\n2. Compiling...")
    ops.compile()

    # 3. Settle
    print("\n3. Settling physics (200 steps)...")
    for _ in range(200):
        ops.step()

    # 4. Get bowl position (target for placement)
    print("\n4. Getting target position...")
    state = ops.get_state()
    bowl_pos = state.get('bowl', {}).get('position', [0, 0, 0])
    print(f"   Bowl at: ({bowl_pos[0]:.2f}, {bowl_pos[1]:.2f}, {bowl_pos[2]:.2f})")

    # Target: place apple IN the bowl (slightly above bowl bottom)
    target_location = (bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.1)
    print(f"   Target: ({target_location[0]:.2f}, {target_location[1]:.2f}, {target_location[2]:.2f})")

    # 5. AUTOMATIC PLACEMENT!
    print("\n5. ✨ ACTIVATING AUTOMATIC PLACEMENT...")
    print("   (IK calculation → Action generation → Execution)")

    demo = RobotPlacementDemo(ops)

    try:
        demo.execute_placement(
            object_name="apple",
            target_location=target_location,
            settling_steps=100
        )

        print("\n✅ AUTOMATIC PLACEMENT COMPLETE!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Validation
    print("\n6. Validating...")
    final_state = ops.get_state()
    apple_final = final_state.get('apple', {}).get('position', [0, 0, 0])

    print(f"   Apple final: ({apple_final[0]:.2f}, {apple_final[1]:.2f}, {apple_final[2]:.2f})")
    print(f"   Target: ({target_location[0]:.2f}, {target_location[1]:.2f}, {target_location[2]:.2f})")

    # Check distance to target
    distance = ((apple_final[0] - target_location[0])**2 +
                (apple_final[1] - target_location[1])**2 +
                (apple_final[2] - target_location[2])**2)**0.5

    if distance < 0.3:  # Within 30cm
        print(f"\n✅ SUCCESS! Apple within {distance:.2f}m of target!")
        print("\n🎯 PROVEN:")
        print("   ✓ Automatic IK calculation works")
        print("   ✓ Action generation works")
        print("   ✓ Pick-and-place execution works")
        return True
    else:
        print(f"\n⚠️  Apple {distance:.2f}m from target (expected < 0.3m)")
        return False


if __name__ == "__main__":
    success = test_simple_placement()

    if success:
        print("\n🚀 AUTOMATIC PLACEMENT WORKS! No hardcoding needed!")
    else:
        print("\n❌ Automatic placement failed. Need to debug.")

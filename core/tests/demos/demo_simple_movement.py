#!/usr/bin/env python3
"""
SIMPLE MOVEMENT TEST - Debug base movement
Just move forward 1m - no objects, no manipulation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_blocks_registry import move_forward


def test_simple_movement():
    """Simplest possible test: robot moves forward 1m"""
    print("\n" + "="*70)
    print("SIMPLE MOVEMENT TEST - Move Forward 1m")
    print("="*70)

    # 1. Create scene (no objects!)
    print("\n1. Creating scene...")
    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="2k_demo",
        save_fps=30
    )
    ops.create_scene(name="simple_movement_test", width=10, length=10, height=3)
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))
    ops.add_overhead_camera()

    # 2. Compile
    print("\n2. Compiling...")
    ops.compile()
    print("   ✓ Scene compiled!")

    # 3. Settle physics
    print("\n3. Settling physics (100 steps)...")
    for _ in range(100):
        ops.step()
    print("   ✓ Physics settled!")

    # 4. Submit movement action
    distance_target = 1.0
    print(f"\n4. Submitting move_forward({distance_target}m, speed=0.3)...")
    block = move_forward(distance=distance_target, speed=0.3)
    ops.submit_block(block)
    print("   ✓ Action submitted!")

    # 5. Execute and track
    print(f"\n5. Executing (max 4000 steps)...")
    start_pos = None

    for step in range(4000):
        result = ops.step()

        # Track odometry
        if 'odometry' in ops.robot.sensors:
            odom = ops.robot.sensors['odometry'].get_data()
            if start_pos is None:
                start_pos = (odom['x'], odom['y'])

            dx = odom['x'] - start_pos[0]
            dy = odom['y'] - start_pos[1]
            distance = (dx**2 + dy**2)**0.5
        else:
            distance = 0.0

        # Print progress every 200 steps
        if step % 200 == 0 and step > 0:
            print(f"   Step {step}: distance={distance:.3f}m, block={block.status}, progress={block.progress:.0f}%")

        # Check completion
        if block.status == 'completed':
            print(f"\n✅ MOVEMENT COMPLETED at step {step}!")
            print(f"   Distance traveled: {distance:.3f}m")
            print(f"   Target: {distance_target:.3f}m")
            print(f"   Error: {abs(distance - distance_target):.3f}m")

            # Validate
            if abs(distance - distance_target) < 0.3:  # 30cm tolerance
                print(f"\n🎉 SUCCESS! Robot moved {distance:.3f}m ≈ {distance_target}m")
                return True
            else:
                print(f"\n⚠️  PARTIAL: Moved {distance:.3f}m (expected {distance_target}m)")
                return False

    # Timeout
    print(f"\n❌ TIMEOUT after 4000 steps!")
    print(f"   Final distance: {distance:.3f}m (expected {distance_target}m)")
    print(f"   Block status: {block.status}")
    print(f"   Block progress: {block.progress:.0f}%")

    if distance < 0.1:
        print("\n💥 ROBOT DIDN'T MOVE AT ALL! Possible issues:")
        print("   - Base actuator not working")
        print("   - Wheels stuck/collision")
        print("   - Control commands not reaching backend")

    return False


if __name__ == "__main__":
    success = test_simple_movement()

    if success:
        print("\n✅ Base movement works! Ready for manipulation.")
    else:
        print("\n❌ Base movement failed. Need to debug before adding manipulation.")
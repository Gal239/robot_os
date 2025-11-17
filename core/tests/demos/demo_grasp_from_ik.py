#!/usr/bin/env python3
"""
GRASP DEMO - Using ReachabilityModal (Pure MOP!)

Steps:
1. Create scene (table + apple) WITHOUT robot
2. Compile to get apple's ACTUAL position
3. Use ReachabilityModal.solve_for_grasp() to calculate robot placement
4. Recreate scene with robot at calculated position
5. Just close gripper - robot already positioned!

Pure MOP: Modals calculate everything!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.main.experiment_ops_unified import ExperimentOps
from core.modals.reachability_modal import ReachabilityModal
from core.modals.stretch.action_modals import GripperMoveTo


def test_grasp_from_ik():
    """Use ReachabilityModal to calculate grasp pose"""
    print("\n" + "="*70)
    print("GRASP DEMO - ReachabilityModal Calculation")
    print("="*70)

    # ================================================================
    # STEP 1: Create scene WITHOUT robot (get apple position first!)
    # ================================================================
    print("\n1. Creating scene WITHOUT robot...")
    ops = ExperimentOps(
        mode="simulated",
        headless=False,
        render_mode="rl_core",
        save_fps=30
    )

    ops.create_scene(name="grasp_test", width=5, length=5, height=3)
    ops.add_asset(asset_name="table", relative_to=(0.6, 0, 0))
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="center")
    ops.compile()

    print("   Scene compiled (no robot yet)")

    # Settle physics
    for _ in range(200):
        ops.step()

    # ================================================================
    # STEP 2: Get apple ACTUAL position (SELF-RENDERING!)
    # ================================================================
    print("\n2. Getting apple position from state...")
    state = ops.get_state()
    apple_state = state.get("apple", {})
    apple_pos = np.array(apple_state["position"])

    print(f"   Apple position: ({apple_pos[0]:.3f}, {apple_pos[1]:.3f}, {apple_pos[2]:.3f})")

    # ================================================================
    # STEP 3: Use ReachabilityModal to calculate robot placement!
    # ================================================================
    print("\n3. Calculating robot placement using ReachabilityModal...")
    ik = ReachabilityModal()

    joint_values = ik.solve_for_grasp(
        target_pos=apple_pos,
        object_width=0.08,  # Apple ~8cm wide
        robot_config={}
    )

    # Extract base position and convert joint values to actuator commands
    robot_pos = joint_values.pop("_base_position")

    # Convert joint names to actuator names (MOP: actuators != joints!)
    initial_state = {
        "lift": joint_values["joint_lift"],
        "arm": joint_values["joint_arm_l0"] * 4,  # Sum of 4 arm segments
        "gripper": joint_values["joint_gripper_finger_left"] * 2,  # Sum of left+right fingers
        "wrist_yaw": joint_values["joint_wrist_yaw"],
        "wrist_pitch": joint_values["joint_wrist_pitch"],
        "wrist_roll": joint_values["joint_wrist_roll"],
    }

    print(f"   Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
    print(f"   Lift: {initial_state['lift']:.3f}m")
    print(f"   Arm: {initial_state['arm']:.3f}m")
    print(f"   Gripper: {initial_state['gripper']:.3f}m")

    # ================================================================
    # STEP 4: Recreate scene with robot at calculated position
    # ================================================================
    print("\n4. Recreating scene with robot at calculated pose...")
    ops.close()

    ops = ExperimentOps(
        mode="simulated",
        headless=False,
        render_mode="rl_core",
        save_fps=30
    )

    ops.create_scene(name="grasp_test", width=5, length=5, height=3)

    # Spawn robot at calculated position with initial_state!
    ops.add_robot(
        robot_name="stretch",
        position=tuple(robot_pos),
        initial_state=initial_state
    )

    ops.add_asset(asset_name="table", relative_to=(0.6, 0, 0))
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="center")
    ops.compile()

    print("   Robot spawned with arm extended, gripper at apple!")

    # Check position IMMEDIATELY after compile (before settling!)
    state_immediate = ops.get_state()
    robot_immediate = state_immediate.get("stretch", {}).get("position", [0, 0, 0])
    print(f"   Robot base IMMEDIATELY after compile: ({robot_immediate[0]:.3f}, {robot_immediate[1]:.3f}, {robot_immediate[2]:.3f})")

    # Settle
    for _ in range(200):
        ops.step()

    # Check position AFTER settling
    state_after = ops.get_state()
    robot_after = state_after.get("stretch", {}).get("position", [0, 0, 0])
    print(f"   Robot base AFTER settling: ({robot_after[0]:.3f}, {robot_after[1]:.3f}, {robot_after[2]:.3f})")

    # ================================================================
    # STEP 5: Verify gripper is at apple
    # ================================================================
    print("\n5. Verifying gripper position...")
    state = ops.get_state()

    # Check robot base position
    robot_state = state.get("stretch", {})
    robot_base_pos = robot_state.get("position", [0, 0, 0])
    print(f"   Robot base at: ({robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f})")

    apple_final = state.get("apple", {}).get("position", [0, 0, 0])
    print(f"   Apple at: ({apple_final[0]:.3f}, {apple_final[1]:.3f}, {apple_final[2]:.3f})")

    # Check gripper position
    gripper_state = state.get("stretch.gripper", {})
    gripper_pos = gripper_state.get("position", [0, 0, 0])
    print(f"   Gripper at: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
    print(f"   Gripper open: {gripper_state.get('aperture', 0):.3f}m")
    print(f"   Gripper holding: {gripper_state.get('holding', False)}")

    # Check actuator states
    print(f"\n   Actuator states:")
    print(f"   Lift: {ops.robot.actuators['lift'].get_position():.3f}m")
    print(f"   Arm: {ops.robot.actuators['arm'].get_position():.3f}m")
    print(f"   Gripper: {ops.robot.actuators['gripper'].get_position():.3f}m")

    # Calculate distance
    import math
    distance = math.sqrt((gripper_pos[0] - apple_final[0])**2 +
                        (gripper_pos[1] - apple_final[1])**2 +
                        (gripper_pos[2] - apple_final[2])**2)
    print(f"\n   Distance to apple: {distance:.3f}m")

    # ================================================================
    # STEP 6: Close gripper to grasp!
    # ================================================================
    print("\n6. Closing gripper...")
    close_action = GripperMoveTo(position=0.0, force_limit=5.0)

    # Connect to robot
    close_action.connect(ops.robot, ops.engine.event_log)

    # Execute
    for step in range(500):
        cmd = close_action.tick()
        if cmd:
            ops.engine.backend.set_controls(cmd)
        ops.step()

        if close_action.status == "completed":
            print(f"   Gripper closed at step {step}")
            break

    # ================================================================
    # STEP 7: Verify grasp using modal properties!
    # ================================================================
    print("\n7. Verifying grasp using modal properties...")
    state = ops.get_state()

    # Check apple contact with gripper (MODAL PROPERTY!)
    apple_state = state.get("apple", {})
    contact_force = apple_state.get("held_by_stretch.gripper", 0.0)

    # Check gripper holding state (MODAL PROPERTY!)
    gripper_state = state.get("stretch.gripper", {})
    is_holding = gripper_state.get("holding", False)
    is_closed = gripper_state.get("closed", False)

    print(f"   Contact force: {contact_force:.2f}N")
    print(f"   Gripper holding: {is_holding}")
    print(f"   Gripper closed: {is_closed}")

    # Validation
    if contact_force > 0.5 and is_holding:
        print(f"\n✅ SUCCESS! Gripper grasped apple!")
        print(f"   Contact force: {contact_force:.2f}N")
        print(f"   Modal properties confirm grasp!")
        return True
    else:
        print(f"\n❌ FAILED! Gripper didn't grasp apple")
        print(f"   Contact force: {contact_force:.2f}N (expected > 0.5N)")
        print(f"   Holding: {is_holding} (expected True)")
        return False


if __name__ == "__main__":
    success = test_grasp_from_ik()

    if success:
        print("\n🎉 GRASP WORKS! ReachabilityModal calculated perfect pose!")
    else:
        print("\n⚙️  Need to debug grasp calculation")

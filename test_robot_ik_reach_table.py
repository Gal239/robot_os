#!/usr/bin/env python3
"""
TEST: Robot uses IK to reach object on table
"""
from core.main.experiment_ops_unified import ExperimentOps

print("\n" + "="*70)
print("TEST: Robot IK - Reach Apple on Table")
print("="*70)

ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core")
ops.create_scene(name="robot_ik_reach", width=10, length=10, height=3)

# Place robot FIRST (for qpos consistency)
print("\n1. Adding robot at (0.0, 0.8, 0.0)...")
ops.add_robot(robot_name="stretch", position=(0.0, 0.8, 0.0))

# Place table at origin
print("2. Adding table at (0.0, 0.0, 0.0)...")
ops.add_asset(asset_name="table", relative_to=(0.0, 0.0, 0.0))

# Place apple ON the table
print("3. Adding apple on table at (0.0, 0.0, 1.0)...")
ops.add_asset(asset_name="apple", relative_to=(0.0, 0.0, 1.0))

ops.compile()

# Check initial positions
state = ops.get_state()
robot_pos = state["stretch.base"]["position"]
apple_pos = state["apple"]["position"]

print(f"\n4. Initial positions:")
print(f"   Robot: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
print(f"   Apple: ({apple_pos[0]:.3f}, {apple_pos[1]:.3f}, {apple_pos[2]:.3f})")

# Calculate target position (slightly above apple)
target_x = apple_pos[0]
target_y = apple_pos[1]
target_z = apple_pos[2] + 0.05  # 5cm above apple

print(f"\n5. Target gripper position: ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

# Use IK to reach target
print("\n6. Using IK to move gripper to target...")
try:
    # Try to reach the target position
    ops.robot.reach_position(
        target_position=(target_x, target_y, target_z),
        max_steps=200
    )
    print("   ✅ IK command sent successfully")
except AttributeError:
    print("   ⚠️  reach_position() not available, trying alternative...")
    # Try alternative approach - set action directly
    try:
        # Queue action to reach position
        action = {
            "type": "reach_position",
            "target": [target_x, target_y, target_z],
            "max_steps": 200
        }
        ops.apply_action(action)
        print("   ✅ Action queued successfully")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Trying manual joint control instead...")

        # Manual approach - extend arm and lift
        for step in range(200):
            # Get current state
            state = ops.get_state()

            # Simple control: extend arm, raise lift
            arm_extension = 0.5  # 50% extended
            lift_height = 0.8    # 80cm lift

            # Apply control (if possible)
            ops.step()

            if step % 50 == 0:
                gripper_pos = state.get("stretch.gripper", {}).get("position", [0, 0, 0])
                print(f"   Step {step}: gripper at ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")

# Run simulation to execute movement
print("\n7. Running simulation (200 steps)...")
for i in range(200):
    ops.step()
    if i % 50 == 0:
        state = ops.get_state()
        # Try to get gripper position
        if "stretch.gripper" in state:
            gripper_pos = state["stretch.gripper"].get("position", [0, 0, 0])
            distance_to_apple = (
                (gripper_pos[0] - apple_pos[0])**2 +
                (gripper_pos[1] - apple_pos[1])**2 +
                (gripper_pos[2] - apple_pos[2])**2
            )**0.5
            print(f"   Step {i:3d}: gripper distance to apple = {distance_to_apple:.3f}m")

# Final check
state = ops.get_state()
print(f"\n8. Final state:")
print(f"   Robot base: ({state['stretch.base']['position'][0]:.3f}, {state['stretch.base']['position'][1]:.3f}, {state['stretch.base']['position'][2]:.3f})")

if "stretch.gripper" in state and "position" in state["stretch.gripper"]:
    gripper_pos = state["stretch.gripper"]["position"]
    distance = (
        (gripper_pos[0] - apple_pos[0])**2 +
        (gripper_pos[1] - apple_pos[1])**2 +
        (gripper_pos[2] - apple_pos[2])**2
    )**0.5
    print(f"   Gripper: ({gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f})")
    print(f"   Distance to apple: {distance:.3f}m")

    if distance < 0.15:
        print(f"\n   ✅ SUCCESS! Gripper within reach of apple!")
    else:
        print(f"\n   ⚠️  Gripper not close enough (need < 0.15m)")

print("\n" + "="*70)
input("Check viewer, then press Enter to close...")
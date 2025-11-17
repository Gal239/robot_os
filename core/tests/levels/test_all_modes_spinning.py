#!/usr/bin/env python3
"""
Test ALL reward modes with spinning action - validates each mode works correctly

Modes tested:
1. Discrete - binary 0/100
2. Convergent - smooth gradient + penalties
3. Achievement - smooth gradient, forgiving (no negative)
"""
import sys
import os
from pathlib import Path

# Enable EGL for GPU-accelerated rendering
os.environ['MUJOCO_GL'] = 'egl'

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_blocks_registry import spin


def test_mode(mode_name, target_degrees=45, action_degrees=45):
    """Test a single reward mode"""

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="rl_core"
    )

    ops.create_scene(f"test_{mode_name}_{action_degrees}", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Add reward with specific mode
    ops.add_reward(
        tracked_asset="stretch.base",
        behavior="rotation",
        target=target_degrees,
        reward=100,
        mode=mode_name,
        id=f"rotation_{mode_name}"
    )

    ops.compile()

    # Submit action
    action_block = spin(degrees=action_degrees)
    ops.submit_block(action_block)

    # Collect trajectory
    trajectory = {
        "mode": mode_name,
        "action": action_degrees,
        "target": target_degrees,
        "rotations": [],
        "rewards": [],
        "steps": 0,
        "final_rotation": 0,
        "final_reward": 0
    }

    max_steps = 300
    for step in range(max_steps):
        result = ops.step()

        # Get current state
        state = ops.get_state()
        rotation = state.get("stretch.base", {}).get("rotation", 0)

        # Get rewards
        total_reward = result.get('reward_total', 0.0)

        trajectory["rotations"].append(rotation)
        trajectory["rewards"].append(total_reward)

        # Check if action completed
        if action_block.status == "completed":
            trajectory["steps"] = step + 1
            trajectory["final_rotation"] = rotation
            trajectory["final_reward"] = total_reward
            break

    return trajectory


def print_trajectory(traj, show_full=False):
    """Print trajectory summary"""
    mode = traj["mode"]
    action = traj["action"]
    target = traj["target"]
    final_rot = traj["final_rotation"]
    final_rew = traj["final_reward"]
    steps = traj["steps"]

    print(f"\n   Mode: {mode}")
    print(f"   Action: spin({action:+.0f}°)")
    print(f"   Target: {target:.0f}°")
    print(f"   ─" * 35)
    print(f"   Steps: {steps}")
    print(f"   Final rotation: {final_rot:+.1f}°")
    print(f"   Final reward: {final_rew:.1f}pts")

    if show_full:
        # Show reward trajectory (sample every 10 steps)
        print(f"\n   Reward trajectory:")
        sample_interval = max(1, steps // 10)
        for i in range(0, min(steps, len(traj["rewards"])), sample_interval):
            rot = traj["rotations"][i] if i < len(traj["rotations"]) else 0
            rew = traj["rewards"][i]
            print(f"      Step {i:3d}: rotation={rot:+6.1f}°, reward={rew:6.1f}pts")


def main():
    print("="*70)
    print("TEST ALL REWARD MODES - SPINNING")
    print("="*70)
    print("\nTesting all reward modes with rotation task")
    print("Target: 45° rotation")
    print("Action: spin(+45°) - perfect action\n")

    # Test 1: Discrete mode
    print("\n" + "="*70)
    print("TEST 1: DISCRETE MODE")
    print("="*70)
    print("Expected: 0pts until target reached, then jump to 100pts")

    traj_discrete = test_mode("discrete", target_degrees=45, action_degrees=45)
    print_trajectory(traj_discrete, show_full=True)

    # Test 2: Convergent mode
    print("\n" + "="*70)
    print("TEST 2: CONVERGENT MODE")
    print("="*70)
    print("Expected: Smooth increase 0→20→40→60→80→100pts")
    print("         Can go NEGATIVE if you overshoot!")

    traj_convergent = test_mode("convergent", target_degrees=45, action_degrees=45)
    print_trajectory(traj_convergent, show_full=True)

    # Test 3: Achievement mode
    print("\n" + "="*70)
    print("TEST 3: ACHIEVEMENT MODE")
    print("="*70)
    print("Expected: Smooth increase like convergent")
    print("         But FORGIVING (no negative rewards)")

    traj_achievement = test_mode("achievement", target_degrees=45, action_degrees=45)
    print_trajectory(traj_achievement, show_full=True)

    # Test overshooting with different modes
    print("\n" + "="*70)
    print("TEST 4: OVERSHOOTING BEHAVIOR")
    print("="*70)
    print("Action: spin(+90°) - overshoot past 45° target")
    print("Testing how each mode handles overshooting\n")

    print("   Discrete mode (overshoot):")
    traj_d_over = test_mode("discrete", target_degrees=45, action_degrees=90)
    print(f"      Final: {traj_d_over['final_rotation']:+.1f}°, {traj_d_over['final_reward']:.1f}pts")

    print("\n   Convergent mode (overshoot):")
    traj_c_over = test_mode("convergent", target_degrees=45, action_degrees=90)
    print(f"      Final: {traj_c_over['final_rotation']:+.1f}°, {traj_c_over['final_reward']:.1f}pts")
    print(f"      ← Notice NEGATIVE reward (penalty for overshooting!)")

    print("\n   Achievement mode (overshoot):")
    traj_a_over = test_mode("achievement", target_degrees=45, action_degrees=90)
    print(f"      Final: {traj_a_over['final_rotation']:+.1f}°, {traj_a_over['final_reward']:.1f}pts")
    print(f"      ← Stays at 0pts or positive (forgiving, no penalty)")

    # Test wrong direction
    print("\n" + "="*70)
    print("TEST 5: WRONG DIRECTION")
    print("="*70)
    print("Action: spin(-45°) - moving AWAY from target")
    print("Testing how each mode handles wrong direction\n")

    print("   Discrete mode (wrong direction):")
    traj_d_wrong = test_mode("discrete", target_degrees=45, action_degrees=-45)
    print(f"      Final: {traj_d_wrong['final_rotation']:+.1f}°, {traj_d_wrong['final_reward']:.1f}pts")

    print("\n   Convergent mode (wrong direction):")
    traj_c_wrong = test_mode("convergent", target_degrees=45, action_degrees=-45)
    print(f"      Final: {traj_c_wrong['final_rotation']:+.1f}°, {traj_c_wrong['final_reward']:.1f}pts")
    print(f"      ← NEGATIVE reward (penalty for moving away!)")

    print("\n   Achievement mode (wrong direction):")
    traj_a_wrong = test_mode("achievement", target_degrees=45, action_degrees=-45)
    print(f"      Final: {traj_a_wrong['final_rotation']:+.1f}°, {traj_a_wrong['final_reward']:.1f}pts")
    print(f"      ← Stays at 0pts (forgiving mode)")

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: MODE COMPARISON")
    print("="*70)

    print(f"\n{'Mode':<15} {'Perfect':<12} {'Overshoot':<12} {'Wrong Dir':<12} {'Best For':<30}")
    print("-" * 85)
    print(f"{'Discrete':<15} {traj_discrete['final_reward']:>8.0f}pts  {traj_d_over['final_reward']:>8.0f}pts  {traj_d_wrong['final_reward']:>8.0f}pts  Binary tasks (on/off)")
    print(f"{'Convergent':<15} {traj_convergent['final_reward']:>8.0f}pts  {traj_c_over['final_reward']:>8.0f}pts  {traj_c_wrong['final_reward']:>8.0f}pts  Precise control (RL)")
    print(f"{'Achievement':<15} {traj_achievement['final_reward']:>8.0f}pts  {traj_a_over['final_reward']:>8.0f}pts  {traj_a_wrong['final_reward']:>8.0f}pts  Exploration (curriculum)")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("1. DISCRETE: All-or-nothing, perfect for binary goals")
    print("2. CONVERGENT: Smooth gradients + penalties, perfect for RL")
    print("3. ACHIEVEMENT: Smooth gradients, forgiving, perfect for learning")
    print("="*70)


if __name__ == "__main__":
    main()

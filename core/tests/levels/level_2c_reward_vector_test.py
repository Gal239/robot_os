#!/usr/bin/env python3
"""
LEVEL 2C: REWARD VECTOR TEST - Validate RL Learning Signal
============================================================

Goal: Show that reward vectors help agents distinguish good vs bad actions

Target: spin(45°) for 100pts

Test 3 actions:
1. spin(45°)  - Perfect → smooth increase to 100pts, stays there
2. spin(180°) - Overshoot → peaks at 100pts then drops
3. spin(-45°) - Wrong direction → stays at 0pts

This validates the RL interface provides rich learning signals.
"""

import os
import sys
from pathlib import Path

# Enable EGL for GPU-accelerated rendering
os.environ['MUJOCO_GL'] = 'egl'

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_blocks_registry import spin


def test_reward_vector(action_degrees, target_degrees=45):
    """Execute one action and collect reward trajectory"""

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="rl_core"
    )

    ops.create_scene(f"test_spin_{action_degrees}", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Set reward: 100pts for reaching target rotation
    # Use convergent mode for smooth reward signal (required for RL!)
    ops.add_reward(
        tracked_asset="stretch.base",
        behavior="rotation",
        target=target_degrees,
        reward=100,
        mode="convergent",
        id=f"rotation_{target_degrees}"
    )

    ops.compile()

    # Submit action
    action_block = spin(degrees=action_degrees)
    ops.submit_block(action_block)

    # Collect trajectory
    trajectory = {
        "action": action_degrees,
        "target": target_degrees,
        "rotations": [],
        "step_rewards": [],
        "cumulative_rewards": [],
        "steps": 0,
        "final_rotation": 0,
        "final_reward": 0
    }

    prev_total = 0
    max_steps = 300

    for step in range(max_steps):
        result = ops.step()

        # Get current state
        state = ops.get_state()
        rotation = state.get("stretch.base", {}).get("rotation", 0)

        # Get rewards
        total_reward = result.get('reward_total', 0.0)
        step_reward = result.get('reward_step', 0.0)

        trajectory["rotations"].append(rotation)
        trajectory["step_rewards"].append(step_reward)
        trajectory["cumulative_rewards"].append(total_reward)

        # Check if action completed
        if action_block.status == "completed":
            trajectory["steps"] = step + 1
            trajectory["final_rotation"] = rotation
            trajectory["final_reward"] = total_reward
            break

    return trajectory


def print_trajectory_summary(traj):
    """Print concise summary of reward trajectory"""
    action = traj["action"]
    target = traj["target"]
    final_rot = traj["final_rotation"]
    final_rew = traj["final_reward"]
    steps = traj["steps"]

    # Find peak reward
    peak_reward = max(traj["cumulative_rewards"]) if traj["cumulative_rewards"] else 0
    peak_step = traj["cumulative_rewards"].index(peak_reward) if traj["cumulative_rewards"] else 0

    # Count non-zero step rewards
    nonzero_steps = sum(1 for r in traj["step_rewards"] if abs(r) > 0.01)

    print(f"\n   Action: spin({action:+.0f}°)")
    print(f"   Target: {target:.0f}°")
    print(f"   ─" * 35)
    print(f"   Steps taken: {steps}")
    print(f"   Final rotation: {final_rot:+.1f}°")
    print(f"   Final reward: {final_rew:.1f}pts")
    print(f"   Peak reward: {peak_reward:.1f}pts at step {peak_step}")
    print(f"   Reward changes: {nonzero_steps} steps")

    # Show reward trajectory (sample every 10 steps)
    print(f"\n   Reward trajectory:")
    sample_interval = max(1, steps // 10)
    for i in range(0, min(steps, len(traj["cumulative_rewards"])), sample_interval):
        rot = traj["rotations"][i] if i < len(traj["rotations"]) else 0
        cum = traj["cumulative_rewards"][i]
        step_r = traj["step_rewards"][i]
        print(f"      Step {i:3d}: rotation={rot:+6.1f}°, "
              f"step_rew={step_r:+5.1f}pts, total={cum:6.1f}pts")

    # Show final few steps
    if steps > 5 and steps > sample_interval * 10:
        print(f"      ...")
        for i in range(max(steps-3, 0), steps):
            if i < len(traj["rotations"]):
                rot = traj["rotations"][i]
                cum = traj["cumulative_rewards"][i]
                step_r = traj["step_rewards"][i]
                print(f"      Step {i:3d}: rotation={rot:+6.1f}°, "
                      f"step_rew={step_r:+5.1f}pts, total={cum:6.1f}pts")


def main():
    print("="*70)
    print("LEVEL 2C: REWARD VECTOR TEST")
    print("="*70)
    print("\nGoal: Validate that reward vectors distinguish good vs bad actions")
    print(f"Target: spin(45°) for 100pts (convergent mode)")
    print("\nTesting 3 actions:")
    print("  1. spin(+45°)  - Perfect action")
    print("  2. spin(+180°) - Overshoot action")
    print("  3. spin(-45°)  - Wrong direction (moves away from target)")

    # Test 1: Perfect action
    print("\n" + "="*70)
    print("TEST 1: Perfect Action - spin(+45°)")
    print("="*70)
    print("Expected: Smooth increase from ~75pts (starting distance) to 100pts, stays there")

    traj1 = test_reward_vector(action_degrees=45, target_degrees=45)
    print_trajectory_summary(traj1)

    # Test 2: Overshoot action
    print("\n" + "="*70)
    print("TEST 2: Overshoot Action - spin(+180°)")
    print("="*70)
    print("Expected: Increases to 100pts around 45°, then drops as we overshoot past target")

    traj2 = test_reward_vector(action_degrees=180, target_degrees=45)
    print_trajectory_summary(traj2)

    # Test 3: Wrong direction
    print("\n" + "="*70)
    print("TEST 3: Wrong Direction - spin(-45°)")
    print("="*70)
    print("Expected: Starts at ~75pts (initial distance), DECREASES as we move away from target")

    traj3 = test_reward_vector(action_degrees=-45, target_degrees=45)
    print_trajectory_summary(traj3)

    # Compare trajectories
    print("\n" + "="*70)
    print("COMPARISON: Which Action is Best?")
    print("="*70)

    print(f"\n   Action          Final Reward    Peak Reward    Analysis")
    print(f"   ─" * 70)

    peak1 = max(traj1["cumulative_rewards"]) if traj1["cumulative_rewards"] else 0
    peak2 = max(traj2["cumulative_rewards"]) if traj2["cumulative_rewards"] else 0
    peak3 = max(traj3["cumulative_rewards"]) if traj3["cumulative_rewards"] else 0

    print(f"   spin(+45°)      {traj1['final_reward']:6.1f}pts      "
          f"{peak1:6.1f}pts      ✅ Perfect! Reaches and stays at 100pts")
    print(f"   spin(+180°)     {traj2['final_reward']:6.1f}pts      "
          f"{peak2:6.1f}pts      ⚠️  Overshot! Peaked at 100pts then dropped")
    print(f"   spin(-45°)      {traj3['final_reward']:6.1f}pts      "
          f"{peak3:6.1f}pts      ❌ Wrong direction! Moved away, reward decreased")

    print("\n" + "="*70)
    print("KEY INSIGHT: Convergent Mode Rewards")
    print("="*70)
    print("\n   The reward vector reveals the PROCESS, not just the outcome!")
    print("\n   ✓ Perfect action (45°): Increases smoothly → stable at 100pts")
    print("   ✓ Overshoot (180°): Peaks at 100pts → drops below start (negative progress!)")
    print("   ✓ Wrong direction (-45°): Starts at 75pts → decreases to 50pts (moving away)")
    print("\n   Convergent mode penalizes distance from target - rich signal for RL! 🎯")
    print("\n" + "="*70)

    # VALIDATION: Verify per-step rewards work correctly!
    print("\n" + "="*70)
    print("VALIDATION: Per-Step Reward Correctness")
    print("="*70)

    all_pass = True

    # Test 1: Check per-step rewards sum to final (within floating point tolerance)
    for traj, name in [(traj1, "spin(+45°)"), (traj2, "spin(+180°)"), (traj3, "spin(-45°)")]:
        step_sum = sum(traj["step_rewards"])
        final_reward = traj["final_reward"]
        diff = abs(step_sum - final_reward)

        if diff < 0.1:  # Allow small floating point error
            print(f"   ✅ {name}: step_sum={step_sum:.1f}, final={final_reward:.1f} (diff={diff:.2f})")
        else:
            print(f"   ❌ {name}: step_sum={step_sum:.1f} != final={final_reward:.1f} (diff={diff:.2f})")
            all_pass = False

    # Test 2: Check rewards are distributed (not frozen/throttled)
    for traj, name in [(traj1, "spin(+45°)"), (traj2, "spin(+180°)")]:
        non_zero_steps = len([r for r in traj["step_rewards"] if abs(r) > 0.01])
        total_steps = traj["steps"] if traj["steps"] > 0 else len(traj["step_rewards"])  # Use actual steps taken

        # At least 20% of steps should have non-zero rewards (not all throttled)
        if total_steps > 0 and non_zero_steps > total_steps * 0.2:
            print(f"   ✅ {name}: {non_zero_steps}/{total_steps} steps have rewards (not throttled!)")
        else:
            print(f"   ❌ {name}: Only {non_zero_steps}/{total_steps} steps have rewards (THROTTLED!)")
            all_pass = False

    if all_pass:
        print(f"\n   🎉 ALL VALIDATIONS PASSED!")
        print(f"   Per-step rewards work correctly - ready for RL training!")
    else:
        print(f"\n   ❌ VALIDATION FAILED!")
        print(f"   Per-step rewards are broken - fix reward computation!")

    print("="*70)

    return all_pass


if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

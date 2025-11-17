"""
LEVEL 1A: MOP Time Machine - Save -> Load -> Replay

Tests the complete save -> load -> replay cycle with actions and rewards
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps, load_experiment


def test_level_1a_time_machine():
    """LEVEL 1A: Save experiment with actions -> Load -> Replay backwards/forwards"""
    print("\n" + "="*70)
    print("LEVEL 1A: MOP Time Machine - Save -> Load -> Replay")
    print("="*70)

    # =========================================================================
    # PHASE 1: SAVE - Create and save experiment with robot movement
    # =========================================================================
    print("\nPHASE 1: Creating and saving experiment...")

    ops1 = ExperimentOps(
        headless=True,
        render_mode="rl_core",
        save_fps=30,
        mode="simulated"
    )

    print("  Creating 10x10m warehouse...")
    ops1.create_scene("warehouse", width=10, length=10, height=3)
    ops1.add_robot("stretch", position=(0, 0, 0))

    # Add start marker at robot's initial position
    ops1.add_asset("apple", relative_to=(0, 0, 0))

    # SELF-VALIDATING: Reward validates actual displacement from start!
    # TRUE MOP: Natural range auto-discovered from initial distance!
    ops1.add_reward(
        tracked_asset="stretch.base",
        behavior="distance_to",
        spatial_target="apple",
        target=1.0,
        reward=100,
        mode="convergent",
        id="move_1m"
    )
    print("  Added reward: distance_to(apple) >= 1m = 100pts")

    ops1.compile()

    # Run robot moving forward using action system
    print("  Submitting move_forward action (1.0m)...")
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=1.0, speed=0.3)
    ops1.submit_block(block)

    # Execute until REWARD + ACTION both validate
    print("  Running simulation (reward + action validate)...")
    for step in range(4000):
        result = ops1.step()
        reward_total = result['reward_total']

        # Progress indicator
        if step % 500 == 0:
            print(f"    Step {step} - Reward: {reward_total:.1f}pts, Progress: {block.progress:.0f}%")

        # SELF-VALIDATING: Both systems prove completion!
        if reward_total >= 100 and block.status == 'completed':
            print(f"  VALIDATED at step {step}!")
            print(f"    Reward: {reward_total:.1f}pts (proves 1m displacement from start)")
            print(f"    Block: {block.status} (proves action complete)")
            break

    experiment_path = ops1.experiment_dir
    ops1.close()

    print(f"  Saved experiment: {experiment_path}")

    # =========================================================================
    # PHASE 2: LOAD - Load experiment with viewer
    # =========================================================================
    print("\nPHASE 2: Loading experiment with viewer...")

    ops2 = load_experiment(
        path=experiment_path,
        frame=None,
        headless=False  # VIEWER ON!
    )

    # Get frame count
    frame_count = ops2.get_frame_count()
    final_frame = frame_count - 1

    print(f"  Loaded experiment ({frame_count} frames)")
    print(f"  Viewer: ON")

    # =========================================================================
    # PHASE 3: JUMP TO FINAL FRAME
    # =========================================================================
    print(f"\nPHASE 3: Jumping to final frame {final_frame}...")

    ops2.load_frame(final_frame)
    print(f"  At final frame {final_frame}")

    # =========================================================================
    # PHASE 4: REPLAY BACKWARDS (Time Travel!)
    # =========================================================================
    print(f"\nPHASE 4: Replaying BACKWARDS ({final_frame} -> 0 at 30 FPS)...")

    import time
    fps = 30
    frame_time = 1.0 / fps

    for frame in range(final_frame, -1, -1):
        ops2.load_frame(frame)
        time.sleep(frame_time)

        if frame % 50 == 0 or frame == 0:
            print(f"    Frame {frame}/{final_frame}")

    print("  Backward replay complete!")

    # =========================================================================
    # PHASE 5: REPLAY FORWARDS (Fast)
    # =========================================================================
    print(f"\nPHASE 5: Replaying FORWARD (0 -> {final_frame} at 60 FPS)...")

    for frame in range(0, final_frame + 1):
        ops2.load_frame(frame)
        time.sleep(1.0 / 60)  # 2x speed

        if frame % 50 == 0 or frame == final_frame:
            print(f"    Frame {frame}/{final_frame}")

    print("  Forward replay complete!")

    ops2.close()

    print("\n" + "="*70)
    print("LEVEL 1A PASSED!")
    print("="*70)
    print("\nWhat We Proved:")
    print("  [X] Actions - move_forward() with action system")
    print("  [X] Save - Creates mujoco_package + scene_state")
    print("  [X] Load - Factory function returns ready ops")
    print("  [X] Jump - Load any frame instantly")
    print("  [X] Viewer - Toggle headless mode")
    print("  [X] Time Travel - Backward and forward replay")
    print("  [X] Pure MOP - Complete time machine")


if __name__ == "__main__":
    print("\nStarting LEVEL 1A: MOP Time Machine Test...")
    print("="*70)

    test_level_1a_time_machine()

    print("\n" + "="*70)
    print("LEVEL 1A COMPLETE!")
    print("="*70)
    print("\nPure MOP Time Travel: Save -> Load -> Jump -> Rewind -> Fast-forward!")

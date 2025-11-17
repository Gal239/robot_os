#!/usr/bin/env python3
"""
SCENE OPERATIONS SIDE TEST: Kitchen Breakfast Scene
====================================================
PURPOSE: Test breakfast scene with object tracking cameras + validation

Creates breakfast scene (apple, banana, mug, bowl on table) with:
- 4 tracking cameras (one per object)
- 3 validation cameras (top/side/detail)
- 10 second cinematic recording
"""

import sys
import time
from pathlib import Path


# Add simulation_center to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_modals import ActionBlock, BaseMoveForward
from core.modals.stretch.action_blocks_registry import move_backward,spin


def test_1_kitchen_breakfast_scene():
    """Test 1: Kitchen Breakfast Scene - Morning table setup with object tracking"""
    print("\n" + "="*70)
    print("TEST 1: Kitchen Breakfast Scene (CINEMATIC + TRACKING)")
    print("="*70)

    # High-quality 30fps HD video recording
    ops = ExperimentOps(mode="simulated",headless=False,render_mode="rl_core",save_fps=30)
    print("\n  Building breakfast scene...")
    ops.create_scene(name="breakfast_scene", width=8, length=8, height=3)
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))
    ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))

    # BEAUTIFUL MOP: Surface position behavior (auto-calculates distance + offset!)
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="top_left")
    ops.add_asset(asset_name="banana", relative_to="table", relation="on_top", surface_position="center")
    ops.add_asset(asset_name="mug", relative_to="table", relation="on_top", surface_position="top_right")
    ops.add_asset(asset_name="bowl", relative_to="table", relation="on_top", surface_position="bottom_left")
    ops.add_asset(asset_name="plate", relative_to="table", relation="on_top", surface_position="bottom_center")
    ops.add_asset(asset_name="spoon", relative_to="table", relation="on_top", surface_position="bottom_right")
    ops.add_asset(asset_name="orange", relative_to="table", relation="on_top", surface_position="center_left")

    # Pure MOP: Objects know what they're stacked on! (SPATIAL PROPERTIES!)    ops.add_reward(tracked_asset="apple", behavior="stacked_on", target="table", reward=100, id="apple_on_table")
    ops.add_reward(tracked_asset="banana", behavior="stacked_on", target="table", reward=100, id="banana_on_table")
    ops.add_reward(tracked_asset="mug", behavior="stacked_on", target="table", reward=100, id="mug_on_table")
    ops.add_reward(tracked_asset="bowl", behavior="stacked_on", target="table", reward=-10, id="bowl_not_on_table")

    # Table knows what's supporting (reciprocal relationship - SPATIAL!)
    ops.add_reward(tracked_asset="table", behavior="supporting", target="apple", reward=50, id="table_supports_apple")
    ops.add_reward(tracked_asset="table", behavior="supporting", target="banana", reward=50, id="table_supports_banana")
    ops.add_reward(tracked_asset="table", behavior="supporting", target="mug", reward=50, id="table_supports_mug")
    ops.add_reward(tracked_asset="table", behavior="supporting", target="bowl", reward=-10, id="table_supports_bowl")

    # FROM FLOOR SIDE OF THINGS
    ops.add_reward(tracked_asset="floor", behavior="supporting", target="apple", reward=-25, id="floor_supports_apple")
    ops.add_reward(tracked_asset="floor", behavior="supporting", target="mug", reward=-25, id="floor_supports_mug")
    ops.add_reward(tracked_asset="floor", behavior="supporting", target="banana", reward=-25, id="floor_supports_banana")
    ops.add_reward(tracked_asset="floor", behavior="supporting", target="bowl", reward=-25, id="floor_supports_bowl")
    # Check table stability (not wobbling with objects on top)
    ops.add_reward(tracked_asset="table", behavior="stable", target=True, reward=50, id="table_stable")

    # NAVIGATION REWARD: Robot gets closer to table
    ops.add_reward(tracked_asset="stretch.base",behavior="distance_to",target="table",tolerance_override=0.2,reward=50,id="navigated")

    # =====DEBUG MOP VALIDATION====

    # FLOOR VALIDATION: Objects that fell should be detected on floor
    ops.add_reward(tracked_asset="apple", behavior="stacked_on", target="floor", reward=-25, id="apple_on_floor")
    ops.add_reward(tracked_asset="mug", behavior="stacked_on", target="floor", reward=-25, id="mug_on_floor")
    ops.add_reward(tracked_asset="banana", behavior="stacked_on", target="floor", reward=-25, id="banana_on_floor")
    ops.add_reward(tracked_asset="bowl", behavior="stacked_on", target="floor", reward=-25, id="bowl_on_floor")

    # ===== TRACKING CAMERAS=====
    print("\n  Adding TRACKING cameras (follow each object)...")
    ops.add_free_camera(camera_id="track_table",track_target="table",distance=1.5,azimuth=135,elevation=-30)  # TODO: Fix furniture tracking
    ops.add_free_camera(camera_id="track_apple",track_target="apple",distance=1.0,azimuth=45,elevation=-20)
    ops.add_free_camera(camera_id="track_banana",track_target="banana",distance=1.0,azimuth=90,elevation=-20)
    ops.add_free_camera(camera_id="track_mug",track_target="mug",distance=1.0,azimuth=135,elevation=-20)
    ops.add_free_camera(camera_id="track_bowl",track_target="bowl",distance=2,azimuth=180,elevation=-15)

    # ===== CINEMATIC CAMERAS=====
    ops.add_overhead_camera()  # Automatically calculates lookat, distance, azimuth, elevation!
    ops.add_free_camera("side_cam",lookat=(2.0, 0.0, 0.8),distance=2.5, azimuth=90, elevation=-20)
    ops.add_free_camera("robot_pov_cam",lookat=(2.0, 0.0, 0.8),distance=1.5, azimuth=0, elevation=-10)
    ops.add_free_camera("robot_cam",lookat=(1.0, 0.0, 0.5), distance=3.0, azimuth=135, elevation=-20)
    ops.compile()

    block = spin(degrees=90, speed=6.0)  # Positive = clockwise to match reward!
    ops.submit_block(block)
    print("\n  ▶️  Running simulation with reward tracking...")
    number_of_steps=1000
    for step in range(number_of_steps):
        ops.step()



    # ===== SEMANTIC VALIDATION (CLEAN API) =====
    result = ops.validate_semantics(expected_on_table=['apple', 'banana', 'mug'],expected_on_floor=['bowl'])

    # ===== LAYER 3: VISUAL VALIDATION (SCREENSHOTS) =====
    ops.save_all_screenshots(frame=10, subdir="all_screenshots")
    ops.close()
    ops.validate_videos(timeout=180)

    # ✨ BEAUTIFUL MOP: Single method validates all videos!
    result = ops.validate_video_files()

    # Check if validation passed
    if not result['valid']:
        return False

    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    screenshots_dir = Path(ops.experiment_dir) / "all_screenshots"

    # ===== VALIDATION SUMMARY =====
    print("\n" + "="*70)
    print(f"👁️ VISUAL VALIDATION:")
    print(f"   Screenshots: {screenshots_dir}")
    print(f"   Videos:      {videos_dir}")
    print(f"\n🔍 MANUAL INSPECTION:")
    print(f"   1. View all screenshots:")
    print(f"      eog {screenshots_dir}/*.jpg")
    print(f"   2. Play tracking videos:")
    print(f"      vlc {videos_dir}/track_apple/track_apple_rgb.mp4")
    print(f"      vlc {videos_dir}/track_banana/track_banana_rgb.mp4")
    print(f"      vlc {videos_dir}/track_mug/track_mug_rgb.mp4")
    print(f"      vlc {videos_dir}/track_bowl/track_bowl_rgb.mp4")
    print(f"   3. Play cinematic videos:")
    print(f"      vlc {videos_dir}/overhead_cam/overhead_cam_rgb.mp4")
    print(f"      vlc {videos_dir}/side_cam/side_cam_rgb.mp4")
    print(f"\n✅ PASS: Kitchen Breakfast Scene")
    print(f"   Experiment: {Path(ops.experiment_dir).name}")
    return True


def test_2_apples_next_to_each_other():
    """Test 2: Three Apples Next to Each Other (CENTER OF TABLE)"""
    print("\n" + "="*70)
    print("TEST 2: Three Apples Next to Each Other (CENTER)")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core", save_fps=30)
    print("\n  Building scene with 3 apples...")
    ops.create_scene("apples_scene", width=8, length=8, height=3)
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))
    ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))

    # BEAUTIFUL MOP SYNTAX: First apple in center
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="center")

    # NEXT TO EACH OTHER: Second apple next to first (auto-distance!)
    ops.add_asset(asset_name="banana", relative_to="apple", relation="next_to")

    # NEXT TO EACH OTHER: Third apple next to second (auto-distance!)
    ops.add_asset(asset_name="mug", relative_to="banana", relation="next_to")

    # Rewards
    ops.add_reward(tracked_asset="apple", behavior="stacked_on", target="table", reward=100, id="apple_on_table")
    ops.add_reward(tracked_asset="banana", behavior="stacked_on", target="table", reward=100, id="banana_on_table")
    ops.add_reward(tracked_asset="mug", behavior="stacked_on", target="table", reward=100, id="mug_on_table")

    ops.add_overhead_camera()
    ops.compile()

    print("\n  Running 100 steps...")
    for step in range(100):
        ops.step()

    state = ops.get_state()

    print("\n" + "="*70)
    print("VALIDATION: 3 Objects Next to Each Other")
    print("="*70)

    # Check all on table
    on_table = []
    for obj in ["apple", "banana", "mug"]:
        if state.get(obj, {}).get("stacked_on_table", False):
            on_table.append(obj)
            print(f"   ✅ {obj}: ON TABLE")
        else:
            print(f"   ❌ {obj}: NOT ON TABLE!")

    if len(on_table) == 3:
        print(f"\n✅ PASSED: All 3 objects on table in a row!")
    else:
        print(f"\n❌ FAILED: Only {len(on_table)}/3 objects on table")

    ops.close()
    return len(on_table) == 3


if __name__ == "__main__":
    test_1_kitchen_breakfast_scene()
    # TODO Later: test_2_apples_next_to_each_other()  # 3 objects next to each other in center

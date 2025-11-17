#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USE CASE DEMO: Kitchen Breakfast Scene
=======================================
DEMONSTRATES: Surface positioning with surface_position parameter

USE CASE: Setting up a breakfast table scene
- Clean MOP API: surface_position="center", "top_left", etc.
- Automatic dimension extraction from asset configs
- Object tracking cameras follow each item
- Validation of object placement
"""

import sys
from pathlib import Path

# Add simulation_center to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_blocks_registry import spin


def stuff_on_table():
    """
    USE CASE: Morning breakfast table setup

    DEMONSTRATES:
    - surface_position API (no manual distance/offset needed!)
    - Automatic surface height extraction from furniture configs
    - Multi-camera tracking for cinematic recordings
    """
    print("\n" + "="*70)
    print("USE CASE DEMO: Kitchen Breakfast Scene")
    print("="*70)
    print("\n( BEAUTIFUL MOP: Using surface_position parameter")
    print("   No manual distance or offset calculations needed!")

    # High-quality 30fps HD video recording
    ops = ExperimentOps(mode="simulated", headless=False, render_mode="rl_core", save_fps=30)

    # Create scene
    print("\n  =� Building breakfast scene...")
    ops.create_scene(name="breakfast_scene", width=8, length=8, height=3)
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))
    ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))

    # ( BEAUTIFUL MOP: Surface position behavior (auto-calculates everything!)
    print("\n  <} Placing objects on table using surface_position...")
    ops.add_asset(asset_name="apple", relative_to="table", relation="on_top", surface_position="top_left")
    ops.add_asset(asset_name="banana", relative_to="table", relation="on_top", surface_position="center")
    ops.add_asset(asset_name="mug", relative_to="table", relation="on_top", surface_position="top_right")
    ops.add_asset(asset_name="bowl", relative_to="table", relation="on_top", surface_position="bottom_left")
    ops.add_asset(asset_name="plate", relative_to="table", relation="on_top", surface_position="bottom_right")
    ops.add_asset(asset_name="spoon", relative_to="table", relation="on_top", offset=(0.3, 0.3, 0))
    ops.add_asset(asset_name="orange", relative_to="table", relation="on_top", offset=(-0.3, 0.3, 0))

    # Setup rewards for validation
    ops.add_reward(tracked_asset="apple", behavior="stacked_on", target="table", reward=100, id="apple_on_table")
    ops.add_reward(tracked_asset="banana", behavior="stacked_on", target="table", reward=100, id="banana_on_table")
    ops.add_reward(tracked_asset="mug", behavior="stacked_on", target="table", reward=100, id="mug_on_table")
    ops.add_reward(tracked_asset="bowl", behavior="stacked_on", target="table", reward=100, id="bowl_on_table")
    ops.add_reward(tracked_asset="plate", behavior="stacked_on", target="table", reward=100, id="plate_on_table")
    ops.add_reward(tracked_asset="spoon", behavior="stacked_on", target="table", reward=100, id="spoon_on_table")
    ops.add_reward(tracked_asset="orange", behavior="stacked_on", target="table", reward=100, id="orange_on_table")

    # Add tracking cameras (follow each object)
    print("\n  =� Adding tracking cameras...")
    ops.add_free_camera(camera_id="track_apple", track_target="apple", distance=1.0, azimuth=45, elevation=-20)
    ops.add_free_camera(camera_id="track_banana", track_target="banana", distance=1.0, azimuth=90, elevation=-20)
    ops.add_free_camera(camera_id="track_mug", track_target="mug", distance=1.0, azimuth=135, elevation=-20)
    ops.add_free_camera(camera_id="track_bowl", track_target="bowl", distance=2, azimuth=180, elevation=-15)

    # Add cinematic cameras
    ops.add_overhead_camera()
    ops.add_free_camera("side_cam", lookat=(2.0, 0.0, 0.8), distance=2.5, azimuth=90, elevation=-20)
    ops.add_free_camera("robot_pov_cam", lookat=(2.0, 0.0, 0.8), distance=1.5, azimuth=0, elevation=-10)

    # Compile scene
    print("\n  � Compiling scene...")
    ops.compile()

    # Run robot action
    print("\n  > Robot spinning 90 degrees...")
    block = spin(degrees=90, speed=6.0)
    ops.submit_block(block)

    # Run simulation
    print("\n  �  Running simulation (1000 steps)...")
    for step in range(1000):
        ops.step()

    # Validate object placement
    print("\n   Validating object placement...")
    result = ops.validate_semantics(
        expected_on_table=['apple', 'banana', 'mug', 'bowl'],
        expected_on_floor=[]
    )

    # Save screenshots and close
    print("\n  =� Saving screenshots...")
    ops.save_all_screenshots(frame=10, subdir="all_screenshots")
    ops.close()

    # Validate videos
    print("\n  <� Validating videos...")
    ops.validate_videos(timeout=180)
    video_result = ops.validate_video_files()

    # Print results
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    screenshots_dir = Path(ops.experiment_dir) / "all_screenshots"

    print("\n" + "="*70)
    print(" DEMO COMPLETE!")
    print("="*70)
    print(f"\n=� Output Locations:")
    print(f"   Screenshots: {screenshots_dir}")
    print(f"   Videos:      {videos_dir}")
    print(f"\n<� View Results:")
    print(f"   eog {screenshots_dir}/*.jpg")
    print(f"   vlc {videos_dir}/track_apple/track_apple_rgb.mp4")
    print(f"\n=� KEY TAKEAWAY:")
    print(f"   surface_position parameter makes object placement clean and simple!")
    print(f"   No manual distance/offset calculations needed.")
    print(f"   Assets declare their own surface geometry via configs.")

    return video_result['valid']


if __name__ == "__main__":
    demo_breakfast_scene()

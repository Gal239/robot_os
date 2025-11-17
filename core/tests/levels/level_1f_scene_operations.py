#!/usr/bin/env python3
"""
LEVEL 1F: CINEMATIC SCENE COMPOSITION WITH VIDEO
=================================================

Comprehensive cinematic tests with multi-angle camera views and video recording:
- Real-world scenarios: kitchen, dining, workspace, storage, market, etc.
- Multi-camera cinematography with camera tours
- High-quality 30 FPS video recording (demo mode)
- Multi-layer validation: Physics + Semantics + Vision + Reasoning + Video Quality

Each test creates a 10-second cinematic video showing the scene from multiple angles!

12 Tests Total - Cinematic Scene Composition! 🎬

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1f_scene_operations.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import numpy as np
import cv2
from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_modals import ArmMoveTo, LiftMoveTo, ActionBlock

# Load tolerances from discovered_tolerances.json (PURE MOP - single source of truth)
TOLERANCE_PATH = Path(__file__).parent.parent.parent / "modals" / "stretch" / "discovered_tolerances.json"
with open(TOLERANCE_PATH) as f:
    TOLERANCES = json.load(f)


# ============================================================================
# TEST 1: KITCHEN BREAKFAST SCENE
# ============================================================================

def test_1_kitchen_breakfast_scene():
    """Test 1: Kitchen Breakfast Scene - Morning table setup with 3 camera angles"""
    print("\n" + "="*70)
    print("TEST 1: Kitchen Breakfast Scene (CINEMATIC)")
    print("="*70)

    # High-quality 30fps HD video recording (headless, fast!)
    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="2k_demo",  # 30fps HD cameras!
        save_fps=30
    )

    print("\n  Building breakfast scene...")
    ops.create_scene("breakfast_scene", width=8, length=8, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    # Fixed! Keyframe generation disabled in xml_resolver.py (lines 235-242 commented)
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_left")
    ops.add_asset("banana", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")
    ops.add_asset("mug", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_right")
    ops.add_asset("bowl", relative_to="table", relation="front", distance=0.8)

    # Add 4 cinematic cameras
    print("  Adding 4 camera angles...")
    # Overhead: AUTO-CALCULATED MuJoCo viewer defaults!
    ops.add_overhead_camera()  # Automatically calculates lookat, distance, azimuth, elevation!
    ops.add_free_camera("side_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=2.5, azimuth=90, elevation=-20)
    ops.add_free_camera("robot_pov_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=1.5, azimuth=0, elevation=-10)
    ops.add_free_camera("robot_cam",
                       lookat=(1.0, 0.0, 0.5),  # Between robot and table
                       distance=3.0, azimuth=135, elevation=-20)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Physics: 200Hz (0.005s timestep), Save: 30fps
    # 2000 steps / 200Hz = 10 seconds real time
    # 2000 steps / (200/30) = 300 frames @ 30fps = 10 second video
    for step in range(10000):
        # Camera tour: overhead → side → robot POV
        if step < 667:
            if step == 0:
                print("     📹 Phase 1: Overhead view (0-3.3s)")
        elif step < 1334:
            if step == 667:
                print("     📹 Phase 2: Side view (3.3-6.6s)")
        else:
            if step == 1334:
                print("     📹 Phase 3: Robot POV (6.6-10s)")

        ops.step()

    # ===== LAYER 1: PHYSICS VALIDATION =====
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    required_objects = ["table", "apple", "banana", "mug", "bowl"]
    for obj in required_objects:
        if obj not in state:
            print(f"     ✗ {obj} not in state")
            return False

    table_z = state["table"]["position"][2]
    apple_z = state["apple"]["position"][2]
    banana_z = state["banana"]["position"][2]
    mug_z = state["mug"]["position"][2]

    print(f"     Table Z: {table_z:.3f}m")
    print(f"     Apple Z: {apple_z:.3f}m")
    print(f"     Banana Z: {banana_z:.3f}m")
    print(f"     Mug Z: {mug_z:.3f}m")

    # Objects should be on table
    if not (apple_z > table_z and banana_z > table_z and mug_z > table_z):
        print("     ✗ Objects not on table surface")
        return False

    print("     ✓ All objects on table surface")

    # ===== LAYER 2: SEMANTIC VALIDATION =====
    print("  2. Semantic Validation:")

    # Check surface positioning
    apple_pos = np.array(state["apple"]["position"])
    banana_pos = np.array(state["banana"]["position"])
    mug_pos = np.array(state["mug"]["position"])
    table_pos = np.array(state["table"]["position"])

    # Apple should be left, banana center, mug right
    apple_x_offset = apple_pos[0] - table_pos[0]
    mug_x_offset = mug_pos[0] - table_pos[0]

    print(f"     Apple X offset: {apple_x_offset:.3f}m (should be < 0, left)")
    print(f"     Mug X offset: {mug_x_offset:.3f}m (should be > 0, right)")

    if not (apple_x_offset < 0 and mug_x_offset > 0):
        print("     ✗ Surface positioning incorrect")
        return False

    print("     ✓ Surface positioning correct (apple left, mug right)")

    # ===== LAYER 3: VISION + VIDEO VALIDATION =====
    print("  3. Vision + Video Validation:")

    # Check video exists (in timeline/cameras/{camera_name}/)
    camera_names = ["overhead_cam", "side_cam", "robot_pov_cam"]
    videos_found = 0

    for cam_name in camera_names:
        # Check for MP4 (converted) or AVI (raw, still converting)
        video_path_mp4 = Path(ops.experiment_dir) / "timeline" / "cameras" / cam_name / f"{cam_name}_rgb.mp4"
        video_path_avi = Path(ops.experiment_dir) / "timeline" / "cameras" / cam_name / f"{cam_name}_rgb.avi"

        if video_path_mp4.exists():
            videos_found += 1
            print(f"     ✓ {cam_name}: {video_path_mp4.name} ({video_path_mp4.stat().st_size // 1024}KB)")
        elif video_path_avi.exists():
            videos_found += 1
            print(f"     ✓ {cam_name}: {video_path_avi.name} ({video_path_avi.stat().st_size // 1024}KB, converting...)")
        else:
            print(f"     ✗ {cam_name}: No video found")

    if videos_found < len(camera_names):
        print(f"     ✗ Only {videos_found}/{len(camera_names)} camera videos found")
        return False

    print(f"     ✓ All {len(camera_names)} camera videos saved")
    print(f"     📹 Videos dir: {Path(ops.experiment_dir) / 'timeline' / 'cameras'}")

    # ===== LAYER 4: REASONING VALIDATION =====
    print("  4. Reasoning Validation:")

    # Bowl should be in front of table
    bowl_pos = np.array(state["bowl"]["position"])
    bowl_y_offset = bowl_pos[1] - table_pos[1]

    print(f"     Bowl Y offset: {bowl_y_offset:.3f}m (should be > 0, front)")

    if bowl_y_offset <= 0:
        print("     ✗ Bowl not in front of table")
        return False

    print("     ✓ Bowl positioned in front (breakfast layout correct)")

    # ===== LAYER 5: VIDEO CONVERSION VALIDATION (MOP!) =====
    print("  5. Video Conversion Validation:")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Validate video files exist using VideoOps
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["overhead_cam", "side_cam", "robot_pov_cam", "robot_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")

    print(f"\n  ✅ PASS: Kitchen Breakfast Scene")
    return True


# ============================================================================
# TEST 2: COFFEE STATION
# ============================================================================

def test_2_coffee_station():
    """Test 2: Coffee Station - Coffee corner with 2 camera angles"""
    print("\n" + "="*70)
    print("TEST 2: Coffee Station (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building coffee station...")
    ops.create_scene("coffee_station", width=8, length=8, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("mug", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")
    ops.add_asset("bowl", relative_to="mug", relation="next_to", distance=0.15)
    ops.add_asset("banana", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_left")

    # Add 2 cameras
    print("  Adding 2 camera angles...")
    ops.add_free_camera("wide_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=2.0, azimuth=45, elevation=-15)
    ops.add_free_camera("close_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=1.0, azimuth=-30, elevation=-25)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: wide → close
    for step in range(2000):
        if step < 1000:
            if step == 0:
                print("     📹 Phase 1: Wide shot (0-5s)")
        else:
            if step == 1000:
                print("     📹 Phase 2: Close-up (5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    if "mug" not in state or "bowl" not in state:
        print("     ✗ Objects missing")
        return False

    mug_pos = np.array(state["mug"]["position"])
    bowl_pos = np.array(state["bowl"]["position"])

    distance = np.linalg.norm(mug_pos[:2] - bowl_pos[:2])
    print(f"     Mug-Bowl distance: {distance:.3f}m (expected ~0.15m)")

    if abs(distance - 0.15) > 0.1:
        print("     ✗ Distance incorrect")
        return False

    print("     ✓ Objects positioned correctly")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["wide_cam", "close_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Coffee Station")
    return True


# ============================================================================
# TEST 3: STORAGE ORGANIZATION
# ============================================================================

def test_3_storage_organization():
    """Test 3: Storage Organization - Bins with contained items, 3 cameras"""
    print("\n" + "="*70)
    print("TEST 3: Storage Organization (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building storage scene...")
    ops.create_scene("storage_scene", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("storage_bin", relative_to=(2.0, -0.5, 0.0))
    ops.add_asset("bowl", relative_to=(2.0, 0.5, 0.0))
    ops.add_asset("apple", relative_to="storage_bin", relation="inside", distance=0)
    ops.add_asset("banana", relative_to="bowl", relation="inside", distance=0)

    # Add 3 cameras
    print("  Adding 3 camera angles...")
    ops.add_free_camera("org_cam",
                       lookat=(2.0, 0.0, 0.3),
                       distance=3.0, azimuth=60, elevation=-30)
    ops.add_free_camera("bin_cam",
                       lookat=(2.0, -0.5, 0.3),
                       distance=1.2, azimuth=45, elevation=-35)
    ops.add_free_camera("bowl_cam",
                       lookat=(2.0, 0.5, 0.3),
                       distance=1.2, azimuth=-45, elevation=-35)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: org → bin → bowl
    for step in range(2000):
        if step < 667:
            if step == 0:
                print("     📹 Phase 1: Organization view (0-3.3s)")
        elif step < 1334:
            if step == 667:
                print("     📹 Phase 2: Bin focus (3.3-6.6s)")
        else:
            if step == 1334:
                print("     📹 Phase 3: Bowl focus (6.6-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    bin_pos = np.array(state["storage_bin"]["position"])
    apple_pos = np.array(state["apple"]["position"])
    bowl_pos = np.array(state["bowl"]["position"])
    banana_pos = np.array(state["banana"]["position"])

    bin_distance = np.linalg.norm(apple_pos - bin_pos)
    bowl_distance = np.linalg.norm(banana_pos - bowl_pos)

    print(f"     Apple-Bin distance: {bin_distance:.3f}m")
    print(f"     Banana-Bowl distance: {bowl_distance:.3f}m")

    if bin_distance >= 0.5 or bowl_distance >= 0.5:
        print("     ✗ Objects not inside containers")
        return False

    print("     ✓ Container behavior validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["org_cam", "bin_cam", "bowl_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Storage Organization")
    return True


# ============================================================================
# TEST 4: DINING TABLE SETUP
# ============================================================================

def test_4_dining_table_setup():
    """Test 4: Dining Table Setup - Formal dining with 4 camera angles"""
    print("\n" + "="*70)
    print("TEST 4: Dining Table Setup (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building dining scene...")
    ops.create_scene("dining_scene", width=8, length=8, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(3.0, 0.0, 0.0))
    ops.add_asset("bowl", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="back")

    # Add 4 cameras for dining perspectives
    print("  Adding 4 camera angles...")
    ops.add_free_camera("diner_left_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=2.0, azimuth=-45, elevation=-15)
    ops.add_free_camera("diner_right_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=2.0, azimuth=45, elevation=-15)
    ops.add_free_camera("host_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=2.5, azimuth=0, elevation=-12)
    ops.add_free_camera("aerial_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=3.5, azimuth=0, elevation=-70)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: left → right → host → aerial
    for step in range(2000):
        if step < 500:
            if step == 0:
                print("     📹 Phase 1: Left diner (0-2.5s)")
        elif step < 1000:
            if step == 500:
                print("     📹 Phase 2: Right diner (2.5-5s)")
        elif step < 1500:
            if step == 1000:
                print("     📹 Phase 3: Host view (5-7.5s)")
        else:
            if step == 1500:
                print("     📹 Phase 4: Aerial view (7.5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    table_z = state["table"]["position"][2]
    bowl_z = state["bowl"]["position"][2]
    apple_z = state["apple"]["position"][2]

    if not (bowl_z > table_z and apple_z > table_z):
        print("     ✗ Objects not on table")
        return False

    print("     ✓ Dining setup validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["diner_left_cam", "diner_right_cam", "host_cam", "aerial_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Dining Table Setup")
    return True


# ============================================================================
# TEST 5: WORKSTATION SETUP
# ============================================================================

def test_5_workstation_setup():
    """Test 5: Workstation Setup - Desk with accessories, 2 cameras"""
    print("\n" + "="*70)
    print("TEST 5: Workstation Setup (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building workstation...")
    ops.create_scene("workstation_scene", width=7, length=7, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.5, 0.0, 0.0))
    ops.add_asset("storage_bin", relative_to="table", relation="left", distance=1.0)
    ops.add_asset("mug", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_right")
    ops.add_asset("bowl", relative_to="table", relation="back", distance=0.5)

    # Add 2 cameras
    print("  Adding 2 camera angles...")
    ops.add_free_camera("user_cam",
                       lookat=(2.5, 0.0, 0.8),
                       distance=1.5, azimuth=0, elevation=-5)
    ops.add_free_camera("productivity_cam",
                       lookat=(2.5, 0.0, 0.8),
                       distance=2.5, azimuth=135, elevation=-20)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: user → productivity
    for step in range(2000):
        if step < 1000:
            if step == 0:
                print("     📹 Phase 1: User view (0-5s)")
        else:
            if step == 1000:
                print("     📹 Phase 2: Productivity angle (5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    table_pos = np.array(state["table"]["position"])
    bin_pos = np.array(state["storage_bin"]["position"])

    x_offset = table_pos[0] - bin_pos[0]
    print(f"     Storage bin X offset: {x_offset:.3f}m (should be ~1.0m, left)")

    if abs(x_offset - 1.0) > 0.3:
        print("     ✗ Bin not positioned correctly")
        return False

    print("     ✓ Workstation layout validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["user_cam", "productivity_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Workstation Setup")
    return True


# ============================================================================
# TEST 6: SNACK SHELF
# ============================================================================

def test_6_snack_shelf():
    """Test 6: Snack Shelf - Depth ordering with 2 cameras"""
    print("\n" + "="*70)
    print("TEST 6: Snack Shelf (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building snack shelf...")
    ops.create_scene("snack_shelf_scene", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="front")
    ops.add_asset("banana", relative_to="apple", relation="back", distance=0.5)
    ops.add_asset("bowl", relative_to="banana", relation="back", distance=0.4)

    # Add 2 cameras
    print("  Adding 2 camera angles...")
    ops.add_free_camera("shelf_front_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=2.0, azimuth=0, elevation=-10)
    ops.add_free_camera("shelf_angle_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=2.2, azimuth=30, elevation=-15)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: front → angle
    for step in range(2000):
        if step < 1000:
            if step == 0:
                print("     📹 Phase 1: Front view (0-5s)")
        else:
            if step == 1000:
                print("     📹 Phase 2: Angled view (5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    apple_pos = np.array(state["apple"]["position"])
    banana_pos = np.array(state["banana"]["position"])
    bowl_pos = np.array(state["bowl"]["position"])

    # Check depth ordering (Y axis)
    y_order = [apple_pos[1], banana_pos[1], bowl_pos[1]]
    print(f"     Y positions: apple={apple_pos[1]:.3f}, banana={banana_pos[1]:.3f}, bowl={bowl_pos[1]:.3f}")

    # Apple should be closest, banana middle, bowl farthest (decreasing Y from robot at origin)
    if not (apple_pos[1] > banana_pos[1] > bowl_pos[1]):
        print("     ✗ Depth ordering incorrect")
        return False

    print("     ✓ Depth ordering validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["shelf_front_cam", "shelf_angle_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Snack Shelf")
    return True


# ============================================================================
# TEST 7: KITCHEN COUNTER SCENE
# ============================================================================

def test_7_kitchen_counter_scene():
    """Test 7: Kitchen Counter Scene - Full counter with cutting board, 3 cameras"""
    print("\n" + "="*70)
    print("TEST 7: Kitchen Counter Scene (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building kitchen counter...")
    ops.create_scene("kitchen_counter_scene", width=8, length=8, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.5, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")
    ops.add_asset("banana", relative_to="apple", relation="next_to", distance=0.15)
    ops.add_asset("mug", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_right")
    ops.add_asset("bowl", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_left")

    # Add 3 cameras
    print("  Adding 3 camera angles...")
    ops.add_free_camera("prep_cam",
                       lookat=(2.5, 0.0, 0.8),
                       distance=1.0, azimuth=0, elevation=-15)
    ops.add_free_camera("counter_wide_cam",
                       lookat=(2.5, 0.0, 0.8),
                       distance=2.5, azimuth=45, elevation=-20)
    ops.add_free_camera("ingredient_cam",
                       lookat=(2.5, 0.0, 0.8),
                       distance=0.8, azimuth=-30, elevation=-25)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: prep → wide → ingredient
    for step in range(2000):
        if step < 667:
            if step == 0:
                print("     📹 Phase 1: Prep view (0-3.3s)")
        elif step < 1334:
            if step == 667:
                print("     📹 Phase 2: Wide counter (3.3-6.6s)")
        else:
            if step == 1334:
                print("     📹 Phase 3: Ingredient close-up (6.6-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    table_z = state["table"]["position"][2]
    apple_z = state["apple"]["position"][2]

    if apple_z <= table_z:
        print("     ✗ Apple not on counter")
        return False

    print("     ✓ Kitchen counter validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["prep_cam", "counter_wide_cam", "ingredient_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Kitchen Counter Scene")
    return True


# ============================================================================
# TEST 8: LIVING ROOM SIDE TABLE
# ============================================================================

def test_8_living_room_side_table():
    """Test 8: Living Room Side Table - Cozy setup with 2 cameras"""
    print("\n" + "="*70)
    print("TEST 8: Living Room Side Table (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building side table scene...")
    ops.create_scene("side_table_scene", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(1.5, 0.0, 0.0))
    ops.add_asset("mug", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="front_right")
    ops.add_asset("bowl", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="back")
    ops.add_asset("banana", relative_to="bowl", relation="inside", distance=0)

    # Add 2 cameras
    print("  Adding 2 camera angles...")
    ops.add_free_camera("couch_cam",
                       lookat=(1.5, 0.0, 0.8),
                       distance=1.2, azimuth=-20, elevation=-5)
    ops.add_free_camera("standing_cam",
                       lookat=(1.5, 0.0, 0.8),
                       distance=1.8, azimuth=45, elevation=-25)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: couch → standing
    for step in range(2000):
        if step < 1000:
            if step == 0:
                print("     📹 Phase 1: Couch view (0-5s)")
        else:
            if step == 1000:
                print("     📹 Phase 2: Standing view (5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    bowl_pos = np.array(state["bowl"]["position"])
    banana_pos = np.array(state["banana"]["position"])

    distance = np.linalg.norm(bowl_pos - banana_pos)
    print(f"     Bowl-Banana distance: {distance:.3f}m")

    if distance >= 0.5:
        print("     ✗ Banana not in bowl")
        return False

    print("     ✓ Side table validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["couch_cam", "standing_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Living Room Side Table")
    return True


# ============================================================================
# TEST 9: PICNIC BASKET SCENE
# ============================================================================

def test_9_picnic_basket_scene():
    """Test 9: Picnic Basket Scene - Outdoor basket with 3 cameras"""
    print("\n" + "="*70)
    print("TEST 9: Picnic Basket Scene (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building picnic scene...")
    ops.create_scene("picnic_scene", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("storage_bin", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="storage_bin", relation="inside", distance=0)
    ops.add_asset("banana", relative_to="storage_bin", relation="inside", distance=0)
    ops.add_asset("bowl", relative_to="storage_bin", relation="next_to", distance=0.3)

    # Add 3 cameras
    print("  Adding 3 camera angles...")
    ops.add_free_camera("picnic_wide_cam",
                       lookat=(2.0, 0.0, 0.3),
                       distance=2.0, azimuth=45, elevation=-20)
    ops.add_free_camera("basket_cam",
                       lookat=(2.0, 0.0, 0.3),
                       distance=1.0, azimuth=0, elevation=-30)
    ops.add_free_camera("ground_cam",
                       lookat=(2.0, 0.0, 0.2),
                       distance=1.5, azimuth=90, elevation=-5)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: wide → basket → ground
    for step in range(2000):
        if step < 667:
            if step == 0:
                print("     📹 Phase 1: Wide picnic view (0-3.3s)")
        elif step < 1334:
            if step == 667:
                print("     📹 Phase 2: Basket focus (3.3-6.6s)")
        else:
            if step == 1334:
                print("     📹 Phase 3: Ground level (6.6-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    bin_pos = np.array(state["storage_bin"]["position"])
    apple_pos = np.array(state["apple"]["position"])

    distance = np.linalg.norm(bin_pos - apple_pos)
    print(f"     Apple-Basket distance: {distance:.3f}m")

    if distance >= 0.5:
        print("     ✗ Apple not in basket")
        return False

    # Basket should be on ground
    if bin_pos[2] >= 0.15:
        print("     ✗ Basket not on ground")
        return False

    print("     ✓ Picnic scene validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["picnic_wide_cam", "basket_cam", "ground_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Picnic Basket Scene")
    return True


# ============================================================================
# TEST 10: THREE-TIER DISPLAY
# ============================================================================

def test_10_three_tier_display():
    """Test 10: Three-Tier Display - Multi-height stacking with 3 cameras"""
    print("\n" + "="*70)
    print("TEST 10: Three-Tier Display (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building three-tier display...")
    ops.create_scene("three_tier_scene", width=7, length=7, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.5, 0.5, 0.0))
    ops.add_asset("storage_bin", relative_to="table", relation="on_top", distance=0.75)
    ops.add_asset("apple", relative_to="storage_bin", relation="on_top", distance=0.15)

    # Add 3 cameras
    print("  Adding 3 camera angles...")
    ops.add_free_camera("display_front_cam",
                       lookat=(2.5, 0.5, 0.8),
                       distance=2.5, azimuth=0, elevation=-15)
    ops.add_free_camera("display_side_cam",
                       lookat=(2.5, 0.5, 0.8),
                       distance=2.5, azimuth=90, elevation=-15)
    ops.add_free_camera("display_detail_cam",
                       lookat=(2.5, 0.5, 1.0),
                       distance=1.0, azimuth=45, elevation=-20)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: front → side → detail
    for step in range(2000):
        if step < 667:
            if step == 0:
                print("     📹 Phase 1: Front view (0-3.3s)")
        elif step < 1334:
            if step == 667:
                print("     📹 Phase 2: Side profile (3.3-6.6s)")
        else:
            if step == 1334:
                print("     📹 Phase 3: Detail close-up (6.6-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    table_z = state["table"]["position"][2]
    bin_z = state["storage_bin"]["position"][2]
    apple_z = state["apple"]["position"][2]

    print(f"     Heights: table={table_z:.3f}, bin={bin_z:.3f}, apple={apple_z:.3f}")

    # Check stacking order
    if not (table_z < bin_z < apple_z):
        print("     ✗ Stacking order incorrect")
        return False

    print("     ✓ Three-tier stacking validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["display_front_cam", "display_side_cam", "display_detail_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Three-Tier Display")
    return True


# ============================================================================
# TEST 11: MARKET STALL
# ============================================================================

def test_11_market_stall():
    """Test 11: Market Stall - Produce stand with 4 cameras"""
    print("\n" + "="*70)
    print("TEST 11: Market Stall (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building market stall...")
    ops.create_scene("market_stall_scene", width=8, length=8, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(3.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_left")
    ops.add_asset("banana", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="top_right")
    ops.add_asset("bowl", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")
    ops.add_asset("storage_bin", relative_to="table", relation="back", distance=0.6)

    # Add 4 cameras
    print("  Adding 4 camera angles...")
    ops.add_free_camera("customer_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=2.0, azimuth=0, elevation=-10)
    ops.add_free_camera("vendor_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=2.0, azimuth=180, elevation=-10)
    ops.add_free_camera("overhead_market_cam",
                       lookat=(3.0, 0.0, 0.8),
                       distance=3.0, azimuth=0, elevation=-75)
    ops.add_free_camera("product_cam",
                       lookat=(3.0, 0.0, 0.85),
                       distance=0.8, azimuth=45, elevation=-20)

    ops.compile()

    print("  🎬 Recording cinematic video (10 seconds @ 30fps)...")

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour: customer → vendor → overhead → product
    for step in range(2000):
        if step < 500:
            if step == 0:
                print("     📹 Phase 1: Customer view (0-2.5s)")
        elif step < 1000:
            if step == 500:
                print("     📹 Phase 2: Vendor perspective (2.5-5s)")
        elif step < 1500:
            if step == 1000:
                print("     📹 Phase 3: Overhead layout (5-7.5s)")
        else:
            if step == 1500:
                print("     📹 Phase 4: Product close-up (7.5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    apple_pos = np.array(state["apple"]["position"])
    banana_pos = np.array(state["banana"]["position"])
    table_pos = np.array(state["table"]["position"])

    # Check left/right positioning
    apple_x = apple_pos[0] - table_pos[0]
    banana_x = banana_pos[0] - table_pos[0]

    print(f"     Apple X offset: {apple_x:.3f}m")
    print(f"     Banana X offset: {banana_x:.3f}m")

    if not (apple_x < 0 and banana_x > 0):
        print("     ✗ Left/right positioning incorrect")
        return False

    print("     ✓ Market stall layout validated")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["customer_cam", "vendor_cam", "overhead_market_cam", "product_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Market Stall")
    return True


# ============================================================================
# TEST 12: ROBOT INTERACTION SCENE
# ============================================================================

def test_12_robot_interaction_scene():
    """Test 12: Robot Interaction Scene - Robot reaches for apple with 4 cameras"""
    print("\n" + "="*70)
    print("TEST 12: Robot Interaction Scene (CINEMATIC)")
    print("="*70)

    ops = ExperimentOps(
        mode="simulated",
        headless=True,
        render_mode="demo",  # 30fps HD!
        save_fps=30
    )

    print("\n  Building robot interaction scene...")
    ops.create_scene("robot_interaction_scene", width=7, length=7, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top",
                  distance=0.75, surface_position="center")

    # Add 4 cameras for action coverage
    print("  Adding 4 camera angles...")
    ops.add_free_camera("third_person_cam",
                       lookat=(1.0, 0.0, 0.5),
                       distance=3.0, azimuth=180, elevation=-15)
    ops.add_free_camera("side_action_cam",
                       lookat=(1.0, 0.0, 0.5),
                       distance=2.5, azimuth=90, elevation=-10)
    ops.add_free_camera("target_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=0.5, azimuth=180, elevation=-5)
    ops.add_free_camera("overhead_action_cam",
                       lookat=(1.0, 0.0, 0.3),
                       distance=3.5, azimuth=0, elevation=-70)

    ops.compile()

    print("  🎬 Recording cinematic video with robot action (10 seconds @ 30fps)...")

    # Submit robot action: reach toward apple
    print("  🤖 Robot action: Extending arm and lifting...")
    arm_action = ArmMoveTo(position=0.4)
    lift_action = LiftMoveTo(height=0.6)
    action_block = ActionBlock(
        id="reach_apple",
        description="Reach toward apple",
        actions=[arm_action, lift_action]
    )
    ops.submit_block(action_block)

    # CINEMATIC VIDEO RECORDING (2000 steps @ 200Hz = 10 seconds, 30fps save)
    # Camera tour with action: third_person → side → target → overhead
    for step in range(2000):
        if step < 500:
            if step == 0:
                print("     📹 Phase 1: Third person (0-2.5s)")
        elif step < 1000:
            if step == 500:
                print("     📹 Phase 2: Side action view (2.5-5s)")
        elif step < 1500:
            if step == 1000:
                print("     📹 Phase 3: Target view (5-7.5s)")
        else:
            if step == 1500:
                print("     📹 Phase 4: Overhead action (7.5-10s)")

        ops.step()

    # Validation
    print("\n  1. Physics Validation:")
    state = ops.get_state()

    arm_extension = state["stretch.arm"]["extension"]
    lift_height = state["stretch.lift"]["height"]

    print(f"     Arm extension: {arm_extension:.3f}m (target: 0.4m)")
    print(f"     Lift height: {lift_height:.3f}m (target: 0.6m)")

    arm_ok = abs(arm_extension - 0.4) < 0.05
    lift_ok = abs(lift_height - 0.6) < 0.08

    if not (arm_ok and lift_ok):
        print("     ✗ Robot action incomplete")
        return False

    print("     ✓ Robot action validated (reaching pose achieved)")

    # Close and trigger async video conversion
    ops.close()

    # MOP: VALIDATE videos converted - crashes if any failed!
    ops.validate_videos(timeout=180)

    # Video validation
    print("  2. Video Validation:")
    from core.video import VideoOps
    videos_dir = Path(ops.experiment_dir) / "timeline" / "cameras"
    camera_names = ["third_person_cam", "side_action_cam", "target_cam", "overhead_action_cam"]

    for cam_name in camera_names:
        mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
        thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

        # Validate MP4
        info = VideoOps.validate_video_file(mp4_path)
        if not info['valid']:
            print(f"     ✗ {cam_name} MP4 invalid: {info['error']}")
            return False

        # Validate thumbnail
        if not thumb_path.exists():
            print(f"     ✗ {cam_name} thumbnail missing!")
            return False

    print(f"     ✓ All {len(camera_names)} videos and thumbnails validated")
    print(f"     📹 Videos dir: {videos_dir}")

    print(f"\n  ✅ PASS: Robot Interaction Scene")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all cinematic scene composition tests"""
    print("\n" + "="*70)
    print("LEVEL 1F: CINEMATIC SCENE COMPOSITION WITH VIDEO")
    print("="*70)
    print("\n🎬 Each test creates a 10-second cinematic video @ 30 FPS!\n")

    tests = [
        ("Test 1: Kitchen Breakfast Scene", test_1_kitchen_breakfast_scene),
        ("Test 2: Coffee Station", test_2_coffee_station),
        ("Test 3: Storage Organization", test_3_storage_organization),
        ("Test 4: Dining Table Setup", test_4_dining_table_setup),
        ("Test 5: Workstation Setup", test_5_workstation_setup),
        ("Test 6: Snack Shelf", test_6_snack_shelf),
        ("Test 7: Kitchen Counter Scene", test_7_kitchen_counter_scene),
        ("Test 8: Living Room Side Table", test_8_living_room_side_table),
        ("Test 9: Picnic Basket Scene", test_9_picnic_basket_scene),
        ("Test 10: Three-Tier Display", test_10_three_tier_display),
        ("Test 11: Market Stall", test_11_market_stall),
        ("Test 12: Robot Interaction Scene", test_12_robot_interaction_scene),
    ]

    results = []

    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total = len(results)
    print(f"\n  Total: {passed_count}/{total} tests passed")

    if passed_count == total:
        print("\n  🎉 ALL TESTS PASSED!")
        print(f"  🎬 {total} cinematic videos created (~2 minutes of footage)!")
    elif passed_count >= total * 0.8:
        print(f"\n  ⚠️  {total - passed_count} test(s) failed")
    else:
        print(f"\n  ❌ {total - passed_count} test(s) failed")


if __name__ == "__main__":
    main()

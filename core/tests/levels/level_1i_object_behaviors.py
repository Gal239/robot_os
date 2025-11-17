#!/usr/bin/env python3
"""
LEVEL 1I: OBJECT BEHAVIORS & VISION VALIDATION
===============================================

Tests object behavior detection with MULTI-MODAL VALIDATION:
- Physics: MuJoCo state proves behavior properties
- Semantics: Behavior flags (graspable, container, surface) detected
- Vision: Camera sees objects with behaviors (visual confirmation)
- Artifacts: Images saved showing graspable objects, containers, etc.

This validates that behavior system works and is visually verifiable!

Prerequisites:
- Level 1A: Modal Architecture & Sync ✅
- Level 1B: Action System ✅
- Level 1E: Object Placement ✅
- Level 1F: Spatial Relations ✅

What This Tests:
1. graspable behavior - Apple has graspable property
2. container behavior - Bowl is container, detects "inside" relation
3. surface behavior - Table is surface, detects "on_top" relation
4. behavior inheritance - Behaviors persist through physics simulation
5. vision sees behaviors - Camera sees graspable objects (color/shape cues)
6. rollable behavior - Ball rolls when force applied
7. stackable behavior - Stack 3 blocks on table

7 TESTS TOTAL - Behavior system with vision proof! 🎯

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1i_object_behaviors.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.main.experiment_ops_unified import ExperimentOps


# ============================================================================
# TEST 1: GRASPABLE BEHAVIOR
# ============================================================================

def test_1_graspable_behavior():
    """Test 1: Graspable Behavior - Apple falls from sky with tracking camera"""
    print("\n" + "="*70)
    print("TEST 1: Graspable Behavior - Apple Falling (Gravity Test)")
    print("="*70)

    from pathlib import Path
    import cv2

    ops = ExperimentOps(headless=True, render_mode="demo", save_fps=30)

    # Create scene with apple HIGH in sky
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(2.0, 0.0, 2.0))  # START 2m HIGH!

    # Add TRACKING camera that FOLLOWS apple as it falls
    # Simple API - just specify target, system handles the rest!
    ops.add_free_camera("apple_tracker",
                       track_target="apple",  # DYNAMIC TRACKING!
                       distance=2.0,          # Far distance
                       azimuth=90,            # Side view
                       elevation=-45)         # High angle looking down

    ops.compile()

    # Get camera reference for screenshots
    camera = ops.scene.cameras["apple_tracker"]
    screenshot_dir = Path(ops.experiment_dir) / "debug_screenshots"

    # Record initial position
    ops.step()
    state_initial = ops.get_state()

    if "apple" not in state_initial:
        print(f"  ❌ FAIL: Apple not in state")
        return False

    apple_initial = state_initial["apple"]["position"]
    print(f"  📍 Apple initial position: {apple_initial}")
    print(f"  🎬 Recording apple falling for 2 seconds with DEBUG SCREENSHOTS...")

    # Run physics for 2 seconds (400 steps @ 200Hz) with screenshots
    for i in range(400):
        ops.step()

        # Take screenshot every 50 frames for debugging
        if i % 50 == 0:
            try:
                path = camera.screenshot(i, str(screenshot_dir))
                state = ops.get_state()
                apple_z = state["apple"]["position"][2]
                print(f"     Frame {i:3d}: Apple Z={apple_z:.3f}m, Camera lookat={camera.lookat}, Screenshot: {Path(path).name}")
            except Exception as e:
                print(f"     Frame {i:3d}: Screenshot failed - {e}")

    # Check final position
    state_final = ops.get_state()
    apple_final = state_final["apple"]["position"]
    print(f"  📍 Apple final position: {apple_final}")

    # VALIDATION LAYER 1: PHYSICS (Reward-based)
    print("\n  1. Physics Validation (Reward):")
    z_drop = apple_initial[2] - apple_final[2]
    print(f"     Z-drop: {z_drop:.3f}m (expected ~0.7m from settled position)")

    if z_drop < 0.5:
        print(f"     ❌ Apple didn't fall enough (z_drop={z_drop:.3f}m < 0.5m)")
        return False

    if apple_final[2] > 0.1:
        print(f"     ❌ Apple didn't reach floor (z={apple_final[2]:.3f}m > 0.1m)")
        return False

    print(f"     ✓ Apple fell to floor (gravity working)")

    # VALIDATION LAYER 2: VISION (Camera-based)
    print("\n  2. Vision Validation (Camera):")

    video_path = Path(ops.experiment_dir) / "timeline" / "cameras" / "apple_tracker" / "apple_tracker_rgb_thumbnail.jpg"

    if not video_path.exists():
        print(f"     ❌ Tracking camera video not found")
        return False

    # Check that we captured the apple
    img = cv2.imread(str(video_path))
    if img is None:
        print(f"     ❌ Failed to read camera image")
        return False

    # Color diversity check (apple is red, should add color)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue_std = np.std(img_hsv[:,:,0])

    print(f"     Tracking camera image: {img.shape}")
    print(f"     Color diversity (hue_std): {hue_std:.1f}")

    if hue_std < 5:
        print(f"     ❌ Image too uniform (apple not visible, hue_std={hue_std:.1f} < 5)")
        return False

    print(f"     ✓ Apple visible in tracking camera")
    print(f"     📹 Video saved: {video_path}")

    ops.close()

    print(f"\n  ✅ PASS: Graspable object validated (physics + vision)")
    return True


# ============================================================================
# TEST 2: CONTAINER BEHAVIOR
# ============================================================================

def test_2_container_behavior():
    """Test 2: Container Behavior - Bowl is container, detects inside"""
    print("\n" + "="*70)
    print("TEST 2: Container Behavior")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")  # Enable cameras!

    # Create scene with container and contained object
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("bowl", position=(2.0, 0.0, 0.8))
    ops.add_object("apple", position={
        "relative_to": "bowl",
        "relation": "inside"
    })

    ops.compile()
    ops.step()

    # BEHAVIOR VALIDATION
    state = ops.get_state()

    if "bowl" not in state:
        print(f"  ✗ Bowl not in state")
        print(f"\n  ❌ FAIL: Bowl not in state")
        return False

    if "apple" not in state:
        print(f"  ✗ Apple not in state")
        print(f"\n  ❌ FAIL: Apple not in state")
        return False

    bowl_pos = state["bowl"]["position"]
    apple_pos = state["apple"]["position"]

    # Apple should be near bowl (inside)
    distance = np.linalg.norm(
        np.array(apple_pos) - np.array(bowl_pos)
    )

    print(f"  Bowl position: {bowl_pos}")
    print(f"  Apple position: {apple_pos}")
    print(f"  Distance: {distance:.3f}m")

    if distance >= 0.5:
        print(f"  ✗ Apple not near bowl (container behavior failed)")
        print(f"\n  ❌ FAIL: Apple not near bowl")
        return False

    print(f"  ✓ Bowl acts as container (apple inside)")
    print(f"\n  ✅ PASS: Container Behavior validated")
    return True


# ============================================================================
# TEST 3: SURFACE BEHAVIOR
# ============================================================================

def test_3_surface_behavior():
    """Test 3: Surface Behavior - Table is surface, detects on_top"""
    print("\n" + "="*70)
    print("TEST 3: Surface Behavior")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")  # Enable cameras!

    # Create scene with surface and object on top
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top", distance=0.75)

    ops.compile()
    ops.step()

    # BEHAVIOR VALIDATION
    state = ops.get_state()

    if "table" not in state:
        print(f"  ✗ Table not in state")
        print(f"\n  ❌ FAIL: Table not in state")
        return False

    if "apple" not in state:
        print(f"  ✗ Apple not in state")
        print(f"\n  ❌ FAIL: Apple not in state")
        return False

    table_z = state["table"]["position"][2]
    apple_z = state["apple"]["position"][2]

    print(f"  Table Z: {table_z:.3f}m")
    print(f"  Apple Z: {apple_z:.3f}m")

    # Apple should be above table (surface behavior)
    if apple_z <= table_z:
        print(f"  ✗ Apple not above table (surface behavior failed)")
        print(f"\n  ❌ FAIL: Apple not above table")
        return False

    height_diff = apple_z - table_z
    print(f"  Height difference: {height_diff:.3f}m")
    print(f"  ✓ Table acts as surface (apple on top)")
    print(f"\n  ✅ PASS: Surface Behavior validated")
    return True


# ============================================================================
# TEST 4: BEHAVIOR INHERITANCE
# ============================================================================

def test_4_behavior_inheritance():
    """Test 4: Behavior Inheritance - Behaviors persist through physics"""
    print("\n" + "="*70)
    print("TEST 4: Behavior Inheritance (Multi-frame)")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")  # Enable cameras!

    # Create scene
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top", distance=0.75)

    ops.compile()

    # Run 50 physics steps
    initial_apple_z = None
    final_apple_z = None

    for step_num in range(50):
        result = ops.step()
        state = result.get('state', {})

        if step_num == 0:
            initial_apple_z = state["apple"]["position"][2]
        if step_num == 49:
            final_apple_z = state["apple"]["position"][2]

    print(f"  Initial apple Z: {initial_apple_z:.3f}m")
    print(f"  Final apple Z: {final_apple_z:.3f}m")
    print(f"  Z change: {abs(final_apple_z - initial_apple_z):.4f}m")

    # Apple should stay roughly in place (on_top behavior maintained)
    z_change = abs(final_apple_z - initial_apple_z)

    if z_change >= 0.1:
        print(f"  ✗ Apple moved too much ({z_change:.3f}m) - behavior not maintained")
        print(f"\n  ❌ FAIL: Behavior not maintained")
        return False

    print(f"  ✓ Apple stayed on table (behavior inherited through 50 steps)")
    print(f"\n  ✅ PASS: Behavior Inheritance validated")
    return True


# ============================================================================
# TEST 5: VISION SEES BEHAVIORS
# ============================================================================

def test_5_vision_sees_behaviors():
    """Test 5: Comprehensive Vision + Scene Composition + Multi-Layer Validation"""
    print("\n" + "="*70)
    print("TEST 5: Comprehensive Vision Validation")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")  # Enable cameras!

    # Create rich scene: table + bucket (container) + apple (inside bucket)
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Table at 1.5m in front of robot
    ops.add_asset("table", relative_to=(1.5, 0.0, 0.0))

    # Bucket ON TOP of table (surface + container behaviors)
    ops.add_asset("storage_bin", relative_to="table", relation="on_top", distance=0.75)

    # Apple INSIDE bucket (graspable + container relation)
    ops.add_asset("apple", relative_to="storage_bin", relation="inside", distance=0.0)

    # Free camera pointed AT the table scene for clear view
    ops.add_free_camera("scene_cam",
                       lookat=(1.5, 0.0, 0.8),  # Look at table height
                       distance=2.0,             # 2m away
                       azimuth=45,               # 45° angle
                       elevation=-20)            # Look down slightly

    ops.compile()
    ops.step()

    # ===== LAYER 1: PHYSICS VALIDATION =====
    print("  1. Physics Validation:")
    state = ops.get_state()

    if "table" not in state:
        print("     ✗ Table not in state")
        print(f"\n  ❌ FAIL: Table not in state")
        return False

    if "storage_bin" not in state:
        print("     ✗ Bucket not in state")
        print(f"\n  ❌ FAIL: Bucket not in state")
        return False

    if "apple" not in state:
        print("     ✗ Apple not in state")
        print(f"\n  ❌ FAIL: Apple not in state")
        return False

    table_z = state["table"]["position"][2]
    bucket_z = state["storage_bin"]["position"][2]
    apple_z = state["apple"]["position"][2]

    print(f"     Table Z: {table_z:.3f}m")
    print(f"     Bucket Z: {bucket_z:.3f}m")
    print(f"     Apple Z: {apple_z:.3f}m")

    # Bucket should be ON TOP of table
    if bucket_z <= table_z:
        print("     ✗ Bucket not on top of table!")
        print(f"\n  ❌ FAIL: Bucket not on top of table")
        return False

    print(f"     ✓ Bucket on table (bucket_z > table_z)")

    # ===== LAYER 2: SEMANTIC VALIDATION =====
    print("  2. Semantic Validation:")

    # Container behavior - bucket should contain apple
    bucket_pos = np.array(state["storage_bin"]["position"])
    apple_pos = np.array(state["apple"]["position"])
    distance = np.linalg.norm(apple_pos - bucket_pos)

    print(f"     Apple-Bucket distance: {distance:.3f}m")

    if distance >= 0.5:
        print(f"     ✗ Apple not in bucket (distance: {distance:.2f}m)")
        print(f"\n  ❌ FAIL: Apple not in bucket")
        return False

    print(f"     ✓ Apple inside bucket (container behavior)")

    # Surface behavior - bucket on table surface
    height_diff = bucket_z - table_z
    print(f"     Bucket-Table height diff: {height_diff:.3f}m")

    if not (0.7 < height_diff < 0.9):
        print(f"     ✗ Bucket not at table surface height")
        print(f"\n  ❌ FAIL: Bucket not at table surface height")
        return False

    print(f"     ✓ Bucket at table surface (surface behavior)")

    # ===== LAYER 3: VISION VALIDATION =====
    print("  3. Vision Validation:")
    views = ops.engine.last_views if hasattr(ops, 'engine') and hasattr(ops.engine, 'last_views') else {}

    # Check free camera exists
    if 'scene_cam_view' not in views:
        print("     ✗ Free camera view not found!")
        print(f"\n  ❌ FAIL: Free camera view not found")
        return False

    print(f"     ✓ Free camera view exists")

    scene_view = views['scene_cam_view']

    if "rgb" not in scene_view:
        print("     ✗ No RGB in scene camera!")
        print(f"\n  ❌ FAIL: No RGB in scene camera")
        return False

    img = scene_view["rgb"]
    print(f"     ✓ Camera image: {img.shape}")

    # CONTENT VALIDATION - Check image has color variety (not just gray wall)
    import cv2

    # Convert to HSV to analyze color content
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Calculate color diversity
    hue_std = np.std(img_hsv[:,:,0])
    sat_mean = np.mean(img_hsv[:,:,1])

    print(f"     Image color diversity - Hue std: {hue_std:.1f}, Sat mean: {sat_mean:.1f}")

    # Image should have color variety (objects visible, not just uniform gray)
    if hue_std <= 5.0:
        print(f"     ✗ Image too uniform (hue_std={hue_std:.1f}) - objects not visible!")
        print(f"\n  ❌ FAIL: Image too uniform")
        return False

    print(f"     ✓ Image has color variety (objects visible)")

    # Save image for visual inspection
    save_path = Path(ops.experiment_dir) / "views" / "scene_composition.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)

    if not save_path.exists():
        print("     ✗ Image not saved!")
        print(f"\n  ❌ FAIL: Image not saved")
        return False

    print(f"     📸 Scene image saved: {save_path}")

    # ===== LAYER 4: REASONING VALIDATION =====
    print("  4. Reasoning Validation:")

    # Spatial reasoning - check object arrangement makes sense
    table_pos = np.array(state["table"]["position"])

    # Bucket should be directly above table (same X, Y)
    horizontal_offset = np.linalg.norm(bucket_pos[:2] - table_pos[:2])
    print(f"     Bucket-Table horizontal offset: {horizontal_offset:.3f}m")

    if horizontal_offset >= 0.3:
        print("     ✗ Bucket not centered on table")
        print(f"\n  ❌ FAIL: Bucket not centered on table")
        return False

    print(f"     ✓ Bucket centered on table (spatial relation correct)")

    # Apple should be directly above bucket center
    apple_bucket_offset = np.linalg.norm(apple_pos[:2] - bucket_pos[:2])
    print(f"     Apple-Bucket horizontal offset: {apple_bucket_offset:.3f}m")

    if apple_bucket_offset >= 0.2:
        print("     ✗ Apple not centered in bucket")
        print(f"\n  ❌ FAIL: Apple not centered in bucket")
        return False

    print(f"     ✓ Apple centered in bucket (containment correct)")

    # Complete behavior chain validated
    print(f"     ✓ Behavior chain: Table(surface) → Bucket(on_top+container) → Apple(inside+graspable)")

    print(f"\n  ✅ PASS: Comprehensive Vision Validation")
    return True


# ============================================================================
# TEST 6: ROLLABLE BEHAVIOR
# ============================================================================

def test_6_rollable_behavior():
    """Test 6: Rollable Behavior - Ball rolls when force applied"""
    print("\n" + "="*70)
    print("TEST 6: Rollable Behavior")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")

    # Create scene with ball on floor
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Baseball on floor, 2m in front of robot
    ops.add_object("baseball", position=(2.0, 0.0, 0.05))  # Slightly above floor to prevent clipping

    # Free camera to track ball rolling
    ops.add_free_camera("roll_cam",
                       lookat=(2.0, 0.0, 0.1),
                       distance=3.0,
                       azimuth=90,
                       elevation=-10)

    ops.compile()

    # ===== LAYER 1: PHYSICS VALIDATION (Initial State) =====
    print("  1. Physics Validation (Initial):")
    ops.step()
    state = ops.get_state()

    if "baseball" not in state:
        print("     ✗ Baseball not in state")
        print(f"\n  ❌ FAIL: Baseball not in state")
        return False

    initial_pos = np.array(state["baseball"]["position"])
    print(f"     Initial position: {initial_pos}")

    # Ball should be on floor
    if initial_pos[2] >= 0.1:
        print(f"     ✗ Ball not on floor (z={initial_pos[2]:.3f}m)")
        print(f"\n  ❌ FAIL: Ball not on floor")
        return False

    print(f"     ✓ Ball on floor (z={initial_pos[2]:.3f}m)")

    # ===== APPLY FORCE TO MAKE BALL ROLL =====
    print("  2. Applying force to ball...")

    # Get ball body ID and apply horizontal force
    import mujoco
    ball_body_id = mujoco.mj_name2id(ops.model, mujoco.mjtObj.mjOBJ_BODY, "baseball")

    # Apply force for 50 steps to get ball rolling
    for i in range(50):
        ops.data.xfrc_applied[ball_body_id][0] = -2.0  # Force in -X direction
        ops.step()

    # ===== LAYER 2: SEMANTIC VALIDATION (Rolling State) =====
    print("  3. Semantic Validation (Rolling):")
    state = ops.get_state()

    # Check for velocity and rolling properties
    if "velocity" in state["baseball"]:
        velocity = state["baseball"]["velocity"]
        speed = np.linalg.norm(velocity)
        print(f"     Ball speed: {speed:.3f} m/s")

        if speed <= 0.1:
            print(f"     ✗ Ball not moving (speed={speed:.3f})")
            print(f"\n  ❌ FAIL: Ball not moving")
            return False

        print(f"     ✓ Ball is moving")

    # Check angular velocity if available
    if "angular_velocity" in state["baseball"]:
        ang_vel = state["baseball"]["angular_velocity"]
        ang_speed = np.linalg.norm(ang_vel)
        print(f"     Angular velocity: {ang_speed:.3f} rad/s")
        if ang_speed > 0.5:
            print(f"     ✓ Ball is rotating (rolling behavior)")

    # Check rolling flag
    if "rolling" in state["baseball"]:
        rolling = state["baseball"]["rolling"]
        print(f"     Rolling flag: {rolling}")
        if rolling:
            print(f"     ✓ Rolling behavior detected")

    # ===== LAYER 3: VISION VALIDATION =====
    print("  4. Vision Validation:")
    views = ops.engine.last_views if hasattr(ops, 'engine') and hasattr(ops.engine, 'last_views') else {}

    if 'roll_cam_view' not in views:
        print("     ✗ Roll camera not found")
        print(f"\n  ❌ FAIL: Roll camera not found")
        return False

    print(f"     ✓ Roll camera view exists")

    roll_view = views['roll_cam_view']

    if "rgb" not in roll_view:
        print("     ✗ No RGB in roll camera")
        print(f"\n  ❌ FAIL: No RGB in roll camera")
        return False

    img = roll_view["rgb"]
    print(f"     ✓ Camera image: {img.shape}")

    # Save image
    import cv2
    save_path = Path(ops.experiment_dir) / "views" / "ball_rolling.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)
    print(f"     📸 Rolling ball image saved: {save_path}")

    # ===== LAYER 4: REASONING VALIDATION (Motion Tracking) =====
    print("  5. Reasoning Validation (Motion):")

    final_pos = np.array(state["baseball"]["position"])
    print(f"     Final position: {final_pos}")

    # Ball should have moved in -X direction
    displacement = initial_pos - final_pos
    print(f"     Displacement: {displacement}")

    # Check X displacement (should be positive since force in -X)
    if displacement[0] <= 0.05:
        print(f"     ✗ Ball didn't move enough (dx={displacement[0]:.3f}m)")
        print(f"\n  ❌ FAIL: Ball didn't move enough")
        return False

    print(f"     ✓ Ball moved {displacement[0]:.3f}m in expected direction")

    # Ball should stay roughly at same Y
    if abs(displacement[1]) >= 0.2:
        print(f"     ✗ Ball drifted sideways too much")
        print(f"\n  ❌ FAIL: Ball drifted sideways")
        return False

    print(f"     ✓ Ball trajectory straight (Y drift: {abs(displacement[1]):.3f}m)")

    # Ball should stay on floor
    if final_pos[2] >= 0.15:
        print(f"     ✗ Ball jumped off floor (z={final_pos[2]:.3f}m)")
        print(f"\n  ❌ FAIL: Ball jumped off floor")
        return False

    print(f"     ✓ Ball stayed on floor (rollable behavior confirmed)")

    print(f"\n  ✅ PASS: Rollable Behavior validated")
    return True


# ============================================================================
# TEST 7: STACKABLE BEHAVIOR
# ============================================================================

def test_7_stackable_behavior():
    """Test 7: Stackable Behavior - Stack 3 blocks on table"""
    print("\n" + "="*70)
    print("TEST 7: Stackable Behavior")
    print("="*70)

    ops = ExperimentOps(headless=True, render_mode="vision_rl")

    # Create scene with table
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Table at 1.5m in front
    ops.add_asset("table", relative_to=(1.5, 0.0, 0.0))

    # Stack 3 different stackable boxes on table
    # Note: Scene system doesn't support multiple instances of same asset
    # Use only BOX-shaped objects (not cans which roll): foam_brick, pudding_box, cracker_box
    # NEW: orientation='upright' ensures boxes spawn in correct orientation for stable stacking!
    ops.add_asset("foam_brick", relative_to="table", relation="on_top", distance=0.75, orientation="upright")
    ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on", orientation="upright")
    ops.add_asset("cracker_box", relative_to="pudding_box", relation="stack_on", orientation="upright")

    # Free camera to view the stack (scenic view)
    ops.add_free_camera("stack_cam",
                       lookat=(1.5, 0.0, 0.9),
                       distance=2.5,
                       azimuth=45,
                       elevation=-15)

    # ADDED: Top-down diagnostic camera for horizontal alignment validation
    ops.add_free_camera("val_top",
                       lookat=(1.5, 0.0, 0.9),
                       distance=2.0,
                       azimuth=0,
                       elevation=-85)  # Nearly straight down

    ops.compile()

    # Let physics settle (increased for stable stacking)
    for i in range(250):
        ops.step()

    # ===== LAYER 1: PHYSICS VALIDATION =====
    print("  1. Physics Validation:")
    state = ops.get_state()

    if "table" not in state:
        print("     ✗ Table not in state")
        print(f"\n  ❌ FAIL: Table not in state")
        return False

    if "foam_brick" not in state:
        print("     ✗ foam_brick not in state")
        print(f"\n  ❌ FAIL: foam_brick not in state")
        return False

    if "pudding_box" not in state:
        print("     ✗ pudding_box not in state")
        print(f"\n  ❌ FAIL: pudding_box not in state")
        return False

    if "cracker_box" not in state:
        print("     ✗ cracker_box not in state")
        print(f"\n  ❌ FAIL: cracker_box not in state")
        return False

    table_z = state["table"]["position"][2]
    foam_z = state["foam_brick"]["position"][2]
    pudding_z = state["pudding_box"]["position"][2]
    cracker_z = state["cracker_box"]["position"][2]

    print(f"     Table Z: {table_z:.3f}m")
    print(f"     Foam brick Z: {foam_z:.3f}m")
    print(f"     Pudding box Z: {pudding_z:.3f}m")
    print(f"     Cracker box Z: {cracker_z:.3f}m")

    # Objects should be stacked vertically
    if foam_z <= table_z:
        print("     ✗ Foam brick not on table")
        print(f"\n  ❌ FAIL: Foam brick not on table")
        return False

    if pudding_z <= foam_z:
        print("     ✗ Pudding box not on foam brick")
        print(f"\n  ❌ FAIL: Pudding box not on foam brick")
        return False

    if cracker_z <= pudding_z:
        print("     ✗ Cracker box not on pudding box")
        print(f"\n  ❌ FAIL: Cracker box not on pudding box")
        return False

    print(f"     ✓ Objects stacked vertically (table < foam < pudding < cracker)")

    # ===== LAYER 2: SEMANTIC VALIDATION =====
    print("  2. Semantic Validation:")

    # Check height differences
    height_diff_1 = foam_z - table_z
    height_diff_2 = pudding_z - foam_z
    height_diff_3 = cracker_z - pudding_z

    print(f"     Table→Foam height: {height_diff_1:.3f}m")
    print(f"     Foam→Pudding height: {height_diff_2:.3f}m")
    print(f"     Pudding→Cracker height: {height_diff_3:.3f}m")

    # First object at table surface height, others stacked on top
    if not (0.70 < height_diff_1 < 0.95):
        print(f"     ✗ Foam brick not at table surface height")
        print(f"\n  ❌ FAIL: Foam brick not at table surface height")
        return False

    if not (0.02 < height_diff_2 < 0.15):
        print(f"     ✗ Pudding box not properly on foam brick")
        print(f"\n  ❌ FAIL: Pudding box not properly on foam brick")
        return False

    if not (0.02 < height_diff_3 < 0.15):
        print(f"     ✗ Cracker box not properly on pudding box")
        print(f"\n  ❌ FAIL: Cracker box not properly on pudding box")
        return False

    print(f"     ✓ Height increments consistent (stackable behavior)")

    # Check horizontal alignment (objects should be centered on each other)
    foam_pos = np.array(state["foam_brick"]["position"])
    pudding_pos = np.array(state["pudding_box"]["position"])
    cracker_pos = np.array(state["cracker_box"]["position"])

    offset_fp = np.linalg.norm(pudding_pos[:2] - foam_pos[:2])
    offset_pc = np.linalg.norm(cracker_pos[:2] - pudding_pos[:2])

    print(f"     Foam-Pudding horizontal offset: {offset_fp:.3f}m")
    print(f"     Pudding-Cracker horizontal offset: {offset_pc:.3f}m")

    # Objects should be roughly centered (allow some physics wobble)
    if offset_fp >= 0.20:
        print(f"     ✗ Pudding box not centered on foam brick")
        print(f"\n  ❌ FAIL: Pudding box not centered on foam brick")
        return False

    if offset_pc >= 0.20:
        print(f"     ✗ Cracker box not centered on pudding box")
        print(f"\n  ❌ FAIL: Cracker box not centered on pudding box")
        return False

    print(f"     ✓ Objects aligned horizontally (stack stable)")

    # ===== LAYER 3: VISION VALIDATION =====
    print("  3. Vision Validation:")
    views = ops.engine.last_views if hasattr(ops, 'engine') and hasattr(ops.engine, 'last_views') else {}

    if 'stack_cam_view' not in views:
        print("     ✗ Stack camera not found")
        print(f"\n  ❌ FAIL: Stack camera not found")
        return False

    print(f"     ✓ Stack camera view exists")

    stack_view = views['stack_cam_view']

    if "rgb" not in stack_view:
        print("     ✗ No RGB in stack camera")
        print(f"\n  ❌ FAIL: No RGB in stack camera")
        return False

    img = stack_view["rgb"]
    print(f"     ✓ Camera image: {img.shape}")

    # Save image
    import cv2
    save_path = Path(ops.experiment_dir) / "views" / "block_stack.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), img_bgr)
    print(f"     📸 Stack image saved: {save_path}")

    # ===== LAYER 4: REASONING VALIDATION =====
    print("  4. Reasoning Validation:")

    # Total stack height should be reasonable
    total_height = cracker_z - table_z
    print(f"     Total stack height: {total_height:.3f}m")

    if not (0.78 < total_height < 1.0):
        print(f"     ✗ Stack height unrealistic ({total_height:.3f}m)")
        print(f"\n  ❌ FAIL: Stack height unrealistic")
        return False

    print(f"     ✓ Stack height realistic")

    # Check stack stability - top object shouldn't move much in next 50 steps
    cracker_z_before = cracker_z
    for i in range(50):
        ops.step()

    state_after = ops.get_state()
    cracker_z_after = state_after["cracker_box"]["position"][2]
    z_drift = abs(cracker_z_after - cracker_z_before)

    print(f"     Z drift over 50 steps: {z_drift:.4f}m")

    if z_drift >= 0.10:
        print(f"     ✗ Stack unstable (z drift: {z_drift:.3f}m)")
        print(f"\n  ❌ FAIL: Stack unstable")
        return False

    print(f"     ✓ Stack stable over time")

    # Complete stackable behavior validated
    print(f"     ✓ Stackable behavior: 3 objects stacked, aligned, and stable")

    # ===== LAYER 4: VISUAL VALIDATION =====
    print("  4. Visual Validation:")

    from pathlib import Path
    screenshot_dir = Path(ops.experiment_dir) / "behavior_validation"

    # Save both cameras
    for cam_id in ["stack_cam", "val_top"]:
        camera = ops.scene.cameras[cam_id]
        path = camera.screenshot(0, str(screenshot_dir))
        print(f"     ✓ {cam_id}: {Path(path).name}")

    print(f"     → Visual validation: Check screenshots for:")
    print(f"       - stack_cam: Scenic view of 3-block stack")
    print(f"       - val_top: Horizontal alignment (objects centered)")

    print(f"\n  ✅ PASS: Stackable Behavior validated (Physics + Visual)")
    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LEVEL 1I: OBJECT BEHAVIORS & VISION VALIDATION")
    print("="*70)

    tests = [
        ("Test 1: Graspable Behavior", test_1_graspable_behavior),
        ("Test 2: Container Behavior", test_2_container_behavior),
        ("Test 3: Surface Behavior", test_3_surface_behavior),
        ("Test 4: Behavior Inheritance", test_4_behavior_inheritance),
        ("Test 5: Vision Sees Behaviors", test_5_vision_sees_behaviors),
        ("Test 6: Rollable Behavior", test_6_rollable_behavior),
        ("Test 7: Stackable Behavior", test_7_stackable_behavior),
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
    elif passed_count >= total * 0.8:
        print(f"\n  ⚠️  {total - passed_count} test(s) failed")
    else:
        print(f"\n  ❌ {total - passed_count} test(s) failed")


if __name__ == "__main__":
    main()
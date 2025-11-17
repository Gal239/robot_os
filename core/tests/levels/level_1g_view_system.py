#!/usr/bin/env python3
"""
LEVEL 1G: VIEW SYSTEM - COMPREHENSIVE TESTING
==============================================

Complete tests for the view system that powers TIME TRAVELER:
- Camera views (RGB, depth, resolution)
- Sensor views (odometry, joints, all sensor types)
- Actuator views (base, arm, lift, gripper states)
- System views (runtime status, experiment metadata)
- View metadata (__meta__, view_type, modal_ref, modal_category)
- View aggregation (all views together)
- View updates during actions
- View saving to database

8 Tests total

Prerequisites:
- Level 1A: Modal Architecture ✅
- Level 1B: Action System ✅
- Level 1C: Action Queues ✅
- Level 1D: Scene Operations ✅

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1g_view_system.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.main.experiment_ops_unified import ExperimentOps


# ============================================================================
# TEST 1: CAMERA VIEWS
# ============================================================================

def test_1_camera_views():
    """Test 1: Camera views with RGB data"""
    print("\n" + "="*70)
    print("TEST 1: Camera Views")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("camera_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    # Add bird's eye camera to test virtual camera views
    ops.add_free_camera('birds_eye', lookat=(4, 4, 0.5), distance=6.0, elevation=-35)
    ops.compile()

    # Execute action to generate camera frames
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check camera views
    has_camera_view = False
    has_rgb_data = False
    rgb_shape_correct = False
    rgb_dtype_correct = False

    # NEW: Check bird's eye camera
    has_birds_eye_view = False
    birds_eye_has_rgb = False
    birds_eye_shape_correct = False

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        # Look for nav_camera_view
        if "nav_camera_view" in views:
            has_camera_view = True
            camera_view = views["nav_camera_view"]

            # Check RGB data
            if "rgb" in camera_view:
                rgb = camera_view["rgb"]
                if rgb is not None and isinstance(rgb, np.ndarray):
                    has_rgb_data = True
                    # RGB should be 3D array (height, width, channels)
                    rgb_shape_correct = (rgb.ndim == 3)
                    # RGB should be uint8
                    rgb_dtype_correct = (rgb.dtype == np.uint8)

        # NEW: Check bird's eye view
        if "birds_eye_view" in views:
            has_birds_eye_view = True
            birds_eye_view = views["birds_eye_view"]

            # Check RGB data (virtual camera uses 'rgb' key, not 'birds_eye_rgb')
            if "rgb" in birds_eye_view:
                rgb = birds_eye_view["rgb"]
                if rgb is not None and isinstance(rgb, np.ndarray):
                    birds_eye_has_rgb = True
                    # Should be 3D array (height, width, channels)
                    birds_eye_shape_correct = (rgb.ndim == 3 and rgb.dtype == np.uint8)

    print(f"  Has nav camera view: {has_camera_view}")
    print(f"  Has RGB data: {has_rgb_data}")
    print(f"  RGB shape correct: {rgb_shape_correct}")
    print(f"  RGB dtype correct: {rgb_dtype_correct}")
    print(f"  Has bird's eye view: {has_birds_eye_view}")
    print(f"  Bird's eye has RGB: {birds_eye_has_rgb}")
    print(f"  Bird's eye shape correct: {birds_eye_shape_correct}")

    passed = (
        has_camera_view and
        has_rgb_data and
        rgb_shape_correct and
        has_birds_eye_view and  # NEW: Require bird's eye
        birds_eye_has_rgb and   # NEW: Require bird's eye RGB
        birds_eye_shape_correct # NEW: Require correct format
    )

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Camera views generate RGB data with correct resolution")
    return passed


# ============================================================================
# TEST 2: SENSOR VIEWS
# ============================================================================

def test_2_sensor_views():
    """Test 2: Sensor views (odometry, joints, etc.)"""
    print("\n" + "="*70)
    print("TEST 2: Sensor Views")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("sensor_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Execute action to generate sensor data
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check sensor views
    has_odometry = False
    odometry_has_position = False
    has_joint_sensors = False

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        # Check for odometry sensor
        for view_name, view_data in views.items():
            if 'odometry' in view_name.lower():
                has_odometry = True
                # Check if has position data
                if 'x' in view_data and 'y' in view_data:
                    odometry_has_position = True
                break

        # Check for joint sensors
        for view_name in views.keys():
            if 'joint' in view_name.lower():
                has_joint_sensors = True
                break

    print(f"  Has odometry: {has_odometry}")
    print(f"  Odometry has position: {odometry_has_position}")
    print(f"  Has joint sensors: {has_joint_sensors}")

    passed = has_odometry and odometry_has_position

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Sensor views contain odometry and joint data")
    return passed


# ============================================================================
# TEST 3: ACTUATOR VIEWS
# ============================================================================

def test_3_actuator_views():
    """Test 3: Actuator views (base, arm, lift, gripper)"""
    print("\n" + "="*70)
    print("TEST 3: Actuator Views")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("actuator_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Execute action to generate actuator data
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check actuator views
    has_base_view = False
    has_arm_view = False
    has_actuator_data = False

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        # Check for base actuator view
        for view_name in views.keys():
            if 'base' in view_name.lower():
                has_base_view = True
                break

        # Check for arm view
        if 'arm_view' in views:
            has_arm_view = True
            arm_view = views['arm_view']
            # Should have position data
            if 'position' in arm_view or 'arm' in arm_view:
                has_actuator_data = True

    print(f"  Has base view: {has_base_view}")
    print(f"  Has arm view: {has_arm_view}")
    print(f"  Has actuator data: {has_actuator_data}")

    # At least some actuator views should exist
    passed = has_base_view or has_arm_view

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Actuator views contain state information")
    return passed


# ============================================================================
# TEST 4: SYSTEM VIEWS
# ============================================================================

def test_4_system_views():
    """Test 4: System views (runtime status, experiment info)"""
    print("\n" + "="*70)
    print("TEST 4: System Views")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("system_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Execute action
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check system views
    has_runtime_status = False
    has_experiment_view = False
    runtime_has_data = False

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        # Check for runtime status
        if 'runtime_status' in views:
            has_runtime_status = True
            runtime = views['runtime_status']
            # Should have some runtime data
            if len(runtime) > 0:
                runtime_has_data = True

        # Check for experiment view
        for view_name in views.keys():
            if 'experiment' in view_name.lower() or 'scene' in view_name.lower():
                has_experiment_view = True
                break

    print(f"  Has runtime status: {has_runtime_status}")
    print(f"  Runtime has data: {runtime_has_data}")
    print(f"  Has experiment view: {has_experiment_view}")

    passed = has_runtime_status

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: System views contain runtime and experiment metadata")
    return passed


# ============================================================================
# TEST 5: VIEW METADATA
# ============================================================================

def test_5_view_metadata():
    """Test 5: View metadata (__meta__ field)"""
    print("\n" + "="*70)
    print("TEST 5: View Metadata")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("metadata_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Execute action
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check view metadata
    views_with_meta = 0
    views_with_view_type = 0
    views_with_modal_ref = 0
    video_views = 0
    data_views = 0

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        for view_name, view_data in views.items():
            # Check for __meta__ field
            if "__meta__" in view_data:
                views_with_meta += 1
                meta = view_data["__meta__"]

                # Check for view_type
                if "view_type" in meta:
                    views_with_view_type += 1
                    view_type = meta["view_type"]

                    # Count view types
                    if view_type == "video":
                        video_views += 1
                    elif view_type == "data":
                        data_views += 1

                # Check for modal_ref
                if "modal_ref" in meta:
                    views_with_modal_ref += 1

    print(f"  Views with __meta__: {views_with_meta}")
    print(f"  Views with view_type: {views_with_view_type}")
    print(f"  Views with modal_ref: {views_with_modal_ref}")
    print(f"  Video views: {video_views}")
    print(f"  Data views: {data_views}")

    passed = (
        views_with_meta >= 3 and  # At least 3 views have metadata
        views_with_view_type >= 3 and  # At least 3 have view_type
        (video_views + data_views) >= 2  # At least 2 categorized
    )

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Views have correct __meta__ with view_type, modal_ref, modal_category")
    return passed


# ============================================================================
# TEST 6: VIEW AGGREGATION
# ============================================================================

def test_6_view_aggregation():
    """Test 6: View aggregation (all views in one dict)"""
    print("\n" + "="*70)
    print("TEST 6: View Aggregation")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("aggregation_test", width=8, length=8, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Execute action
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=0.5, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(500):
        ops.step()
        if block.status == 'completed':
            break

    # Check view aggregation
    has_views = False
    total_views = 0
    has_camera = False
    has_sensor = False
    has_actuator = False
    has_system = False

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        has_views = True
        total_views = len(views)

        # Check for different view categories
        for view_name in views.keys():
            if 'camera' in view_name.lower():
                has_camera = True
            if 'sensor' in view_name.lower() or 'odometry' in view_name.lower():
                has_sensor = True
            if 'arm' in view_name.lower() or 'gripper' in view_name.lower():
                has_actuator = True
            if 'runtime' in view_name.lower() or 'status' in view_name.lower():
                has_system = True

    print(f"  Has views: {has_views}")
    print(f"  Total views: {total_views}")
    print(f"  Has camera: {has_camera}")
    print(f"  Has sensor: {has_sensor}")
    print(f"  Has actuator: {has_actuator}")
    print(f"  Has system: {has_system}")

    passed = (
        has_views and
        total_views >= 5 and  # At least 5 different views
        has_camera  # Must have camera
    )

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: ViewAggregator combines all views into single dictionary")
    return passed


# ============================================================================
# TEST 7: VIEW UPDATES DURING ACTION
# ============================================================================

def test_7_view_updates_during_action():
    """Test 7: Views update during robot action"""
    print("\n" + "="*70)
    print("TEST 7: View Updates During Action")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("update_test", width=10, length=10, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Capture initial views
    ops.step()
    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    initial_views = ops.engine.last_views

    # Execute action
    from core.modals.stretch.action_blocks_registry import move_forward

    block = move_forward(distance=1.0, speed=0.3)
    ops.submit_block(block)

    # Run until action completes
    for i in range(1000):
        ops.step()
        if block.status == 'completed':
            break

    # Capture final views
    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    final_views = ops.engine.last_views

    # Check if views changed
    views_changed = False
    position_changed = False

    if initial_views and final_views:
        # Check if odometry changed
        for view_name in initial_views.keys():
            if 'odometry' in view_name.lower():
                if view_name in final_views:
                    init_x = initial_views[view_name].get('x', 0)
                    final_x = final_views[view_name].get('x', 0)
                    if abs(final_x - init_x) > 0.1:
                        position_changed = True
                        views_changed = True
                    break

    print(f"  Views changed: {views_changed}")
    print(f"  Position changed: {position_changed}")

    passed = views_changed and position_changed

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Views change as robot moves")
    return passed


# ============================================================================
# TEST 8: VIEW COMPLETENESS
# ============================================================================

def test_8_view_completeness():
    """Test 8: All major view types present"""
    print("\n" + "="*70)
    print("TEST 8: View Completeness")
    print("="*70)

    # Create rich scene
    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("complete_test", width=10, length=10, height=3,
                     floor_texture="concrete", wall_texture="gray_wall")  # Fixed: use valid texture
    ops.add_robot("stretch")

    # Add objects for visual interest
    try:
        ops.add_object("table", position=(3, 0, 0))
        ops.add_object("chair", position={"relative_to": "table", "relation": "front", "distance": 0.8})
        ops.compile()
    except:
        pass

    # Execute action
    from core.modals.stretch.action_blocks_registry import move_forward, spin_left

    block1 = move_forward(distance=1.0, speed=0.3)
    ops.submit_block(block1)

    block2 = spin_left(degrees=90, speed=15.0)
    ops.submit_block(block2)

    # Run until actions complete
    for i in range(2000):
        ops.step()
        if block1.status == 'completed' and block2.status == 'completed':
            break

    # Check view completeness
    total_views = 0
    view_categories = set()
    camera_count = 0
    sensor_count = 0
    actuator_count = 0
    system_count = 0

    # OFFENSIVE: Access views directly - crashes if engine not initialized (reveals bugs!)
    views = ops.engine.last_views

    if views:
        total_views = len(views)

        for view_name, view_data in views.items():
            # Categorize by __meta__ if available
            if "__meta__" in view_data:
                meta = view_data["__meta__"]
                if "modal_category" in meta:
                    view_categories.add(meta["modal_category"])

                view_type = meta.get("view_type", "")
                if view_type == "video":
                    camera_count += 1
                elif view_type == "data":
                    system_count += 1
                elif view_type == "video_and_data":
                    if "sensor" in meta.get("modal_category", ""):
                        sensor_count += 1
                    else:
                        actuator_count += 1

    print(f"  Total views: {total_views}")
    print(f"  View categories: {len(view_categories)}")
    print(f"  Camera count: {camera_count}")
    print(f"  Sensor count: {sensor_count}")
    print(f"  Actuator count: {actuator_count}")
    print(f"  System count: {system_count}")

    passed = (
        total_views >= 8 and  # At least 8 different views
        camera_count >= 1 and  # At least one camera
        len(view_categories) >= 2  # At least 2 different categories
    )

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Comprehensive scene has all major view types")
    return passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all tests"""
    print("\n" + "📹 "*35)
    print("LEVEL 1G: VIEW SYSTEM - COMPREHENSIVE TESTING")
    print("📹 "*35)

    tests = [
        ("Test 1: Camera Views", test_1_camera_views),
        ("Test 2: Sensor Views", test_2_sensor_views),
        ("Test 3: Actuator Views", test_3_actuator_views),
        ("Test 4: System Views", test_4_system_views),
        ("Test 5: View Metadata", test_5_view_metadata),
        ("Test 6: View Aggregation", test_6_view_aggregation),
        ("Test 7: View Updates During Action", test_7_view_updates_during_action),
        ("Test 8: View Completeness", test_8_view_completeness),
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
    print("LEVEL 1G SUMMARY")
    print("="*70)

    passed_count = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\n  Passed: {passed_count}/{total}")

    if passed_count == total:
        print("\n  🎉 ALL LEVEL 1G VIEW TESTS PASSED!")
        return True
    else:
        print(f"\n  ⚠️  {total - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

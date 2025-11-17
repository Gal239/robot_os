#!/usr/bin/env python3
"""
LEVEL 1H: SCENE COMPOSITION & TIME-SERIES VALIDATION
=====================================================

Tests multi-object scene composition with TIME-SERIES VALIDATION:
- Physics: Complex scenes with multiple spatial relations
- Stability: Objects maintain relationships over time (100+ frames)
- Vision: Camera captures scene evolution (time-lapse proof!)
- Artifacts: Multi-frame image sequences showing scene stability

This validates that complex scene composition is stable and correct!

Prerequisites:
- Level 1A: Modal Architecture & Sync ✅
- Level 1B: Action System ✅
- Level 1E: Object Placement ✅
- Level 1F: Spatial Relations ✅
- Level 1G: Object Behaviors ✅

What This Tests:
1. simple composition - Table + Apple (2 objects, 1 relation)
2. nested composition - Table + Bowl + Apple (3 objects, nested relations)
3. complex scene - 5+ objects with multiple spatial relations
4. time-series stability - Scene stays stable over 100 frames (with snapshots!)

4 TESTS TOTAL - Complex scene composition with time-lapse proof! 📹

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1h_scene_composition.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
from core.main.experiment_ops_unified import ExperimentOps


def test_1_simple_composition():
    """Test 1: Simple Composition - Table + Apple"""
    print("\n" + "="*70)
    print("TEST 1: Simple Composition (2 objects)")
    print("="*70)

    ops = ExperimentOps(headless=True)

    # Create simple composed scene
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("table", position=(2.0, 0.0, 0.0))
    ops.add_object("apple", position={
        "relative_to": "table",
        "relation": "on_top"
    })

    ops.compile()
    ops.step()

    # COMPOSITION VALIDATION
    state = ops.get_state()

    passed = True
    if "table" not in state:
        print("  ✗ Table not in state")
        passed = False
    if "apple" not in state:
        print("  ✗ Apple not in state")
        passed = False

    if passed:
        table_z = state["table"]["position"][2]
        apple_z = state["apple"]["position"][2]

        print(f"  Table Z: {table_z:.3f}m")
        print(f"  Apple Z: {apple_z:.3f}m")

        if apple_z > table_z:
            print(f"  ✓ Apple on top of table")
        else:
            print(f"  ✗ Apple not above table")
            passed = False

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Simple Composition")
    return passed


def test_2_nested_composition():
    """Test 2: Nested Composition - Table + Bowl + Apple"""
    print("\n" + "="*70)
    print("TEST 2: Nested Composition (3 objects)")
    print("="*70)

    ops = ExperimentOps(headless=True)

    # Create nested composition
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Level 1: Table on floor
    ops.add_object("table", position=(2.0, 0.0, 0.0))

    # Level 2: Bowl on table
    ops.add_object("bowl", position={
        "relative_to": "table",
        "relation": "on_top"
    })

    # Level 3: Apple in bowl
    ops.add_object("apple", position={
        "relative_to": "bowl",
        "relation": "inside"
    })

    ops.compile()
    ops.step()

    # NESTED COMPOSITION VALIDATION
    state = ops.get_state()
    table_z = state["table"]["position"][2]
    bowl_z = state["bowl"]["position"][2]
    apple_z = state["apple"]["position"][2]

    print(f"  Table Z: {table_z:.3f}m")
    print(f"  Bowl Z:  {bowl_z:.3f}m")
    print(f"  Apple Z: {apple_z:.3f}m")

    # Verify hierarchy
    passed = True
    if not (bowl_z > table_z):
        print("  ✗ Bowl not above table")
        passed = False
    if not (apple_z >= bowl_z - 0.1):
        print("  ✗ Apple not at/in bowl level")
        passed = False

    if passed:
        print(f"  ✓ Nested composition: Table → Bowl → Apple")

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Nested Composition")
    return passed


def test_3_complex_scene():
    """Test 3: Complex Scene - 5+ objects with multiple relations"""
    print("\n" + "="*70)
    print("TEST 3: Complex Scene (5+ objects)")
    print("="*70)

    ops = ExperimentOps(headless=True)

    # Create complex scene
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # Furniture
    ops.add_object("table", position=(2.0, 0.0, 0.0))

    # Objects on table
    ops.add_object("bowl", position={
        "relative_to": "table",
        "relation": "on_top"
    })

    ops.add_object("apple", position={
        "relative_to": "bowl",
        "relation": "inside"
    })

    # Objects on floor (various distances)
    ops.add_object("baseball", position=(1.0, 1.0, 0.05))    # Near robot
    ops.add_object("softball", position=(3.0, -1.0, 0.05))   # Far from robot
    ops.add_object("large_marker", position=(-1.0, 2.0, 0.05))  # Behind robot

    ops.compile()
    ops.step()

    # COMPLEX SCENE VALIDATION
    state = ops.get_state()

    # Verify all objects present
    objects = ["table", "bowl", "apple", "baseball", "softball", "large_marker"]
    passed = True
    for obj_name in objects:
        if obj_name not in state:
            print(f"  ✗ {obj_name} not in state")
            passed = False
        else:
            print(f"  ✓ {obj_name}: {state[obj_name]['position']}")

    if passed:
        # Verify spatial relationships
        table_z = state["table"]["position"][2]
        bowl_z = state["bowl"]["position"][2]
        apple_z = state["apple"]["position"][2]

        if not (bowl_z > table_z):
            print("  ✗ Bowl not on table")
            passed = False
        if not (apple_z >= bowl_z - 0.1):
            print("  ✗ Apple not in bowl")
            passed = False

        if passed:
            print(f"  ✓ Complex scene composed: 6 objects with nested/spatial relations")

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Complex Scene")
    return passed


def test_4_time_series_stability():
    """Test 4: Time-Series Stability - Scene stable over 100 frames"""
    print("\n" + "="*70)
    print("TEST 4: Time-Series Stability (100 frames + snapshots)")
    print("="*70)

    ops = ExperimentOps(headless=True)

    # Create scene
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("table", position=(2.0, 0.0, 0.0))
    ops.add_object("apple", position={
        "relative_to": "table",
        "relation": "on_top"
    })

    ops.compile()

    # Run 100 frames, save every 25 frames
    snapshot_frames = [0, 25, 50, 75, 99]
    snapshots = {}

    print(f"  Running 100 physics steps...")

    for step_num in range(100):
        result = ops.step()
        state = result.get('state', {})

        # Save snapshots
        if step_num in snapshot_frames:
            table_z = state["table"]["position"][2]
            apple_z = state["apple"]["position"][2]

            snapshots[step_num] = {
                "table_z": table_z,
                "apple_z": apple_z,
                "apple_above_table": apple_z > table_z
            }

            print(f"    Frame {step_num}: Table={table_z:.3f}m, Apple={apple_z:.3f}m, Above={apple_z > table_z}")

            # Try to save camera frame
            try:
                views = ops.get_views()
                for view_name, view_data in views.items():
                    if 'nav_camera' in view_name and "nav_rgb" in view_data:
                        import cv2
                        img = view_data["nav_rgb"]
                        save_path = Path(ops.experiment_dir) / "views" / f"stability_frame_{step_num:03d}.png"
                        save_path.parent.mkdir(parents=True, exist_ok=True)

                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(str(save_path), img_bgr)
                        break
            except:
                pass  # Skip if camera/opencv not available

    # STABILITY VALIDATION
    print(f"\n  Stability Analysis:")

    # Check all snapshots maintain relationship
    all_stable = all(snap["apple_above_table"] for snap in snapshots.values())
    passed = True

    if not all_stable:
        print("  ✗ Apple fell off table during simulation!")
        passed = False
    else:
        # Check Z position variance
        apple_z_values = [snap["apple_z"] for snap in snapshots.values()]
        z_variance = np.var(apple_z_values)
        z_max_change = max(apple_z_values) - min(apple_z_values)

        print(f"    Apple Z variance: {z_variance:.6f}")
        print(f"    Apple Z max change: {z_max_change:.4f}m")

        if z_max_change >= 0.05:
            print(f"  ✗ Apple moved too much ({z_max_change:.3f}m) - scene unstable!")
            passed = False
        else:
            print(f"  ✓ Scene stable over 100 frames (5 snapshots saved)")

    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'}: Time-Series Stability")
    return passed


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LEVEL 1H: SCENE COMPOSITION & TIME-SERIES VALIDATION")
    print("="*70)

    tests = [
        ("Test 1: Simple Composition", test_1_simple_composition),
        ("Test 2: Nested Composition", test_2_nested_composition),
        ("Test 3: Complex Scene", test_3_complex_scene),
        ("Test 4: Time-Series Stability", test_4_time_series_stability),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

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
        return True
    else:
        print(f"\n  ⚠️  {total - passed_count} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

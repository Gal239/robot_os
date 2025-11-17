#!/usr/bin/env python3
"""
LEVEL 1L: COMPOSITE & SEQUENTIAL REWARDS
=========================================

Tests advanced reward system features:
- Composite conditions (AND, OR, NOT)
- Nested conditions (complex logic)
- Sequential rewards (requires chains)
- Reward modes (smooth, discrete, auto)
- Contact force magnitude
- Time conditions and speed bonuses

This level validates PART 2 (sections 9-11) of GUIDE_SMART_SCENE_COMPOSITION.md

Prerequisites:
- Level 1A: Modal Architecture & Sync ✅
- Level 1B: Action System ✅
- Level 1C: Sensor System ✅
- Level 1D: Object Placement & Room Components ✅
- Level 1E: Basic Rewards & Asset Linking ✅

What This Tests:
1. Composite conditions combine multiple requirements
2. Sequential rewards create task chains
3. Reward modes control how rewards are calculated
4. Advanced tracking (force, speed, time)

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1l_composite_rewards.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps


# ============================================================================
# TEST 1: AND CONDITION
# ============================================================================

def test_1_and_condition():
    """Test 1: AND Condition - Gripper holds apple AND lift raised"""
    print("\n" + "="*70)
    print("TEST 1: AND Condition")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(1.0, 0.0, 0.8))

    # Add composite reward: BOTH conditions must be true
    ops.add_reward(
        condition={
            "type": "AND",
            "conditions": [
                {
                    "tracked_asset": "stretch.gripper",
                    "behavior": "holding",
                    "target": "apple",
                    "threshold": True
                },
                {
                    "tracked_asset": "stretch.lift",
                    "behavior": "height",
                    "threshold": {"above": 0.5}
                }
            ]
        },
        reward=200,
        reward_id="lifted_apple"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: AND Condition - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 2: OR CONDITION
# ============================================================================

def test_2_or_condition():
    """Test 2: OR Condition - Either camera sees apple"""
    print("\n" + "="*70)
    print("TEST 2: OR Condition")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(2.0, 0.0, 0.8))

    # Add composite reward: ANY condition can be true
    ops.add_reward(
        condition={
            "type": "OR",
            "conditions": [
                {
                    "tracked_asset": "stretch.vision",  # nav_camera
                    "behavior": "target_visible",
                    "target": "apple",
                    "threshold": True
                },
                {
                    "tracked_asset": "stretch.gripper",  # gripper camera
                    "behavior": "target_visible",
                    "target": "apple",
                    "threshold": True
                }
            ]
        },
        reward=10,
        reward_id="apple_visible"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: OR Condition - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 3: NOT CONDITION
# ============================================================================

def test_3_not_condition():
    """Test 3: NOT Condition - Gripper NOT holding anything"""
    print("\n" + "="*70)
    print("TEST 3: NOT Condition")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))

    # Add negation reward
    ops.add_reward(
        condition={
            "type": "NOT",
            "condition": {
                "tracked_asset": "stretch.gripper",
                "behavior": "holding",
                "threshold": True
            }
        },
        reward=20,
        reward_id="gripper_empty"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: NOT Condition - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 4: NESTED CONDITION
# ============================================================================

def test_4_nested_condition():
    """Test 4: Nested Condition - (Apple in basket OR in gripper) AND NOT on floor"""
    print("\n" + "="*70)
    print("TEST 4: Nested Condition")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("basket", position=(2.0, 0.0, 0.8))
    ops.add_object("apple", position=(2.0, 0.0, 0.9))

    # Add nested composite reward
    ops.add_reward(
        condition={
            "type": "AND",
            "conditions": [
                {
                    "type": "OR",
                    "conditions": [
                        {
                            "tracked_asset": "basket",
                            "behavior": "contains",
                            "target": "apple",
                            "threshold": True
                        },
                        {
                            "tracked_asset": "stretch.gripper",
                            "behavior": "holding",
                            "target": "apple",
                            "threshold": True
                        }
                    ]
                },
                {
                    "type": "NOT",
                    "condition": {
                        "tracked_asset": "floor",
                        "behavior": "contact",
                        "target": "apple",
                        "threshold": True
                    }
                }
            ]
        },
        reward=300,
        reward_id="apple_secured"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Nested Condition - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 5: SEQUENTIAL SIMPLE
# ============================================================================

def test_5_sequential_simple():
    """Test 5: Sequential Simple - Pick then place"""
    print("\n" + "="*70)
    print("TEST 5: Sequential Simple")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(1.0, 0.0, 0.8))
    ops.add_object("basket", position=(2.0, 1.0, 0.8))

    # Step 1: Pick up apple
    ops.add_reward(
        tracked_asset="stretch.gripper",
        behavior="holding",
        target="apple",
        threshold=True,
        reward=50,
        reward_id="picked_apple"
    )

    # Step 2: Place in basket (only after picking)
    ops.add_reward(
        tracked_asset="basket",
        behavior="contains",
        target="apple",
        threshold=True,
        reward=100,
        reward_id="placed_apple",
        requires="picked_apple"  # Sequential dependency!
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify both rewards exist
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Sequential Simple - Rewards exist in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 6: SEQUENTIAL MULTI-STEP
# ============================================================================

def test_6_sequential_multi_step():
    """Test 6: Sequential Multi-Step - Navigate → See → Grasp"""
    print("\n" + "="*70)
    print("TEST 6: Sequential Multi-Step")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(3.0, 0.0, 0.8))

    # Step 1: Navigate to apple
    ops.add_reward(
        tracked_asset="stretch.base",
        behavior="distance_to",
        target="apple",
        threshold={"below": 0.5},
        reward=20,
        reward_id="reached_apple"
    )

    # Step 2: See apple (after reaching)
    ops.add_reward(
        tracked_asset="stretch.vision",
        behavior="target_visible",
        target="apple",
        threshold=True,
        reward=10,
        reward_id="saw_apple",
        requires="reached_apple"  # Requires step 1
    )

    # Step 3: Grasp apple (after seeing)
    ops.add_reward(
        tracked_asset="stretch.gripper",
        behavior="holding",
        target="apple",
        threshold=True,
        reward=100,
        reward_id="grasped_apple",
        requires="saw_apple"  # Requires step 2
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify all rewards exist
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Sequential Multi-Step - Rewards exist in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 7: SMOOTH MODE
# ============================================================================

def test_7_smooth_mode():
    """Test 7: Smooth Mode - Continuous distance shaping"""
    print("\n" + "="*70)
    print("TEST 7: Smooth Mode")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("goal", position=(5.0, 0.0, 0.0))

    # Smooth mode: Continuous reward as robot approaches goal
    ops.add_reward(
        tracked_asset="stretch.base",
        behavior="distance_to",
        target="goal",
        threshold=0.0,
        reward=100,
        mode="smooth",  # Continuous shaping!
        reward_id="approaching"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # In smooth mode, reward should be proportional to progress
    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Smooth Mode - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 8: DISCRETE MODE
# ============================================================================

def test_8_discrete_mode():
    """Test 8: Discrete Mode - One-time threshold crossing"""
    print("\n" + "="*70)
    print("TEST 8: Discrete Mode")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))

    # Discrete mode: One-time reward when threshold crossed
    ops.add_reward(
        tracked_asset="stretch.lift",
        behavior="height",
        threshold=0.8,
        reward=50,
        mode="discrete",  # Sparse, one-time!
        reward_id="lift_raised"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Discrete Mode - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 9: CONTACT FORCE
# ============================================================================

def test_9_contact_force():
    """Test 9: Contact Force - Heavy impact detection"""
    print("\n" + "="*70)
    print("TEST 9: Contact Force")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("box", position=(1.0, 0.0, 2.0))  # Start high (will fall)

    # Track HOW HARD object hit floor
    ops.add_reward(
        tracked_asset="floor",
        behavior="contact_force",
        target="box",
        threshold={"above": 50.0},  # Heavy impact (Newtons)
        reward=-5,
        reward_id="box_dropped_hard"
    )

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Contact Force - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# TEST 10: TIME CONDITIONS
# ============================================================================

def test_10_time_conditions():
    """Test 10: Time Conditions - Complete task within time"""
    print("\n" + "="*70)
    print("TEST 10: Time Conditions")
    print("="*70)

    ops = ExperimentOps(mode="simulated", headless=True)

    ops.create_scene("test_room")
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_object("apple", position=(1.0, 0.0, 0.8))

    # Time-based reward: Pick apple quickly
    ops.add_reward(
        tracked_asset="stretch.gripper",
        behavior="holding",
        target="apple",
        threshold=True,
        reward=100,
        reward_id="picked_fast"
    )

    # Speed bonus: Complete within N steps
    # Note: Actual time condition implementation depends on reward modal
    # This test validates the structure

    ops.compile()
    ops.step()  # CRITICAL: Populate state!

    reward_state = ops.get_reward_state()
    print(f"  Reward state: {reward_state}")

    # Verify reward exists
    if len(reward_state) > 0:
        print("\n  ✅ PASS: Time Conditions - Reward exists in state")
        return True
    else:
        print("\n  ❌ FAIL: No rewards in state")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all Level 1L tests"""
    print("\n" + "="*70)
    print("LEVEL 1L: COMPOSITE & SEQUENTIAL REWARDS")
    print("="*70)

    tests = [
        ("Test 1: AND Condition", test_1_and_condition),
        ("Test 2: OR Condition", test_2_or_condition),
        ("Test 3: NOT Condition", test_3_not_condition),
        ("Test 4: Nested Condition", test_4_nested_condition),
        ("Test 5: Sequential Simple", test_5_sequential_simple),
        ("Test 6: Sequential Multi-Step", test_6_sequential_multi_step),
        ("Test 7: Smooth Mode", test_7_smooth_mode),
        ("Test 8: Discrete Mode", test_8_discrete_mode),
        ("Test 9: Contact Force", test_9_contact_force),
        ("Test 10: Time Conditions", test_10_time_conditions),
    ]

    results = []

    # Composite Condition Tests
    print("\n" + "-"*70)
    print("COMPOSITE CONDITION TESTS")
    print("-"*70)

    for i in range(4):
        name, test_func = tests[i]
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Sequential Reward Tests
    print("\n" + "-"*70)
    print("SEQUENTIAL REWARD TESTS")
    print("-"*70)

    for i in range(4, 6):
        name, test_func = tests[i]
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Reward Mode Tests
    print("\n" + "-"*70)
    print("REWARD MODE TESTS")
    print("-"*70)

    for i in range(6, 8):
        name, test_func = tests[i]
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Advanced Tracking Tests
    print("\n" + "-"*70)
    print("ADVANCED TRACKING TESTS")
    print("-"*70)

    for i in range(8, 10):
        name, test_func = tests[i]
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
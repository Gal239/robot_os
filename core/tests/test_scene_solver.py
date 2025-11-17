"""
TEST: Scene Solver System - PURE MOP Auto-Placement

Tests the complete scene solver system:
1. Robot capability discovery (from actuators)
2. Asset dimension extraction (from MuJoCo or XML)
3. Task-based placement calculation (grasp, inspect)
4. Integration with ExperimentOps

PURE MOP: NO HARDCODING!
- Robot capabilities from actuator.range
- Asset dimensions from get_asset_dimensions()
- Task solvers self-calculate placement
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
import math


def test_robot_reach_capabilities():
    """Test 1: Robot capability discovery - PURE MOP!

    Verifies that robot.get_reach_capabilities() extracts capabilities
    from actuators WITHOUT hardcoding.
    """
    print("\n" + "="*80)
    print("TEST 1: Robot Capability Discovery")
    print("="*80)

    # Create robot
    from core.main.robot_ops import create_robot
    robot = create_robot("stretch", "stretch_1")

    # AUTO-DISCOVERY: Get capabilities from actuators
    caps = robot.get_reach_capabilities()

    print(f"\n✅ Discovered capabilities:")
    print(f"   Horizontal reach: {caps.get('horizontal_reach', 'N/A')} {caps.get('arm_unit', '')}")
    print(f"   Vertical reach: {caps.get('vertical_reach', 'N/A')} {caps.get('lift_unit', '')}")
    print(f"   Gripper width: {caps.get('gripper_width', 'N/A')} {caps.get('gripper_unit', '')}")
    print(f"   Mobile base: {caps.get('mobile_base', False)}")

    if 'workspace_bounds' in caps:
        print(f"\n✅ Workspace bounds:")
        bounds = caps['workspace_bounds']
        print(f"   X: [{bounds['x_min']:.2f}, {bounds['x_max']:.2f}]")
        print(f"   Y: [{bounds['y_min']:.2f}, {bounds['y_max']:.2f}]")
        print(f"   Z: [{bounds['z_min']:.2f}, {bounds['z_max']:.2f}]")

    # Validate capabilities were discovered
    assert 'horizontal_reach' in caps, "Failed to discover arm reach!"
    assert 'vertical_reach' in caps, "Failed to discover lift reach!"
    # Note: mobile_base may or may not be present (Stretch has freejoint but not mobile actuator)

    # Validate reasonable values (Stretch robot specific)
    assert 0.4 < caps['horizontal_reach'] < 0.6, f"Unexpected arm reach: {caps['horizontal_reach']}"
    assert 0.9 < caps['vertical_reach'] < 1.2, f"Unexpected lift reach: {caps['vertical_reach']}"

    print("\n✅ TEST 1 PASSED: Robot capabilities auto-discovered!")


def test_placement_surface_info():
    """Test 2: Asset surface info extraction - PURE MOP!

    Verifies that placement.get_surface_info() extracts dimensions
    from MuJoCo model WITHOUT hardcoding.
    """
    print("\n" + "="*80)
    print("TEST 2: Asset Surface Info Extraction")
    print("="*80)

    # Create scene with table
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.compile()

    # Get surface info (runtime extraction from MuJoCo)
    table_placement = ops.scene.find("table")
    runtime_state = {'model': ops.backend.model}

    info = table_placement.get_surface_info(ops.scene, runtime_state)

    print(f"\n✅ Extracted table surface info:")
    print(f"   Position: {info['position']}")
    print(f"   Dimensions: {info['dimensions']}")
    print(f"   Surface Z: {info['surface_z']}")

    # Validate extraction
    assert 'position' in info, "Failed to extract position!"
    assert 'dimensions' in info, "Failed to extract dimensions!"
    assert 'width' in info['dimensions'], "Missing width dimension!"
    assert 'depth' in info['dimensions'], "Missing depth dimension!"
    assert 'height' in info['dimensions'], "Missing height dimension!"

    # Validate position matches placement
    assert info['position'] == (2.0, 0.0, 0.0), f"Position mismatch: {info['position']}"

    # Validate dimensions are reasonable for a table
    dims = info['dimensions']
    assert 0.5 < dims['width'] < 2.0, f"Unexpected table width: {dims['width']}"
    assert 0.5 < dims['depth'] < 2.0, f"Unexpected table depth: {dims['depth']}"
    assert 0.5 < dims['height'] < 1.0, f"Unexpected table height: {dims['height']}"

    print("\n✅ TEST 2 PASSED: Asset dimensions auto-extracted!")

    ops.close()


def test_scene_solver_grasp_task():
    """Test 3: Scene solver grasp task - PURE MOP!

    Verifies that scene solver calculates robot placement for grasping
    WITHOUT hardcoding dimensions or reach distances.
    """
    print("\n" + "="*80)
    print("TEST 3: Scene Solver - Grasp Task")
    print("="*80)

    # Create scene with table and apple
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # MOP: Compile scene first so dimensions are available!
    ops.compile()

    # Calculate optimal placement for grasping apple
    print("\n🤖 Calculating robot placement for grasping apple...")

    placement = ops.solve_robot_placement(
        robot_id="stretch",
        task="grasp",
        target_asset="apple"
    )

    print(f"\n✅ Calculated placement:")
    print(f"   Position: {placement['position']}")
    print(f"   Orientation: {placement['orientation']}")
    print(f"   Joint positions: {len(placement['initial_state'])} joints")

    # Print joint positions
    for joint_name, value in placement['initial_state'].items():
        print(f"      {joint_name}: {value:.3f}")

    # Validate placement structure
    assert 'position' in placement, "Missing position in placement!"
    assert 'orientation' in placement, "Missing orientation in placement!"
    assert 'initial_state' in placement, "Missing initial_state in placement!"

    # Validate position is reasonable (should be behind apple for comfortable reach)
    pos = placement['position']
    assert isinstance(pos, tuple) and len(pos) == 3, "Position must be (x, y, z) tuple!"

    # Validate joint positions contain expected actuators (MOP names!)
    joints = placement['initial_state']
    assert 'lift' in joints, "Missing lift actuator!"
    assert 'arm' in joints, "Missing arm actuator!"

    # Validate lift is within range (0 to 1.1m for Stretch)
    assert 0.0 <= joints['lift'] <= 1.2, f"Lift out of range: {joints['lift']}"

    print("\n✅ TEST 3 PASSED: Scene solver calculated valid placement!")

    ops.close()


def test_scene_solver_integration():
    """Test 4: Complete integration - PURE MOP!

    Verifies complete workflow:
    1. Calculate placement with solve_robot_placement()
    2. Apply placement with add_robot()
    3. Compile and verify robot is positioned correctly
    """
    print("\n" + "="*80)
    print("TEST 4: Scene Solver Integration")
    print("="*80)

    # Create scene with table and apple
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # MOP: Compile scene first (need dimensions!)
    ops.compile()

    # Calculate placement
    print("\n🤖 Step 1: Calculate optimal placement")
    placement = ops.solve_robot_placement(
        robot_id="stretch",
        task="grasp",
        target_asset="apple"
    )

    # Add robot with calculated placement
    print("\n🤖 Step 2: Add robot with calculated placement")
    ops.add_robot(
        "stretch",
        position=placement['position'],
        orientation=placement['orientation'],
        initial_state=placement['initial_state']
    )

    # Compile scene
    print("\n🤖 Step 3: Compile scene")
    ops.compile()

    # Step physics to populate state
    ops.step()

    # Verify robot position
    print("\n🤖 Step 4: Verify robot position")
    state = ops.get_state()

    # MOP: Robot base position is under 'stretch.base', not 'stretch'!
    robot_base_state = state.get('stretch.base', {})
    robot_pos = robot_base_state.get('position')

    print(f"\n✅ Robot state after compilation:")
    print(f"   Position: {robot_pos}")

    # Validate robot was positioned correctly
    assert robot_pos is not None, "Robot position not in state!"

    # Position should match calculated placement (approximately)
    expected_pos = placement['position']
    for i in range(3):
        assert abs(robot_pos[i] - expected_pos[i]) < 0.1, \
            f"Position mismatch axis {i}: {robot_pos[i]} vs {expected_pos[i]}"

    print("\n✅ TEST 4 PASSED: Complete integration successful!")

    ops.close()


def test_hybrid_api_one_step():
    """Test 5: HYBRID API - ONE-STEP method!

    Verifies that add_robot_for_task() does everything in one step.
    """
    print("\n" + "="*80)
    print("TEST 5: HYBRID API - ONE-STEP Method")
    print("="*80)

    # Create scene with table and apple
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # MOP: Compile scene first (need dimensions for solver!)
    ops.compile()

    # ONE-STEP: Add robot for task (calculates + adds in one call!)
    print("\n🤖 ONE-STEP API: add_robot_for_task()")

    robot = ops.add_robot_for_task(
        robot_name="stretch",
        task="grasp",
        target_asset="apple"
    )

    # Compile and verify
    ops.compile()
    ops.step()  # Step physics to populate state
    state = ops.get_state()

    # MOP: Robot base position is under 'stretch.base', not 'stretch'!
    robot_base_state = state.get('stretch.base', {})
    robot_pos = robot_base_state.get('position')

    print(f"\n✅ Robot added and positioned:")
    print(f"   Position: {robot_pos}")

    # Validate robot was added
    assert robot is not None, "Robot not returned!"
    assert robot_pos is not None, "Robot position not in state!"

    print("\n✅ TEST 5 PASSED: ONE-STEP API works!")

    ops.close()


def test_hybrid_api_two_step():
    """Test 6: HYBRID API - TWO-STEP method!

    Verifies that solve_robot_placement() + add_robot(**placement) works.
    """
    print("\n" + "="*80)
    print("TEST 6: HYBRID API - TWO-STEP Method")
    print("="*80)

    # Create scene with table and apple
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # MOP: Compile scene first!
    ops.compile()

    # TWO-STEP: Calculate placement first
    print("\n🤖 TWO-STEP API: solve_robot_placement() + add_robot(**placement)")
    print("\n   Step 1: Calculate placement")

    placement = ops.solve_robot_placement(
        robot_id="stretch",
        task="grasp",
        target_asset="apple"
    )

    print(f"\n   Calculated: {placement['position']}")

    # Step 2: Apply with unpacking
    print("\n   Step 2: Add robot with **placement unpacking")

    robot = ops.add_robot("stretch", **placement)

    # Compile and verify
    ops.compile()
    ops.step()  # Step physics to populate state
    state = ops.get_state()

    # MOP: Robot base position is under 'stretch.base', not 'stretch'!
    robot_base_state = state.get('stretch.base', {})
    robot_pos = robot_base_state.get('position')

    print(f"\n✅ Robot added and positioned:")
    print(f"   Position: {robot_pos}")

    # Validate robot was added
    assert robot is not None, "Robot not returned!"
    assert robot_pos is not None, "Robot position not in state!"

    # Position should match calculated placement
    expected_pos = placement['position']
    for i in range(3):
        assert abs(robot_pos[i] - expected_pos[i]) < 0.1, \
            f"Position mismatch axis {i}: {robot_pos[i]} vs {expected_pos[i]}"

    print("\n✅ TEST 6 PASSED: TWO-STEP API works!")

    ops.close()


def test_scene_solver_inspect_task():
    """Test 5: Scene solver inspect task - PURE MOP!

    Verifies that scene solver calculates robot placement for inspection
    (camera positioning for visual observation).
    """
    print("\n" + "="*80)
    print("TEST 5: Scene Solver - Inspect Task")
    print("="*80)

    # Create scene with table and apple
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # MOP: Compile scene first!
    ops.compile()

    # Calculate optimal placement for inspecting apple
    print("\n🤖 Calculating robot placement for inspecting apple...")

    placement = ops.solve_robot_placement(
        robot_id="stretch",
        task="inspect",
        target_asset="apple"
    )

    print(f"\n✅ Calculated placement:")
    print(f"   Position: {placement['position']}")
    print(f"   Orientation: {placement['orientation']}")
    print(f"   Joint positions: {len(placement['initial_state'])} joints")

    # Validate placement structure
    assert 'position' in placement, "Missing position in placement!"
    assert 'orientation' in placement, "Missing orientation in placement!"
    assert 'initial_state' in placement, "Missing initial_state in placement!"

    # Validate joints for inspection (camera should be raised, arm retracted)
    joints = placement['initial_state']

    # Lift should be raised to target height (MOP actuator name!)
    assert 'lift' in joints, "Missing lift actuator!"

    # Arm should be retracted for clear view (MOP actuator name!)
    assert 'arm' in joints, "Missing arm actuator!"
    assert joints['arm'] == 0.0, "Arm should be retracted for inspection!"

    # Head should be oriented toward target (MOP actuator names!)
    assert 'head_pan' in joints, "Missing head_pan actuator!"
    assert 'head_tilt' in joints, "Missing head_tilt actuator!"

    print("\n✅ TEST 5 PASSED: Inspect task placement calculated!")

    ops.close()


if __name__ == "__main__":
    """Run all scene solver tests - NO PYTEST!"""
    print("\n" + "="*80)
    print("SCENE SOLVER SYSTEM - PURE MOP TESTS")
    print("="*80)

    # Test 1: Robot capability discovery
    test_robot_reach_capabilities()

    # Test 2: Asset surface info extraction
    test_placement_surface_info()

    # Test 3: Scene solver grasp task
    test_scene_solver_grasp_task()

    # Test 4: Complete integration
    test_scene_solver_integration()

    # Test 5: HYBRID API - ONE-STEP
    test_hybrid_api_one_step()

    # Test 6: HYBRID API - TWO-STEP
    test_hybrid_api_two_step()

    # Test 7: Inspect task
    test_scene_solver_inspect_task()

    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n🧵 PURE MOP VERIFIED:")
    print("   ✓ Robot capabilities auto-discovered from actuators")
    print("   ✓ Asset dimensions auto-extracted from MuJoCo model")
    print("   ✓ Task solvers self-calculate placement")
    print("   ✓ NO HARDCODING - all data discovered at runtime")
    print("   ✓ MODAL-TO-MODAL communication working")
    print("\n🎯 HYBRID API VERIFIED:")
    print("   ✓ ONE-STEP: add_robot_for_task() - simple and fast")
    print("   ✓ TWO-STEP: solve_robot_placement() + add_robot(**placement) - detailed control")
    print("   ✓ Dict unpacking with **placement works perfectly")
    print("="*80 + "\n")

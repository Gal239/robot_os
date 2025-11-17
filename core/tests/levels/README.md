# LEVEL TESTS - MOP Showcase Refactor Plan

## Philosophy: Real Use Cases, Not System Tests

**OLD Approach**: Test infrastructure ("does reward exist?", "can we create scene?")
**NEW Approach**: Real robotics scenarios with complex motion, manipulation, and self-validation

Every test must:
- ✅ Do something REAL (navigate, grasp, stack, search, manipulate)
- ✅ Use multiple capabilities together
- ✅ Show **low code** (5-50 lines) achieving **complex behavior**
- ✅ Self-validate through **rewards + vision proof**

---

## Complete Level Structure

### LEVEL 1A: Foundation (Infrastructure - Keep as is)
**File**: `level_1a_basic_infrastructure_test_new.py`
**Purpose**: Modal architecture, auto-discovery, database
**Status**: ✅ Keep - tests infrastructure (needed for foundation)

---

### LEVEL 1B: Action System
**File**: `level_1b_action_system.py`
**Current**: 18 system tests, 1682 lines
**Target**: 5-6 REAL use cases, ~400 lines

**NEW Tests (Real Scenarios):**
1. **Navigate to Target** - Move forward + rotate to reach specific position (15 lines)
2. **Reach and Grasp Object** - Arm extend + lift adjust + gripper close (20 lines)
3. **Pick and Place** - Lift object, navigate, lower, release (25 lines)
4. **Emergency Stop** - Interrupt action with higher priority task (15 lines)
5. **Multi-Actuator Coordination** - Simultaneous arm+lift+base (20 lines)
6. **Self-Correcting Behavior** - Action fails, retry with adjustment (25 lines)

**Showcase**: "Real manipulation workflows with 4-way self-validation"
- Reward triggers
- Block status = completed
- Block progress = 100%
- Action queue empty

---

### LEVEL 1C: Action Queues
**File**: `level_1c_action_queues.py`
**Current**: 19 infrastructure tests, 1739 lines
**Target**: 4-5 REAL scenarios, ~350 lines

**NEW Tests (Real Scenarios):**
1. **Sequential Task Chain** - Navigate → reach → grasp → lift (must complete in order) (30 lines)
2. **Parallel Multi-Actuator** - Extend arm WHILE raising lift WHILE rotating (20 lines)
3. **Priority Interrupt** - Task A running, emergency → task B → resume A (25 lines)
4. **Complex Choreography** - 10+ action sequence with dependencies (40 lines)
5. **Recovery from Failure** - Queue handles blocked action, re-plans (30 lines)

**Showcase**: "Real task orchestration with automatic queue management"

---

### LEVEL 1D: Sensor System
**File**: `level_1d_sensor_system.py`
**Current**: 13 trackable_behaviors tests, 996 lines
**Target**: 5-6 REAL perception scenarios, ~400 lines

**NEW Tests (Real Scenarios):**
1. **Follow Odometry Path** - Navigate using odometry feedback (25 lines)
2. **LIDAR Obstacle Avoidance** - Detect wall/object, adjust navigation (30 lines)
3. **IMU Stability Check** - Validate robot stable during manipulation (20 lines)
4. **Camera-Guided Reach** - Use camera to locate object, guide arm (35 lines)
5. **Multi-Sensor Fusion** - Combine odometry + LIDAR + camera (40 lines)
6. **Sensor-Triggered Behavior** - Change action based on sensor reading (25 lines)

**Showcase**: "Real perception-driven behaviors"

---

### LEVEL 1E: Rewards ✅ DONE
**File**: `level_1e_basic_rewards.py`
**Status**: ✅ Already refactored (10 tests, 434 lines)
**Tests**: Self-validating actions through rewards
**Showcase**: "Rewards PROVE actions worked - no manual validation!"

---

### LEVEL 1F: Scene Operations
**File**: `level_1f_scene_operations.py`
**Current**: 16 infrastructure tests, 737 lines
**Target**: 5-6 REAL scene building, ~400 lines

**NEW Tests (Real Scenarios):**
1. **Kitchen Scene** - Table, counters, objects on surfaces (25 lines)
2. **Obstacle Course** - Scattered objects robot must navigate around (30 lines)
3. **Shelf with Items** - Multiple objects at different heights (25 lines)
4. **Nested Containers** - Bowl on table, objects in bowl (3-level nesting) (20 lines)
5. **Dynamic Scene** - Objects can roll, fall, collide (physics validation) (35 lines)
6. **Multi-Room Environment** - Robot navigates between areas (40 lines)

**Showcase**: "Build complex realistic environments with declarative API"

---

### LEVEL 1G: View System
**File**: `level_1g_view_system.py`
**Current**: 8 infrastructure tests, 662 lines
**Target**: 4-5 REAL observation scenarios, ~350 lines

**NEW Tests (Real Scenarios):**
1. **Multi-Angle Object Tracking** - 3 cameras track object during manipulation (30 lines)
2. **Navigation with Vision** - Camera views guide robot through space (35 lines)
3. **Manipulation Verification** - Cameras capture before/after grasp (25 lines)
4. **Time-Lapse Validation** - Views prove stability over 100+ steps (30 lines)
5. **Complete Scene Capture** - Views from all modals (robot + scene state) (40 lines)

**Showcase**: "Vision proves behaviors work - TIME TRAVELER ready!"

---

### LEVEL 1H: Spatial Relations ✅ ALREADY GOOD
**File**: `level_1h_spatial_relations.py`
**Current**: 6 tests, 409 lines (multi-modal validation)
**Target**: 6-7 tests, ~450 lines (keep current + maybe 1 complex scenario)
**Status**: ✅ Minimal changes needed
**Showcase**: "Spatial understanding - Physics + Vision proof"

---

### LEVEL 1I: Object Behaviors
**File**: `level_1i_object_behaviors.py`
**Current**: 7 behavior tests, 868 lines
**Target**: 5 REAL behavior scenarios, ~500 lines

**NEW Tests (Real Scenarios):**
1. **Grasp Graspable** - Find and pick up object using "graspable" behavior (30 lines)
2. **Fill Container** - Place object inside bowl using "container" behavior (35 lines)
3. **Stack on Surface** - Build tower using "surface" + "stackable" behaviors (40 lines)
4. **Roll Sphere** - Push ball, physics validates "rollable" behavior (25 lines)
5. **Nested Behaviors** - Bowl (container+surface) holds apple (graspable), on table (45 lines)

**Showcase**: "Behaviors create emergent interactions"

---

### LEVEL 1K: Object Placement
**File**: `level_1k_object_placement.py`
**Current**: 13 placement tests, 1244 lines (LONGEST!)
**Target**: 5-6 REAL placement scenarios, ~500 lines

**NEW Tests (Real Scenarios):**
1. **Build Kitchen Table Scene** - Plate, cup, utensils correctly placed (30 lines)
2. **Stack Blocks Tower** - 4-5 objects stacked with stability validation (35 lines)
3. **Populate Shelf** - Multiple objects at different heights (30 lines)
4. **Create Obstacle Field** - Random object placement, test navigation (35 lines)
5. **Nested Containers Setup** - Box contains bowl contains apple (30 lines)
6. **Dynamic Placement Validation** - Objects stay stable over time (40 lines)

**Showcase**: "Complex placement with one-line API calls"

---

### LEVEL 1L: Composite Rewards
**File**: `level_1l_composite_rewards.py`
**Current**: 10 logic gate tests, 655 lines
**Target**: 5-6 REAL multi-objective tasks, ~450 lines

**NEW Tests (Real Scenarios):**
1. **Grasp AND Lift Task** - Both must succeed (AND gate) (30 lines)
2. **Reach Apple OR Banana** - Either succeeds (OR gate) (25 lines)
3. **Avoid Obstacles** - NOT(collision) while navigating (NOT gate) (30 lines)
4. **Sequential Manipulation** - Grasp → lift → navigate → place (chain) (40 lines)
5. **Complex Task Logic** - (grasp AND lift) OR (navigate AND avoid) + time limits (50 lines)
6. **Multi-Objective Optimization** - Balance speed vs safety vs success (45 lines)

**Showcase**: "Complex task logic self-validates through reward gates"

---

### LEVEL 1M: Final Boss ✅ DONE
**File**: `level_1m_final_boss.py`
**Status**: ✅ Already refactored (5 tests, 432 lines)
**Tests**: Ultimate showcase of all capabilities
**Showcase**: "5-50 lines achieves production-ready scenarios"

---

### LEVEL 1N: Grasping (IK System)
**File**: `level_1n_grasping.py`
**Current**: 6 IK tests, 476 lines
**Target**: 5 REAL whole-body manipulation, ~400 lines

**NEW Tests (Real Scenarios):**
1. **Reach High Shelf** - Full body coordination (base+lift+arm+wrist) (35 lines)
2. **Grasp from Floor** - Low reach requiring base adjustment (30 lines)
3. **Reach Around Obstacle** - IK finds path around blocking object (40 lines)
4. **Bimanual-Like Sequence** - Use gripper, adjust, re-grasp (complex) (35 lines)
5. **Long-Reach Task** - Maximum extension with stability validation (30 lines)
6. **Adaptive Grasping** - Try multiple poses if first fails (40 lines)

**Showcase**: "IK enables complex manipulation automatically"

---

### LEVEL 1O: Scene Composition ✅ ALREADY GOOD
**File**: `level_1o_scene_composition.py`
**Current**: 4 time-series tests, 347 lines
**Target**: 4-5 tests, ~400 lines (keep current + enhance)
**Status**: ✅ Minimal changes needed
**Showcase**: "Scene stability over time - physics converges"

---

### LEVEL 1O: Realistic Integration
**File**: `level_1o_realistic_integration.py`
**Current**: 1 giant test, 445 lines
**Target**: 4 REAL complete scenarios, ~600 lines

**NEW Tests (Real Scenarios):**
1. **Delivery Task** - Navigate to pickup, grasp, deliver, place (60 lines)
2. **Clean Room Task** - Pick up scattered objects, place in container (70 lines)
3. **Fetch Task** - Search for object, navigate, grasp, return (65 lines)
4. **Sorting Task** - Pick objects, categorize, place in correct bins (80 lines)

**Showcase**: "End-to-end real robotics tasks with complete validation"

---

## Total Impact

### Before Refactor:
- **Files**: 14 levels
- **Tests**: ~132 tests (mostly system/infrastructure)
- **Lines**: ~11,000 lines
- **Focus**: "Does feature X work?"

### After Refactor:
- **Files**: 14 levels
- **Tests**: 70-75 tests (all REAL use cases)
- **Lines**: ~5,500 lines (50% reduction!)
- **Focus**: "Can robot do complex task X?"

**Reduction**: ~45% fewer tests, ~50% fewer lines, 100% more REAL

---

## Key MOP Patterns in ALL Tests

### 1. Multi-Modal Validation
```python
# ===== LAYER 1: PHYSICS VALIDATION =====
assert abs(extension - 0.3) < 0.05

# ===== LAYER 2: SEMANTIC VALIDATION =====
assert state["stretch.arm"]["at_target"] == True

# ===== LAYER 3: VISION VALIDATION =====
cv2.imwrite("proof.png", camera_view["rgb"])

# ===== LAYER 4: REASONING VALIDATION =====
assert physics_valid and semantic_valid

# ===== LAYER 5: ACTION IMAGE =====
assert not np.array_equal(before_img, after_img)
```

### 2. Self-Validation Through Rewards
```python
ops.add_reward("stretch.gripper", "holding", target="apple", reward=100)
total_reward = sum(ops.step() for _ in range(500))
assert total_reward >= 100  # Reward PROVES task succeeded!
```

### 3. Declarative API (Say What, Not How)
```python
# 1 line creates complex scene!
ops.add_asset("apple", relative_to="table", relation="on_top")
```

### 4. Low Code, Complex Behavior
```python
# 5-50 lines per test
# But each test does REAL robotics task
# (navigate, grasp, manipulate, validate)
```

---

## Implementation Order

### Phase 1: Aggressive Consolidation (4 files)
1. level_1b_action_system.py (18 → 6 tests)
2. level_1c_action_queues.py (19 → 5 tests)
3. level_1k_object_placement.py (13 → 6 tests)
4. level_1f_scene_operations.py (16 → 6 tests)

### Phase 2: Moderate Consolidation (2 files)
5. level_1i_object_behaviors.py (7 → 5 tests)
6. level_1o_realistic_integration.py (1 → 4 tests)

### Phase 3: Enhance/Simplify (5 files)
7. level_1d_sensor_system.py (13 → 6 tests)
8. level_1g_view_system.py (8 → 5 tests)
9. level_1l_composite_rewards.py (10 → 6 tests)
10. level_1n_grasping.py (6 → 6 tests, enhance)
11. level_1h_spatial_relations.py (6 → 7 tests, minimal)
12. level_1o_scene_composition.py (4 → 5 tests, minimal)

### Phase 4: Already Done ✅
- level_1e_basic_rewards.py ✅
- level_1m_final_boss.py ✅

---

## Success Criteria

Each refactored test file must:
- ✅ Show REAL robotics scenarios (not infrastructure tests)
- ✅ Demonstrate low code (5-50 lines per test)
- ✅ Self-validate through rewards + vision
- ✅ Use multi-modal validation where appropriate
- ✅ Save visual proof (camera images)
- ✅ Showcase MOP philosophy throughout

---

## Example Transformation

### BEFORE (System Test):
```python
def test_arm_extension_exists():
    """Test: Can we track arm extension?"""
    ops.add_robot("stretch")
    ops.compile()
    state = ops.get_state()

    # Just checks if feature exists
    assert "stretch.arm" in state
    assert "extension" in state["stretch.arm"]
    print("✅ Arm extension trackable")
```
**Focus**: Infrastructure ("does X exist?")
**Lines**: 10 lines, trivial

---

### AFTER (Real Use Case):
```python
def test_reach_and_grasp_apple():
    """Real Scenario: Apple on table, robot reaches and grasps it

    Showcases:
    - Scene composition (1 line)
    - Self-validating rewards
    - Multi-modal validation
    - Complex manipulation in <30 lines
    """
    ops = ExperimentOps(headless=True, render_mode="vision_rl")
    ops.create_scene("kitchen", width=5, length=5)
    ops.add_robot("stretch")

    # Build scene (declarative!)
    ops.add_asset("table", relative_to=(2, 0, 0))
    ops.add_asset("apple", relative_to="table", relation="on_top")

    # Add camera for vision validation
    ops.add_free_camera("proof_cam", lookat=(2,0,0.8), distance=2)

    # Self-validating reward
    ops.add_reward("stretch.gripper", "holding", target="apple", reward=100)
    ops.compile()

    # Execute complex manipulation
    ops.submit_block(reach_and_grasp(target="apple"))

    total_reward = 0
    for _ in range(500):
        result = ops.step()
        total_reward += result.get('reward', 0.0)

    # ===== MULTI-MODAL VALIDATION =====
    # 1. PHYSICS: Check gripper force
    state = ops.get_state()
    force = state["stretch.gripper"]["force"]
    assert force > 0.1, "Physics validates grasp"

    # 2. SEMANTIC: Reward triggered
    assert total_reward >= 100, "Reward validates behavior"

    # 3. VISION: Save proof
    img = ops.engine.last_views["proof_cam_view"]["rgb"]
    cv2.imwrite("apple_grasped.png", img)

    print("✅ Real task complete: Robot grasped apple from table!")
    print(f"   Physics ✓  Semantic ✓  Vision ✓")
```
**Focus**: Real manipulation task
**Lines**: 30 lines, COMPLEX behavior
**Validation**: Multi-modal proof

---

## Vision for Each Level

**Level 1A**: Foundation (infrastructure - needed)
**Level 1B**: Real manipulation workflows
**Level 1C**: Real task orchestration
**Level 1D**: Real perception-driven behaviors
**Level 1E**: Self-validation through rewards ✅
**Level 1F**: Build complex realistic scenes
**Level 1G**: Vision proves everything (TIME TRAVELER)
**Level 1H**: Spatial understanding with proof
**Level 1I**: Behavior-driven emergent interactions
**Level 1K**: One-line complex placement
**Level 1L**: Multi-objective task logic
**Level 1M**: Ultimate integration showcase ✅
**Level 1N**: Whole-body IK manipulation
**Level 1O (composition)**: Time-series stability
**Level 1O (integration)**: End-to-end real tasks

---

**Total Vision**: Every level shows "LOW CODE → COMPLEX BEHAVIOR → SELF-VALIDATES"

This is Modal-Oriented Programming!

# SYNTAX GUIDE: Correct Robot OS API Usage

This document provides the **CORRECT** syntax for Robot OS APIs, extracted from working test files.

**Reference files** (these tests PASS):
- `level_1b_action_system.py` ✅
- `level_1e_basic_rewards.py` ✅
- `level_1h_spatial_relations.py` ✅
- `level_1i_object_behaviors.py` ✅

---

## 1. Reward System Syntax

### ✅ CORRECT: Short Form (Positional Arguments)

```python
# Pattern: ops.add_reward(asset, behavior, threshold, reward=X, id="name")

# Actuator rewards
ops.add_reward("stretch.arm", "extension", 0.3, reward=100, id="arm_extended")
ops.add_reward("stretch.lift", "height", 0.5, reward=100, id="lift_raised")
ops.add_reward("stretch.base", "rotation", 90.0, reward=100, id="rotated_90")
ops.add_reward("stretch.gripper", "closed", True, reward=100, id="gripper_closed")

# Asset-to-asset rewards
ops.add_reward("stretch.base", "distance_to", target="apple",
               threshold=2.5, reward=100, id="near_apple")
ops.add_reward("apple", "height_above", target="floor",
               threshold=0.5, reward=100, id="off_floor")

# With dependencies
ops.add_reward("stretch.arm", "extension", 0.3, reward=50, id="step1")
ops.add_reward("stretch.lift", "height", 0.5, reward=50,
               requires="step1", id="step2")
```

### ✅ CORRECT: Long Form (Named Arguments)

```python
# Pattern: ops.add_reward(tracked_asset="...", behavior="...", threshold=..., ...)

ops.add_reward(
    tracked_asset="stretch.arm",
    behavior="extension",
    threshold=0.3,
    reward=100,
    mode="smooth",
    id="arm_ok"
)

ops.add_reward(
    tracked_asset="stretch.base",
    behavior="rotation",
    threshold=90.0,
    reward=100,
    mode="smooth",
    id="rotate_90"
)
```

### ❌ WRONG: Common Mistakes

```python
# ❌ WRONG: Using "position" for head_pan (use "angle_rad" instead!)
ops.add_reward("stretch.head_pan", "position", 0.0, reward=30, id="scanned")

# ✅ CORRECT: Use "angle_rad" for head actuators
ops.add_reward("stretch.head_pan", "angle_rad", 0.0, reward=30, id="scanned")

# ❌ WRONG: Passing tuple for position (should be single number or use distance_to)
ops.add_reward("stretch.base", "position", (None, 5.0, None), reward=50, id="moved")

# ✅ CORRECT: Use "distance_to" with target for spatial rewards
ops.add_reward("stretch.base", "distance_to", target="goal", threshold=5.0, reward=50, id="moved")

# ❌ WRONG: Using degrees for rotation (system uses RADIANS!)
ops.add_reward("stretch.base", "rotation", 90.0, reward=100, id="rotated")  # This is 90 radians!

# ✅ CORRECT: Convert degrees to radians OR check working tests for actual values
import math
ops.add_reward("stretch.base", "rotation", math.radians(90), reward=100, id="rotated")

# ❌ WRONG: Wrong parameter names
ops.add_reward(asset="stretch.arm", property="extension", value=0.3)  # Wrong param names!
```

---

## 2. Available Behaviors (Properties) by Asset Type

**Source**: `/simulation_center/core/behaviors/ROBOT_BEHAVIORS.json`

### Stretch Robot Actuators

```python
# Arm
"stretch.arm"
  - "extension" : float (0.0 to 0.52m)
  - "extended" : bool (True when extended past threshold, default 0.3)

# Lift
"stretch.lift"
  - "height" : float (0.0 to 1.1m)
  - "raised" : bool (True when raised past threshold, default 0.5)

# Base
"stretch.base"
  - "rotation" : float (radians, not degrees!)
  - "position" : float (m/s) - velocity-based, AVOID for rewards
  - "at_location" : float (meters, for distance checking)
  - "distance_to" : requires target= parameter (use this for spatial rewards!)

# Gripper
"stretch.gripper"
  - "aperture" : float (meters)
  - "closed" : bool (True/False)
  - "holding" : bool (requires target= parameter)

# Head Pan
"stretch.head_pan"
  - "angle_rad" : float (radians, NOT "position"!)

# Head Tilt
"stretch.head_tilt"
  - "angle_rad" : float (radians, NOT "position"!)

# Wrist Actuators
"stretch.wrist_yaw", "stretch.wrist_pitch", "stretch.wrist_roll"
  - "angle_rad" : float (radians)
```

### Scene Objects/Assets

```python
# Any asset (apple, table, etc.)
"asset_name"
  - "height_above" : requires target= parameter (e.g., target="floor")
  - "distance_to" : requires target= parameter (e.g., target="table")
  - "position" : tuple (x, y, z)
```

### Room Components

```python
# Room parts are automatically tracked
"floor"
  - "position" : tuple

"wall_north", "wall_south", "wall_east", "wall_west"
  - "position" : tuple
```

---

## 3. Action Block Syntax

### ✅ CORRECT: Creating Action Blocks

```python
# Simple action block
from core.modals.stretch.action_modals import ArmMoveTo, ActionBlock

block = ActionBlock(
    id="extend_arm",
    actions=[ArmMoveTo(position=0.3)]
)
ops.submit_block(block)

# Sequential execution
block = ActionBlock(
    id="sequential_task",
    execution_mode="sequential",  # Actions wait for each other
    actions=[
        ArmMoveTo(position=0.3),
        LiftMoveTo(position=0.5),
        GripperMoveTo(position=-0.1)
    ]
)

# Parallel execution
block = ActionBlock(
    id="parallel_task",
    execution_mode="parallel",  # Actions run simultaneously
    actions=[
        ArmMoveTo(position=0.3),
        LiftMoveTo(position=0.5),
        BaseMoveTo(rotation=45.0)
    ]
)

# With priority
block = ActionBlock(
    id="emergency",
    actions=[ArmMoveTo(position=0.0)],
    priority=10  # Higher priority interrupts lower
)

# Replace current actions
block = ActionBlock(
    id="emergency_stop",
    actions=[ArmMoveTo(position=0.0)],
    replace_current=True  # Cancels all current actions
)
```

### ✅ CORRECT: Registry Actions (Pre-built)

```python
from core.modals.stretch.action_blocks_registry import move_forward, spin_left

# These are already ActionBlocks, just submit them
forward_block = move_forward(distance=2.0, speed=0.3)
ops.submit_block(forward_block)

rotate_block = spin_left(degrees=90, speed=6.0)
ops.submit_block(rotate_block)
```

### ❌ WRONG: Common Mistakes

```python
# ❌ WRONG: Trying to import extend_arm, raise_lift (don't exist!)
from core.modals.stretch.action_blocks_registry import extend_arm, raise_lift

# ✅ CORRECT: Create them manually
block = ActionBlock(id="extend", actions=[ArmMoveTo(position=0.3)])
```

---

## 4. Scene Creation Syntax

### ✅ CORRECT: Scene Creation

```python
# MUST include height!
ops.create_scene("my_scene", width=6, length=6, height=3)

# With textures
ops.create_scene("my_scene", width=5, length=5, height=3,
                floor_texture="floor_tiles",
                wall_texture="concrete")
```

### ❌ WRONG: Common Mistakes

```python
# ❌ WRONG: Missing height parameter
ops.create_scene("my_scene", width=6, length=6)  # Will fail!

# ✅ CORRECT
ops.create_scene("my_scene", width=6, length=6, height=3)
```

---

## 5. Asset Management Syntax

### ✅ CORRECT: Adding Assets

```python
# Absolute position
ops.add_asset("table", relative_to=(2, 0, 0))

# Relative to another asset
ops.add_asset("apple", relative_to="table", relation="on_top")

# Stacking
ops.add_asset("box1", relative_to="table", relation="on_top")
ops.add_asset("box2", relative_to="box1", relation="stack_on")

# Inside container
ops.add_asset("apple", relative_to="bowl", relation="inside")

# Robot position
ops.add_robot("stretch", position=(0, 0, 0))

# Robot with initial state
ops.add_robot("stretch", position=(0, 0, 0),
              initial_state={'arm': 0.4, 'lift': 0.8})
```

---

## 6. Camera Syntax

### ✅ CORRECT: Adding Cameras

```python
# Free camera (not attached)
ops.add_free_camera("my_cam",
                   lookat=(2, 0, 0.8),  # Point to look at
                   distance=3,           # Distance from lookat
                   azimuth=45,          # Horizontal angle
                   elevation=20)         # Vertical angle (optional)

# Multiple cameras
ops.add_free_camera("cam1", lookat=(2,0,0.8), distance=2, azimuth=45)
ops.add_free_camera("cam2", lookat=(2,0,0.8), distance=3, azimuth=135)
```

### ✅ CORRECT: Accessing Camera Views

```python
# After compile and step
ops.compile()
ops.step()

# Get camera view
img = ops.engine.last_views["my_cam_view"]["rgb"]  # Note: "_view" suffix!
depth = ops.engine.last_views["my_cam_view"]["depth"]

# Save image
import cv2
cv2.imwrite("proof.png", img)
```

---

## 7. State Access Syntax

### ✅ CORRECT: Getting State

```python
state = ops.get_state()

# Actuator state
arm_ext = state["stretch.arm"]["extension"]
lift_height = state["stretch.lift"]["height"]
base_rot = state["stretch.base"]["rotation"]
gripper_closed = state["stretch.gripper"]["closed"]

# Asset state
apple_pos = state["apple"]["position"]
table_pos = state["table"]["position"]

# Check if behavior exists
at_target = state["stretch.arm"].get("at_target", False)
```

---

## 8. Action Block Status & Progress

### ✅ CORRECT: Checking Completion

```python
# Submit block
block = ActionBlock(id="my_action", actions=[ArmMoveTo(position=0.3)])
ops.submit_block(block)

# Check status in loop
for _ in range(500):
    ops.step()

    # Check completion
    if block.status == "completed":
        print("Action done!")
        break

    # Check progress
    print(f"Progress: {block.progress}%")
```

### ✅ CORRECT: 4-Way MOP Validation

```python
# After action completes, validate with 4 checks:

# 1. Reward triggered
assert total_reward >= 100

# 2. Block status
assert block.status == "completed"

# 3. Block progress
assert block.progress >= 100

# 4. Queue state
queue_empty = len(ops.queue_manager.action_queue) == 0
assert queue_empty
```

---

## 9. Experiment Initialization

### ✅ CORRECT: ExperimentOps Setup

```python
# Headless (no viewer)
ops = ExperimentOps(headless=True)

# With vision/RL rendering
ops = ExperimentOps(headless=True, render_mode="vision_rl")

# With viewer window
ops = ExperimentOps(headless=False, render_mode="rl_core")

# With custom save settings
ops = ExperimentOps(headless=True, save_fps=30)

# With mode
ops = ExperimentOps(mode="simulated", headless=True)
```

---

## 10. Common Patterns from Working Tests

### Pattern 1: Simple Actuator Movement with Reward

```python
ops = ExperimentOps(headless=True)
ops.create_scene("test", width=5, length=5, height=3)
ops.add_robot("stretch")
ops.add_reward("stretch.arm", "extension", 0.3, reward=100, id="extended")
ops.compile()

block = ActionBlock(id="extend", actions=[ArmMoveTo(position=0.3)])
ops.submit_block(block)

total_reward = 0
for _ in range(500):
    result = ops.step()
    total_reward += result.get('reward', 0.0)
    if block.status == 'completed':
        break

assert total_reward >= 100  # Self-validated!
```

### Pattern 2: Distance-Based Reward

```python
ops.add_robot("stretch", position=(0, 0, 0))
ops.add_asset("apple", relative_to=(2, 0, 0))

# Asset linking through reward
ops.add_reward("stretch.base", "distance_to", target="apple",
               threshold=1.5, reward=100, id="near_apple")
ops.compile()

# Move toward apple
block = move_forward(distance=1.0)
ops.submit_block(block)

for _ in range(1000):
    result = ops.step()
    # Check reward...
```

### Pattern 3: Sequential Dependencies

```python
# Step 1 must complete before step 2
ops.add_reward("stretch.base", "distance_to", target="table",
               threshold=1.0, reward=50, id="navigated")
ops.add_reward("stretch.arm", "extension", 0.4,
               reward=50, requires="navigated", id="reached")
ops.add_reward("stretch.gripper", "holding", target="apple",
               threshold=True, reward=100, requires="reached", id="grasped")
```

### Pattern 4: Multi-Modal Validation

```python
# ===== 5-WAY VALIDATION =====

# 1. Physics
arm_ext = state["stretch.arm"]["extension"]
assert abs(arm_ext - 0.3) < 0.05

# 2. Semantic (reward)
assert total_reward >= 100

# 3. Vision
img = ops.engine.last_views["cam_view"]["rgb"]
cv2.imwrite("proof.png", img)

# 4. Reasoning
reasoning_valid = (arm_ext > 0.25) and (total_reward >= 100)
assert reasoning_valid

# 5. Action Image
images_different = not np.array_equal(before_img, after_img)
assert images_different
```

---

## 11. Available Atomic Actions

**Source**: `/simulation_center/core/modals/stretch/action_modals.py`

### All Atomic Actions Available

```python
from core.modals.stretch.action_modals import (
    # Arm actions
    ArmMoveTo, ArmMoveBy,

    # Lift actions
    LiftMoveTo, LiftMoveBy,

    # Gripper actions
    GripperMoveTo, GripperMoveBy,

    # Head actions
    HeadPanMoveTo, HeadPanMoveBy,
    HeadTiltMoveTo, HeadTiltMoveBy,

    # Wrist actions
    WristYawMoveTo, WristYawMoveBy,
    WristPitchMoveTo, WristPitchMoveBy,
    WristRollMoveTo, WristRollMoveBy,

    # Base actions
    BaseMoveForward, BaseMoveBackward,
    BaseRotateBy, BaseMoveTo,

    # Speaker
    SpeakerPlay
)
```

### Action Usage Patterns

```python
# Pattern: <Actuator>MoveTo(position=value)
arm_action = ArmMoveTo(position=0.3)  # Move arm to 0.3m extension
lift_action = LiftMoveTo(position=0.5)  # Move lift to 0.5m height
gripper_action = GripperMoveTo(position=-0.1)  # Close gripper

# Pattern: <Actuator>MoveBy(delta=value)
arm_action = ArmMoveBy(delta=0.1)  # Extend arm by 0.1m more
head_action = HeadPanMoveBy(delta=0.5)  # Rotate head by 0.5 radians

# Base actions (special cases)
forward = BaseMoveForward(distance=2.0, speed=0.3)
rotate = BaseRotateBy(angle=1.57, speed=6.0)  # ~90 degrees in radians
```

---

## 12. ActionBlock Analysis & Queue Monitoring

### ActionBlock Properties (Self-Reporting!)

**MOP Pattern**: ActionBlocks self-report their status and progress from their actions.

```python
from core.modals.stretch.action_modals import ActionBlock, ArmMoveTo

# Create ActionBlock
block = ActionBlock(
    id="my_task",
    execution_mode="sequential",  # or "parallel"
    actions=[
        ArmMoveTo(position=0.3),
        LiftMoveTo(position=0.5)
    ]
)

# Submit to queue
ops.submit_block(block)

# MOP: Block self-reports status and progress!
print(f"Status: {block.status}")        # 'pending', 'executing', 'completed', 'failed'
print(f"Progress: {block.progress}%")   # 0.0 to 100.0

# Status logic:
# - 'failed' if ANY action failed
# - 'executing' if ANY action executing
# - 'completed' if ALL actions completed
# - 'pending' otherwise

# Progress: Average of all action progress values
```

### Monitoring ActionBlock During Execution

```python
# Pattern 1: Simple status check
for step in range(1000):
    result = ops.step()

    if block.status == "completed":
        print("Action done!")
        break

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}: {block.status}, {block.progress}%")

# Pattern 2: Full 4-way MOP validation
total_reward = 0
for step in range(1000):
    result = ops.step()
    total_reward += result.get('reward', 0.0)

    # Check ALL 4 conditions
    if (total_reward >= 100 and
        block.status == "completed" and
        block.progress >= 100):
        print("✅ FULLY VALIDATED!")
        break
```

### Queue Analysis (MOP - Easy to Inspect!)

```python
# Access queue manager
queue = ops.queue_manager.action_queue

# Check queue status
print(f"Queue length: {len(queue)}")
print(f"Queue empty: {len(queue) == 0}")

# Inspect queue contents
for i, block in enumerate(queue):
    print(f"Block {i}: {block.id}")
    print(f"  Status: {block.status}")
    print(f"  Progress: {block.progress}%")
    print(f"  Actions: {len(block.actions)}")

    # Inspect individual actions
    for j, action in enumerate(block.actions):
        print(f"    Action {j}: {action.__class__.__name__}")
        print(f"      Status: {action.status}")
        print(f"      Progress: {action.progress}%")

# Check if specific block is in queue
is_queued = block in ops.queue_manager.action_queue
```

### ActionBlock Configuration Options

```python
# Sequential execution (default)
block = ActionBlock(
    id="sequential_task",
    execution_mode="sequential",  # Actions run one after another
    actions=[ArmMoveTo(position=0.3), LiftMoveTo(position=0.5)]
)

# Parallel execution
block = ActionBlock(
    id="parallel_task",
    execution_mode="parallel",  # Actions run simultaneously
    actions=[ArmMoveTo(position=0.3), LiftMoveTo(position=0.5)]
)

# Priority execution
block = ActionBlock(
    id="urgent_task",
    push_before_others=True,  # Jump to front of queue
    actions=[ArmMoveTo(position=0.0)]
)

# Replace current execution
block = ActionBlock(
    id="emergency_stop",
    replace_current=True,  # Cancel all current actions
    actions=[ArmMoveTo(position=0.0)]
)
```

### Queue Analysis is MOP-Easy!

The queue gives you **ALL information** you need:

✅ **Block level**: `block.status`, `block.progress`, `block.id`
✅ **Action level**: `action.status`, `action.progress`, `action.__class__.__name__`
✅ **Queue level**: `len(queue)`, iterate through all blocks
✅ **Self-reporting**: No complex analysis needed - modals tell you everything!

**Example: Complete queue analysis**
```python
def analyze_queue(ops):
    """Complete queue analysis - MOP makes it easy!"""
    queue = ops.queue_manager.action_queue

    print(f"\n📊 QUEUE ANALYSIS")
    print(f"Total blocks: {len(queue)}")

    for i, block in enumerate(queue):
        print(f"\nBlock {i+1}: {block.id}")
        print(f"  Status: {block.status}")
        print(f"  Progress: {block.progress:.1f}%")
        print(f"  Mode: {block.execution_mode}")
        print(f"  Actions: {len(block.actions)}")

        for j, action in enumerate(block.actions):
            print(f"    {j+1}. {action.__class__.__name__}: "
                  f"{action.status} ({action.progress:.1f}%)")

    if len(queue) == 0:
        print("  ✅ Queue empty - all tasks completed!")

# Use it
analyze_queue(ops)
```

---

## Summary of Key Rules

1. **Always include `height` in `create_scene()`**
2. **Use short form reward syntax**: `ops.add_reward(asset, behavior, threshold, reward=X, id="name")`
3. **For asset linking, use `target=` parameter**: `distance_to`, `height_above`, `holding`
4. **Threshold must be numeric or boolean**, never tuple
5. **Camera views have `_view` suffix**: `"my_cam_view"`, not `"my_cam"`
6. **ActionBlock IDs must be unique**
7. **Use `tracked_asset=` in long form, NOT `asset=`**
8. **Head actuators need specific behaviors, NOT generic `"position"`**
9. **`extend_arm()` and `raise_lift()` DON'T exist** - create ActionBlocks manually
10. **Always compile before stepping**: `ops.compile()` then `ops.step()`

---

## When in Doubt, Copy from These Working Tests

- **Level 1B** (`level_1b_action_system.py`): Actuator control patterns
- **Level 1E** (`level_1e_basic_rewards.py`): Reward system patterns
- **Level 1H** (`level_1h_spatial_relations.py`): Spatial relationship patterns
- **Level 1I** (`level_1i_object_behaviors.py`): Object behavior patterns

These files have PASSING tests - their syntax is CORRECT!
---

## 13. ops.step() Return Value - CRITICAL

**CRITICAL: The return dict structure from `ops.step()`**

```python
result = ops.step()
# result is a dict with these keys:
# - 'reward_step': float (delta for THIS step - use for RL accumulation!)
# - 'reward_total': float (cumulative total for display)
# - 'rewards': dict (per-reward breakdown)
# - 'step_count': int
# - 'state': dict
# - 'elapsed_time': float

# ✅ CORRECT: Use 'reward_step' for accumulation
total_reward = 0
for _ in range(500):
    result = ops.step()
    total_reward += result['reward_step']  # CORRECT!

# ✅ ALSO CORRECT: Comprehension style
total_reward = sum(ops.step()['reward_step'] for _ in range(500))

# ❌ WRONG: 'reward' key doesn't exist!
total_reward += result['reward']  # KeyError!

# ❌ VERY WRONG: Using .get() hides bugs silently!
total_reward += result.get('reward', 0.0)  # Returns 0.0, masks bug!
```

**Why this matters:**
- Using `.get('reward', 0.0)` will SILENTLY return 0.0 because 'reward' key doesn't exist
- This hides bugs and makes debugging impossible
- ALWAYS use explicit `['reward_step']` to crash immediately if key is wrong

---

## 14. CRITICAL: Spatial vs Non-Spatial Behaviors

**IMPORTANT: Some modals have BOTH regular and `_spatial` versions!**

```python
# ❌ WRONG: robot_base doesn't have "distance_to"!
ops.add_reward("stretch.base", "distance_to", target="apple", ...)

# ✅ CORRECT: robot_base has "at_location" (boolean)
ops.add_reward("stretch.base", "at_location", target="apple", threshold=1.0, ...)

# ✅ ALSO CORRECT: Use spatial modal for distance tracking
# (Note: This may require spatial modal to be enabled)
ops.add_reward("stretch.base_spatial", "distance_to", target="apple", ...)
```

**Available robot_base behaviors** (from ROBOT_BEHAVIORS.json):
- `"position"` - velocity (m/s) - for movement speed
- `"rotation"` - angle (radians) - for orientation  
- `"at_location"` - boolean (meters threshold) - for "near target" check

**Spatial variants** (if available):
- `robot_arm_spatial`: Has `"position"` (XYZ) and `"distance_to"`
- `robot_gripper_spatial`: Has `"position"` (XYZ) and `"distance_to"`  
- `robot_head_spatial`: Has `"position"` (XYZ) and `"distance_to"`

**Rule**: ALWAYS check ROBOT_BEHAVIORS.json to see what properties exist!

# SMART SCENE COMPOSITION
## Build Scenes + Define Rewards - The Complete Guide

---

## What is Smart Scene Composition?

**Smart Scene Composition** is how you build simulation environments and define task goals in a unified, declarative way.

**Two sides of the same coin:**
1. **Scene** - Physical world (robots, objects, room)
2. **Rewards** - Task goals (what you want to happen)

Both use the same **uniform asset interface** - everything is trackable!

---

## Table of Contents

**PART 1: BUILDING SCENES**
1. [Creating Rooms](#1-creating-rooms)
2. [Adding Robots](#2-adding-robots)
3. [Adding Objects](#3-adding-objects)
4. [Relative Positioning](#4-relative-positioning)
5. [Room Components as Assets](#5-room-components-as-assets)

**PART 2: DEFINING REWARDS**
6. [Basic Rewards](#6-basic-rewards)
7. [Asset Linking](#7-asset-linking-target-parameter)
8. [Room Component Tracking](#8-room-component-tracking)
9. [Composite Conditions](#9-composite-conditions)
10. [Sequential Rewards](#10-sequential-rewards)
11. [Reward Modes](#11-reward-modes)

**PART 3: VIEWS & INSPECTION**
12. [Scene Views](#12-scene-views)
13. [Reward Views](#13-reward-views)

**PART 4: COMPLETE EXAMPLES**
14. [Real-World Scenarios](#14-real-world-scenarios)

---

# PART 1: BUILDING SCENES

---

## 1. Creating Rooms

Rooms define the environment space.

### Quick Start - Use Default Room

```python
from core.main.scene_ops import create_scene

# Create scene with default 5x5x3m room
scene = create_scene("kitchen")
```

### Create Custom Room

```python
from core.modals.room_modal import RoomModal

# Define room
room = RoomModal(
    name="warehouse",
    width=20,     # 20m wide
    length=20,    # 20m long
    height=5,     # 5m tall
    floor_texture="concrete",
    wall_texture="industrial",
    openings=[
        {"wall": "north", "width_m": 2.0, "height_m": 3.0, "state": "open"},
        {"wall": "east", "width_m": 1.2, "height_m": 2.5, "state": "closed"}
    ]
)

# Save for reuse (modal generates JSON!)
room.save()
# Creates: assets/rooms/warehouse.json

# Use in scene
from core.modals.scene_modal import Scene
scene = Scene(room)
```

### Load Saved Room

```python
# Later, load the generated room
scene = create_scene("warehouse")  # Auto-loads from assets/rooms/warehouse.json
```

### Override Room Properties

```python
# Load room but change dimensions
scene = create_scene("kitchen", width=10, length=8, height=4)
```

---

## 2. Adding Robots

Robots are auto-wrapped as trackable assets.

### Add Full Robot

```python
from core.main.robot_ops import create_robot

robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))

# Robot components now trackable:
# - stretch.arm, stretch.gripper, stretch.lift, etc. (actuators)
# - stretch.nav_camera, stretch.lidar, stretch.imu, etc. (sensors)
```

### Customize Robot Before Adding

```python
# Vision-only task
robot = create_robot("stretch") \
    .remove_actuators(["speaker"]) \
    .sensors_only(["nav_camera", "d405_camera"]) \
    .basic_views_only()

scene.add_robot(robot, relative_to=(0, 0, 0))
```

---

## 3. Adding Objects

Objects are auto-discovered from registry.

### Simple Add

```python
# Add at absolute position
scene.add("table_1", position=(2, 0, 0))  # x=2m, y=0, z=0
scene.add("apple", position=(2, 0, 0.8))  # On top of table
```

### With Initial State

```python
# Set initial joint positions
scene.add("door",
    position=(5, 0, 0),
    initial_state={"door_hinge": 1.57}  # 90 degrees open
)
```

### List Available Objects

```python
from core import registry

print(registry.list_available("objects"))
# ['apple', 'banana', 'cup', 'mug', 'ball', 'box', ...]

print(registry.list_available("furniture"))
# ['table_1', 'chair', 'desk', 'shelf', ...]
```

---

## 4. Relative Positioning

Place objects relative to other assets.

### On Top

```python
# Place apple on table
scene.add("table_1", position=(2, 0, 0))
scene.add("apple", position={
    "relative_to": "table_1",
    "relation": "on_top"
})
```

### In Front / Behind / Left / Right

```python
# Place box in front of robot
scene.add("box", position={
    "relative_to": "stretch.base",
    "relation": "front",
    "distance": 0.5  # 0.5m in front
})

# Place goal marker behind robot
scene.add("goal", position={
    "relative_to": "stretch.base",
    "relation": "back",
    "distance": 2.0
})
```

### In Gripper

```python
# Start with object in gripper
scene.add("tool", position={
    "relative_to": "stretch.gripper",
    "relation": "in_gripper"
})
```

### Inside Container

```python
# Place apple inside basket
scene.add("basket", position=(3, 0, 0.8))
scene.add("apple", position={
    "relative_to": "basket",
    "relation": "inside"
})
```

---

## 5. Room Components as Assets

**Key insight:** Floor, walls, ceiling are trackable assets!

### What Room Declares

```python
# When you create a scene, room auto-declares these components:
scene.assets.keys()
# Includes:
# - "floor"         (behaviors: ["surface", "spatial"])
# - "wall_north"    (behaviors: ["spatial", "room_boundary"])
# - "wall_south"    (behaviors: ["spatial", "room_boundary"])
# - "wall_east"     (behaviors: ["spatial", "room_boundary"])
# - "wall_west"     (behaviors: ["spatial", "room_boundary"])
# - "ceiling"       (behaviors: ["surface", "spatial"])
```

### Use Room Components in Rewards

```python
# Check if object fell on floor
scene.add_reward(
    tracked_asset="floor",
    behavior="contact",
    target="apple",
    threshold=True,
    reward=-10
)

# Check distance from floor (object height!)
scene.add_reward(
    tracked_asset="apple",
    behavior="distance_to",
    target="floor",
    threshold={"above": 0.5},
    reward=20
)
```

---

# PART 2: DEFINING REWARDS

---

## 6. Basic Rewards

### Boolean Condition

```python
# Reward when gripper is closed
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="closed",
    threshold=True,
    reward=10,
    id="gripper_closed"
)
```

### Numeric Threshold

```python
# Reward when arm extends past 0.5m
scene.add_reward(
    tracked_asset="stretch.arm",
    behavior="extension",
    threshold=0.5,
    reward=20,
    id="arm_extended"
)
```

### Range Condition

```python
# Reward when lift is between 0.3m and 0.8m
scene.add_reward(
    tracked_asset="stretch.lift",
    behavior="height",
    threshold={"above": 0.3, "below": 0.8},
    reward=15,
    id="lift_mid_height"
)
```

---

## 7. Asset Linking (Target Parameter)

The `target` parameter links assets together - **this is the magic!**

### Object Manipulation

```python
# Gripper holding specific object
scene.add_reward(
    tracked_asset="stretch.gripper",  # Asset A
    behavior="holding",
    target="apple",                   # Asset B (linked!)
    threshold=True,
    reward=100,
    id="holding_apple"
)
```

### Distance-Based

```python
# Robot within 0.5m of goal
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="goal_marker",
    threshold={"below": 0.5},
    reward=50,
    id="reached_goal"
)
```

### Vision-Based

```python
# Camera sees target
scene.add_reward(
    tracked_asset="stretch.nav_camera",
    behavior="target_visible",
    target="person",
    threshold=True,
    reward=10,
    id="person_detected"
)
```

### Container Checks

```python
# Basket contains apple
scene.add_reward(
    tracked_asset="basket",
    behavior="contains",
    target="apple",
    threshold=True,
    reward=200,
    id="apple_in_basket"
)
```

---

## 8. Room Component Tracking

Floor, walls, ceiling are **trackable assets** - you can link them!

### Object on Floor (Contact)

```python
# Penalty when apple touches floor
scene.add_reward(
    tracked_asset="floor",       # Room component as asset!
    behavior="contact",
    target="apple",              # Object linked to floor
    threshold=True,
    reward=-10,
    id="apple_dropped"
)
```

### Object Height (Distance from Floor)

```python
# Reward for lifting apple off floor
scene.add_reward(
    tracked_asset="apple",
    behavior="distance_to",
    target="floor",              # Floor as target!
    threshold={"above": 0.5},    # More than 0.5m from floor
    reward=30,
    id="apple_lifted"
)
```

### Robot Collision with Wall

```python
# Penalty for hitting wall
scene.add_reward(
    tracked_asset="wall_north",
    behavior="object_contact",
    target="stretch.base",
    threshold=True,
    reward=-5,
    id="wall_collision"
)
```

### Ceiling Impact

```python
# Penalty if object hits ceiling
scene.add_reward(
    tracked_asset="ceiling",
    behavior="contact",
    target="ball",
    threshold=True,
    reward=-20,
    id="ceiling_hit"
)
```

### Contact Force Magnitude

```python
# Track HOW HARD object hit floor
scene.add_reward(
    tracked_asset="floor",
    behavior="contact_force",
    target="box",
    threshold={"above": 50.0},  # Heavy impact (Newtons)
    reward=-5,
    id="box_dropped_hard"
)
```

---

## 9. Composite Conditions

Build complex conditions from simple ones.

### AND - All Must Be True

```python
from core.modals.reward_modal import AND

# Gripper holds apple AND lift is raised
scene.add_reward(
    condition=AND([
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
    ]),
    reward=200,
    id="lifted_apple"
)
```

### OR - Any Can Be True

```python
from core.modals.reward_modal import OR

# Either camera sees apple
scene.add_reward(
    condition=OR([
        {
            "tracked_asset": "stretch.nav_camera",
            "behavior": "target_visible",
            "target": "apple",
            "threshold": True
        },
        {
            "tracked_asset": "stretch.d405_camera",
            "behavior": "target_visible",
            "target": "apple",
            "threshold": True
        }
    ]),
    reward=10,
    id="apple_visible"
)
```

### NOT - Negation

```python
from core.modals.reward_modal import NOT

# Reward when gripper NOT holding anything
scene.add_reward(
    condition=NOT({
        "tracked_asset": "stretch.gripper",
        "behavior": "holding",
        "threshold": True
    }),
    reward=20,
    id="gripper_empty"
)
```

### Nested Composition

```python
# (Apple in basket OR in gripper) AND (NOT on floor)
scene.add_reward(
    condition=AND([
        OR([
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
        ]),
        NOT({
            "tracked_asset": "floor",
            "behavior": "contact",
            "target": "apple",
            "threshold": True
        })
    ]),
    reward=300,
    id="apple_secured"
)
```

---

## 10. Sequential Rewards

Use `requires` to create task sequences.

### Simple Sequence

```python
# Step 1: Pick up apple
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="apple",
    threshold=True,
    reward=50,
    id="picked_apple"
)

# Step 2: Place in basket (only after picking)
scene.add_reward(
    tracked_asset="basket",
    behavior="contains",
    target="apple",
    threshold=True,
    reward=100,
    id="placed_apple",
    requires="picked_apple"  # Only evaluate after "picked_apple"
)
```

### Multi-Step Navigation

```python
# Step 1: Navigate to apple
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="apple",
    threshold={"below": 0.5},
    reward=20,
    id="reached_apple"
)

# Step 2: See apple (after reaching)
scene.add_reward(
    tracked_asset="stretch.nav_camera",
    behavior="target_visible",
    target="apple",
    threshold=True,
    reward=10,
    id="saw_apple",
    requires="reached_apple"
)

# Step 3: Grasp apple (after seeing)
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="apple",
    threshold=True,
    reward=100,
    id="grasped_apple",
    requires="saw_apple"
)
```

---

## 11. Reward Modes

### Auto Mode (Default)

System chooses based on property type:
- Boolean → discrete
- Numeric → smooth

```python
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    threshold=True,
    reward=100,
    mode="auto"  # Default
)
```

### Smooth Mode (Continuous Shaping)

Reward proportional to progress.

```python
# Continuous reward as robot approaches goal
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="goal",
    threshold=0.0,
    reward=100,
    mode="smooth"
)
# Returns: 100 * (1 - distance/max_distance)
```

### Discrete Mode (Sparse, One-Time)

Reward only when threshold first met.

```python
# One-time reward when threshold crossed
scene.add_reward(
    tracked_asset="stretch.lift",
    behavior="height",
    threshold=0.8,
    reward=50,
    mode="discrete"
)
```

---

# PART 3: VIEWS & INSPECTION

---

## 12. Scene Views

Scenes have multiple views for different purposes.

### Asset List View

```python
# See all trackable assets
print(scene.assets.keys())
# dict_keys([
#   'table_1', 'apple', 'basket',                    # Objects
#   'stretch.arm', 'stretch.gripper', 'stretch.lift', # Robot actuators
#   'stretch.nav_camera', 'stretch.lidar',            # Robot sensors
#   'floor', 'wall_north', 'wall_south',              # Room components
#   'wall_east', 'wall_west', 'ceiling'
# ])
```

### Component State View (Data)

```python
# Human-readable component state
gripper = scene.assets["stretch.gripper"]
print(gripper.get_data())
# {
#   "gripper": {
#     "aperture": 0.15,
#     "closed": False,
#     "holding": True,
#     "position": [0.5, 0.3, 0.9],
#     "distance_to": 0.12
#   }
# }
```

### Component State View (RL)

```python
# Flat, normalized vector for RL training
gripper = scene.assets["stretch.gripper"]
rl_obs = gripper.get_rl()
# array([0.75, 0.0, 1.0, 0.5, 0.3, 0.9, 0.12])
# All values normalized using min/max from BEHAVIORS.json
```

### Room View (Data)

```python
# Room dimensions and properties
print(scene.room.get_data())
# {
#   "name": "kitchen",
#   "width": 5.0,
#   "length": 5.0,
#   "height": 3.0,
#   "openings": [
#     {"wall": "north", "width_m": 0.9, "height_m": 2.0, "state": "closed"}
#   ],
#   "floor_texture": "wood_floor",
#   "wall_texture": "gray_wall"
# }
```

### Room View (RL)

```python
# Normalized room dimensions
rl_room = scene.room.get_rl()
# array([0.25, 0.25, 0.6, 1.0])
# [width_norm, length_norm, height_norm, opening_count]
```

### Placement View

```python
# See all object placements
for placement in scene.placements:
    print(f"{placement.name}: {placement.position}")
# table_1: (2, 0, 0)
# apple: (2, 0, 0.8)
# basket: (3, 0, 0.8)
```

---

## 13. Reward Views

Rewards have multiple views for inspection and debugging.

### Reward State View

```python
# Get current state of all rewards
state = scene.reward_modal.get_data()

print(state)
# {
#   "picked_apple": {
#     "satisfied": True,
#     "progress": 1.0,
#     "total_reward": 50,
#     "first_satisfied": 347  # timestep when first triggered
#   },
#   "placed_apple": {
#     "satisfied": False,
#     "progress": 0.6,  # 60% progress toward condition
#     "total_reward": 0
#   },
#   "apple_dropped": {
#     "satisfied": False,
#     "progress": 0.0,
#     "total_reward": 0
#   }
# }
```

### Timeline View

```python
# Visual timeline of when rewards triggered
timeline = scene.reward_modal.timeline()
print(timeline)

# Example output:
# picked_apple:   [----------✓=================]  t=347, reward=+50
# placed_apple:   [------------------------]  (in progress, 60%)
# apple_dropped:  [----------]  (not triggered)
```

### Individual Condition View

```python
# Inspect specific condition
condition = scene.reward_modal.conditions["picked_apple"]
print(f"ID: {condition.id}")
print(f"Satisfied: {condition.was_satisfied}")
print(f"Reward: {condition.reward}")
print(f"Mode: {condition.mode}")
```

---

# PART 4: COMPLETE EXAMPLES

---

## 14. Real-World Scenarios

### Scenario 1: Pick and Place

```python
from core.main.scene_ops import create_scene
from core.main.robot_ops import create_robot

# Build scene
scene = create_scene("kitchen")
robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))

scene.add("table_1", position=(2, 0, 0))
scene.add("apple", position={"relative_to": "table_1", "relation": "on_top"})
scene.add("basket", position=(3, 1, 0.8))

# Define task rewards
# Shaping: Approach apple
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="distance_to",
    target="apple",
    threshold=0.0,
    reward=10,
    mode="smooth",
    id="approaching"
)

# Step 1: Pick up apple
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="apple",
    threshold=True,
    reward=50,
    id="picked"
)

# Step 2: Place in basket
scene.add_reward(
    tracked_asset="basket",
    behavior="contains",
    target="apple",
    threshold=True,
    reward=100,
    id="placed",
    requires="picked"
)

# Safety: Penalty for dropping
scene.add_reward(
    tracked_asset="floor",
    behavior="contact",
    target="apple",
    threshold=True,
    reward=-50,
    id="dropped"
)

# Compile and run
xml_path = scene.compile_scene()
print(f"Scene ready: {xml_path}")
```

### Scenario 2: Door Opening with Safety

```python
# Build scene
scene = create_scene("warehouse", width=10, length=10)
robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))

scene.add("door", position=(5, 0, 0), initial_state={"door_hinge": 0.0})

# Task sequence
# Step 1: Approach door
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="door",
    threshold={"below": 1.0},
    reward=20,
    id="approached_door"
)

# Step 2: Grasp handle
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="door_handle",
    threshold=True,
    reward=50,
    id="grasped_handle",
    requires="approached_door"
)

# Step 3: Open door (> 45 degrees)
scene.add_reward(
    tracked_asset="door",
    behavior="angle",
    threshold={"above": 0.785},  # radians
    reward=100,
    id="door_opened",
    requires="grasped_handle"
)

# Safety: Don't hit door
scene.add_reward(
    condition=AND([
        {
            "tracked_asset": "door",
            "behavior": "contact",
            "target": "stretch.base",
            "threshold": True
        },
        NOT({
            "tracked_asset": "stretch.gripper",
            "behavior": "holding",
            "target": "door_handle",
            "threshold": True
        })
    ]),
    reward=-20,
    id="hit_door"
)

# Safety: Don't hit walls
for wall in ["wall_north", "wall_south", "wall_east", "wall_west"]:
    scene.add_reward(
        tracked_asset=wall,
        behavior="object_contact",
        target="stretch.base",
        threshold=True,
        reward=-10,
        id=f"hit_{wall}"
    )
```

### Scenario 3: Multi-Object Cleanup

```python
# Build scene
scene = create_scene("playroom")
robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))

# Scattered toys
toys = ["toy_car", "toy_block", "toy_ball"]
positions = [(1, 1, 0), (2, -1, 0), (-1, 2, 0)]
for toy, pos in zip(toys, positions):
    scene.add(toy, position=pos)

# Toy box for storage
scene.add("toy_box", position=(3, 3, 0))

# Task: Collect all toys
for toy in toys:
    # Shaping: Approach toy
    scene.add_reward(
        tracked_asset="stretch.gripper",
        behavior="distance_to",
        target=toy,
        threshold=0.0,
        reward=5,
        mode="smooth",
        id=f"approaching_{toy}"
    )

    # Pick up toy
    scene.add_reward(
        tracked_asset="stretch.gripper",
        behavior="holding",
        target=toy,
        threshold=True,
        reward=30,
        id=f"picked_{toy}"
    )

    # Place in box
    scene.add_reward(
        tracked_asset="toy_box",
        behavior="contains",
        target=toy,
        threshold=True,
        reward=70,
        id=f"stored_{toy}",
        requires=f"picked_{toy}"
    )

    # Penalty: Don't drop on floor
    scene.add_reward(
        tracked_asset="floor",
        behavior="contact",
        target=toy,
        threshold=True,
        reward=-20,
        id=f"dropped_{toy}"
    )

# Completion bonus: All toys collected
scene.add_reward(
    condition=AND([
        {"tracked_asset": "toy_box", "behavior": "contains", "target": "toy_car", "threshold": True},
        {"tracked_asset": "toy_box", "behavior": "contains", "target": "toy_block", "threshold": True},
        {"tracked_asset": "toy_box", "behavior": "contains", "target": "toy_ball", "threshold": True}
    ]),
    reward=200,
    id="cleanup_complete"
)
```

### Scenario 4: Vision-Guided Navigation with Room Awareness

```python
# Build scene
scene = create_scene("office", width=8, length=8)
robot = create_robot("stretch").sensors_only(["nav_camera", "lidar"])
scene.add_robot(robot, relative_to=(0, 0, 0))

# Goal: Navigate to person
scene.add("person", position=(6, 6, 0))

# Shaping: Keep person visible
scene.add_reward(
    tracked_asset="stretch.nav_camera",
    behavior="target_visible",
    target="person",
    threshold=True,
    reward=10,
    mode="smooth",
    id="tracking"
)

# Shaping: Get closer
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="person",
    threshold=0.0,
    reward=20,
    mode="smooth",
    id="approaching"
)

# Goal: Stop at safe distance (1.5m)
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="person",
    threshold={"above": 1.2, "below": 1.8},
    reward=100,
    id="reached"
)

# Safety: Don't get too close
scene.add_reward(
    tracked_asset="stretch.base",
    behavior="distance_to",
    target="person",
    threshold={"below": 0.8},
    reward=-50,
    id="too_close"
)

# Safety: Avoid walls (room awareness!)
for wall in ["wall_north", "wall_south", "wall_east", "wall_west"]:
    scene.add_reward(
        tracked_asset="stretch.base",
        behavior="distance_to",
        target=wall,
        threshold={"below": 0.5},
        reward=-10,
        mode="smooth",
        id=f"too_close_{wall}"
    )

    scene.add_reward(
        tracked_asset=wall,
        behavior="object_contact",
        target="stretch.base",
        threshold=True,
        reward=-30,
        id=f"hit_{wall}"
    )
```

---

## Best Practices

### 1. Start Simple, Add Complexity

```python
# ✓ Start with basic scene
scene = create_scene("kitchen")
robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))
scene.add("apple", position=(1, 0, 0.8))

# ✓ Add basic reward
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="apple",
    threshold=True,
    reward=100
)

# ✓ Then add shaping
scene.add_reward(
    tracked_asset="stretch.gripper",
    behavior="distance_to",
    target="apple",
    threshold=0.0,
    reward=10,
    mode="smooth"
)

# ✓ Then add safety
scene.add_reward(
    tracked_asset="floor",
    behavior="contact",
    target="apple",
    reward=-50
)
```

### 2. Use Room Components for Safety

```python
# Always penalize drops
scene.add_reward(tracked_asset="floor", behavior="contact", target="fragile", reward=-100)

# Always penalize wall collisions
for wall in ["wall_north", "wall_south", "wall_east", "wall_west"]:
    scene.add_reward(tracked_asset=wall, behavior="object_contact", target="robot", reward=-10)
```

### 3. Balance Shaping and Sparse Rewards

```python
# Shaping: Small, frequent (guides learning)
scene.add_reward(..., reward=10, mode="smooth")

# Sparse: Large, rare (defines goal)
scene.add_reward(..., reward=100, mode="discrete")
```

### 4. Use Descriptive IDs

```python
# ✓ GOOD
id="picked_apple_from_table"
id="placed_cup_on_plate"
id="avoided_wall_collision"

# ✗ BAD
id="r1"
id="cond2"
id="test"
```

### 5. Test Incrementally

```python
# Build scene
scene = create_scene("test")
scene.add_robot(robot)
scene.add("apple")

# Test compilation
xml_path = scene.compile_scene()
print(f"✓ Scene compiles: {xml_path}")

# Test assets
print(f"✓ Assets: {list(scene.assets.keys())}")

# Test single reward
scene.add_reward(tracked_asset="stretch.gripper", behavior="closed", threshold=True, reward=10)
print("✓ Reward added")

# Test views
print(scene.reward_modal.get_data())
```

---

## Summary: The Complete Workflow

```python
# 1. CREATE SCENE
scene = create_scene("kitchen", width=5, length=5, height=3)

# 2. ADD ROBOT
robot = create_robot("stretch")
scene.add_robot(robot, relative_to=(0, 0, 0))

# 3. ADD OBJECTS
scene.add("table_1", position=(2, 0, 0))
scene.add("apple", position={"relative_to": "table_1", "relation": "on_top"})
scene.add("basket", position=(3, 1, 0.8))

# 4. ADD REWARDS
scene.add_reward(tracked_asset="stretch.gripper", behavior="holding", target="apple", reward=50, id="picked")
scene.add_reward(tracked_asset="basket", behavior="contains", target="apple", reward=100, id="placed", requires="picked")
scene.add_reward(tracked_asset="floor", behavior="contact", target="apple", reward=-50, id="dropped")

# 5. COMPILE
xml_path = scene.compile_scene()

# 6. RUN
from core.runtime.runtime_engine import RuntimeEngine
engine = RuntimeEngine(scene)
engine.reset()

# 7. INSPECT
print(scene.assets.keys())  # All trackable assets
print(scene.reward_modal.get_data())  # Reward state
print(scene.reward_modal.timeline())  # Visual timeline
```

---

## Next Steps

- **QUICKSTART.md** - Get started in 5 minutes
- **API_REFERENCE.md** - Full API documentation
- **ADDING_COMPONENTS.md** - Extend the system
- **MODAL_ORIENTED_PROGRAMMING.md** - Understand the philosophy
- **ARCHITECTURE.md** - Learn how it works internally

---

**Smart Scene Composition: Build Once, Track Everything** ✨
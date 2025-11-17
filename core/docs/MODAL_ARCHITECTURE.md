rl # MODAL ARCHITECTURE: How the Simulation Framework Works

> **A Simulation-Only Framework for Robotic Research**
> This framework is designed for physics-based simulation using MuJoCo. It is NOT designed to control real physical robots.

---

## What Is Modal-Oriented Programming?

**Modal-Oriented Programming (MOP)** is the architectural pattern powering this simulation framework. Instead of writing glue code to connect components, you compose **modals** - self-aware data structures that:

- **Auto-discover** their own structure from XML
- **Self-generate** their behavior definitions
- **Self-render** in multiple formats (data, RL vectors, XML)
- **Compose** like LEGO blocks with zero configuration
- **Communicate** directly with other modals

Think of modals as smart building blocks that know how to snap together and work together automatically.

---

## The Core Components

The framework has **five core components** that work together like LEGO blocks:

1. **Scene Modal** - The orchestrator that holds everything
2. **Asset Modal** - Self-discovering objects and furniture
3. **Robot Modal** - Dynamic self-building robots
4. **Reward Modal** - Intelligent validation system
5. **Smart Actions** - Self-validating execution blocks

Let's explore each one:

---

## The Four Core Modals

### 1. Scene Modal - The Orchestrator

The **Scene Modal** is the central orchestrator that holds everything together.

**What it contains:**
```python
@dataclass
class Scene:
    room: RoomModal              # The physical space
    placements: List[Placement]  # Spatial relations
    assets: Dict[str, Asset]     # All objects (tables, chairs, etc.)
    robots: Dict[str, Any]       # Robot information
    cameras: Dict[str, Camera]   # All cameras
    conditions: Dict[str, Any]   # Reward conditions
    reward_modal: RewardModal    # Reward system
```

**Key responsibilities:**

1. **Spatial Composition** - Manages object placement with semantic relations:
   ```python
   ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
   ops.add_asset("foam_brick", relative_to="table", relation="on_top")
   ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")
   ```
   The Scene Modal calculates heights automatically - no manual math needed!

2. **Component Wrapping** - Wraps robot components as trackable assets:
   ```python
   # Robot has actuators: arm, lift, gripper, base, etc.
   # Scene wraps each as: "stretch.arm", "stretch.lift", etc.
   # Result: Uniform interface for ALL components!
   ```

3. **State Management** - Provides unified access to all simulation state:
   ```python
   state = ops.get_state()
   # Returns:
   # {
   #   "stretch.arm": {"extension": 0.3, "at_target": True, ...},
   #   "stretch.gripper": {"closed": True, "holding": True, ...},
   #   "table": {"position": [2.0, 0.0, 0.8], ...},
   #   ...
   # }
   ```

**Why it's powerful:**
- ONE interface for robots, sensors, objects, and furniture
- Spatial relations handled declaratively (no coordinate math!)
- Complete state available through single `get_state()` call

---

### 2. Asset Modal - Self-Discovering Components

The **Asset Modal** represents any object in the simulation (furniture, objects, even robot parts).

**Component structure:**
```python
@dataclass
class Component:
    name: str                          # Component identifier
    behaviors: List[str]               # What it can do
    trackable_behaviors: List[str]     # What can be tracked
    geom_names: List[str]              # MuJoCo geometry IDs
    joint_names: List[str]             # MuJoCo joint IDs
    site_names: List[str]              # MuJoCo site IDs (attachment points)
```

**Auto-discovery process:**

1. **XML Parsing** - Reads the MuJoCo XML definition:
   ```xml
   <!-- table.xml -->
   <body name="table">
     <geom name="table_top" type="box" size="0.5 0.8 0.02"/>
     <geom name="table_leg_1" type="cylinder" size="0.05 0.4"/>
     <site name="place_on_top" pos="0 0 0.5"/>
   </body>
   ```

2. **Semantic Inference** - Infers behaviors from naming patterns:
   ```python
   # "place_on_top" site → surface behavior
   # "grasp_*" site → graspable behavior
   # "*_hinge" joint → hinged behavior
   ```

3. **Behavior Definition** - Generates JSON specification:
   ```json
   {
     "surface": {
       "description": "Object with surface for placement",
       "properties": {
         "placement_height": {"unit": "meters", "default": null},
         "surface_area": {"unit": "meters²", "default": null}
       }
     }
   }
   ```

**Self-rendering in multiple formats:**
```python
asset = ops.scene.assets["table"]

# 1. Human-readable data
data = asset.get_data()
# {"position": [2.0, 0.0, 0.8], "type": "furniture", ...}

# 2. RL training vector
rl_vec = asset.get_rl()
# [2.0, 0.0, 0.8, ...]  # Normalized, flat vector

# 3. XML for MuJoCo
xml = asset.render_xml()
# <body name="table">...</body>
```

**Why it's powerful:**
- Add object to scene → behaviors auto-discovered from XML
- No manual configuration needed
- Same modal renders in 3+ formats automatically

---

### 3. Robot Modal - Dynamic Self-Building

The **Robot Modal** represents the simulated robot and all its components.

**Structure:**
```python
@dataclass
class Robot:
    name: str                           # Robot identifier
    robot_type: str                     # "stretch", "franka", etc.
    sensors: Dict[str, SensorModal]     # All sensors
    actuators: Dict[str, ActuatorModal] # All actuators
    actions: Dict[str, ActionBlock]     # Available actions
```

**Auto-discovery process:**

1. **XML Scanning** - Loads robot XML and discovers all actuators:
   ```python
   # Finds in stretch.xml:
   # - arm (prismatic joint)
   # - lift (prismatic joint)
   # - gripper (finger joints)
   # - base (mobile base - wheels)
   # - head_pan, head_tilt (revolute joints)
   ```

2. **Sensor Discovery** - Discovers all sensors with `.behaviors` field:
   ```python
   class NavCamera(BaseModel):
       sensor_id: Literal['nav'] = 'nav'
       rgb_image: Any
       depth_image: Any

       # Self-declares trackability!
       behaviors: List[str] = ["vision", "robot_head_spatial"]
       site_names: List[str] = ["nav_camera_site"]
   ```

3. **Behavior Generation** - Auto-generates ROBOT_BEHAVIORS.json:
   ```python
   robot.create_robot_asset_package()
   # Scans ALL actuators + sensors with .behaviors
   # Generates complete behavior definitions
   # Single source of truth: THE MODALS!
   ```

**Component wrapping:**
```python
# Scene wraps robot components as trackable assets:
ops.add_robot("stretch")

# Creates trackable assets:
# - "stretch.arm" (ActuatorComponent)
# - "stretch.gripper" (ActuatorComponent)
# - "stretch.nav_camera" (SensorComponent)
# - etc.

# Now you can track them in rewards:
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
ops.add_reward("stretch.nav_camera", "target_visible", True, reward=50)
```

**Why it's powerful:**
- Robot self-discovers ALL components from XML
- Add sensor to robot → automatically trackable
- Uniform API: robot parts treated like any other asset

---

### 4. Reward Modal - Intelligent Validation

The **Reward Modal** provides self-validating behavior verification through rewards.

**Condition types:**

1. **Simple Condition** - Single behavior threshold:
   ```python
   ops.add_reward(
       tracked_asset="stretch.arm",
       behavior="extension",
       threshold=0.3,
       reward=100,
       id="arm_extended"
   )
   # Triggers when arm extension >= 0.3 meters
   ```

2. **Composite Condition** - AND/OR/NOT logic:
   ```python
   # Both conditions must be met
   ops.add_reward("stretch.gripper", "holding", True, target="apple", reward=0, id="grasping")
   ops.add_reward("apple", "height_above", 0.5, target="floor", reward=0, id="lifted")

   ops.add_reward_composite(
       operator="AND",
       conditions=["grasping", "lifted"],
       reward=200,
       id="apple_lifted"
   )
   # Triggers only when BOTH grasping AND lifted
   ```

3. **Sequence Condition** - Multi-stage tasks:
   ```python
   ops.add_reward_sequence(
       sequence=["approach_table", "grasp_apple", "lift_apple"],
       reward=300,
       id="pick_task"
   )
   # Must complete in order: approach → grasp → lift
   ```

**Physics-aware tolerances:**
```python
# Actuators auto-discover their position tolerance from physics
actuator_comp = ActuatorComponent(
    name="lift",
    position=0.497,        # Actual position from MuJoCo
    range=(0.0, 1.095),    # Valid range
    tolerance=0.0143       # Auto-measured motor accuracy
)

# Reward system validates with tolerance:
ops.add_reward("stretch.lift", "position", 0.5, reward=100)
# Checks: 0.497 >= (0.5 - 0.0143) ✓ SUCCESS!
# Accounts for motor inaccuracy, gravity, compliance
```

**Self-validation pattern:**
```python
# Traditional approach (error-prone):
arm.extend(0.3)
time.sleep(2.0)
actual = arm.get_position()
assert abs(actual - 0.3) < 0.05  # Manual check!

# Modal approach (self-validating):
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
ops.submit_block(extend_arm(extension=0.3))
total_reward = sum(ops.step()['reward'] for _ in range(500))
# If reward >= 100 → action validated automatically!
```

**Why it's powerful:**
- Actions validate themselves through rewards
- Physics tolerances handled automatically
- Complex logic (AND/OR/NOT, sequences) built-in
- No manual threshold checking needed

---

## LEGO Composition: How Modals Work Together

### The Complete Flow

Here's how modals compose from scene creation to runtime:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SCENE CREATION                                           │
│    User: ops.create_scene("kitchen", width=8, length=8)     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SCENE MODAL SELF-BUILDS                                  │
│    - Creates RoomModal (floor, walls, ceiling)              │
│    - Initializes empty asset dict                           │
│    - Initializes empty reward dict                          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ADD FURNITURE                                            │
│    User: ops.add_asset("table", relative_to=(2, 0, 0))      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. ASSET MODAL AUTO-DISCOVERS                               │
│    - Loads table.xml from registry                          │
│    - Extracts geoms, joints, sites                          │
│    - Infers behaviors from naming                           │
│    - Scene adds to assets dict                              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ADD ROBOT                                                │
│    User: ops.add_robot("stretch")                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. ROBOT MODAL SELF-BUILDS                                  │
│    - Loads stretch.xml                                      │
│    - Discovers actuators (arm, lift, gripper, base)         │
│    - Discovers sensors (cameras, LIDAR, IMU)                │
│    - Generates ROBOT_BEHAVIORS.json                         │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. SCENE WRAPS ROBOT COMPONENTS                             │
│    - Wraps each actuator as trackable asset:                │
│      "stretch.arm", "stretch.lift", etc.                    │
│    - Wraps sensors with .behaviors as trackable assets:     │
│      "stretch.nav_camera", etc.                             │
│    - Result: Uniform interface for ALL components!          │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. ADD REWARDS                                              │
│    User: ops.add_reward("stretch.arm", "extension", 0.3)    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 9. REWARD MODAL VALIDATES                                   │
│    - Finds "stretch.arm" in scene.assets                    │
│    - Discovers "extension" behavior                         │
│    - Creates condition with physics tolerance               │
│    - Scene adds to reward_modal.conditions                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 10. COMPILE                                                 │
│     User: ops.compile()                                     │
│     - Scene generates complete MuJoCo XML                   │
│     - All modals render themselves to XML                   │
│     - Physics engine initialized                            │
│     - Database created for persistence                      │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 11. RUNTIME - MODAL-TO-MODAL COMMUNICATION                  │
│     User: ops.step()                                        │
│     ┌─────────────────────────────────────────┐             │
│     │ MuJoCo Physics Simulation (1 step)      │             │
│     └──────────────┬──────────────────────────┘             │
│                    ▼                                        │
│     ┌─────────────────────────────────────────┐             │
│     │ Scene Modal syncs from MuJoCo           │             │
│     │ - Extracts all qpos, qvel, contacts     │             │
│     └──────────────┬──────────────────────────┘             │
│                    ▼                                        │
│     ┌─────────────────────────────────────────┐             │
│     │ Asset Modals update their state         │             │
│     │ - Each asset calls get_data()           │             │
│     │ - Modals compute their behaviors        │             │
│     └──────────────┬──────────────────────────┘             │
│                    ▼                                        │
│     ┌─────────────────────────────────────────┐             │
│     │ Reward Modal evaluates conditions       │             │
│     │ - Checks each condition against state   │             │
│     │ - Awards rewards when thresholds met    │             │
│     └──────────────┬──────────────────────────┘             │
│                    ▼                                        │
│     ┌─────────────────────────────────────────┐             │
│     │ Return complete state + rewards         │             │
│     └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

### Key LEGO Principles

**1. No Glue Code** - Modals find each other automatically:
```python
# Scene finds asset by name
ops.add_reward("stretch.arm", "extension", 0.3)
# Scene: "stretch.arm exists? Yes! Has 'extension' behavior? Yes! Create condition."
```

**2. Uniform Interface** - Everything uses same API:
```python
state = ops.get_state()
# Robot components: state["stretch.arm"]["extension"]
# Furniture:        state["table"]["position"]
# Objects:          state["apple"]["position"]
# All accessed the same way!
```

**3. Self-Sync** - Modals update themselves automatically:
```python
ops.step()  # Physics simulation
# Each modal automatically:
# 1. Queries MuJoCo for its state
# 2. Updates its internal data
# 3. Computes its behaviors
# 4. Makes itself available via get_data()
```

**4. Add Once, Use Everywhere** - Single declaration propagates:
```python
# Add sensor with .behaviors field:
class NewSensor(BaseModel):
    behaviors: List[str] = ["temperature_sensing"]
    site_names: List[str] = ["temp_sensor_site"]

# Automatically:
# ✓ Robot discovers it during XML scan
# ✓ Generation creates behavior JSON
# ✓ Scene wraps it as trackable asset
# ✓ Rewards can use it: "robot.temp_sensor"
# ONE declaration, FIVE effects!
```

---

## 5. Smart Actions - Self-Validating Execution

Beyond the four core modals, the framework includes **Smart Actions** - action blocks that know how to validate themselves and track their own progress.

**Action Modal structure:**
```python
@dataclass
class ActionBlock:
    id: str                      # Action identifier
    description: str             # Human-readable description
    actions: List[Action]        # List of actions to execute
    status: str                  # "pending", "running", "completed", "failed"
    progress: float              # 0-100% completion
    execution_mode: str          # "sequential" or "parallel"
```

**Self-validation through status tracking:**

```python
# Example: Rotate robot 180 degrees
block = spin_left(degrees=180)
ops.submit_block(block)

for step in range(3000):
    ops.step()

    # Smart Action tracks its own progress!
    print(f"Status: {block.status}")      # "running" → "completed"
    print(f"Progress: {block.progress}%") # 0% → 100%

    # Action validates itself when complete
    if block.status == 'completed' and block.progress >= 100:
        print("✅ Action self-validated!")
        break
```

**Multi-level validation:**

Smart Actions validate themselves through **4 independent metrics**:

1. **Block Status** - Action knows if it completed:
   ```python
   block.status  # "pending" → "running" → "completed"
   ```

2. **Block Progress** - Action tracks percentage complete:
   ```python
   block.progress  # 0.0 → 25.0 → 50.0 → 100.0
   ```

3. **Physics Measurement** - Actual physical state:
   ```python
   # For rotation action, measure actual rotation
   rotation_deg = get_rotation_from_start()  # 0° → 90° → 180°
   ```

4. **Reward Validation** - Reward system confirms success:
   ```python
   ops.add_reward("stretch.base", "rotation", 180.0, reward=100)
   # If reward >= 100 → rotation validated!
   ```

**Complete validation example:**
```python
# Add reward for validation
ops.add_reward(
    tracked_asset="stretch.base",
    behavior="rotation",
    threshold=180.0,
    reward=100.0,
    id="rotate_180"
)

# Submit smart action
block = spin_left(degrees=180)
ops.submit_block(block)

# Run with 4-way validation
for step in range(3000):
    result = ops.step()

    # 1. Block self-reports status
    if block.status == 'completed':
        print("✅ 1/4: Block status = completed")

    # 2. Block self-reports progress
    if block.progress >= 100:
        print("✅ 2/4: Block progress = 100%")

    # 3. Physics measurement
    rotation_deg = measure_rotation()
    if rotation_deg >= 180:
        print("✅ 3/4: Physics = 180°")

    # 4. Reward validation
    if result['reward_total'] >= 100:
        print("✅ 4/4: Reward = 100pts")

    # All 4 metrics agree → action validated!
    if (block.status == 'completed' and
        block.progress >= 100 and
        rotation_deg >= 180 and
        result['reward_total'] >= 100):
        print("✅ COMPLETE VALIDATION: All 4 metrics confirm success!")
        break
```

**Why Smart Actions matter:**

- **Self-Aware** - Actions know their own state (no external tracking needed)
- **Self-Validating** - Actions validate themselves (no manual checks)
- **Progress Tracking** - Real-time feedback (0-100%)
- **Multi-Modal Validation** - 4 independent proofs of correctness
- **Composable** - Smart Actions work with reward system seamlessly

**Action types:**

```python
# Movement actions
move_forward(distance=1.0)    # Self-validates distance traveled
spin_left(degrees=90)         # Self-validates rotation angle

# Manipulation actions
extend_arm(extension=0.3)     # Self-validates arm extension
close_gripper()               # Self-validates gripper closure

# Complex sequences
pick_object(target="apple")   # Multi-stage: approach → grasp → lift
place_object(target="bowl")   # Multi-stage: move → open → release
```

Each action knows:
- What it needs to do
- How to measure progress
- When it's complete
- How to validate success

**No manual validation needed** - the action tells YOU when it's done!

---

## Complete Code Example

Here's a complete example showing all four modals working together:

```python
from simulation_center.core.main.experiment_ops_unified import ExperimentOps
import cv2

# Initialize experiment
ops = ExperimentOps(headless=True, render_mode="vision_rl")

# ============================================================================
# SCENE MODAL - Create environment
# ============================================================================
ops.create_scene("kitchen", width=8, length=8, height=3)
# Scene Modal auto-creates:
# ✓ Floor, walls, ceiling (all trackable!)
# ✓ Empty asset dict
# ✓ Empty reward dict

# ============================================================================
# ASSET MODAL - Add furniture (auto-discovery)
# ============================================================================
ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
# Asset Modal:
# ✓ Loads table.xml
# ✓ Discovers geoms, joints, sites
# ✓ Infers "surface" behavior from "place_on_top" site
# ✓ Calculates height = 0.8m

# Stack objects (spatial relations)
ops.add_asset("foam_brick", relative_to="table", relation="on_top")
ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")
ops.add_asset("bowl", relative_to="pudding_box", relation="stack_on")
# Asset Modal:
# ✓ Calculates heights automatically
# ✓ No manual coordinate math!

# Add object inside container
ops.add_object("apple", position={"relative_to": "bowl", "relation": "inside"})
# Asset Modal:
# ✓ Positions apple inside bowl
# ✓ Validates containment

# ============================================================================
# ROBOT MODAL - Add robot (self-building)
# ============================================================================
ops.add_robot("stretch", position=(0, 0, 0))
# Robot Modal:
# ✓ Loads stretch.xml
# ✓ Discovers 13 actuators (arm, lift, gripper, base, etc.)
# ✓ Discovers 5 sensors (nav_camera, gripper_camera, etc.)
# ✓ Generates ROBOT_BEHAVIORS.json (18 behaviors!)
# Scene wraps as trackable assets:
# ✓ "stretch.arm", "stretch.gripper", "stretch.nav_camera", etc.

# ============================================================================
# Add cameras for vision validation
# ============================================================================
ops.add_free_camera("view1", lookat=(2.0, 0.0, 0.8), distance=3.0, azimuth=45)
ops.add_free_camera("view2", lookat=(2.0, 0.0, 0.8), distance=3.5, azimuth=135)

# ============================================================================
# REWARD MODAL - Add self-validating rewards
# ============================================================================

# Simple condition: gripper holding apple
ops.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    threshold=True,
    target="apple",
    reward=100,
    id="grasp"
)
# Reward Modal:
# ✓ Finds "stretch.gripper" in scene.assets
# ✓ Discovers "holding" behavior
# ✓ Creates condition with physics tolerance
# ✓ Validates when gripper.holding == True

# Composite condition: apple lifted high (requires grasp first!)
ops.add_reward(
    tracked_asset="apple",
    behavior="height_above",
    threshold=1.0,
    target="floor",
    reward=100,
    requires="grasp",  # Dependency!
    id="lift"
)
# Reward Modal:
# ✓ Only checks if "grasp" condition met first
# ✓ Validates apple is 1m above floor

# ============================================================================
# COMPILE - All modals render themselves to MuJoCo XML
# ============================================================================
ops.compile()
# Scene Modal:
# ✓ Generates complete MuJoCo XML from all modals
# Asset/Robot Modals:
# ✓ Each renders itself to XML
# MuJoCo:
# ✓ Compiles physics simulation
# Database:
# ✓ Created for experiment persistence

# ============================================================================
# RUNTIME - Modal-to-modal communication
# ============================================================================

# Submit action (robot moves forward)
from simulation_center.core.modals.stretch.action_blocks_registry import move_forward
block = move_forward(distance=1.5)
ops.submit_block(block)

# Run simulation with self-validation
for step in range(1000):
    result = ops.step()

    # Modal-to-modal communication happens automatically:
    # 1. MuJoCo simulates physics
    # 2. Scene Modal syncs from MuJoCo
    # 3. Asset Modals compute behaviors (holding, position, etc.)
    # 4. Reward Modal evaluates conditions
    # 5. Database saves state

    # Check if task complete (self-validated by rewards!)
    if result['reward_total'] >= 200:
        print(f"✅ Task completed at step {step}!")
        print(f"   - Grasped apple: {result['reward_breakdown']['grasp']}")
        print(f"   - Lifted apple:  {result['reward_breakdown']['lift']}")
        break

# ============================================================================
# SELF-RENDERING - Get state in multiple formats
# ============================================================================

# 1. Human-readable data
state = ops.get_state()
print(f"Gripper holding: {state['stretch.gripper']['holding']}")
print(f"Apple height: {state['apple']['position'][2]}")

# 2. RL training vector (auto-normalized)
rl_obs = ops.view_aggregator.get_obs()
# Returns flat, normalized vector for RL training

# 3. Visual proof (cameras auto-configured)
cv2.imwrite("proof_view1.png", ops.engine.last_views["view1_view"]["rgb"])
cv2.imwrite("proof_view2.png", ops.engine.last_views["view2_view"]["rgb"])

print("✅ Complete multi-modal validation:")
print("   1. Physics: MuJoCo verified grasping and lifting")
print("   2. Semantic: Behaviors validated (holding=True, height=1.0m)")
print("   3. Vision: Cameras captured visual proof")
print("   4. Reasoning: Spatial/temporal consistency maintained")
print("   5. Rewards: Self-validation confirmed success (200 points)")
```

**Result:**
- Complex robotic task in ~60 lines
- 4-level object stack (table → brick → pudding → bowl → apple)
- Robot with 18 auto-discovered behaviors
- Multi-camera observation
- Self-validating rewards
- Complete database persistence
- **Zero manual configuration!**

---

## The 5 MOP Principles in Action

### 1. Auto-Discovery
```python
# You write:
ops.add_robot("stretch")

# System auto-discovers:
# - 13 actuators from XML
# - 5 sensors with .behaviors
# - 18 total behaviors
# - Joint names, geom names, site names
# - Action blocks
# ZERO manual configuration!
```

### 2. Self-Generation
```python
# Modals generate their own JSON specifications:
robot.create_robot_asset_package()
# Scans sensors/actuators → generates ROBOT_BEHAVIORS.json
# Single source of truth: THE MODALS!
# Not humans writing JSON!
```

### 3. Self-Rendering
```python
# Same modal, multiple formats:
asset = ops.scene.assets["table"]

data = asset.get_data()        # Human-readable
rl_vec = asset.get_rl()        # RL training
xml = asset.render_xml()       # MuJoCo XML
# One modal, three+ views!
```

### 4. LEGO Composition
```python
# Add field to sensor → system auto-integrates:
class NewSensor(BaseModel):
    behaviors: List[str] = ["my_new_behavior"]

# Automatically:
# ✓ Robot discovers it
# ✓ JSON generated
# ✓ Scene wraps it
# ✓ Rewards can use it
# Add once, use everywhere!
```

### 5. Modal-to-Modal Communication
```python
# Modals talk directly, no coordinator:
ops.step()
# MuJoCo → Scene Modal → Asset Modals → Reward Modal
# Each modal syncs itself!
# Each modal computes itself!
# ZERO glue code!
```

---

## Why This Architecture?

### Minimal Code, Maximum Capability

**Traditional robotics framework:**
```python
# 50+ lines just to set up environment
env = gym.make("FetchReach-v1")
env.reset()
# Then manually:
# - Track rewards
# - Validate actions
# - Save/load state
# - Configure cameras
# - Sync physics
```

**This framework (Modal-Oriented):**
```python
# 5 lines - everything auto-configured
ops = ExperimentOps()
ops.create_scene("room", width=5, length=5)
ops.add_robot("stretch")
ops.compile()
ops.step()  # Auto-syncs, auto-validates, auto-saves!
```

### Self-Validating Tests

**Traditional:**
```python
# Manual validation (error-prone)
robot.move_arm(0.3)
time.sleep(2.0)
actual = robot.get_arm_position()
assert abs(actual - 0.3) < 0.05
```

**Modal-Oriented:**
```python
# Self-validating through rewards
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
# Reward triggers → action validated!
# ZERO manual checks!
```

### Multi-Modal Validation

Every behavior validated through **5 modalities**:

1. **Physics** - MuJoCo state proves behavior worked
2. **Semantic** - Modal behaviors confirm correctness
3. **Vision** - Cameras capture visual proof
4. **Reasoning** - Spatial/temporal consistency validated
5. **Rewards** - Self-validation confirms success

All available automatically through modal self-declaration!

---

## What This Framework Does (and Doesn't Do)

### ✅ What It DOES

- **Physics-based simulation** using MuJoCo
- **Robotic manipulation** research in simulation
- **RL training** for simulated robots
- **Multi-modal validation** (physics + vision + semantics)
- **Experiment reproducibility** with complete persistence
- **Self-validating tests** through reward systems

### ❌ What It DOES NOT Do

- **Control real physical robots** - This is simulation-only
- **Sim-to-real transfer** - No real robot interfaces
- **Hardware integration** - No physical actuator/sensor drivers
- **Real-world deployment** - Not designed for production robotics

**This framework is for simulation research only.** If you need to control real robots, you'll need a different framework with hardware interfaces.

---

## Next Steps

- **[API_QUICKSTART.md](./API_QUICKSTART.md)** - 10 minimal code examples
- **[MODAL_ORIENTED_PROGRAMMING.md](../MODAL_ORIENTED_PROGRAMMING.md)** - Deep dive into MOP philosophy
- **[ROBOT_OS_OVERVIEW.md](./ROBOT_OS_OVERVIEW.md)** - Complete system overview
- **[CAPABILITY_MATRIX.md](./CAPABILITY_MATRIX.md)** - All capabilities listed
- **[VALIDATION_FRAMEWORK.md](./VALIDATION_FRAMEWORK.md)** - 5-way validation explained

---

## Summary

**Modal-Oriented Programming** is about composing self-aware data structures:

- **Scene Modal** orchestrates everything
- **Asset Modal** self-discovers from XML
- **Robot Modal** self-builds from components
- **Reward Modal** self-validates behaviors
- **Smart Actions** track progress and validate themselves

They snap together like LEGO blocks with **zero configuration** and communicate **directly with each other**.

The result: Complex robotic simulation scenarios in **3-50 lines** of clean, declarative code, with **actions that know when they're done**.

**Remember:** This is a simulation framework for robotics research, not for controlling real physical robots.

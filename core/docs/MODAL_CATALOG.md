# MODAL CATALOG: Self-Declaring Components

## What is a Modal?

A **modal** is a self-contained component that:
1. **Declares** what it provides (behaviors, properties)
2. **Syncs** automatically with MuJoCo physics
3. **Validates** itself through exposed behaviors
4. **Composes** with other modals

Think of modals as **self-aware components** that know what they are and what they can do.

---

## Core Modals

### RobotModal
**What:** Complete robot with all subsystems
**Self-Declares:** All sensors, all actuators, odometry
**Behaviors:** Auto-discovered from child modals
**Example:**
```python
ops.add_robot("stretch")
# Auto-discovers:
# - Base (wheels, odometry)
# - Arm (joints, extension)
# - Gripper (holding, force)
# - Lift (height)
# - Head (pan, tilt, camera)
# - All sensors (LIDAR, IMU, etc.)
```

---

### ActuatorModal
**What:** Controllable robot component
**Self-Declares:** Position, velocity, limits, at_target
**Behaviors:**
- `extension` (arm)
- `rotation` (base)
- `height` (lift)
- `at_target` (goal reached)

**Example:**
```python
# Modal self-declares "extension" behavior
state = ops.get_state()
arm_extension = state["stretch.arm"]["extension"]
at_target = state["stretch.arm"]["at_target"]
```

---

### SensorModal
**What:** Perception component
**Self-Declares:** Measurements, valid_reading, sensor_type
**Behaviors:**
- `position` (odometry)
- `distance` (LIDAR)
- `acceleration` (IMU)
- `rgb`, `depth` (camera)

**Example:**
```python
# Modal self-declares "position" from odometry
odometry = state["stretch.odometry"]["position"]
```

---

### SceneModal
**What:** Complete environment
**Self-Declares:** Room components (floor, walls, ceiling)
**Behaviors:**
- `width`, `length`, `height`
- `floor`, `walls`, `ceiling` as trackable assets
- Spatial relations (on_top, stack_on, inside)

**Example:**
```python
# Scene modal auto-creates room components
ops.create_scene("room", width=8, length=8)
# Now "floor", "wall_north", etc. are trackable!
```

---

### AssetModal
**What:** Objects in the scene
**Self-Declares:** Position, orientation, dimensions, behaviors
**Behaviors:**
- `graspable` (can be picked up)
- `container` (can hold objects)
- `surface` (can support objects)
- `rollable` (sphere physics)
- `stackable` (can stack on)

**Example:**
```python
# Asset modal declares its behaviors
state = ops.get_state()
bowl_behaviors = state["bowl"]["behaviors"]
# Contains: ["container", "surface", "graspable"]
```

---

### RewardModal
**What:** Validation/objective component
**Self-Declares:** Conditions, thresholds, progress
**Behaviors:**
- `triggered` (condition met)
- `progress` (partial completion)
- `value` (reward amount)

**Example:**
```python
# Reward modal validates through behavior
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
# Modal automatically checks threshold and awards!
```

---

## Modal Composition

Modals **compose hierarchically**:

```
RobotModal ("stretch")
├── BaseModal
│   ├── WheelMotorModal (left)
│   ├── WheelMotorModal (right)
│   └── OdometryModal
├── ArmModal
│   └── JointMotorModals (x4)
├── GripperModal
│   └── ForceModal
├── LiftModal
│   └── JointMotorModal
└── HeadModal
    ├── PanMotorModal
    ├── TiltMotorModal
    └── CameraModal
        ├── RGBModal
        └── DepthModal
```

**Each modal:**
- Self-declares its behaviors
- Auto-syncs with MuJoCo
- Accessible via dot notation: `stretch.arm.joint_1`

---

## Self-Declaration Pattern

All modals follow the **self-declaration pattern**:

```python
class ArmModal(ActuatorModal):
    def __init__(self):
        # Declare behaviors this modal provides
        self.behaviors = {
            "extension": self.get_extension,
            "at_target": self.check_target,
            "joint_positions": self.get_joints
        }

    def get_extension(self) -> float:
        """Auto-syncs with MuJoCo"""
        return self.mujoco_data.qpos[self.joint_ids].sum()

    def check_target(self) -> bool:
        """Self-validates"""
        return abs(self.get_extension() - self.target) < self.tolerance
```

**Result:**
- `ops.get_state()["stretch.arm"]["extension"]` → auto-synced
- `ops.get_state()["stretch.arm"]["at_target"]` → self-validated
- **Zero manual configuration!**

---

## Modal Benefits

### 1. Auto-Discovery
```python
ops.add_robot("stretch")
# All modals discovered automatically
# All behaviors accessible immediately
```

### 2. Self-Validation
```python
# Modal knows when target reached
if state["stretch.arm"]["at_target"]:
    print("Validated by modal!")
```

### 3. Behavior Composition
```python
# Gripper modal composes force + position
state["stretch.gripper"]["holding"]  # Semantic
state["stretch.gripper"]["force"]    # Physics
```

### 4. Declarative Access
```python
# Dot notation reflects modal hierarchy
stretch.base.left_wheel.velocity
stretch.arm.joint_1.position
stretch.head.camera.rgb
```

---

## Modal vs Traditional

### Traditional (Manual):
```python
# Manual setup
arm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "arm")
pos = data.qpos[arm_id]
target = 0.3
at_target = abs(pos - target) < 0.05

# Manual sync
data.ctrl[arm_id] = target
mujoco.mj_step(model, data)
pos = data.qpos[arm_id]  # Re-read manually
```

### Modal (Auto):
```python
# Modal auto-syncs
state = ops.get_state()
pos = state["stretch.arm"]["extension"]  # Auto-synced!
at_target = state["stretch.arm"]["at_target"]  # Auto-validated!

# Modal handles MuJoCo internally
```

**Result:** 80% less code, 100% more capability!

---

## Adding Custom Modals

You can create custom modals:

```python
from core.modals.base_modal import SensorModal

class CustomSensorModal(SensorModal):
    def __init__(self):
        super().__init__()
        self.behaviors = {
            "my_measurement": self.get_measurement,
            "is_valid": self.validate
        }

    def get_measurement(self) -> float:
        # Auto-syncs with MuJoCo
        return self.mujoco_data.sensordata[self.sensor_id]

    def validate(self) -> bool:
        # Self-validates
        return self.get_measurement() > 0.0
```

**Result:** Custom modal integrates seamlessly with existing system!

---

## Modal Philosophy

> **"Modals self-declare, self-sync, and self-validate - enabling zero-config robotic programming."**

This is the power of Modal-Oriented Programming:
- **No manual setup** - modals discover themselves
- **No manual sync** - modals auto-sync with physics
- **No manual validation** - modals self-validate through behaviors

**The result:** Clean, declarative code that focuses on **what** you want, not **how** to do it.

---

## Cross-References

- **[ROBOT_OS_OVERVIEW.md](./ROBOT_OS_OVERVIEW.md)** - MOP philosophy
- **[VALIDATION_FRAMEWORK.md](./VALIDATION_FRAMEWORK.md)** - How modals enable 5-way validation
- **[API_QUICKSTART.md](./API_QUICKSTART.md)** - Using modals in practice
- **[CAPABILITY_MATRIX.md](./CAPABILITY_MATRIX.md)** - Complete capability list

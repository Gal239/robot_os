# ROBOT OS: A Modal-Oriented Operating System for Robotics

## What Is This?

**Robot OS** is not just a framework - it's a complete operating system for robotic simulation and control built on **Modal-Oriented Programming (MOP)**.

Think of it as:
- **Linux** for robot control (complete, self-contained, composable)
- **React** for physics simulation (declarative, self-validating)
- **Docker** for experiment reproducibility (containerized scenes, portable)

## The Core Philosophy

### Traditional Robotics Framework:
```python
# 50+ lines to set up environment
env = gym.make("FetchReach-v1")
env.reset()
# Manually track rewards
# Manually validate actions
# Manually save/load state
# Manually configure cameras
# Manually sync physics
```

### Robot OS (MOP):
```python
# 5 lines - everything auto-configured
ops = ExperimentOps()
ops.create_scene("room", width=5, length=5)
ops.add_robot("stretch")
ops.compile()  # Auto-discovers modals, sensors, actuators
ops.step()      # Auto-syncs, auto-validates, auto-saves
```

**Why?** Because modals **self-declare** their properties, capabilities, and validation logic.

---

## Modal-Oriented Programming (MOP)

### What is a Modal?

A **modal** is a self-contained unit that:
1. **Declares** what it is (sensor, actuator, asset, etc.)
2. **Exposes** its behaviors (extension, rotation, holding, etc.)
3. **Validates** itself (rewards, thresholds, constraints)
4. **Syncs** automatically (MuJoCo ↔ Python state)

### Example: Arm Modal

```python
class ArmModal(ActuatorModal):
    def __init__(self):
        self.behaviors = {
            "extension": self.get_extension,  # Self-declares behavior
            "at_target": self.check_target     # Self-validates
        }

    def get_extension(self) -> float:
        return self.mujoco_data.qpos[self.joint_id]  # Auto-syncs
```

**Result:** When you write `ops.add_reward("stretch.arm", "extension", 0.3, reward=100)`, the system:
1. Finds the arm modal
2. Discovers the "extension" behavior
3. Auto-validates when threshold met
4. Auto-awards reward
5. **Zero manual configuration!**

---

## Why Is This Powerful?

### 1. Minimal Code, Maximum Capability

**Stack 3 Objects with Auto-Height Calculation:**
```python
ops.add_asset("foam_brick", relative_to="table", relation="on_top")
ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")
ops.add_asset("cracker_box", relative_to="pudding_box", relation="stack_on")
```

3 lines → Perfect stack with automatic height calculation, collision detection, and stability validation.

### 2. Self-Validating Through Rewards

**Traditional:**
```python
# Manual validation (error-prone!)
arm.extend(0.3)
time.sleep(2.0)
actual = arm.get_position()
assert abs(actual - 0.3) < 0.05, "Failed!"
```

**Robot OS:**
```python
# Reward validates automatically
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
# If reward received → action validated! Zero manual checks.
```

### 3. Multi-Modal Validation

Every capability can be validated through **5 modalities**:
1. **Physics**: MuJoCo state proves behavior
2. **Semantic**: Modal behaviors confirm correctness
3. **Vision**: Camera captures visual proof
4. **Reasoning**: Spatial/temporal consistency validated
5. **Action Image**: Action execution captured visually

### 4. Automatic Everything

- **Auto-Discovery**: Sensors/actuators found automatically
- **Auto-Sync**: MuJoCo ↔ Python state synchronized every step
- **Auto-Save**: Complete experiment saved to database
- **Auto-Load**: Resume from any previous state
- **Auto-Validate**: Rewards confirm action completion
- **Auto-Configure**: Cameras, sensors, rewards all self-setup

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ROBOT OS LAYERS                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  USER API (ExperimentOps)                             │  │
│  │  • create_scene()   • add_robot()   • add_reward()   │  │
│  │  • compile()        • step()        • get_state()    │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  MODAL LAYER (Auto-Discovery)                         │  │
│  │  • RobotModal  • SensorModal  • ActuatorModal         │  │
│  │  • AssetModal  • SceneModal   • RewardModal           │  │
│  │  [Self-declaring, self-validating, self-syncing]     │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  PHYSICS ENGINE (MuJoCo)                              │  │
│  │  • Rigid body dynamics  • Collision detection         │  │
│  │  • Sensor simulation    • Actuator control            │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  DATABASE LAYER (Auto-Persistence)                    │  │
│  │  • Experiment metadata  • Views (TIME TRAVELER)       │  │
│  │  • Rewards tracking     • Complete reproducibility    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Complete Example: "Hello Robot OS"

```python
from simulation_center.core.main.experiment_ops_unified import ExperimentOps

# 1. Initialize
ops = ExperimentOps(headless=True, render_mode="vision_rl")

# 2. Build scene (auto-creates room with floor, walls, ceiling)
ops.create_scene("kitchen", width=8, length=8, height=3)

# 3. Add robot (auto-discovers all sensors + actuators)
ops.add_robot("stretch", position=(0, 0, 0))

# 4. Add furniture (spatial relations auto-calculated)
ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))

# 5. Stack objects (heights auto-calculated)
ops.add_asset("foam_brick", relative_to="table", relation="on_top")
ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")

# 6. Add nested containers
ops.add_asset("bowl", relative_to="pudding_box", relation="stack_on")
ops.add_object("apple", position={"relative_to": "bowl", "relation": "inside"})

# 7. Add free camera (auto-configured)
ops.add_free_camera("view1", lookat=(2.0, 0.0, 0.8), distance=3.0, azimuth=45)

# 8. Add reward (auto-validates through modal behaviors)
ops.add_reward(
    tracked_asset="stretch.gripper",
    behavior="holding",
    target="apple",
    threshold=True,
    reward=100,
    id="apple_grasped"
)

# 9. Compile (auto-discovery happens here!)
ops.compile()

# 10. Execute action
from simulation_center.core.modals.stretch.action_blocks_registry import move_forward
block = move_forward(distance=1.0, speed=0.3)
ops.submit_block(block)

# 11. Run simulation (auto-sync, auto-validate, auto-save)
for step in range(1000):
    result = ops.step()
    if result['reward'] > 0:
        print(f"✅ Validated by reward system at step {step}!")
        break

# 12. Save experiment (auto-persisted to database)
# Experiment ID: ops.experiment_dir
# Complete reproducibility - can resume from any step!
```

**Result:** In ~40 lines, you created:
- A rich 3D environment
- A 4-level object stack (table → foam → pudding → bowl → apple)
- Spatial relationships (on_top, stack_on, inside)
- Free camera for vision validation
- Self-validating reward system
- Complete database persistence
- Action execution system

---

## What Makes This an "OS"?

### 1. Complete Abstraction
Like Linux abstracts hardware, Robot OS abstracts:
- Physics engines (MuJoCo, PyBullet, Isaac, etc.)
- Robots (Stretch, Franka, UR5, etc.)
- Sensors (cameras, LIDAR, IMU, etc.)
- Actuators (joints, grippers, mobile bases, etc.)

### 2. Self-Contained
Everything needed for robotic simulation/control in one system:
- Scene management ✓
- Robot control ✓
- Sensor fusion ✓
- Reward validation ✓
- Vision processing ✓
- Database persistence ✓
- Experiment reproducibility ✓

### 3. Declarative API
Like React's JSX, you declare what you want, not how to do it:
```python
ops.add_asset("apple", relative_to="table", relation="on_top")
# System figures out: height, collision, physics, rendering
```

### 4. Composable
Modals compose like Unix pipes:
```python
# Robot modal contains:
├── Base modal (odometry, movement)
├── Arm modal (extension, joints)
├── Gripper modal (holding, force)
├── Camera modal (RGB, depth)
└── Sensor modals (LIDAR, IMU, etc.)

# Each auto-discovered, auto-synced, auto-validated!
```

---

## Progressive Capability Ladder

Robot OS is designed to be learned progressively:

**Level 1A** → Modal architecture (foundation)
**Level 1B** → Action system (robot control)
**Level 1C** → Action queues (complex orchestration)
**Level 1D** → Sensor system (perception)
**Level 1E** → Rewards (validation)
**Level 1F** → Scene composition (spatial relations)
**Level 1G** → View system (TIME TRAVELER)
**Level 1H** → Vision validation (explainability)
**Level 1I** → Object behaviors (physics reasoning)
**Level 1K** → Advanced placement (multi-modal validation)
**Level 1L** → Composite rewards (logic gates)
**Level 1M** → **FINAL BOSS** (complete integration)

Each level builds specific skills, enabling increasingly sophisticated behaviors with **minimal code**.

---

## The "Wow Factor"

### Stack 3 Objects: 3 Lines
```python
ops.add_asset("foam_brick", relative_to="table", relation="on_top")
ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")
ops.add_asset("cracker_box", relative_to="pudding_box", relation="stack_on")
```
**Automatic:** Height calculation, collision detection, stability validation, 100-step physics verification

### Self-Validating Test: 5 Lines
```python
ops.add_reward("stretch.arm", "extension", 0.3, reward=100)
block = extend_arm(extension=0.3)
ops.submit_block(block)
total_reward = sum(ops.step()['reward'] for _ in range(500))
assert total_reward >= 100, "Action self-validated!"
```
**Automatic:** Threshold detection, reward triggering, action validation - **zero manual checks!**

### Vision Validation: 3 Lines
```python
ops.add_free_camera("cam", lookat=(2,0,0.8), distance=3.0)
ops.step()
cv2.imwrite("proof.png", ops.engine.last_views["cam_view"]["rgb"])
```
**Automatic:** Camera setup, rendering, image capture, proof saving

---

## Next Steps

- **[CAPABILITY_MATRIX.md](./CAPABILITY_MATRIX.md)** - Complete list of all capabilities
- **[VALIDATION_FRAMEWORK.md](./VALIDATION_FRAMEWORK.md)** - 5-way validation explained
- **[API_QUICKSTART.md](./API_QUICKSTART.md)** - 10 minimal code examples
- **[MODAL_CATALOG.md](./MODAL_CATALOG.md)** - All modals + their behaviors

---

## Philosophy

> **"The best code is no code."**
> Robot OS makes complex robotic behaviors achievable in 3-50 lines through self-declaring modals, auto-validation, and multi-modal verification.

**Traditional Framework:** 200+ lines for basic manipulation
**Robot OS:** 15 lines for complex multi-stage task

This is the power of Modal-Oriented Programming.

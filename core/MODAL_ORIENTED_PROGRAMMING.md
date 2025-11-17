# MODAL-ORIENTED PROGRAMMING
## Self-Aware Data Structures That Dream

---

## What is Modal-Oriented Programming?

**Modal-Oriented Programming (MOP)** is a paradigm where data structures are self-aware: they know their own structure, generate their own specifications, render themselves in multiple formats, and compose with zero configuration.

**The key insight:** Instead of writing code to manipulate passive data, you write **modals** - data structures that understand themselves well enough to teach the system how to use them.

---

## 🧵 THE SILK THREAD

Every modal follows 5 core principles - THE SILK THREAD that connects the entire system:

### 1. **AUTO-DISCOVERY**
Modals discover their own structure from minimal input.

**Example - Assets from XML:**
```python
# Traditional: Manually configure every property
asset_config = {
    "components": {
        "door": {
            "behaviors": ["hinged", "lockable"],
            "geom_names": ["door_frame", "door_panel"],
            "joint_names": ["door_hinge"],
            # ... 50 more lines of manual config
        }
    }
}

# Modal-Oriented: Asset discovers itself from XML
asset = Asset("door", {})  # Just the name!
# Asset:
# 1. Loads door.xml from registry (filesystem scan)
# 2. Extracts all geoms, joints, sites using XMLResolver
# 3. Infers behaviors from naming patterns:
#    - "hinge" in joint name → hinged behavior
#    - "grasp_*" site → graspable behavior
# 4. Reads property definitions from BEHAVIORS.json
# NO MANUAL CONFIGURATION!
```

**The Magic - Semantic Naming:**
```xml
<!-- door.xml -->
<mujoco>
  <body name="door">
    <geom name="door_panel" .../>
    <joint name="door_hinge" type="hinge"/>
    <site name="grasp_handle"/>
    <site name="place_on_top"/>
  </body>
</mujoco>
```

The system understands:
- `*_hinge` → hinged behavior
- `grasp_*` → graspable
- `place_*` → surface

**LLM can reason about this!** It sees `grasp_handle` and knows "this is where you grasp."

---

### 2. **SELF-GENERATION**
Modals create their own JSON specifications - not humans!

**The Problem with Traditional Systems:**
```python
# Human manually writes robot_behaviors.json:
{
  "vision": {
    "description": "Visual sensing",
    "properties": {
      "target_visible": {"unit": "boolean", "default": null},
      "target_distance": {"unit": "meters", "default": null}
    }
  }
}
# Then manually writes sensor code:
class Camera:
    def __init__(self):
        self.behaviors = ["vision"]  # Must match JSON!

# PROBLEM: Two sources of truth! Drift inevitable!
```

**Modal-Oriented Solution:**
```python
# Sensor self-declares behaviors:
class NavCamera(BaseModel):
    sensor_id: Literal['nav'] = 'nav'
    rgb_image: Any
    depth_image: Any
    timestamp: float

    # Self-declare trackability
    behaviors: List[str] = ["vision", "robot_head_spatial"]
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["nav_camera_site"]

# Robot scans sensors and generates JSON:
robot = create_robot("stretch")
package = robot.create_robot_asset_package()
# Scans ALL sensors with .behaviors → auto-generates JSON!
# Single source of truth: THE MODAL!
```

**Key Principle:** "Modal create the json not the other way around"

---

### 3. **SELF-RENDERING**
Modals know how to display themselves in multiple formats.

**Every modal has views:**
```python
component = asset.components["gripper"]

# Human-readable view
data = component.get_data()
# {
#   "aperture": 0.15,
#   "closed": False,
#   "holding": True,
#   "position": [0.5, 0.3, 0.9],
#   "distance_to": 0.12
# }

# RL training view (auto-normalized from BEHAVIORS.json)
rl = component.get_rl()
# [0.75, 0.0, 1.0, 0.5, 0.3, 0.9, 0.12]  # Flat, normalized vector

# XML view
xml = asset.render_xml()
# <body name="gripper">...</body>

# Timeline view (for rewards)
timeline = reward.timeline()
# Visual representation of reward progress over time

# Validation view (simulation: sensor vs physics ground truth!)
state = extract_robot_gripper(model, data, component, all_assets, robot)
# {
#   "closed": True,
#   "holding": True,
#   "_sensor_force": 10.2,    # What sensor reports
#   "_physics_force": 10.5,   # Ground truth from MuJoCo
#   "_validated": True        # Sensor matches physics!
# }
```

**The Dual-Signal Pattern (Simulation Only):**

In simulation, we have BOTH signals:
- **Sensor**: What the robot's sensor reports (mimics real robot)
- **Physics**: Ground truth from MuJoCo contacts

This enables **self-validation**: the modal compares sensor vs physics and tells you if they match!

```python
# Force-based gripping extractor - SELF-VALIDATING!
def extract_robot_gripper(model, data, component, all_assets, robot):
    # 1. Get SENSOR signal (if robot available)
    sensor_force = robot.sensors['gripper_force'].get_data()['force_left']

    # 2. Get PHYSICS signal (ground truth from contacts)
    physics_force = compute_contact_force(model, data, component.geom_names)

    # 3. SELF-VALIDATE: Compare them
    validated = abs(sensor_force - physics_force) < 1.0  # Within 1N

    # 4. Use sensor (real robot mode) or physics (sim-only mode)
    holding_force = sensor_force if sensor_force > 0.1 else physics_force

    return {
        "closed": holding_force > 2.0,
        "_sensor_force": sensor_force,
        "_physics_force": physics_force,
        "_validated": validated  # Modal tells you if sensor works!
    }
```

**Why This Matters:**
- **Sensor Development**: Build/tune sensors in sim, validate against physics
- **Debugging**: If `_validated=False`, sensor is misconfigured
- **Sim-to-Real**: Same code runs in sim (dual-signal) and real robot (sensor-only)
- **Self-Debugging**: Modal tells you if it's broken!

**One modal, five+ formats - no manual conversion!**

---

### 3.5. **SELF-PRESENTATION** (The Elegance Layer)
Modals introduce themselves gracefully - both to humans AND other modals.

**The Problem with Traditional Objects:**
```python
# What is this?
>>> my_object
<__main__.TrainingConfig object at 0x7f8b4c3a1d90>
# Useless! No information!

# Have to manually inspect
>>> print(my_object.name, my_object.algorithm, my_object.params)
"Lesson 1" "PPO" {"lr": 0.0003}
# Tedious and error-prone
```

**Modal-Oriented (Self-Presenting):**
```python
# Modals introduce themselves!
>>> lesson
LessonModal('Spin 15°', 15.0°, easy)
# INSTANT understanding!

>>> policy
PolicyModal(algorithm='PPO', lr=0.0003)
# Clean and informative!

>>> curriculum
CurriculumModal('Spin Curriculum', 3 lessons)
# Tells you EXACTLY what it is!

# Developer view (repr) shows more detail
>>> repr(lesson)
LessonModal(name='Spin 15°', target_angle=15.0, action_space=['base'], difficulty='easy')
```

**The Implementation:**
```python
class LessonModal(BaseModel):
    name: str
    target_angle: float
    action_space: List[str]

    def __str__(self) -> str:
        """SELF-PRESENTATION: How I introduce myself to users!"""
        return f"LessonModal('{self.name}', {self.target_angle}°, {self.get_difficulty()})"

    def __repr__(self) -> str:
        """SELF-PRESENTATION: How I introduce myself to developers!"""
        return (
            f"LessonModal(name='{self.name}', target_angle={self.target_angle}, "
            f"action_space={self.action_space}, difficulty='{self.get_difficulty()}')"
        )
```

**Why This Matters for LEGO Composition:**

When modals compose with each other, they can introduce themselves:
```python
# TrainingSessionModal printing its configuration
>>> session = TrainingSessionModal(curriculum=curr, policy=pol)
>>> session.run()

📋 Session Configuration:
  Curriculum: CurriculumModal('Spin Curriculum', 3 lessons)
  Policy: PolicyModal(algorithm='PPO', lr=0.0003)
  Baselines: 1
    - VanillaBaselineModal(action_space=FULL)
```

**Modals talking to modals!** Each modal can inspect others by just printing them:
```python
# Inside CurriculumModal.train()
for lesson in self.lessons:
    print(f"Training on: {lesson}")  # Self-presenting!
    # Output: "Training on: LessonModal('Spin 15°', 15.0°, easy)"
```

**Benefits:**
- **Debugging**: Instantly see what modal you're working with
- **Logging**: Modal introduces itself in logs (no manual formatting!)
- **LEGO Composition**: Modals discover each other's identity without inspection
- **LLM Understanding**: LLMs can read modal identities from logs
- **Human-Friendly**: Print any modal, get useful info (not memory address!)

**The Principle:** "Every modal must be able to introduce itself elegantly, both to humans and to other modals."

---

### 4. **LEGO COMPOSITION**
Components snap together with zero configuration.

**The Old Way:**
```python
# Add new sensor → update 5 files:
# 1. sensors.py - define sensor class
# 2. robot_config.json - register sensor
# 3. scene_wrapper.py - add wrapping logic
# 4. reward_system.py - add tracking code
# 5. behaviors.json - add behavior definition
# Miss one? Silent failure or crash!
```

**The Modal Way:**
```python
# 1. Add .behaviors field to sensor Pydantic model:
class NewSensor(BaseModel):
    behaviors: List[str] = ["temperature_sensing"]
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["temp_sensor_site"]

# 2. That's it! System auto-integrates:
# ✓ Robot scans sensors → finds NewSensor
# ✓ Generation scans behaviors → creates JSON entry
# ✓ Scene wraps sensor → trackable asset
# ✓ Rewards can use it → temperature.hot_detected
# ONE CHANGE, FIVE EFFECTS!
```

**LEGO Principle:** Add field → system includes it. Remove field → system ignores it.

---

### 5. **MODAL-TO-MODAL COMMUNICATION**
Modals communicate directly - NO GLUE CODE!

**The Old Way (Glue Everywhere):**
```python
# Manual mapping in 5 places:
# 1. Gym wrapper extracts observations
obs = np.concatenate([
    robot.sensors['lidar'].read(),  # Hardcoded!
    robot.sensors['imu'].read(),
    robot.actuators['arm'].position
])

# 2. Gym wrapper applies actions
robot.actuators['arm'].move_to(action[0])  # Hardcoded index!
robot.actuators['lift'].move_to(action[1])

# 3. Reward computer extracts state
reward = compute_distance(robot.get_position(), target)  # Manual!

# 4. Termination checker
if robot.get_rotation() > threshold:  # Manual check!
    done = True
```

**The Modal Way (Direct Communication):**
```python
# Modals talk to each other - NO GLUE!

# GymBridge → ExperimentOps → Modals
obs = view_aggregator.get_obs()  # ViewAggregator asks sensors!
ops.apply_action(action, actuators_active)  # Actuators execute!
reward, info = ops.evaluate_rewards()  # RewardModals compute!
done = ops.check_termination()  # RewardModals know!

# ZERO manual mapping!
# ZERO hardcoded indices!
# ZERO glue code!
```

**The Pattern:**
```
GymBridge: "Execute action"
     ↓
ExperimentOps: "Which actuators?"
     ↓
ActuatorModals: "I execute my action!" (SELF-EXECUTING!)
     ↓
MuJoCo: Physics simulation
     ↓
SensorModals: "I sync my state!" (SELF-SYNCING!)
     ↓
RewardModals: "I compute my reward!" (SELF-COMPUTING!)
     ↓
RewardModals: "I know if task done!" (SELF-TERMINATING!)
```

**No coordinator! Modals communicate peer-to-peer!**

---

## Core Patterns

### Pattern 1: Generic Category Computation

**The Old Way (Hardcoded):**
```python
def get_direction(angle):
    if angle < -0.5:
        return "right"
    elif angle < 0.5:
        return "forward"
    elif angle < 1.5:
        return "left"
    else:
        return "backward"
# Add new category? Edit code!
# Change threshold? Edit code!
# LLM can't see the logic!
```

**Modal-Oriented (Data-Driven):**
```json
{
  "direction": {
    "category_source": "angle_rad",
    "category_thresholds": [
      {"max": -0.5, "value": "right"},
      {"max": 0.5, "value": "forward"},
      {"max": 1.5, "value": "left"},
      {"max": 3.14, "value": "backward"}
    ]
  }
}
```

```python
# Generic computation reads from BEHAVIORS.json:
def get_state(self, prop_name):
    prop_data = BEHAVIORS[self.behavior]["properties"][prop_name]
    category_source = prop_data["category_source"]
    source_value = self.get_state(category_source)

    for threshold in prop_data["category_thresholds"]:
        if source_value <= threshold["max"]:
            return threshold["value"]
```

**Benefits:**
- Add new category → just edit JSON (no code changes!)
- Change threshold → just edit JSON
- LLM can read the logic directly
- Single source of truth

---

### Pattern 2: Modal Trust (No isinstance/hasattr Hacks)

**The Old Way (Defensive):**
```python
# Constantly checking types:
if isinstance(component, RobotComponent):
    # Handle robot component
elif isinstance(component, ObjectComponent):
    # Handle object component
elif isinstance(component, dict):
    # Handle dict
else:
    # Handle unknown?

# Constantly checking attributes:
if hasattr(sensor, 'behaviors'):
    if sensor.behaviors:
        # Trackable
```

**Modal-Oriented (Trust):**
```python
# If sensor has .behaviors, it's trackable. PERIOD.
for sensor in robot.sensors.values():
    if hasattr(sensor, 'behaviors') and sensor.behaviors:
        # Wrap as trackable asset
        # System trusts the modal declaration!
```

**Why this matters:**
- Modal declares: "I have behaviors, I'm trackable"
- System trusts: "You have behaviors? You're trackable!"
- No polymorphic hacks, no type checking
- Duck typing + Pydantic validation = safe trust

---

### Pattern 3: Offensive Programming

**The Old Way (Defensive):**
```python
try:
    behavior_data = BEHAVIORS.get(behavior_name, {})
    properties = behavior_data.get("properties", {})
    # Silent failure - missing behavior → empty dict
except Exception as e:
    logger.warning(f"Behavior lookup failed: {e}")
    return default_value
# Developer never knows something is wrong!
```

**Modal-Oriented (Offensive):**
```python
# Crash with educational error:
if behavior_name not in BEHAVIORS:
    available = list(BEHAVIORS.keys())
    raise KeyError(
        f"Behavior '{behavior_name}' not found!\n"
        f"Available behaviors: {available}\n"
        f"Did you forget to regenerate ROBOT_BEHAVIORS.json?\n"
        f"Run: python3 -m core.ops.config_generator"
    )
# Crash educates developer HOW TO FIX IT!
```

**Benefits:**
- Errors happen at development time (not production!)
- Error messages teach how to fix
- No silent failures
- No defensive cruft hiding bugs

---

### Pattern 4: MOP Anti-Patterns (What NOT to Do)

**Learn from real bugs! The `KeyError: 'left_wheel_vel'` incident.**

#### Anti-Pattern 1: External Config Lookup (GLUE CODE)
```python
# ❌ BAD - External JSON dependency
config = load_config("robot.json")
value = config["components"][name]  # Can be incomplete!

# ✅ GOOD - Self-contained modal
value = component.placement_site  # Component knows itself!
```

**Why Bad**: External files drift from source of truth. Config had 10 components, robot XML had 12 → crash!

#### Anti-Pattern 2: Hardcoded Lists (Manual Maintenance)
```python
# ❌ BAD - Manual list, drifts from XML
"components": ["arm", "lift", "gripper"]  # Forgot wheels!

# ✅ GOOD - Auto-discovery
specs = _discover_from_xml()  # Scans ALL actuators
```

**Why Bad**: Humans forget to update parallel lists. Adding actuator to XML requires ALSO updating JSON → guaranteed drift!

#### Anti-Pattern 3: Defensive .get() (Hides Bugs)
```python
# ❌ BAD - Silent failure if data missing
value = data.get("position", 0.0)  # Wrong default masks bug!

# ✅ GOOD - Crash if data missing (forces fix)
value = data["position"]  # Modal MUST be complete!
```

**Why Bad**: Defaults hide incomplete modals. Better to crash during development than fail silently in production!

#### Real Bug: The KeyError Incident

**What Happened**:
- `config.json` had HARDCODED 10 components (manual list)
- Robot XML auto-discovered 12 actuators (including `left_wheel_vel`, `right_wheel_vel`)
- Scene tried to lookup `left_wheel_vel` in config → **KeyError!**

**Root Cause**:
1. **GLUE CODE**: External config.json instead of self-contained modal
2. **MANUAL LIST**: Someone typed 10 components, forgot wheels
3. **NO VALIDATION**: System didn't detect incomplete config

**The Fix**:
1. Added `placement_site` field to ActuatorComponent modal
2. Removed config.json dependency
3. Actuators now **SELF-DECLARE** their placement_site
4. Impossible to have incomplete data (auto-discovered from XML!)

**Lesson**: External configs are technical debt. Self-contained modals eliminate entire class of bugs!

---

#### Real Bug 2: The 'text' Actuator Incident (Test 14 Crash)

**What Happened**:
- Test 14 `test_speaker_integration` crashed with: `ValueError: Actuator 'text' not found in model`
- The test submitted `SpeakerPlay(text="Starting arm movement")`
- The action executed, but sent WRONG dict format to MuJoCo backend!

**Root Cause Analysis** (Full trace):
1. **Test submits**: `ops.submit_action(SpeakerPlay(text="Starting arm movement"))`
   → level_1c_action_queues.py:973

2. **ExecutionQueue.get_next_commands()** calls `action.tick()`
   → execution_queue_modal.py:262

3. **Action.tick()** calls `normalize_command(self._get_command())`
   → action_modals.py:233

4. **SpeakerPlay._get_command()** returns WRONG format:
   ```python
   return {'text': self.text, 'type': 'tts'}  # ❌ NOT MuJoCo format!
   ```
   → action_modals.py:904

5. **normalize_command()** checks `isinstance(cmd, dict)` → returns as-is
   → action_modals.py:195-196

6. **Commands dict updated**: `commands['text'] = 'Starting arm movement'`
   → execution_queue_modal.py:269

7. **Backend tries to send**: `set_controls({'text': 'Starting arm movement'})`
   → runtime_engine.py:769

8. **MuJoCo backend crashes**: `raise ValueError(f"Actuator 'text' not found in model")`
   → mujoco_backend.py:160

**Why It Crashed So Late**:
- **NO EARLY VALIDATION!** `SpeakerPlay` claims to be a valid action
- `_get_command()` returned WRONG dict format (application data, not MuJoCo commands)
- Code blindly passed dict through until MuJoCo rejected it
- **If code was truly OFFENSIVE**, it would crash at `_get_command()` with:
  `"SpeakerPlay is virtual (no MuJoCo actuator) - cannot send commands to backend!"`

**The Root MOP Violation**:
```python
# ❌ BAD - Virtual action returns app-level dict
def _get_command(self):
    return {'text': self.text, 'type': 'tts'}  # NOT MuJoCo format!

# ✅ GOOD - Virtual actions must return empty dict OR self-skip
def _get_command(self):
    # Speaker is VIRTUAL (no MuJoCo representation)
    # Return empty dict - runtime will skip (no joint_names)
    return {}
```

**The Fix Options**:
1. **Option A**: Return `{}` from `_get_command()` (skip backend send)
2. **Option B**: Check `execution_type == "non_actuator"` in `tick()` and skip command generation
3. **Option C**: Make backend smarter - detect virtual actuators (already has `if not actuator.joint_names: continue`)

**Current Status**: Backend already skips virtual actuators (runtime_engine.py:367-368)!
**Real Bug**: `SpeakerPlay._get_command()` returns wrong dict format. Should return `{'speaker': <audio_command>}` or `{}`.

**Lesson**: Virtual components (speaker, logger, etc.) need special handling. Don't send application data to physics backend!

---

### Pattern 5: Pure MOP - Instance Flow (No Type Loss)

**The Problem: Type Conversion Loses Data**
```python
# BAD: Converting instances to dicts loses type information!
def _build_from_config(self):
    components = {}
    for name, comp_data in self.config['components'].items():
        if isinstance(comp_data, Component):
            # Convert to dict for "flexibility"
            comp_dict = {
                'name': comp_data.name,
                'behaviors': comp_data.behaviors,
                'geom_names': comp_data.geom_names,
                # LOST: ActuatorComponent.tolerance field!
            }
            components[name] = Component(**comp_dict)
        # ...
# Result: ActuatorComponent(tolerance=0.0143) → Component (no tolerance!)
# Reward system can't validate positions with physics tolerance!
```

**The Modal Solution: Instances Flow Directly**
```python
# GOOD: Pure MOP - instances flow without conversion!
def _normalize_components(self):
    """Convert dicts to instances in ONE place"""
    normalized = {}
    for comp_name, comp_data in self.config['components'].items():
        if isinstance(comp_data, Component):
            # Already an instance - use directly!
            normalized[comp_name] = comp_data
        else:
            # Dict from JSON - convert once
            normalized[comp_name] = Component(...)
    self.config['components'] = normalized

def _build_from_config(self) -> Dict[str, Component]:
    """OFFENSIVE + Pure MOP - instances only!"""
    components = {}
    for comp_name, comp_data in self.config['components'].items():
        # OFFENSIVE: Crash if not instance!
        assert isinstance(comp_data, Component), (
            f"MOP violation! Expected Component instance, got {type(comp_data)}"
        )
        # Pure MOP: Instance flows directly - NO conversion!
        components[comp_name] = comp_data
    return components
```

**Real Victory: Tolerance-Aware Rewards (Level 1B)**
```python
# Scene creates ActuatorComponent with auto-discovered tolerance:
actuator_comp = ActuatorComponent(
    name="lift",
    behaviors=["lift"],
    position=0.497,      # Actual position from physics
    range=(0.0, 1.095),
    tolerance=0.0143     # Auto-measured from position error
)

# Asset receives instance - no conversion!
asset = Asset("stretch", config={'components': {'lift': actuator_comp}})
# actuator_comp flows through unchanged

# Reward system validates with tolerance!
reward = RewardModal(tracked_asset="stretch.lift", behavior="position")
result = reward.check(target=0.5)
# Checks: 0.497 >= (0.5 - 0.0143) = 0.4857 ✓ SUCCESS!
```

**Benefits:**
- **No type loss**: ActuatorComponent instances preserve all fields (tolerance, position, range)
- **Auto-discovery**: Tolerance (0.0143m) measured from actual physics, not hardcoded
- **Self-validation**: Reward system validates position against physics-based tolerance
- **LEGO composition**: Add field to ActuatorComponent → flows through entire system automatically
- **ONE path**: No if/else for dict vs instance → fewer bugs

**The Principle:** "Modals communicate via instances, not dicts. Once an instance, always an instance."

---

## The Generation Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MODAL DECLARATION (Single Source of Truth)              │
│    Sensors/Actuators self-declare behaviors via Pydantic   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. GENERATION (Modal → JSON)                                │
│    robot.create_robot_asset_package():                      │
│    - Scans ALL actuators                                    │
│    - Scans ALL sensors with .behaviors                      │
│    - Generates behavior definitions                         │
│    - Generates component mappings                           │
│    - OFFENSIVE validation (crashes if incomplete)           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. ROBOT_BEHAVIORS.json (Generated Output)                  │
│    18 behaviors: 13 actuators + 5 sensors                   │
│    Vision, distance_sensing, motion_sensing, tactile, etc.  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. RUNTIME (Scene Auto-Wraps)                               │
│    scene.add_robot():                                       │
│    - Wraps actuators as trackable assets                    │
│    - Wraps sensors with .behaviors as trackable assets      │
│    - Uniform interface: ALL components have same API        │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. REWARDS (Use Trackable Assets)                           │
│    scene.add_reward("stretch.nav_camera", "target_visible") │
│    → Works! Camera auto-tracked because it has .behaviors   │
└─────────────────────────────────────────────────────────────┘
```

**Key:** Modals → JSON → Runtime (ONE DIRECTION!)

---

## Comparison: Traditional vs Modal-Oriented

### Adding a New Temperature Sensor

**Traditional OOP:**
```
1. Write sensor class (sensors/temperature.py)
2. Register in robot config (configs/robot.json)
3. Add to sensor factory (sensor_factory.py)
4. Update scene wrapper (scene_wrapper.py)
5. Add behavior definition (behaviors.json)
6. Update reward system (reward_tracker.py)
7. Update documentation (docs/sensors.md)
8. Write tests (tests/test_temperature.py)

Files touched: 8
Lines of code: ~300
Time: 2 hours
Risk: High (miss one integration point → silent failure)
```

**Modal-Oriented:**
```python
# 1. Add sensor to sensors_modals.py:
class TemperatureSensor(BaseModel):
    temperature: float
    timestamp: float

    # Self-declare trackability
    behaviors: List[str] = ["temperature_sensing"]
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["temp_sensor_site"]

# 2. Regenerate:
# python3 -m core.ops.config_generator

# That's it! System auto-integrates:
# ✓ Generation scans → creates behavior JSON
# ✓ Scene wraps → trackable asset
# ✓ Rewards work → temperature.hot_detected
# ✓ LEGO composition → just works!

Files touched: 1
Lines of code: ~12
Time: 5 minutes
Risk: Low (OFFENSIVE validation catches mistakes)
```

---

## PURE MOP RL ARCHITECTURE - THE ULTIMATE LEGO

**The Thesis**: RL training is the ULTIMATE test of MODAL-TO-MODAL COMMUNICATION. If 8+ modals can snap together with ZERO GLUE CODE, the architecture is PURE MOP.

---

### The Problem with Traditional RL (200+ Lines of Glue)

**Traditional RL setup:**
```python
# 1. Manual observation space (50 lines)
def get_obs():
    lidar = robot.sensors['lidar'].read()  # Hardcoded!
    imu = robot.sensors['imu'].read()
    joints = [robot.joints[i].position for i in range(15)]  # Magic 15!
    return np.concatenate([lidar, imu, joints])  # Hope dimensions match!

# 2. Manual action space (30 lines)
def apply_action(action):
    for i, joint in enumerate(robot.joints):
        joint.set_position(action[i])  # Which joint is index i?

# 3. Manual reward computation (40 lines)
def get_reward():
    pos = robot.get_base_position()
    target_pos = get_target_position()
    dist = np.linalg.norm(pos - target_pos)
    return -dist

# 4. Manual termination check (20 lines)
def is_done():
    if robot.get_rotation() > threshold:
        return True
    return False

# 5. Manual episode reset (60 lines)
def reset():
    # Recompile entire scene (300ms!)
    scene.compile()
    # Reset all states manually
    for joint in robot.joints:
        joint.reset()
    # etc...
```

**Problems:**
- ❌ 200+ lines of manual mapping
- ❌ Hardcoded sensor/actuator indices
- ❌ Recompile entire scene every episode (300ms+)
- ❌ Reward logic scattered everywhere
- ❌ No termination condition
- ❌ Magic numbers (15D, 42D, etc.)
- ❌ Can't reuse for different robots

---

### The PURE MOP Solution (ZERO GLUE!)

**PURE MOP RL - 4 NEW METHODS:**

```python
class ExperimentOps:
    """ExperimentOps with PURE MOP RL API - Modals communicate directly!"""

    def reset(self):
        """Modals reset themselves - 10-30x FASTER than recompile!"""
        self.backend.reset_to_keyframe('initial')  # 10ms vs 300ms!
        self.robot.reset()  # Robot modal resets
        self.scene.reward_modal.reset()  # Reward modal resets
        self.sync_from_mujoco()
        # MODAL-TO-MODAL: Each modal knows how to reset itself!

    def apply_action(self, action: np.ndarray, actuators_active: List[str]):
        """Modals execute actions - NO MANUAL MAPPING!"""
        active_actuators = {
            name: actuator
            for name, actuator in self.robot.actuators.items()
            if actuators_active is None or name in actuators_active
        }

        commands = {}
        for i, (name, actuator) in enumerate(active_actuators.items()):
            if i < len(action):
                target = float(action[i])
                commands[name] = actuator.move_to(target)  # Modal executes!

        self.backend.set_controls(commands)
        # MODAL-TO-MODAL: Actuator modals execute their own actions!

    def evaluate_rewards(self) -> Tuple[float, Dict]:
        """RewardModals compute their own rewards - SELF-COMPUTING!"""
        state = self.get_state()
        total_reward = self.engine.reward_computer.compute(
            state,
            self.scene.reward_modal,
            current_time,
            self.scene.reward_modal.start_time
        )
        # MODAL-TO-MODAL: Reward modals compute themselves!
        return total_reward, {"total": total_reward}

    def check_termination(self) -> bool:
        """RewardModals know when task complete - SELF-TERMINATING!"""
        state = self.get_state()
        for condition in self.scene.reward_modal.conditions:
            if hasattr(condition, 'is_complete') and condition.is_complete(state):
                return True
        # MODAL-TO-MODAL: Reward modals know when done!
        return False
```

**RewardModal gets `is_complete()` method:**
```python
class Condition:
    """Reward condition - now SELF-TERMINATING!"""

    def is_complete(self, state: Dict) -> bool:
        """I know when my goal is achieved - PURE MOP!"""
        result = self.check(state)
        return result.is_met
        # SELF-DESCRIBING: Modal knows its completion condition!
```

**GymBridge becomes THIN CONNECTOR:**
```python
class GymBridge(gym.Env):
    """PURE MOP GymBridge - Just connects modals!"""

    def reset(self):
        """Modals reset - 10-30x FASTER!"""
        self.ops.reset()  # NOT compile()!
        return self._get_obs(), {}
        # Was: self.ops.compile() (300ms)
        # Now: self.ops.reset() (10ms)

    def step(self, action):
        """Modals execute everything - NO GLUE!"""
        # PURE MOP: Modals execute actions
        self.ops.apply_action(action, self.actuators_active)

        # Execute physics
        self.ops.step()

        # PURE MOP: Modals compute rewards
        reward, reward_info = self.ops.evaluate_rewards()

        # Get observation
        obs = self._get_obs()

        # PURE MOP: Modals check termination
        terminated = self.ops.check_termination()

        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {"reward_info": reward_info}
```

**The Result:**
- ✅ GymBridge: 30 lines (was 60+ lines of glue!)
- ✅ Episode reset: 10ms (was 300ms+)
- ✅ Action execution: modals do it
- ✅ Reward computation: modals do it
- ✅ Termination check: modals do it
- ✅ ZERO manual mapping
- ✅ ZERO hardcoded indices
- ✅ ZERO glue code

---

### The SILK THREAD in RL

**Every principle applies:**

#### 1. AUTO-DISCOVERY
```python
# Action space discovered from actuators
env = GymBridge(ops, actuators_active=['base'])  # 1D action space!
# Observation space discovered from ViewAggregator
obs_dim = view_aggregator.get_obs_dim()  # 186D discovered!
# Reward discovered from RewardModal behavior
ops.add_reward("stretch.base", behavior="rotation", threshold=90)
```

#### 2. SELF-GENERATION
```python
# SceneModal generates MuJoCo XML
# RobotModal generates actuators/sensors
# RewardModal generates reward computation
# ALL modals create their own representations!
```

#### 3. SELF-RENDERING
```python
# GymBridge renders as gymnasium.Env
# ViewAggregator renders observations
# RewardModal renders rewards
# Actuators render actions
# ALL modals can execute/display themselves!
```

#### 4. LEGO COMPOSITION
```python
# 8+ modals snap together:
train_atomic_skill(...)
# Scene + Robot + Sensors + Actuators + Rewards + VLM + ViewAgg + GymBridge
# ZERO glue code!
```

#### 5. MODAL-TO-MODAL COMMUNICATION
```python
# Modals talk directly:
GymBridge → ExperimentOps → ActuatorModals (execute!)
GymBridge → ExperimentOps → RewardModals (compute!)
GymBridge → ExperimentOps → RewardModals (terminate!)
# NO coordinator! Peer-to-peer!
```

---

### Performance Benefits

**Episode Reset Speed:**
```
OLD (compile every episode):
- Recompile scene: 300ms
- Total: 300ms/episode
- 1000 episodes: 5 minutes just resetting!

NEW (keyframe reset):
- Reset keyframe: 10ms
- Total: 10ms/episode
- 1000 episodes: 10 seconds!
- 30x FASTER!
```

**Code Complexity:**
```
OLD:
- GymBridge: 60+ lines of glue code
- Manual action mapping: 20 lines
- Manual reward extraction: 15 lines
- Manual termination: 10 lines
- Total: 105+ lines

NEW:
- GymBridge: 30 lines (calls ops methods)
- Action mapping: 1 line (ops.apply_action)
- Reward computation: 1 line (ops.evaluate_rewards)
- Termination check: 1 line (ops.check_termination)
- Total: 33 lines
- 3x CLEANER!
```

---

### The Proof: Level 2A Curriculum Learning

**Complete RL experiment in 10 lines:**
```python
from core.main.experiment_ops_unified import ExperimentOps
from core.runtime.gym_bridge import GymBridge
from stable_baselines3 import PPO

# 1. Create scene (modals auto-configure)
ops = ExperimentOps(headless=True)
ops.create_scene("empty_room", width=5, length=5, height=2)
ops.add_robot("stretch", position=(2.5, 2.5, 0))
ops.add_reward("stretch.base", "rotation", threshold=90, reward=100, id="spin_90")
ops.compile()

# 2. Create gym env (modals auto-discover spaces)
env = GymBridge(ops, actuators_active=['base'], obs_config='baseline')
# Action space: 1D (discovered from actuators_active!)
# Obs space: 186D (discovered from sensors + actuators!)

# 3. Train (PPO uses modal-provided spaces)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Done! 10 lines, ZERO manual configuration!
```

**What just happened:**
- ✅ Scene created and compiled (SceneModal)
- ✅ Robot added with actuators (RobotModal)
- ✅ Reward configured (RewardModal)
- ✅ Action space discovered (ActuatorModal)
- ✅ Obs space discovered (ViewAggregator)
- ✅ Training executed (GymBridge + PPO)
- ✅ ZERO glue code
- ✅ ZERO manual introspection
- ✅ PURE MODAL-TO-MODAL COMMUNICATION!

---

## Why This Matters for AI/LLM

Traditional code:
```python
# What does this mean?
if state[42] > threshold[3]:
    return actions[7]
# LLM: ??? (Opaque indices, magic numbers)
```

Modal-Oriented code:
```python
# What does this mean?
if gripper.closed and gripper.holding:
    return "success"
# LLM: "Gripper is closed AND holding object → task succeeded!"
```

**Self-describing systems = LLM-native systems**

LLM can:
- Understand sensor capabilities from semantic names
- Create reward conditions from behavior definitions
- Compose scenarios using uniform API
- Debug by reading JSON (not reverse-engineering code)
- Write RL training code (10 lines!)

---

## The Philosophy in One Sentence

**"Modals are data structures that know themselves well enough to teach the system how to use them."**

---

## Key Takeaways

1. **Modals create JSONs** (not the other way around)
2. **Add field → system includes it** (LEGO composition)
3. **Semantic naming = LLM understanding** (grasp_center, place_on_table)
4. **Crashes educate developers** (OFFENSIVE programming)
5. **Trust modal declarations** (no isinstance/hasattr hacks)
6. **Single source of truth** (BEHAVIORS.json, not scattered code)
7. **One modal, many views** (data, RL, XML, timeline)
8. **Modals communicate directly** (PURE MODAL-TO-MODAL, zero glue!)
9. **Simulation validates sensors** (dual-signal pattern: sensor + physics ground truth)
10. **Instances flow directly** (Pure MOP: no dict conversions, no type loss, ONE path only)
11. **RL training is PURE LEGO** (10 lines, 30x faster resets, zero glue code!)
12. **Avoid external configs** (GLUE CODE anti-pattern - see KeyError incident above!)
13. **Auto-discover, don't hardcode** (Manual lists drift - modals scan XML/source of truth)
14. **Offensive > Defensive** (Crash with helpful error > silent failure with defaults)

---

## 🔌 CONNECTING TO SIMULATION_CENTER FROM EXTERNAL PROJECTS

**The Ultimate LEGO Test**: Can another project snap into simulation_center with ZERO config and ZERO glue code?

**Answer: YES! See `vibe_robotics` - Complete AI ↔ Robot simulation bridge in ~720 lines!**

---

### Pattern: External Project Integration (PURE MOP!)

**The Traditional Way (Configuration Hell):**
```python
# 1. Install simulation_center as dependency
# 2. Write 50+ lines of configuration
# 3. Create adapter classes (100+ lines)
# 4. Map data structures between systems (200+ lines)
# 5. Write integration tests
# 6. Update when simulation_center changes
# Total: 500+ lines of glue code!
```

**The Modal Way (LEGO COMPOSITION!):**
```python
# 1. Import ExperimentOps
from simulation_center.core.main.experiment_ops_unified import ExperimentOps

# 2. Import Orchestrator (AI agents)
from ai_orchestration.core.orchestrator import Orchestrator

# 3. Compose them in your coordinator
class VibeOps:
    """THIN coordinator - just composes 3 ops layers!"""
    def __init__(self, headless: bool = True):
        # LEGO COMPOSITION - Three ops layers snap together!
        self.experiment_ops = ExperimentOps(headless=headless)
        self.orchestrator = Orchestrator()
        self.websocket_server = None

# That's it! ZERO config, ZERO glue code!
```

---

### Real Example: vibe_robotics (Complete System in 4 Hours!)

**What We Built:**
- ✅ WebSocket server (AI agents ↔ Simulation bridge)
- ✅ REST API (session management, scene parsing)
- ✅ AI agent tools (query simulation state, control robot)
- ✅ Complete end-to-end flow (8/8 tests passing!)

**Code Metrics:**
```
NEW Code Written:      ~720 lines
Code REUSED:           ~15,000+ lines (from simulation_center!)
Reuse Ratio:           22x!!!
Build Time:            ~4 hours
Total Test Time:       10.10 seconds
All Tests Passing:     8/8 (100%)
```

---

### Key Pattern: Reuse ExperimentOps + ViewAggregator

**ExperimentOps gives you EVERYTHING:**
```python
class VibeOps:
    def __init__(self, headless: bool = True):
        # Reuse ExperimentOps - gets you ~15,000+ lines for free!
        self.experiment_ops = ExperimentOps(headless=headless)

    def create_scene(self, name: str, width: float, length: float, height: float):
        """Delegate to ExperimentOps - NO glue code!"""
        self.experiment_ops.create_scene(name, width, length, height)

    def add_robot(self, robot_type: str, position: tuple = (0, 0, 0)):
        """Delegate to ExperimentOps - NO glue code!"""
        self.experiment_ops.add_robot(robot_type, position=position)

    def compile(self):
        """Delegate to ExperimentOps - NO glue code!"""
        self.experiment_ops.compile()

    def step(self):
        """Delegate to ExperimentOps - NO glue code!"""
        self.experiment_ops.step()

    def get_vibe_views(self) -> Dict[str, Any]:
        """Get views - ViewAggregator creates from modals!"""
        return self.experiment_ops.view_aggregator.create_views()
```

**ViewAggregator gives you 24+ views for FREE:**
- Robot sensors (cameras, LiDAR, IMU, gripper forces)
- Robot state (joint positions, velocities, base position)
- Scene state (furniture positions, room layout)
- All generated from modals with `get_data()` - SELF-RENDERING!

---

### Key Pattern: Modals Teach External Systems (SELF-GENERATION!)

**Traditional: Manual Documentation**
```json
// You write this by hand (drifts from code!)
{
  "actions": {
    "BaseMoveTo": {
      "description": "Move robot base",
      "params": {"x": "number", "y": "number"}
    }
  }
}
```

**MOP: Modals Generate Their Own Docs**
```python
def get_action_guide(self) -> Dict[str, Any]:
    """Modals TEACH AI agents what they can do - SELF-GENERATION!"""

    # AUTO-DISCOVERY: Robot modal knows its own actions!
    robot = self.experiment_ops.robot
    for action_name, action_block in robot.actions.items():
        # Action modal describes itself!
        action_guide["available_actions"][action_name] = {
            "type": action_block.id,
            "description": action_block.description,
            "example": f'{{"type": "{action_name}", "params": {{...}}}}'
        }

    # AUTO-DISCOVERY: Scene modal knows its assets!
    for asset_name, asset in self.experiment_ops.scene.assets.items():
        # Asset modal describes itself!
        asset_data = asset.get_data() if hasattr(asset, 'get_data') else {}
        action_guide["available_furniture"].append({
            "name": asset_name,
            "position": asset_data.get("position"),
            "type": asset_data.get("type")
        })

    # AUTO-DISCOVERY: Room modal knows its layout!
    room = self.experiment_ops.scene.room
    action_guide["scene_layout"] = {
        "width": room.width,
        "length": room.length,
        "height": room.height
    }

    return action_guide
```

**Result: AI agents get complete, always-up-to-date documentation!**
- 17 actions discovered automatically
- Furniture positions auto-tracked
- Room layout auto-described
- ZERO manual documentation!

---

### Key Pattern: Modal-to-Modal Communication Across Projects

**The Flow (ZERO Glue Code!):**
```
AI Agent → WebSocket → VibeOps
VibeOps → ExperimentOps → Modals (execute actions!)
VibeOps → ViewAggregator → Modals (render views!)
WebSocket → Broadcast → AI Agent (30 FPS!)
```

**Example: AI Agent Controls Robot**
```python
# AI agent tool (vibe_tools.py)
async def submit_action_block(vibe_ops, action_block: Dict) -> Dict:
    """AI agent controls robot - PURE MOP!"""
    # Convert dict to ActionBlock modal
    from simulation_center.core.modals.stretch.action_modals import ActionBlock
    action = ActionBlock(**action_block)

    # ExperimentOps executes - MODAL-TO-MODAL!
    vibe_ops.experiment_ops.submit_block(action)

    return {"status": "submitted"}
```

**No adapter classes! No data conversion! Just modals talking to modals!**

---

### Key Pattern: Reuse Testing Infrastructure

**Reuse TestModal pattern from simulation_center:**
```python
# vibe_robotics/core/tests/test_complete_system.py
import pytest
from simulation_center.core.main.experiment_ops_unified import ExperimentOps

@pytest.mark.asyncio
async def test_complete_end_to_end_flow():
    """Test COMPLETE flow - proves EVERYTHING works!"""

    # 1. Create VibeOps (LEGO COMPOSITION!)
    ops = VibeOps(headless=True)

    # 2. Create scene (delegates to ExperimentOps)
    ops.create_scene("kitchen", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.compile()

    # 3. Get action guide (SELF-GENERATION!)
    guide = ops.get_action_guide()
    assert "available_actions" in guide
    assert len(guide["available_actions"]) == 17  # Auto-discovered!

    # 4. Get views (ViewAggregator!)
    views = await get_vibe_views(ops)
    assert len(views) == 24  # Auto-generated from modals!

    # 5. Submit action (AI agent tool)
    action_block = {
        "id": "move_forward",
        "description": "Move robot forward",
        "execution_mode": "sequential",
        "actions": [
            {"type": "BaseMoveTo", "params": {"x": 1.0, "y": 1.0, "z": 0.0}}
        ]
    }
    result = await submit_action_block(ops, action_block)
    assert result["status"] == "submitted"

    # 6. Step simulation
    for i in range(10):
        ops.step()

    # 7. Get updated views
    updated_views = ops.get_vibe_views()
    assert len(updated_views) == 24

    # ✅ COMPLETE E2E FLOW WORKS!
```

**Result: 8/8 tests passing, complete system validated!**

---

### Benefits of Connecting to simulation_center

**1. Massive Code Reuse:**
- ExperimentOps: Complete experiment management (~5,000 lines)
- ViewAggregator: View creation and distribution (~1,000 lines)
- All modals with `get_data()`: Scene, Robot, Room, Asset, Reward (~3,000 lines)
- All Ops layers: RobotOps, SceneOps, StreamOps, TrainingOps (~6,000 lines)
- MuJoCo backend and physics engine (~2,000 lines)
- **Total: ~15,000+ lines REUSED!**

**2. Zero Configuration:**
- No config files needed
- No adapter classes
- No data mapping
- Just import and compose!

**3. Auto-Discovery:**
- Actions discovered from robot.actions
- Views discovered from modals' get_data()
- Furniture discovered from scene.assets
- Room layout discovered from scene.room

**4. Self-Generation:**
- Modals generate their own docs (action guides)
- Modals render their own views (ViewAggregator)
- Modals validate themselves (get_data())

**5. Modal-to-Modal Communication:**
- External project → ExperimentOps → Modals
- No glue code between projects
- Just delegate to ExperimentOps methods!

---

### How to Connect Your Project

**Step 1: Import the Ops layers you need**
```python
from simulation_center.core.main.experiment_ops_unified import ExperimentOps
```

**Step 2: Create thin coordinator that composes ops**
```python
class YourOps:
    def __init__(self):
        # LEGO COMPOSITION!
        self.experiment_ops = ExperimentOps(headless=True)
        # Add your own ops layers here
```

**Step 3: Delegate to ExperimentOps methods**
```python
    def create_scene(self, *args, **kwargs):
        """Just delegate - NO glue code!"""
        self.experiment_ops.create_scene(*args, **kwargs)
```

**Step 4: Use ViewAggregator for views**
```python
    def get_views(self):
        """Modals render themselves - SELF-RENDERING!"""
        return self.experiment_ops.view_aggregator.create_views()
```

**Step 5: Add your custom functionality**
```python
    def your_custom_method(self):
        """Your project-specific logic here"""
        # Access simulation state via ExperimentOps
        robot_pos = self.experiment_ops.robot.get_data()["position"]
        # Do your custom logic
        return result
```

**That's it! You now have a complete simulation system in ~50 lines!**

---

### Real Metrics from vibe_robotics

```
Component               Lines    Source
─────────────────────────────────────────────
VibeOps                  280     NEW (coordinator)
WebSocketServer          120     NEW (bridge)
AI Agent Tools            90     NEW (tools)
FastAPI Server           180     NEW (REST API)
Test Suite               450     NEW (validation)
─────────────────────────────────────────────
Total NEW                720     ~4 hours work

Component               Lines    Source
─────────────────────────────────────────────
ExperimentOps          5,000+    REUSED!
ViewAggregator         1,000+    REUSED!
Modals (get_data)      3,000+    REUSED!
All Ops layers         6,000+    REUSED!
MuJoCo backend         2,000+    REUSED!
─────────────────────────────────────────────
Total REUSED          15,000+    FREE!

Reuse Ratio: 22x (wrote 720, got 15,000!)
```

---

### The Principle

**"simulation_center is LEGO - external projects snap in with ZERO config!"**

- ✅ Import ExperimentOps
- ✅ Compose with your ops
- ✅ Delegate to ExperimentOps methods
- ✅ Use ViewAggregator for views
- ✅ Modals do the rest!

**NO adapters! NO glue code! NO configuration! Pure LEGO COMPOSITION!**

---

## 🎨 UI FRAMEWORK - MODAL-ORIENTED INTERFACES

**See `vibe_robotics/MODAL_ORIENTED_PROGRAMMING.md` for complete UI Framework documentation!**

**Quick Summary**: UI is just modals rendering themselves for humans via browsers.

### The Pattern

```
Backend Modal (Python)          Frontend View (JavaScript)
─────────────────────          ──────────────────────────
VibeSessionModal                SessionCard
  .get_data()          →          .render(data)

ExperimentOps                   RobotVisualization
  .view_aggregator
    .create_views()    →          .renderViews(views)
```

### Technology: Pure Vanilla Web

- **NO React, NO npm, NO build tools** (same philosophy as backend MOP!)
- HTML5 + CSS3 + Vanilla JavaScript (ES6+)
- WebSocket for real-time streaming (30 FPS)
- D3.js for visualization
- Lucide Icons (NOT emojis!)

### Multiple Views

- `/simulation` - Scene control, robot visualization
- `/orchestration` - AI agents, task graphs
- `/vibe` - Combined dashboard (simulation + orchestration)
- `/rl` (future) - RL training metrics

### All 5 MOP Principles Apply to UI!

1. **AUTO-DISCOVERY** - UI auto-discovers fields from `get_data()`
2. **SELF-GENERATION** - Components generate their own HTML
3. **SELF-RENDERING** - Same modal → multiple views (card, graph, timeline)
4. **LEGO COMPOSITION** - Add field to backend → auto-appears in UI
5. **MODAL-TO-MODAL** - WebSocket → Card → Graph (direct communication!)

**Complete details**: See `vibe_robotics/MODAL_ORIENTED_PROGRAMMING.md` UI Framework section!

---

## Next Steps

- **MOP_RL_ARCHITECTURE.md** - Deep dive into the RL modal-to-modal architecture
- **ADDING_COMPONENTS.md** - See LEGO composition in action
- **QUICKSTART.md** - Build your first scene in 5 minutes
- **ARCHITECTURE.md** - Understand the internals
- **vibe_robotics/** - See complete external project integration example!
- **vibe_robotics/MODAL_ORIENTED_PROGRAMMING.md** - Complete UI Framework guide!
- **vibe_robotics/RL_CENTER_INTEGRATION.md** - RL team integration guide!

---

**Modal-Oriented Programming: Self-Aware Data That Dreams** 🧵

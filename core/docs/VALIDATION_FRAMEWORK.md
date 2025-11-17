# 5-WAY VALIDATION FRAMEWORK

## The Problem with Traditional Robotics Testing

**Traditional Approach:**
```python
# Execute action
robot.move_forward(1.0)
time.sleep(2.0)

# Manual validation (hope it worked!)
actual_pos = robot.get_position()
expected_pos = start_pos + 1.0
assert abs(actual_pos - expected_pos) < 0.1, "FAILED!"
```

**Issues:**
- ❌ Single validation method (position only)
- ❌ No visual proof
- ❌ No semantic meaning
- ❌ Can't explain WHY it failed
- ❌ Not reproducible

---

## The Robot OS Solution: 5-Way Validation

Every capability can be validated through **5 independent modalities**, providing **explainable, reproducible, multi-modal proof** that behaviors work correctly.

```
┌──────────────────────────────────────────────────────────┐
│                 5-WAY VALIDATION                         │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  1. PHYSICS       →  MuJoCo state proves behavior        │
│  2. SEMANTIC      →  Modal behaviors confirm correctness │
│  3. VISION        →  Camera captures visual proof        │
│  4. REASONING     →  Spatial/temporal consistency        │
│  5. ACTION IMAGE  →  Action execution captured visually  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## Layer 1: PHYSICS VALIDATION

**What:** Direct interrogation of MuJoCo physics state
**Why:** Physics engine is ground truth
**When:** Always - foundation of all validation

### Example: Verify Object Stack

```python
# Physics proves vertical ordering
state = ops.get_state()
table_z = state["table"]["position"][2]
foam_z = state["foam_brick"]["position"][2]
pudding_z = state["pudding_box"]["position"][2]
cracker_z = state["cracker_box"]["position"][2]

# Assert physical constraint
assert table_z < foam_z < pudding_z < cracker_z, "Stack order violated!"

# Assert height increments
height_diff_1 = foam_z - table_z
height_diff_2 = pudding_z - foam_z
height_diff_3 = cracker_z - pudding_z

assert 0.70 < height_diff_1 < 0.95, "Table surface height wrong"
assert 0.02 < height_diff_2 < 0.15, "Stack spacing wrong"
assert 0.02 < height_diff_3 < 0.15, "Stack spacing wrong"
```

**Proves:** Objects are physically stacked in correct order with correct heights

---

## Layer 2: SEMANTIC VALIDATION

**What:** Check modal-declared behaviors for correctness
**Why:** Confirms system understands the semantics of what's happening
**When:** When behavior detection matters (holding, rolling, etc.)

### Example: Verify Gripper Holding

```python
# Semantic validation - gripper knows it's holding
state = ops.get_state()
gripper_state = state["stretch.gripper"]

# Modal self-declares "holding" behavior
if "holding" in gripper_state:
    is_holding = gripper_state["holding"]
    held_object = gripper_state.get("held_object", None)

    assert is_holding == True, "Gripper should be holding"
    assert held_object == "apple", "Should be holding apple specifically"

# Also check contact forces (semantic meaning)
if "contact_force" in gripper_state:
    force = gripper_state["contact_force"]
    assert force > 0.1, "Should have contact force when holding"
```

**Proves:** System semantically understands the "holding" relationship

---

## Layer 3: VISION VALIDATION

**What:** Camera captures visual proof of behavior
**Why:** Humans can visually verify correctness
**When:** Critical for explainability and debugging

### Example: Verify Stack with Camera Proof

```python
# Add free camera
ops.add_free_camera("stack_cam",
                   lookat=(2.0, 0.0, 0.8),
                   distance=3.0,
                   azimuth=45,
                   elevation=-25)

ops.step()

# Get camera view
views = ops.engine.last_views
stack_view = views['stack_cam_view']

if "rgb" not in stack_view:
    raise AssertionError("No RGB in camera view")

img = stack_view["rgb"]
print(f"✓ Camera image: {img.shape}")

# Color diversity check - proves objects visible
import cv2
import numpy as np

img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hue_std = np.std(img_hsv[:,:,0])

assert hue_std > 10.0, f"Image too uniform (hue_std={hue_std:.1f}) - objects not visible!"

# Save visual proof
save_path = Path(ops.experiment_dir) / "views" / "stack_proof.png"
save_path.parent.mkdir(parents=True, exist_ok=True)
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imwrite(str(save_path), img_bgr)

print(f"📸 Visual proof saved: {save_path}")
```

**Proves:** Human-verifiable visual evidence that stack exists and is correctly composed

---

## Layer 4: REASONING VALIDATION

**What:** Verify spatial/temporal consistency and logical constraints
**Why:** Catches edge cases physics/vision might miss
**When:** Complex scenarios with multiple objects/constraints

### Example: Verify 100-Step Stability

```python
# Record initial positions
initial_positions = {
    "foam_brick": state["foam_brick"]["position"][2],
    "pudding_box": state["pudding_box"]["position"][2],
    "cracker_box": state["cracker_box"]["position"][2]
}

# Run 100 physics steps
for step in range(100):
    ops.step()

# Check final positions
state_after = ops.get_state()
final_positions = {
    "foam_brick": state_after["foam_brick"]["position"][2],
    "pudding_box": state_after["pudding_box"]["position"][2],
    "cracker_box": state_after["cracker_box"]["position"][2]
}

# Calculate z-drift for each object
max_drift = 0.0
for obj_name in initial_positions:
    z_drift = abs(final_positions[obj_name] - initial_positions[obj_name])
    print(f"{obj_name} z-drift: {z_drift:.4f}m")
    max_drift = max(max_drift, z_drift)

assert max_drift < 0.1, f"Stack unstable (max drift: {max_drift:.4f}m)"

# Reasoning: If stack drifted > 0.1m over 100 steps, it's unstable
# even if it looks correct in a single frame
```

**Proves:** Stack is not just visually correct, but physically stable over time

---

## Layer 5: ACTION IMAGE (NEW!)

**What:** Capture visual sequence of action execution
**Why:** Proves action succeeded through visual difference
**When:** Action validation, debugging, explainability

### Example: Verify Arm Extension with Before/After Images

```python
# Add camera focused on robot arm
ops.add_free_camera("arm_cam",
                   lookat=(0.0, 0.0, 0.5),  # Look at arm
                   distance=2.0,
                   azimuth=90,
                   elevation=0)

ops.step()

# BEFORE action
before_img = ops.engine.last_views["arm_cam_view"]["rgb"]

# Execute action
from simulation_center.core.modals.stretch.action_modals import ArmMoveTo, ActionBlock
action = ArmMoveTo(position=0.3)
block = ActionBlock(id="extend", actions=[action])
ops.submit_block(block)

for _ in range(500):
    result = ops.step()
    if action.status == 'completed':
        break

# AFTER action
after_img = ops.engine.last_views["arm_cam_view"]["rgb"]

# Verify images are different (action happened visually!)
import numpy as np
images_different = not np.array_equal(before_img, after_img)
assert images_different, "Images should differ after action!"

# Save action sequence proof
import cv2
save_dir = Path(ops.experiment_dir) / "views" / "action_sequence"
save_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(save_dir / "arm_before.png"), cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(save_dir / "arm_after.png"), cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))

print(f"📸 Action sequence saved:")
print(f"  - Before: {save_dir / 'arm_before.png'}")
print(f"  - After:  {save_dir / 'arm_after.png'}")
```

**Proves:** Visual evidence that action execution changed robot state

---

## Complete 5-Way Validation Example

Here's a complete test showcasing all 5 validation layers:

```python
def test_complete_5way_validation():
    """Complete example: Stack 3 objects with 5-way validation"""

    ops = ExperimentOps(headless=True, render_mode="vision_rl")

    # Build scene
    ops.create_scene("test_room", width=6, length=6, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))

    # Add free camera for vision/action validation
    ops.add_free_camera("stack_cam",
                       lookat=(2.0, 0.0, 0.8),
                       distance=3.0,
                       azimuth=45,
                       elevation=-25)

    ops.compile()
    ops.step()

    # ===== ACTION IMAGE: BEFORE =====
    before_img = ops.engine.last_views["stack_cam_view"]["rgb"]
    cv2.imwrite("before_stacking.png", cv2.cvtColor(before_img, cv2.COLOR_RGB2BGR))

    # Stack 3 objects
    ops.add_asset("foam_brick", relative_to="table", relation="on_top", distance=0.75)
    ops.add_asset("pudding_box", relative_to="foam_brick", relation="stack_on")
    ops.add_asset("cracker_box", relative_to="pudding_box", relation="stack_on")

    ops.compile()
    ops.step()

    # ===== LAYER 1: PHYSICS VALIDATION =====
    print("1. Physics Validation:")
    state = ops.get_state()

    table_z = state["table"]["position"][2]
    foam_z = state["foam_brick"]["position"][2]
    pudding_z = state["pudding_box"]["position"][2]
    cracker_z = state["cracker_box"]["position"][2]

    assert table_z < foam_z < pudding_z < cracker_z, "Stack order violated"
    print(f"   ✓ Physics proves stack order: table < foam < pudding < cracker")

    # ===== LAYER 2: SEMANTIC VALIDATION =====
    print("2. Semantic Validation:")

    height_diff_1 = foam_z - table_z
    height_diff_2 = pudding_z - foam_z
    height_diff_3 = cracker_z - pudding_z

    assert 0.70 < height_diff_1 < 0.95, "Table surface height wrong"
    assert 0.02 < height_diff_2 < 0.15, "Stack spacing 1 wrong"
    assert 0.02 < height_diff_3 < 0.15, "Stack spacing 2 wrong"
    print(f"   ✓ Semantic: Height increments realistic (stack_on working)")

    # ===== LAYER 3: VISION VALIDATION =====
    print("3. Vision Validation:")

    stack_view = ops.engine.last_views["stack_cam_view"]
    img = stack_view["rgb"]

    # Color diversity check
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hue_std = np.std(img_hsv[:,:,0])

    assert hue_std > 10.0, f"Image too uniform (hue_std={hue_std:.1f})"
    print(f"   ✓ Vision: Color diversity {hue_std:.1f} (objects visible)")

    # Save proof
    cv2.imwrite("stack_proof.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"   📸 Visual proof saved: stack_proof.png")

    # ===== LAYER 4: REASONING VALIDATION (100-STEP STABILITY) =====
    print("4. Reasoning Validation (100 steps):")

    initial_z = {
        "foam_brick": foam_z,
        "pudding_box": pudding_z,
        "cracker_box": cracker_z
    }

    for _ in range(100):
        ops.step()

    state_after = ops.get_state()
    max_drift = 0.0

    for obj in initial_z:
        final_z = state_after[obj]["position"][2]
        drift = abs(final_z - initial_z[obj])
        max_drift = max(max_drift, drift)

    assert max_drift < 0.1, f"Stack unstable (drift: {max_drift:.4f}m)"
    print(f"   ✓ Reasoning: Stack stable over 100 steps (max drift: {max_drift:.4f}m)")

    # ===== LAYER 5: ACTION IMAGE =====
    print("5. Action Image Validation:")

    after_img = ops.engine.last_views["stack_cam_view"]["rgb"]

    # Images should differ (stacking happened!)
    images_different = not np.array_equal(before_img, after_img)
    assert images_different, "Images should differ after stacking"

    cv2.imwrite("after_stacking.png", cv2.cvtColor(after_img, cv2.COLOR_RGB2BGR))
    print(f"   ✓ Action Image: Visual change detected")
    print(f"   📸 Before/after sequence saved")

    print("\n✅ COMPLETE 5-WAY VALIDATION PASSED!")
    print("   Physics ✓  Semantic ✓  Vision ✓  Reasoning ✓  Action Image ✓")
    return True
```

---

## When to Use Which Validation Layers

### All 5 Layers (Maximum Confidence)
- Critical tests (Final Boss)
- Complex multi-object scenarios
- Long-running behaviors (stability tests)
- When creating documentation/tutorials
- **Example:** Stack 3 objects with 100-step stability

### 4 Layers (Physics + Semantic + Vision + Reasoning)
- Object behaviors (graspable, rollable, container)
- Spatial relationship validation
- Action completion verification
- **Example:** Object placement with vision proof

### 3 Layers (Physics + Semantic + Vision)
- Scene composition tests
- Basic manipulation
- Sensor validation
- **Example:** Place apple on table, verify with camera

### 2 Layers (Physics + Semantic)
- Quick action tests
- Reward validation
- Simple movement
- **Example:** Arm extends to 0.3m

### 1 Layer (Physics Only)
- Unit tests
- Low-level actuator control
- Performance benchmarks
- **Example:** Joint position accuracy

---

## Validation Layers Comparison Table

| Validation Layer | Cost | Confidence | Use Case | Example |
|------------------|------|------------|----------|---------|
| Physics | Low | Medium | Always use | Check position |
| Semantic | Low | Medium | Behavior detection | Check "holding" |
| Vision | Medium | High | Explainability | Save image proof |
| Reasoning | Low | High | Stability/consistency | 100-step drift |
| Action Image | Medium | Very High | Action verification | Before/after images |

**Cost:** Computational overhead
**Confidence:** How certain you are behavior is correct

---

## The Power of Multi-Modal Validation

### Single-Modal (Traditional):
```python
# Only physics
actual_pos = robot.get_position()
assert abs(actual_pos - target_pos) < 0.1
# ❌ What if sensor is wrong?
# ❌ What if position drifts later?
# ❌ No visual proof
```

### Multi-Modal (Robot OS):
```python
# Physics proves
assert abs(state["arm"]["extension"] - 0.3) < 0.05

# Semantics confirms
assert state["arm"]["at_target"] == True

# Vision captures proof
cv2.imwrite("arm_extended.png", camera_view["rgb"])

# Reasoning validates persistence
# (run 100 more steps, check drift < 0.05m)

# Action image shows change
# (before_img != after_img)

# ✅ 5 independent proofs!
```

---

## Benefits of 5-Way Validation

1. **Explainability**: Visual proof humans can verify
2. **Reproducibility**: Complete evidence saved to database
3. **Confidence**: 5 independent confirmations
4. **Debugging**: If one layer fails, others show where
5. **Documentation**: Automatic proof generation
6. **Trust**: Multi-modal evidence builds confidence

---

## Next Steps

- See **[API_QUICKSTART.md](./API_QUICKSTART.md)** for minimal code examples
- See **[CAPABILITY_MATRIX.md](./CAPABILITY_MATRIX.md)** for complete capability list
- See Level 1M (Final Boss) for complete 5-way validation examples

---

## Philosophy

> **"One proof is evidence. Five independent proofs is certainty."**
>
> Robot OS validates behaviors through Physics, Semantics, Vision, Reasoning, and Action Images - providing **explainable, reproducible, multi-modal certainty** that your robot works correctly.

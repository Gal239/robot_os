# Action API Reference

## ActionBlock - Required Parameters

From `/home/gal-labs/PycharmProjects/echo_robot/simulation_center/core/modals/stretch/action_modals.py:991-1002`:

```python
class ActionBlock(BaseModel):
    id: str                                              # REQUIRED
    description: str = ""                                # Optional (defaults to empty)
    execution_mode: Literal['sequential', 'parallel'] = 'sequential'  # Optional
    push_before_others: bool = False                     # Optional
    replace_current: bool = False                        # Optional
    end_with_thinking: bool = False                      # Optional
    actions: List[Action] = []                           # Optional (defaults to empty list)
```

## Creating ActionBlock - CORRECT USAGE

**Example from test_complete_system.py:255-262:**

```python
from simulation_center.core.modals.stretch.action_modals import ActionBlock, LiftMoveTo

# ✅ CORRECT - id parameter is REQUIRED
action = ActionBlock(
    id="lift_up",                          # REQUIRED
    description="Lift arm up",            # Optional but good practice
    actions=[LiftMoveTo(position=1.0)]
)
ops.submit_action_block(action)

# ❌ WRONG - Missing id parameter
action = ActionBlock(actions=[LiftMoveTo(position=1.0)])  # ValidationError: id Field required
```

## Action Classes - Required Parameters

### PositionMoveToBase (LiftMoveTo, ArmMoveTo, WristYaw, etc.)

**Required:**
- `position: float` - Target position in meters or radians

**Optional:**
- `actuator_id: str` - Auto-detected from action type
- `tolerance: float` - Completion tolerance (has defaults)
- `timeout: float` - Max time to complete (has defaults)

**Example:**
```python
# ✅ CORRECT
LiftMoveTo(position=1.0)      # position is required
ArmMoveTo(position=0.3)       # position is required

# ❌ WRONG
LiftMoveTo(height=1.0)        # No 'height' parameter - use 'position'
ArmMoveTo(extension=0.3)      # No 'extension' parameter - use 'position'
```

### BaseMoveForward / BaseMoveBackward

**Required:**
- `distance: float` - Distance to move in meters

**Optional:**
- `speed: float` - Movement speed (has default)
- `tolerance: float` - Completion tolerance (has default)

**Example:**
```python
# ✅ CORRECT
BaseMoveForward(distance=0.5)
BaseMoveBackward(distance=0.5)

# ❌ WRONG
BaseMove(forward=0.5)         # No BaseMove class - use BaseMoveForward
```

### BaseRotateBy

**Required:**
- `angle: float` - Angle to rotate in radians

**Optional:**
- `speed: float` - Rotation speed (has default)

**Example:**
```python
# ✅ CORRECT
BaseRotateBy(angle=1.57)      # 90 degrees in radians

# ❌ WRONG
BaseRotateBy(degrees=90)      # Use radians, not degrees
```

## Complete Working Example

```python
#!/usr/bin/env python3
from simulation_center.core.main.experiment_ops_unified import ExperimentOps
from simulation_center.core.modals.stretch.action_modals import (
    ActionBlock, LiftMoveTo, ArmMoveTo, BaseMoveForward
)

# Create experiment
ops = ExperimentOps(headless=True, render_mode="demo")
ops.create_scene('demo', width=5, length=5, height=3)
ops.add_robot('stretch', position=(0, 0, 0))
ops.compile()

# Submit actions - CORRECT API USAGE
for step in range(1000):
    if step == 0:
        # ✅ REQUIRED: id parameter
        # ✅ REQUIRED: position parameter (not height/extension/etc)
        action = ActionBlock(
            id="lift_up",
            description="Lift arm to 1.0m",
            actions=[LiftMoveTo(position=1.0)]
        )
        ops.submit_action_block(action)

    elif step == 200:
        action = ActionBlock(
            id="arm_extend",
            actions=[ArmMoveTo(position=0.3)]
        )
        ops.submit_action_block(action)

    elif step == 400:
        action = ActionBlock(
            id="move_forward",
            actions=[BaseMoveForward(distance=0.5)]
        )
        ops.submit_action_block(action)

    ops.step()
```

## Quick Reference

| Action Class | Required Parameter | Example |
|--------------|-------------------|---------|
| `ActionBlock` | `id: str` | `ActionBlock(id="my_action", actions=[...])` |
| `LiftMoveTo` | `position: float` | `LiftMoveTo(position=1.0)` |
| `ArmMoveTo` | `position: float` | `ArmMoveTo(position=0.3)` |
| `BaseMoveForward` | `distance: float` | `BaseMoveForward(distance=0.5)` |
| `BaseMoveBackward` | `distance: float` | `BaseMoveBackward(distance=0.5)` |
| `BaseRotateBy` | `angle: float` | `BaseRotateBy(angle=1.57)` |
| `WristYaw` | `position: float` | `WristYaw(position=0.5)` |
| `WristPitch` | `position: float` | `WristPitch(position=0.3)` |
| `WristRoll` | `position: float` | `WristRoll(position=0.2)` |
| `HeadPan` | `position: float` | `HeadPan(position=0.0)` |
| `HeadTilt` | `position: float` | `HeadTilt(position=-0.5)` |
| `GripperClose` | None (no params) | `GripperClose()` |
| `GripperOpen` | None (no params) | `GripperOpen()` |

## Error Messages Mean

**"Field required [type=missing, input_value={'actions': [...]}, input_type=dict]"**
- You forgot a required parameter
- Check the error to see which field is missing
- Example: `id Field required` means you need to add `id="some_id"`

**"Extra inputs are not permitted"**
- You used a parameter name that doesn't exist
- Check the Quick Reference table for correct parameter names
- Example: Using `height` instead of `position`

## Source of Truth

- ActionBlock definition: `simulation_center/core/modals/stretch/action_modals.py:991-1002`
- Test examples: `vibe_robotics/core/tests/test_complete_system.py:238-277`
- Action base classes: `simulation_center/core/modals/stretch/action_modals.py`

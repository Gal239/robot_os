"""
UNIFIED ACTION REGISTRY - All robot actions in one place
Both atomic (parameterizable) and complex (multi-step) actions
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Callable, Optional

from .action_modals import (
    ActionBlock,
    ArmMoveTo, ArmMoveBy,
    LiftMoveTo, LiftMoveBy,
    BaseMoveForward, BaseMoveBackward, BaseRotateBy,
    HeadPanMoveTo, HeadPanMoveBy,
    HeadTiltMoveTo, HeadTiltMoveBy,
    WristYawMoveTo, WristYawMoveBy,
    WristPitchMoveTo, WristPitchMoveBy,
    WristRollMoveTo, WristRollMoveBy,
    GripperMoveTo, GripperMoveBy,
    SpeakerPlay
)


# ============================================
# DIFFERENTIAL DRIVE ACTION BLOCKS - MOP: Pre-configured parallel wheels!
# ============================================

def move_forward(distance: float, speed: float = 0.1) -> ActionBlock:
    """Move forward using differential drive kinematics

    MOP: ONE action controls BOTH wheels using differential drive.
    No more parallel wheel actions - coordinated control in single action!

    Args:
        distance: Distance to move in meters
        speed: Linear velocity in m/s (default 0.1, max ~0.1 due to gear=3 and ±6 clamp)

    Returns:
        ActionBlock with SINGLE multi-actuator action (MOP!)

    Example:
        # Creates ONE action that SELF-DECLARES control of both wheels
        block = move_forward(distance=2.0, speed=0.1)
        # ActionExecutor reads actuator_ids and applies dict commands - NO GLUE!
    """
    return ActionBlock(
        id=f"move_forward_{distance}m",
        description=f"Move forward {distance}m at {speed}m/s using differential drive",
        execution_mode="parallel",  # SINGLE action!
        actions=[
            BaseMoveForward(distance=distance, speed=speed)  # ONE action, BOTH wheels!
        ]
    )


def move_backward(distance: float, speed: float = 0.1) -> ActionBlock:
    """Move backward using differential drive kinematics

    MOP: ONE action controls BOTH wheels using differential drive.

    Args:
        distance: Distance to move backward in meters
        speed: Linear velocity in m/s (default 0.1, max ~0.1 due to gear=3 and ±6 clamp)

    Returns:
        ActionBlock with SINGLE multi-actuator action (MOP!)
    """
    return ActionBlock(
        id=f"move_backward_{distance}m",
        description=f"Move backward {distance}m at {speed}m/s using differential drive",
        execution_mode="sequential",  # SINGLE action!
        actions=[
            BaseMoveBackward(distance=distance, speed=speed)  # ONE action, BOTH wheels!
        ]
    )


# Note: spin_left() and spin_right() removed - use spin() with signed degrees instead
# spin(degrees=-90)  → counter-clockwise (was spin_left)
# spin(degrees=90)   → clockwise (was spin_right)


# ============================================
# ATOMIC ACTION SPACES - PURE MOP SELF-DECLARATION!
# ============================================
# Actions self-declare their parameter ranges (Box)
# - Most actions: Discover from actuator.range
# - Special actions (like spin): Hardcoded (different units than actuators)

import gymnasium as gym
import numpy as np

ATOMIC_ACTION_SPACES = {
    # ROTATION: Hardcoded degrees (actuators are rad/s velocity, not degrees!)
    # INT because degrees are whole numbers! 721 discrete values: -360 to +360
    'atomic_rotation': gym.spaces.Box(
        low=np.array([-360], dtype=np.int32),
        high=np.array([360], dtype=np.int32),
        dtype=np.int32
    ),

    # MOVEMENT: Hardcoded meters (could discover from scene size later)
    'atomic_movement': gym.spaces.Box(
        low=np.array([0.0], dtype=np.float32),
        high=np.array([2.0], dtype=np.float32),
        dtype=np.float32
    ),

    # OTHER ACTIONS: Discover from actuators (MOP!)
    'atomic_arm_extension': 'discover_from_actuator:arm',
    'atomic_lift': 'discover_from_actuator:lift',
    'atomic_gripper': 'discover_from_actuator:gripper',
    'atomic_head_pan': 'discover_from_actuator:head_pan',
    'atomic_head_tilt': 'discover_from_actuator:head_tilt',
    'atomic_wrist_yaw': 'discover_from_actuator:wrist_yaw',
    'atomic_wrist_pitch': 'discover_from_actuator:wrist_pitch',
    'atomic_wrist_roll': 'discover_from_actuator:wrist_roll',
}


# ============================================
# ACTION SPACE RESOLVER - PURE MOP!
# ============================================

def get_action_space(action_name: str, robot=None) -> gym.spaces.Box:
    """Get action space by name - PURE MOP! No reconstruction, no dtype loss!

    This is the SINGLE SOURCE OF TRUTH for action spaces.
    Registry self-declares ranges AND dtypes - ExperimentOps just asks and receives!

    Args:
        action_name: e.g., "atomic_rotation", "atomic_arm_extension"
        robot: Optional robot modal for actuator discovery

    Returns:
        gym.spaces.Box with CORRECT dtype (int32 for rotation, float32 for others)

    Example:
        # Returns Box(-360, 360, dtype=int32) - 721 discrete values!
        space = get_action_space("atomic_rotation")

        # Returns Box from actuator.range
        space = get_action_space("atomic_arm_extension", robot)
    """
    if action_name not in ATOMIC_ACTION_SPACES:
        available = list_atomic_actions()
        raise ValueError(
            f"Unknown action: '{action_name}'\n"
            f"Available atomic actions: {available}"
        )

    box_def = ATOMIC_ACTION_SPACES[action_name]

    # Case 1: Already a complete Box - return as-is! (PRESERVES DTYPE!)
    if isinstance(box_def, gym.spaces.Box):
        return box_def

    # Case 2: Discovery string - resolve from robot actuator
    elif isinstance(box_def, str) and box_def.startswith('discover_from_actuator:'):
        actuator_name = box_def.split(':')[1]

        if robot is None:
            raise ValueError(
                f"Need robot to discover '{action_name}' from actuator '{actuator_name}'\n"
                f"Call: get_action_space('{action_name}', robot)"
            )

        if actuator_name not in robot.actuators:
            raise ValueError(
                f"Actuator '{actuator_name}' not found in robot\n"
                f"Available actuators: {list(robot.actuators.keys())}"
            )

        # Discover range from actuator modal (PURE MOP!)
        low, high = robot.actuators[actuator_name].range
        return gym.spaces.Box(
            low=np.array([low], dtype=np.float32),
            high=np.array([high], dtype=np.float32),
            dtype=np.float32
        )

    else:
        raise ValueError(
            f"Unknown box definition type for '{action_name}': {type(box_def)}\n"
            f"Expected: gym.spaces.Box or 'discover_from_actuator:...'"
        )


def list_atomic_actions() -> List[str]:
    """List all atomic action names - SELF-DOCUMENTING

    Returns:
        List of action names that can be passed to get_action_space()
    """
    return list(ATOMIC_ACTION_SPACES.keys())


def spin(radians: Optional[float] = None, degrees: Optional[float] = None, speed: float = 6.0) -> ActionBlock:
    """Unified spin action - SIGN determines direction!

    REVOLUTIONARY: ONE action, not two! The NUMBER itself contains direction.
    - Positive degrees/radians → clockwise (spin_right)
    - Negative degrees/radians → counter-clockwise (spin_left)

    This is CLEANER for RL:
    - Box(-360, 360) instead of Discrete(2) + Box(0, 360)
    - The robot learns: "15 means right, -15 means left"
    - Natural number semantics!

    Args:
        radians: Signed rotation angle in radians (positive=clockwise, negative=counter-clockwise)
        degrees: Signed rotation angle in degrees (positive=clockwise, negative=counter-clockwise)
        speed: Base wheel velocity magnitude (always positive, range 0-6, default 6.0)

    Returns:
        ActionBlock with wheels configured for rotation (direction from sign)

    Examples:
        spin(degrees=15)   → clockwise 15°
        spin(degrees=-15)  → counter-clockwise 15°
        spin(degrees=360)  → full rotation clockwise
        spin(degrees=-180) → half rotation counter-clockwise
    """
    # Determine direction from sign
    value = degrees if degrees is not None else radians
    if value is None:
        raise ValueError("Must provide either radians or degrees")

    # Positive → clockwise (right), Negative → counter-clockwise (left)
    if value >= 0:
        # Clockwise: left wheel backward, right wheel forward (FIX: swapped speeds!)
        return ActionBlock(
            id=f"spin_{degrees or radians}",
            description=f"Spin {degrees}° (clockwise)" if degrees else f"Spin {radians}rad (clockwise)",
            execution_mode="parallel",
            actions=[
                BaseRotateBy(radians=radians, degrees=degrees, speed=-speed, actuator_id='left_wheel_vel'),  # BACKWARD
                BaseRotateBy(radians=radians, degrees=degrees, speed=speed, actuator_id='right_wheel_vel')   # FORWARD
            ]
        )
    else:
        # Counter-clockwise: left wheel forward, right wheel backward (FIX: swapped speeds!)
        # Pass negative values directly - BaseRotateBy handles sign correctly!
        return ActionBlock(
            id=f"spin_{degrees or radians}",
            description=f"Spin {degrees}° (counter-clockwise)" if degrees else f"Spin {radians}rad (counter-clockwise)",
            execution_mode="parallel",
            actions=[
                BaseRotateBy(radians=radians, degrees=degrees, speed=speed, actuator_id='left_wheel_vel'),   # FORWARD
                BaseRotateBy(radians=radians, degrees=degrees, speed=-speed, actuator_id='right_wheel_vel')  # BACKWARD
            ]
        )



# ============================================
# END OF FILE
# ============================================
# Note: Static ACTION_BLOCKS_REGISTRY removed (unused).
# Use function-based API instead: move_forward(), spin(), etc.


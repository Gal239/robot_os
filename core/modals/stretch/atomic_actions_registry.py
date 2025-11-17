"""
ATOMIC ACTIONS REGISTRY - Parameterizable action builders
These are the fundamental building blocks for all robot actions
"""

from typing import Dict, Callable, Any
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


def create_atomic_action(action_class, action_id: str, description: str, **default_params):
    """Factory for creating parameterizable atomic actions"""
    def builder(**kwargs):
        # Merge defaults with provided params
        params = {**default_params, **kwargs}
        return ActionBlock(
            id=action_id,
            description=description,
            push_before_others=False,
            end_with_thinking=False,
            actions=[action_class(**params)]
        )
    return builder




# ============================================
# ATOMIC ACTIONS - Parameterizable building blocks
# ============================================

ATOMIC_ACTIONS = {
    # ARM ACTIONS
    'arm_move_to': create_atomic_action(
        ArmMoveTo, 'arm_move_to', 'Move arm to position',
        position=0.0  # Default: retracted
    ),
    'arm_move_by': create_atomic_action(
        ArmMoveBy, 'arm_move_by', 'Move arm by distance',
        distance=0.1  # Default: 10cm
    ),

    # LIFT ACTIONS
    'lift_move_to': create_atomic_action(
        LiftMoveTo, 'lift_move_to', 'Move lift to height',
        position=0.5  # Default: middle
    ),
    'lift_move_by': create_atomic_action(
        LiftMoveBy, 'lift_move_by', 'Move lift by distance',
        distance=0.1  # Default: 10cm
    ),

    # BASE ACTIONS
    'base_forward': create_atomic_action(
        BaseMoveForward, 'base_forward', 'Move forward',
        distance=0.5  # Default: 50cm
    ),
    'base_backward': create_atomic_action(
        BaseMoveBackward, 'base_backward', 'Move backward',
        distance=0.5  # Default: 50cm
    ),
    'base_rotate': create_atomic_action(
        BaseRotateBy, 'base_rotate', 'Rotate base',
        radians=1.57  # Default: 90 degrees
    ),

    # HEAD PAN ACTIONS
    'head_pan_to': create_atomic_action(
        HeadPanMoveTo, 'head_pan_to', 'Pan head to angle',
        position=0.0  # Default: center
    ),
    'head_pan_by': create_atomic_action(
        HeadPanMoveBy, 'head_pan_by', 'Pan head by angle',
        radians=0.1  # Default: small turn
    ),

    # HEAD TILT ACTIONS
    'head_tilt_to': create_atomic_action(
        HeadTiltMoveTo, 'head_tilt_to', 'Tilt head to angle',
        position=0.0  # Default: level
    ),
    'head_tilt_by': create_atomic_action(
        HeadTiltMoveBy, 'head_tilt_by', 'Tilt head by angle',
        radians=0.1  # Default: small tilt
    ),

    # WRIST YAW ACTIONS
    'wrist_yaw_to': create_atomic_action(
        WristYawMoveTo, 'wrist_yaw_to', 'Rotate wrist yaw to angle',
        position=0.0  # Default: center
    ),
    'wrist_yaw_by': create_atomic_action(
        WristYawMoveBy, 'wrist_yaw_by', 'Rotate wrist yaw by angle',
        radians=0.1  # Default: small rotation
    ),

    # WRIST PITCH ACTIONS
    'wrist_pitch_to': create_atomic_action(
        WristPitchMoveTo, 'wrist_pitch_to', 'Pitch wrist to angle',
        position=0.0  # Default: level
    ),
    'wrist_pitch_by': create_atomic_action(
        WristPitchMoveBy, 'wrist_pitch_by', 'Pitch wrist by angle',
        radians=0.1  # Default: small pitch
    ),

    # WRIST ROLL ACTIONS
    'wrist_roll_to': create_atomic_action(
        WristRollMoveTo, 'wrist_roll_to', 'Roll wrist to angle',
        position=0.0  # Default: level
    ),
    'wrist_roll_by': create_atomic_action(
        WristRollMoveBy, 'wrist_roll_by', 'Roll wrist by angle',
        radians=0.1  # Default: small roll
    ),

    # GRIPPER ACTIONS
    'gripper_move_to': create_atomic_action(
        GripperMoveTo, 'gripper_move_to', 'Move gripper to position',
        position=0.0  # Default: neutral
    ),
    'gripper_move_by': create_atomic_action(
        GripperMoveBy, 'gripper_move_by', 'Move gripper by distance',
        distance=0.01  # Default: 1cm
    ),

    # SPEAKER ACTION
    'speak': create_atomic_action(
        SpeakerPlay, 'speak', 'Say text',
        text=""  # Must be provided
    ),
}


def get_atomic_action(name: str, **params) -> ActionBlock:
    """Get parameterized atomic action"""
    if name not in ATOMIC_ACTIONS:
        raise KeyError(f"Unknown atomic action: {name}")
    return ATOMIC_ACTIONS[name](**params)


def list_atomic_actions() -> list:
    """List all available atomic actions"""
    return list(ATOMIC_ACTIONS.keys())
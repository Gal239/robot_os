"""
TYPED EVENT SYSTEM - Type-Safe Event Communication
==================================================

Replaces raw dict events with properly typed dataclass events.

BEFORE (Raw Dicts):
    event_bus.emit("step", {"step_count": 10, "reward": 5.0})
    # No type safety, easy to misspell keys, no IDE support

AFTER (Typed Events):
    event = StepEvent(step_count=10, reward=5.0)
    event_bus.emit(event)
    # Type-checked, IDE autocomplete, validated at runtime

MOP PATTERN: Events ARE MODALS
- Self-Building: Events validate themselves
- Self-Rendering: Events serialize to dict
- Self-Composing: Events form event streams

Key Benefits:
1. Type Safety: Catch errors at development time
2. IDE Support: Autocomplete for event fields
3. Documentation: Event schema is code
4. Validation: Invalid events rejected
5. Refactoring: Change propagates automatically

Event Types:
- StepEvent: Simulation step completed
- RewardEvent: Reward changed
- ActionEvent: Action submitted/completed
- StateEvent: State changed
- ErrorEvent: Error occurred
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class EventType(Enum):
    """Event type enum for type-safe event handling"""
    STEP = "step"
    REWARD = "reward"
    ACTION = "action"
    STATE = "state"
    ERROR = "error"
    SENSOR = "sensor"
    ACTUATOR = "actuator"


@dataclass
class BaseEvent:
    """Base class for all typed events

    All events inherit from this and add their own fields.
    Provides common functionality for serialization and validation.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dict for compatibility with EventBus

        Returns:
            Dict representation of event
        """
        return asdict(self)

    def get_type(self) -> EventType:
        """Get event type

        Returns:
            EventType enum value
        """
        # Override in subclasses
        raise NotImplementedError("Subclass must implement get_type()")


@dataclass
class StepEvent(BaseEvent):
    """Event fired after each simulation step

    Fields:
        step_count: Current step number
        reward: Current total reward
        state: Full state dict
        elapsed_time: Time since experiment start (seconds)
    """
    step_count: int
    reward: float
    state: Dict[str, Any]
    elapsed_time: float

    def get_type(self) -> EventType:
        return EventType.STEP


@dataclass
class RewardEvent(BaseEvent):
    """Event fired when reward changes

    Fields:
        reward: New total reward value
        delta: Change in reward this step
        step_count: Step number when reward changed
        reward_breakdown: Optional dict of individual reward values
    """
    reward: float
    delta: float
    step_count: int
    reward_breakdown: Optional[Dict[str, float]] = None

    def get_type(self) -> EventType:
        return EventType.REWARD


@dataclass
class ActionEvent(BaseEvent):
    """Event fired when action is submitted or completed

    Fields:
        action_id: Unique action identifier
        action_type: Type of action (e.g., "move_arm", "grasp")
        status: Action status ("submitted", "executing", "completed", "failed")
        actuator_id: Which actuator this action targets
        params: Action parameters
        step_count: Step number
    """
    action_id: str
    action_type: str
    status: str  # "submitted" | "executing" | "completed" | "failed"
    actuator_id: str
    params: Dict[str, Any]
    step_count: int

    def get_type(self) -> EventType:
        return EventType.ACTION


@dataclass
class StateEvent(BaseEvent):
    """Event fired when specific state changes

    Fields:
        asset_id: Which asset changed (e.g., "stretch.arm", "apple")
        behavior: Which behavior changed (e.g., "extension", "position")
        old_value: Previous value
        new_value: New value
        step_count: Step number
    """
    asset_id: str
    behavior: str
    old_value: Any
    new_value: Any
    step_count: int

    def get_type(self) -> EventType:
        return EventType.STATE


@dataclass
class SensorEvent(BaseEvent):
    """Event fired when sensor data is updated

    Fields:
        sensor_id: Sensor identifier (e.g., "head_camera", "wrist_camera")
        sensor_type: Type of sensor ("camera", "force", "imu", etc.)
        data: Sensor data (varies by type)
        step_count: Step number
        timestamp: Real-world timestamp
    """
    sensor_id: str
    sensor_type: str
    data: Any
    step_count: int
    timestamp: float

    def get_type(self) -> EventType:
        return EventType.SENSOR


@dataclass
class ActuatorEvent(BaseEvent):
    """Event fired when actuator state changes

    Fields:
        actuator_id: Actuator identifier (e.g., "stretch.arm.extension")
        actuator_type: Type of actuator ("prismatic", "revolute", etc.)
        position: Current position
        velocity: Current velocity
        force: Current force/torque
        step_count: Step number
    """
    actuator_id: str
    actuator_type: str
    position: float
    velocity: float
    force: float
    step_count: int

    def get_type(self) -> EventType:
        return EventType.ACTUATOR


@dataclass
class ErrorEvent(BaseEvent):
    """Event fired when error occurs

    Fields:
        error_type: Type of error ("physics", "action", "state", etc.)
        message: Error message
        details: Additional error details
        step_count: Step number when error occurred
        recoverable: Whether error is recoverable
    """
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    step_count: Optional[int] = None
    recoverable: bool = True

    def get_type(self) -> EventType:
        return EventType.ERROR


# === TYPE GUARD HELPERS ===

def is_step_event(event: BaseEvent) -> bool:
    """Check if event is StepEvent"""
    return isinstance(event, StepEvent)


def is_reward_event(event: BaseEvent) -> bool:
    """Check if event is RewardEvent"""
    return isinstance(event, RewardEvent)


def is_action_event(event: BaseEvent) -> bool:
    """Check if event is ActionEvent"""
    return isinstance(event, ActionEvent)


def is_state_event(event: BaseEvent) -> bool:
    """Check if event is StateEvent"""
    return isinstance(event, StateEvent)


def is_sensor_event(event: BaseEvent) -> bool:
    """Check if event is SensorEvent"""
    return isinstance(event, SensorEvent)


def is_actuator_event(event: BaseEvent) -> bool:
    """Check if event is ActuatorEvent"""
    return isinstance(event, ActuatorEvent)


def is_error_event(event: BaseEvent) -> bool:
    """Check if event is ErrorEvent"""
    return isinstance(event, ErrorEvent)


# === EVENT VALIDATION ===

def validate_event(event: BaseEvent) -> bool:
    """Validate event has all required fields

    Args:
        event: Event to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        # Check event has get_type method
        event.get_type()

        # Check event can convert to dict
        event.to_dict()

        return True
    except Exception:
        return False


# === BACKWARDS COMPATIBILITY ===

def dict_to_event(event_name: str, data: Dict[str, Any]) -> BaseEvent:
    """Convert legacy dict event to typed event

    For backwards compatibility with old code that uses dicts.

    Args:
        event_name: Event name ("step", "reward", etc.)
        data: Event data dict

    Returns:
        Typed event instance
    """
    event_name = event_name.lower()

    if event_name == "step":
        return StepEvent(
            step_count=data["step_count"],  # OFFENSIVE - crash if missing!
            reward=data["reward"],  # OFFENSIVE - crash if missing!
            state=data["state"],  # OFFENSIVE - crash if missing!
            elapsed_time=data["elapsed_time"]  # OFFENSIVE - crash if missing!
        )
    elif event_name == "reward":
        return RewardEvent(
            reward=data["reward"],  # OFFENSIVE - crash if missing!
            delta=data["delta"],  # OFFENSIVE - crash if missing!
            step_count=data["step_count"],  # OFFENSIVE - crash if missing!
            reward_breakdown=data.get("reward_breakdown")  # LEGITIMATE - optional field
        )
    elif event_name == "action":
        return ActionEvent(
            action_id=data["action_id"],  # OFFENSIVE - crash if missing!
            action_type=data["action_type"],  # OFFENSIVE - crash if missing!
            status=data["status"],  # OFFENSIVE - crash if missing!
            actuator_id=data["actuator_id"],  # OFFENSIVE - crash if missing!
            params=data["params"],  # OFFENSIVE - crash if missing!
            step_count=data["step_count"]  # OFFENSIVE - crash if missing!
        )
    elif event_name == "state":
        return StateEvent(
            asset_id=data["asset_id"],  # OFFENSIVE - crash if missing!
            behavior=data["behavior"],  # OFFENSIVE - crash if missing!
            old_value=data["old_value"],  # OFFENSIVE - crash if missing!
            new_value=data["new_value"],  # OFFENSIVE - crash if missing!
            step_count=data["step_count"]  # OFFENSIVE - crash if missing!
        )
    elif event_name == "error":
        return ErrorEvent(
            error_type=data["error_type"],  # OFFENSIVE - crash if missing!
            message=data["message"],  # OFFENSIVE - crash if missing!
            details=data.get("details"),  # LEGITIMATE - optional field
            step_count=data.get("step_count"),  # LEGITIMATE - optional field
            recoverable=data.get("recoverable", True)  # LEGITIMATE - has sensible default
        )
    else:
        # Unknown event type - create generic event
        raise ValueError(f"Unknown event type: {event_name}")


if __name__ == "__main__":
    """Example usage"""
    print("=" * 80)
    print("TYPED EVENT SYSTEM - Examples")
    print("=" * 80)

    # Create typed events
    step_event = StepEvent(
        step_count=100,
        reward=50.0,
        state={"robot": {"position": [1, 2, 3]}},
        elapsed_time=10.5
    )

    reward_event = RewardEvent(
        reward=75.0,
        delta=25.0,
        step_count=101,
        reward_breakdown={"distance": 50.0, "grasp": 25.0}
    )

    action_event = ActionEvent(
        action_id="action_123",
        action_type="move_arm",
        status="executing",
        actuator_id="stretch.arm",
        params={"extension": 0.5},
        step_count=100
    )

    print("\n✅ Created typed events:")
    print(f"  - {step_event.get_type().value}: step {step_event.step_count}")
    print(f"  - {reward_event.get_type().value}: reward {reward_event.reward}")
    print(f"  - {action_event.get_type().value}: {action_event.action_type}")

    # Validate events
    print("\n✅ Validating events:")
    for event in [step_event, reward_event, action_event]:
        valid = validate_event(event)
        print(f"  - {event.get_type().value}: {'✓ valid' if valid else '✗ invalid'}")

    # Convert to dict (for EventBus compatibility)
    print("\n✅ Convert to dict:")
    print(f"  - StepEvent: {step_event.to_dict()}")

    # Backwards compatibility
    print("\n✅ Backwards compatibility (dict → event):")
    legacy_dict = {"step_count": 200, "reward": 100.0, "state": {}, "elapsed_time": 20.0}
    converted = dict_to_event("step", legacy_dict)
    print(f"  - Converted dict to {type(converted).__name__}")

    print("\n✅ Typed Event System working!")

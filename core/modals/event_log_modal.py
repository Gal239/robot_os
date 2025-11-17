"""
EVENT LOG MODAL - Time-stamped event tracking for sensors, behaviors, and actions
OFFENSIVE & ELEGANT: Self-describing event stream, queryable history

Philosophy:
- Actions are dumb reflexes (don't emit, just execute)
- Sensors/behaviors are smart observers (emit when state changes)
- EventLog is a view (accessible to tests, RL, LLM)
- ALWAYS-ON: EventLog is instantiated by RuntimeEngine (no optional checks!)
- 90-95% coverage (handles most scenarios elegantly)

Events track:
1. Sensor changes: Force sensor goes 0 → 5.2N
2. Behavior changes: Door opens 0% → 85%
3. Action completion: arm_move_to finishes
4. Action failures: gripper_close fails (stuck)

Usage in tests:
    result = ops.run_simulation(scene, duration=10.0)
    events = result.views['event_log'].query(source="door.hinged")
    assert any(e.data["open"]["new"] >= 80 for e in events)

Usage in RL:
    recent_events = views['event_log'].query(since=current_time - 0.1)
    sensor_changes = [e for e in recent_events if e.event_type == "sensor_change"]
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
import time


class Event(BaseModel):
    """Single event in time - OFFENSIVE & ELEGANT

    Self-describing: Contains everything needed to understand what happened

    Examples:
        # Sensor change
        Event(
            timestamp=0.523,
            source="gripper_force",
            event_type="sensor_change",
            data={"force_left": {"old": 0.0, "new": 5.2}}
        )

        # Behavior change
        Event(
            timestamp=1.245,
            source="door.hinged",
            event_type="behavior_change",
            data={"open": {"old": 0.0, "new": 85.0}}
        )

        # Action complete
        Event(
            timestamp=2.100,
            source="arm_move_to_abc123",
            event_type="action_complete",
            data={"final_position": 0.3, "duration": 1.5}
        )

        # Action failed
        Event(
            timestamp=3.200,
            source="gripper_close_xyz789",
            event_type="action_failed",
            data={"reason": "stuck", "progress": 45.0}
        )
    """
    timestamp: float = Field(..., description="When event occurred (simulation time)")
    source: str = Field(..., description="Who emitted event (sensor_id, asset.behavior, action_id)")
    event_type: Literal["sensor_change", "behavior_change", "action_complete", "action_failed", "custom"] = Field(
        ..., description="Type of event"
    )
    data: Dict[str, Any] = Field(..., description="Event-specific data")

    # Optional metadata
    step: Optional[int] = Field(None, description="Simulation step number")
    episode: Optional[int] = Field(None, description="Episode number (for RL)")


class EventLog(BaseModel):
    """Time-stamped event log - OFFENSIVE & ELEGANT

    Self-describing: Complete history of simulation events
    Queryable: Filter by source, type, time range
    View-compatible: Can be rendered/serialized

    Benefits:
    - Complete observability: See EXACTLY what changed when
    - Debugging: Replay event sequence to understand failures
    - RL training: Learn from event patterns (not just final state)
    - Testing: Assert on event sequences (not just final state)
    - LLM context: Describe what happened using event stream

    Usage:
        # Add event
        log.add_event(Event(timestamp=1.5, source="door", event_type="behavior_change",
                           data={"open": {"old": 0, "new": 50}}))

        # Query events
        door_events = log.query(source="door")
        recent = log.query(since=10.0)
        changes = log.query(event_type="behavior_change")

        # Get last change
        last_force = log.get_last_change("gripper_force", "force_left")
        if last_force:
            print(f"Force changed from {last_force.data['force_left']['old']} to {last_force.data['force_left']['new']}")
    """
    events: List[Event] = Field(default_factory=list, description="Chronological event list")
    max_events: int = Field(10000, description="Max events to keep (prevent memory explosion)")

    def add_event(self, event: Event):
        """Add event to log - OFFENSIVE

        Maintains chronological order
        Trims old events if max_events exceeded
        """
        self.events.append(event)

        # Trim if too many events (keep most recent)
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def query(
        self,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        before: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """Query events with filters - OFFENSIVE

        Args:
            source: Filter by source (e.g., "gripper_force", "door.hinged")
            event_type: Filter by type (e.g., "sensor_change", "behavior_change")
            since: Only events after this timestamp
            before: Only events before this timestamp
            limit: Max events to return

        Returns:
            List of matching events (chronological order)

        Examples:
            # All door events
            log.query(source="door.hinged")

            # Recent sensor changes
            log.query(event_type="sensor_change", since=current_time - 1.0)

            # Last 10 events
            log.query(limit=10)
        """
        result = self.events

        # Filter by source
        if source is not None:
            result = [e for e in result if e.source == source]

        # Filter by event type
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]

        # Filter by time range
        if since is not None:
            result = [e for e in result if e.timestamp >= since]

        if before is not None:
            result = [e for e in result if e.timestamp <= before]

        # Limit results
        if limit is not None:
            result = result[-limit:]  # Take most recent

        return result

    def get_last_change(self, source: str, field: Optional[str] = None) -> Optional[Event]:
        """Get last change event for source/field - OFFENSIVE

        Args:
            source: Event source (sensor_id, asset.behavior)
            field: Optional field name (e.g., "force_left", "open")

        Returns:
            Most recent change event, or None if not found

        Examples:
            # Last force sensor change
            last_force = log.get_last_change("gripper_force", "force_left")

            # Last door behavior change (any field)
            last_door = log.get_last_change("door.hinged")
        """
        # Get all events from source
        source_events = [e for e in self.events if e.source == source]

        # Filter by field if specified
        if field is not None:
            source_events = [e for e in source_events if field in e.data]

        # Return most recent
        return source_events[-1] if source_events else None

    def get_changes_since(self, source: str, since: float) -> Dict[str, Any]:
        """Get all changes for source since timestamp - OFFENSIVE

        Returns dict of field → latest value

        Example:
            changes = log.get_changes_since("door.hinged", t=10.0)
            # {"open": 85.0, "closed": False, "angle": 1.2}
        """
        changes = {}

        events = self.query(source=source, since=since)
        for event in events:
            # Extract "new" values from change events
            for field, value_dict in event.data.items():
                if isinstance(value_dict, dict) and "new" in value_dict:
                    changes[field] = value_dict["new"]

        return changes

    def count_events(
        self,
        source: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None
    ) -> int:
        """Count matching events - OFFENSIVE

        Useful for:
        - RL rewards: "How many times did gripper touch object?"
        - Testing: "Did door open event occur?"
        - Debugging: "How many action failures in last 10 seconds?"
        """
        return len(self.query(source=source, event_type=event_type, since=since))

    def clear(self):
        """Clear all events - OFFENSIVE

        Useful for:
        - Episode resets in RL
        - Test setup/teardown
        """
        self.events = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/replay - OFFENSIVE"""
        return {
            "events": [e.model_dump() for e in self.events],
            "total_events": len(self.events)
        }

    def summary(self, since: Optional[float] = None) -> Dict[str, int]:
        """Get event summary - OFFENSIVE

        Returns:
            Dict of event_type → count

        Example:
            summary = log.summary(since=10.0)
            # {"sensor_change": 42, "behavior_change": 15, "action_complete": 3}
        """
        events = self.query(since=since) if since else self.events
        summary = {}

        for event in events:
            event_type = event.event_type
            summary[event_type] = summary.get(event_type, 0) + 1

        return summary


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_sensor_change_event(
    timestamp: float,
    sensor_id: str,
    field: str,
    old_value: Any,
    new_value: Any,
    step: Optional[int] = None
) -> Event:
    """Helper to create sensor change event - ELEGANT

    Example:
        event = create_sensor_change_event(
            timestamp=1.5,
            sensor_id="gripper_force",
            field="force_left",
            old_value=0.0,
            new_value=5.2
        )
    """
    return Event(
        timestamp=timestamp,
        source=sensor_id,
        event_type="sensor_change",
        data={field: {"old": old_value, "new": new_value}},
        step=step
    )


def create_behavior_change_event(
    timestamp: float,
    asset_id: str,
    behavior: str,
    field: str,
    old_value: Any,
    new_value: Any,
    step: Optional[int] = None
) -> Event:
    """Helper to create behavior change event - ELEGANT

    Example:
        event = create_behavior_change_event(
            timestamp=2.1,
            asset_id="door",
            behavior="hinged",
            field="open",
            old_value=0.0,
            new_value=85.0
        )
    """
    return Event(
        timestamp=timestamp,
        source=f"{asset_id}.{behavior}",
        event_type="behavior_change",
        data={field: {"old": old_value, "new": new_value}},
        step=step
    )


def create_action_event(
    timestamp: float,
    action_id: str,
    event_type: Literal["action_complete", "action_failed"],
    data: Dict[str, Any],
    step: Optional[int] = None
) -> Event:
    """Helper to create action event - ELEGANT

    Example:
        event = create_action_event(
            timestamp=3.5,
            action_id="arm_move_to_xyz",
            event_type="action_complete",
            data={"final_position": 0.3, "duration": 1.2}
        )
    """
    return Event(
        timestamp=timestamp,
        source=action_id,
        event_type=event_type,
        data=data,
        step=step
    )

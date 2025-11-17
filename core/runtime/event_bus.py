"""
EVENT BUS - Central event system for RuntimeEngine
OFFENSIVE & ELEGANT: Publish-subscribe pattern for runtime events

Pattern: Components emit events, subscribers listen and react
Events: step, reward, action_complete, condition_met, episode_end, error
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
import time


@dataclass
class Event:
    """Event data structure - OFFENSIVE

    Every event has:
    - type: Event type string (e.g., "step", "reward", "action_complete")
    - data: Event-specific data dict
    - timestamp: Unix timestamp when event occurred
    """
    type: str  # Event type identifier
    data: Dict[str, Any] = field(default_factory=dict)  # Event-specific data
    timestamp: float = field(default_factory=time.time)  # When event occurred

    def __repr__(self) -> str:
        return f"Event(type='{self.type}', data_keys={list(self.data.keys())}, timestamp={self.timestamp:.3f})"


class EventBus:
    """Central event bus for RuntimeEngine - OFFENSIVE

    Design:
    - Components emit() events with type and data
    - Subscribers on() register callbacks for specific event types
    - EventBus maintains history for debugging/replay
    - Subscribers can filter by event type or data content

    Common Event Types:
    - "step": Emitted after each physics step (data: step_count, reward, state)
    - "reward": Emitted when reward changes (data: reward, delta, conditions_met)
    - "action_complete": Emitted when action finishes (data: action_name, duration, success)
    - "condition_met": Emitted when reward condition satisfied (data: condition_id, timestamp)
    - "episode_end": Emitted when episode terminates (data: total_reward, step_count, success)
    - "error": Emitted on runtime errors (data: error_type, message, traceback)
    """

    def __init__(self, history_limit: int = 1000):
        """Initialize event bus

        Args:
            history_limit: Max events to keep in history (for replay/debugging)
        """
        self.subscribers: Dict[str, List[Callable]] = {}  # {event_type: [callbacks]}
        self.history: List[Event] = []  # Recent events
        self.history_limit = history_limit

    def on(self, event_type: str, callback: Callable[[Event], None]):
        """Subscribe to event type - OFFENSIVE

        Args:
            event_type: Event type to listen for (e.g., "step", "reward")
            callback: Function to call when event occurs (receives Event object)

        Example:
            # Log every step
            bus.on("step", lambda e: print(f"Step {e.data['step_count']}: reward={e.data['reward']}"))

            # Save when episode ends
            bus.on("episode_end", lambda e: save_results(e.data))

            # Alert on errors
            bus.on("error", lambda e: send_alert(e.data['message']))
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []

        self.subscribers[event_type].append(callback)

    def emit(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Emit event to all subscribers - OFFENSIVE

        Args:
            event_type: Type of event (e.g., "step", "reward")
            data: Event-specific data dict

        Example:
            # Emit step event
            bus.emit("step", {
                "step_count": 42,
                "reward": 150.5,
                "state": {...}
            })

            # Emit action complete
            bus.emit("action_complete", {
                "action_name": "arm_move_to(0.5)",
                "duration": 2.3,
                "success": True
            })
        """
        # Create event
        event = Event(type=event_type, data=data or {})

        # Add to history
        self.history.append(event)

        # Trim history if too long
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]

        # Call all subscribers for this event type
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    # Don't crash RuntimeEngine if subscriber fails
                    print(f"WARNING: Event subscriber for '{event_type}' failed: {e}")

    def get_history(self, event_type: Optional[str] = None,
                    limit: Optional[int] = None) -> List[Event]:
        """Get event history - OFFENSIVE

        Args:
            event_type: Filter by event type (None = all events)
            limit: Max events to return (None = all)

        Returns:
            List of Event objects in chronological order
        """
        # Filter by type if specified
        if event_type is not None:
            events = [e for e in self.history if e.type == event_type]
        else:
            events = self.history

        # Limit if specified
        if limit is not None:
            events = events[-limit:]

        return events

    def clear_history(self):
        """Clear event history - OFFENSIVE"""
        self.history = []

    def has_subscribers(self, event_type: str) -> bool:
        """Check if any subscribers for event type - OFFENSIVE"""
        return event_type in self.subscribers and len(self.subscribers[event_type]) > 0


# === USAGE PATTERN ===

"""
# In RuntimeEngine.__init__():
self.event_bus = EventBus()

# Subscribe to events
self.event_bus.on("step", lambda e: print(f"Step {e.data['step_count']}"))
self.event_bus.on("episode_end", lambda e: self._save_results(e.data))

# In RuntimeEngine.step():
def step(self):
    # ... step logic ...

    # Emit step event
    self.event_bus.emit("step", {
        "step_count": self.step_count,
        "reward": self.current_reward,
        "state": state
    })

    # Emit reward event if changed
    if reward != self.last_reward:
        self.event_bus.emit("reward", {
            "reward": reward,
            "delta": reward - self.last_reward,
            "conditions_met": [...]
        })

# User can subscribe externally
engine.on("step", lambda e: logger.log(e.data))
engine.on("action_complete", lambda e: ui.update(e.data))
"""


# === TEST ===

if __name__ == "__main__":
    print("=== Testing EventBus ===")

    # Create bus
    bus = EventBus()

    # Track events
    step_events = []
    reward_events = []
    action_events = []

    # Subscribe
    bus.on("step", lambda e: step_events.append(e))
    bus.on("reward", lambda e: reward_events.append(e))
    bus.on("action_complete", lambda e: action_events.append(e))

    print(f"Subscribers: step={bus.has_subscribers('step')}, reward={bus.has_subscribers('reward')}")

    # Emit events
    print("\nEmitting events...")
    for i in range(5):
        bus.emit("step", {"step_count": i, "reward": i * 10.0})

        if i % 2 == 0:
            bus.emit("reward", {"reward": i * 10.0, "delta": 10.0})

    bus.emit("action_complete", {"action_name": "arm_move_to(0.5)", "duration": 2.3, "success": True})

    # Check results
    print(f"\nStep events received: {len(step_events)}")
    print(f"Reward events received: {len(reward_events)}")
    print(f"Action events received: {len(action_events)}")

    # Check history
    print(f"\nTotal history: {len(bus.history)} events")
    step_history = bus.get_history("step")
    print(f"Step events in history: {len(step_history)}")

    # Print sample events
    print(f"\nSample step event: {step_events[0]}")
    print(f"Sample reward event: {reward_events[0]}")
    print(f"Sample action event: {action_events[0]}")

    # Test error handling (subscriber fails)
    print("\nTesting error handling...")
    bus.on("error", lambda e: [][1])  # Will crash
    bus.emit("error", {"message": "Test error"})
    print("✓ EventBus continued after subscriber failure")

    print("\n✓ EventBus test complete!")

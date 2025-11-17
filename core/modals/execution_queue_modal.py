"""
Execution Queue Modal - Track actuator queues and block execution
Offensive Programming: Single source of truth, no hardcoding
"""

from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Set
from datetime import datetime


class ActuatorState(BaseModel):
    """State of a single actuator"""
    actuator_id: str
    is_available: bool = True
    current_action: Optional[Any] = None
    current_block_id: Optional[int] = None
    queue: List[Any] = []


class BlockRecord(BaseModel):
    """Record of a submitted block"""
    block_id: int
    block_name: str
    submitted_at: datetime
    execution_mode: str
    status: str = "queued"  # queued, executing, completed, cancelled
    actions_total: int = 0
    actions_completed: int = 0


class ExecutionQueueModal(BaseModel):
    """Dynamic queue state - initialized with robot actuators"""

    # Actuator states
    actuators: Dict[str, ActuatorState] = {}

    # Block tracking
    blocks: Dict[int, BlockRecord] = {}
    next_block_id: int = 0

    # Event log
    event_log: List[Dict] = []

    def init_from_actuators(self, actuators: Dict):
        """Initialize queues for all robot actuators"""
        for actuator_id in actuators:
            self.actuators[actuator_id] = ActuatorState(actuator_id=actuator_id)

    def submit_block(self, block) -> int:
        """Submit an ActionBlock for execution"""
        block_id = self.next_block_id
        self.next_block_id += 1

        # Record block
        self.blocks[block_id] = BlockRecord(
            block_id=block_id,
            block_name=block.id,
            submitted_at=datetime.now(),
            execution_mode=block.execution_mode,
            actions_total=len(block.actions)
        )

        # Handle replace_current - clear everything
        if block.replace_current:
            for actuator in self.actuators.values():
                actuator.queue.clear()
                actuator.current_action = None
                actuator.is_available = True

            # Cancel all pending blocks
            for b in self.blocks.values():
                if b.status in ['queued', 'executing']:
                    b.status = 'cancelled'

        # Queue actions with dependencies
        dependencies: Set[str] = set()

        for action in block.actions:
            # MOP: Action declares its actuators!
            actuator_ids = action.actuator_ids

            # Skip actions with no actuators
            if not actuator_ids:
                continue

            # Set dependencies for sequential mode (BEFORE looping actuators!)
            if block.execution_mode == "sequential":
                action._wait_for = dependencies.copy()
                # Add ALL actuators this action controls to dependencies
                for actuator_id in actuator_ids:
                    dependencies.add(actuator_id)
            else:
                action._wait_for = set()

            # Tag action with block_id
            action._block_id = block_id

            # FIX: Add action to ALL actuators it controls (not just first!)
            for actuator_id in actuator_ids:
                # Ensure actuator exists
                if actuator_id not in self.actuators:
                    self.actuators[actuator_id] = ActuatorState(actuator_id=actuator_id)

                # Add to actuator's queue
                if block.push_before_others:
                    self.actuators[actuator_id].queue.insert(0, action)
                else:
                    self.actuators[actuator_id].queue.append(action)

        return block_id

    def get_next_action(self, actuator_id: str) -> Optional[Any]:
        """Get next action that can run on this actuator"""
        if actuator_id not in self.actuators:
            return None

        actuator = self.actuators[actuator_id]
        if not actuator.is_available or not actuator.queue:
            return None

        # Check if next action's dependencies are met
        next_action = actuator.queue[0]
        wait_for = getattr(next_action, '_wait_for', set())

        # Check if any dependency is still running
        for dep_actuator_id in wait_for:
            if dep_actuator_id in self.actuators:
                if not self.actuators[dep_actuator_id].is_available:
                    return None  # Must wait

        # Can run!
        return actuator.queue.pop(0)

    def start_action(self, actuator_id: str, action: Any):
        """Mark action as started on actuator - OFFENSIVE validation!"""
        # OFFENSIVE: Verify actuator exists
        if actuator_id not in self.actuators:
            raise RuntimeError(
                f"QUEUE STATE ERROR!\n"
                f"  Trying to start action on unknown actuator: {actuator_id}\n"
                f"  Known actuators: {list(self.actuators.keys())}\n"
                f"\n"
                f"  FIX: Call init_from_actuators() before starting actions"
            )

        actuator = self.actuators[actuator_id]

        # OFFENSIVE: Verify actuator is available
        if not actuator.is_available:
            raise RuntimeError(
                f"QUEUE STATE ERROR!\n"
                f"  Trying to start action on busy actuator: {actuator_id}\n"
                f"  Current action: {actuator.current_action}\n"
                f"  is_available: {actuator.is_available}\n"
                f"\n"
                f"  FIX: Call complete_action() before starting new action"
            )

        # MOP-CORRECT: Update state atomically
        actuator.current_action = action
        actuator.is_available = False
        actuator.current_block_id = getattr(action, '_block_id', None)

        # Update block status
        block_id = getattr(action, '_block_id', None)
        if block_id is not None and block_id in self.blocks:
            if self.blocks[block_id].status == 'queued':
                self.blocks[block_id].status = 'executing'

    def complete_action(self, actuator_id: str):
        """Mark current action as complete - OFFENSIVE validation!"""
        # OFFENSIVE: Verify actuator exists
        if actuator_id not in self.actuators:
            raise RuntimeError(
                f"QUEUE STATE ERROR!\n"
                f"  Trying to complete action on unknown actuator: {actuator_id}\n"
                f"  Known actuators: {list(self.actuators.keys())}\n"
                f"\n"
                f"  FIX: Call init_from_actuators() first"
            )

        actuator = self.actuators[actuator_id]
        action = actuator.current_action

        # Update block progress if action exists
        if action:
            block_id = getattr(action, '_block_id', None)
            if block_id is not None and block_id in self.blocks:
                self.blocks[block_id].actions_completed += 1

                # Check if block is fully complete
                if self.blocks[block_id].actions_completed == self.blocks[block_id].actions_total:
                    self.blocks[block_id].status = 'completed'
                    # CRITICAL: Remove completed block from queue!
                    del self.blocks[block_id]

        # MOP-CORRECT: Clear state atomically (MUST set both!)
        actuator.current_action = None
        actuator.is_available = True
        actuator.current_block_id = None

        # OFFENSIVE: Verify state is consistent
        assert actuator.current_action is None, "State corruption: current_action not cleared!"
        assert actuator.is_available is True, "State corruption: is_available not set!"
        assert actuator.current_block_id is None, "State corruption: current_block_id not cleared!"

    def get_next_commands(self, robot, event_log=None) -> Dict[str, float]:
        """SELF-COORDINATE: Tick each action once, collect all commands (PURE MOP!)

        MOP Principle: Queue modal knows action→actuator mapping.
        Calls each action.tick() ONCE, gets dict of commands, returns unified dict.
        Runtime is DUMB - just applies commands!

        Why this is PURE MOP:
        - Actions SELF-DECLARE what they control (actuator_ids property)
        - Actions SELF-EXECUTE and return dict of commands (_get_command)
        - Queue SELF-COORDINATES by calling each action once
        - NO isinstance checks anywhere!

        Multi-actuator actions (like BaseMoveForward):
        - Stored in queue of ALL actuators it controls (lines 99-108)
        - tick() called ONCE per step, returns {'left_wheel_vel': 1.5, 'right_wheel_vel': 1.5}
        - All commands collected into unified dict
        - Same action instance appears in multiple actuator queues, but ticked only once!

        Args:
            robot: Robot with actuators (for accessing actuator properties)
            event_log: Optional event log for action tracking

        Returns:
            Dict of {actuator_id: command} for ALL executing actions
        """
        commands = {}
        ticked_actions = set()  # Track which actions already ticked this cycle

        # Iterate all actuators - actions may appear in MULTIPLE queues (multi-actuator)
        for actuator_id, actuator_state in self.actuators.items():
            action = actuator_state.current_action

            if action is None:
                # No action executing - try start next
                next_action = self.get_next_action(actuator_id)
                if next_action:
                    # Check if this action instance already started on another actuator
                    action_id = id(next_action)
                    if action_id not in ticked_actions:
                        # PURE MOP: SELF-CONNECT action to robot (once!)
                        next_action.connect(robot, event_log)
                        ticked_actions.add(action_id)

                    # Start action (queue modal updates state!)
                    self.start_action(actuator_id, next_action)
                    action = next_action
                else:
                    # No actions - skip (hold commands added by runtime)
                    continue

            # Check if already ticked (multi-actuator actions)
            action_id = id(action)
            if action_id in ticked_actions:
                continue  # Already ticked, skip to avoid duplicate execution

            # PURE MOP: Action self-executes via tick()!
            command_dict = action.tick()
            ticked_actions.add(action_id)

            # Action returns dict - add all commands
            # (Single-actuator actions return {actuator_id: command})
            # (Multi-actuator actions return {act1: cmd1, act2: cmd2})
            if isinstance(command_dict, dict):
                commands.update(command_dict)
            else:
                # Legacy support: single float return (should not happen with new actions)
                commands[actuator_id] = command_dict

            # Check if action completed itself
            if action.status in ['completed', 'failed']:
                # Mark complete in ALL actuators (multi-actuator fix!)
                for act_id, act_state in self.actuators.items():
                    if act_state.current_action is action:
                        self.complete_action(act_id)

        return commands

    def get_summary(self) -> Dict:
        """Get current queue state summary"""
        return {
            'actuators': {
                aid: {
                    'available': state.is_available,
                    'current': state.current_action.__class__.__name__ if state.current_action else None,
                    'queue_length': len(state.queue)
                }
                for aid, state in self.actuators.items()
            },
            'blocks': {
                bid: {
                    'name': block.block_name,
                    'status': block.status,
                    'progress': f"{block.actions_completed}/{block.actions_total}"
                }
                for bid, block in self.blocks.items()
                if block.status in ['queued', 'executing']
            }
        }
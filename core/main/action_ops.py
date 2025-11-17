"""
ACTION OPS - Unified API for action system

Following StateOps/VideoOps pattern - wraps RuntimeEngine.ActionExecutor.
Consolidates 6 scattered access patterns into 1 clean API.

Usage:
    ops.actions.get_status()              # What's executing?
    ops.actions.get_available_actions()   # Discovery
    ops.actions.submit(block)             # Execute
"""

from typing import Dict, List, Optional, Any
from core.modals.stretch.action_modals import ActionBlock


class ActionOps:
    """Action system operations - OFFENSIVE & MOP

    Provides clean API for action discovery, inspection, and control.
    """

    def __init__(self, engine):
        """Initialize with RuntimeEngine"""
        self.engine = engine
        self.action_executor = engine.action_executor
        self.queue_modal = engine.action_executor.queue_modal
        self.robot = None  # Set after compile

    # === DISCOVERY ===

    def get_available_actions(self) -> Dict[str, type]:
        """Get all available action types - AUTODISCOVERY"""
        assert self.robot is not None, "❌ Call ops.compile() first"
        return self.robot.actions.copy()

    def get_action_params(self, action_name: str) -> Dict:
        """Get parameters for action type"""
        assert self.robot is not None, "❌ Call ops.compile() first"

        if action_name not in self.robot.actions:
            available = list(self.robot.actions.keys())
            raise KeyError(
                f"❌ Action '{action_name}' not found!\n"
                f"Available: {available}"
            )

        return self._extract_params(self.robot.actions[action_name])

    # === QUEUE INSPECTION ===

    def get_status(self) -> Dict:
        """Get complete action system status - MAIN API

        Returns:
            Dict with actuators, blocks, and summary stats

        Example:
            status = ops.actions.get_status()
            print(f"Executing: {status['summary']['total_executing']}")
        """
        raw = self.queue_modal.get_summary()

        # Add summary stats
        executing = sum(1 for a in raw['actuators'].values() if not a['available'])
        pending = len(raw['blocks'])

        return {
            'actuators': raw['actuators'],
            'blocks': raw['blocks'],
            'summary': {
                'total_executing': executing,
                'total_pending': pending
            }
        }

    def get_executing_actions(self) -> List[Dict]:
        """Get currently executing actions"""
        executing = []
        for actuator_id, state in self.queue_modal.actuators.items():
            if state.current_action:
                action = state.current_action
                executing.append({
                    'actuator': actuator_id,
                    'action_type': action.__class__.__name__,
                    'target': getattr(action, 'target', getattr(action, 'position', None)),
                    'status': action.status,
                    'progress': action.progress
                })
        return executing

    def get_pending_blocks(self) -> List[Dict]:
        """Get pending action blocks"""
        pending = []
        for block_id, block in self.queue_modal.blocks.items():
            if block.status in ['queued', 'executing']:
                pending.append({
                    'block_id': block_id,
                    'name': block.block_name,
                    'status': block.status,
                    'progress': f"{block.actions_completed}/{block.actions_total}",
                    'actions_total': block.actions_total,
                    'actions_completed': block.actions_completed
                })
        return pending

    def get_block_info(self, block_id: int) -> Dict:
        """Get info about specific block - OFFENSIVE"""
        if block_id not in self.queue_modal.blocks:
            available = list(self.queue_modal.blocks.keys())
            raise KeyError(f"❌ Block {block_id} not found! Available: {available}")
        return self.queue_modal.blocks[block_id].dict()

    # === SUBMIT/CONTROL ===

    def submit(self, action_or_block) -> int:
        """Submit action or block - SMART (auto-wraps actions in blocks)

        Example:
            block_id = ops.actions.submit(move_forward(distance=2.0))
        """
        if isinstance(action_or_block, ActionBlock):
            return self.action_executor.submit_block(action_or_block)
        else:
            # Wrap single action in block
            block = ActionBlock(
                id=f"single_{action_or_block.id}",
                description=f"Single: {action_or_block.__class__.__name__}",
                actions=[action_or_block]
            )
            return self.action_executor.submit_block(block)

    def cancel_block(self, block_id: int):
        """Cancel pending block - OFFENSIVE"""
        if block_id not in self.queue_modal.blocks:
            raise KeyError(f"❌ Block {block_id} not found")

        block = self.queue_modal.blocks[block_id]
        block.status = 'cancelled'

        # Remove from actuator queues
        for actuator in self.queue_modal.actuators.values():
            actuator.queue = [a for a in actuator.queue
                            if getattr(a, '_block_id', None) != block_id]

    def clear_queue(self):
        """Clear all pending actions - WARNING: Cancels everything!"""
        for actuator in self.queue_modal.actuators.values():
            actuator.queue.clear()

        for block in self.queue_modal.blocks.values():
            if block.status in ['queued', 'executing']:
                block.status = 'cancelled'

    # === HELPERS ===

    def _extract_params(self, action_class) -> Dict:
        """Extract parameters from Pydantic model"""
        schema = action_class.model_json_schema()
        params = {}

        # Skip internal fields
        skip = {'id', 'status', 'actuator_ids', 'required_sensors',
                'required_actuators', 'progress'}

        for field_name, field_info in schema.get('properties', {}).items():
            if field_name in skip:
                continue

            params[field_name] = {
                'type': field_info.get('type', 'unknown'),
                'required': field_name in schema.get('required', []),
                'description': field_info.get('description', ''),
                'default': field_info.get('default', None)
            }

        return params

    def __str__(self):
        """MOP self-presentation"""
        if not self.robot:
            return "ActionOps(not_compiled)"

        status = self.get_status()
        return (
            f"ActionOps("
            f"executing={status['summary']['total_executing']}, "
            f"pending={status['summary']['total_pending']}, "
            f"actions={len(self.robot.actions)})"
        )

    def __repr__(self):
        return self.__str__()
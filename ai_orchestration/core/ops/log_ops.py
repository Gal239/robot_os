"""
LOG OPS - Operations on LogModal
Handles event logging and persistence
"""

from typing import Dict, Optional
from ..modals import LogModal, ToolType


class LogOps:
    """
    Operations layer for LogModal
    Handles logging and event management
    """

    def __init__(self, modal: LogModal):
        self.modal = modal

    @classmethod
    def create_new(cls) -> 'LogOps':
        """Create new LogOps with empty modal"""
        return cls(LogModal())

    # ========== RENDERING (MOP - requires graph_modal) ==========

    def render_for_display(self, graph_modal) -> str:
        """Get human-readable log (GENERATED from graph_modal)"""
        return self.modal.render_for_display(graph_modal)

    def render_for_json(self, graph_modal) -> Dict:
        """Get full JSON serialization (GENERATED from graph_modal)"""
        return self.modal.render_for_json(graph_modal)

    # ========== LOG FORMATTING (reads graph state) ==========

    def format_meta_log(self, graph_modal) -> str:
        """Format meta log from graph state (delegates to modal)"""
        return self.modal.format_meta_log(graph_modal)

    def format_task_log(self, graph_modal, task_id: str) -> str:
        """Format task log from graph state (delegates to modal)"""
        return self.modal.format_task_log(graph_modal, task_id)

    def build_log_context(self, graph_modal, task_id: str) -> str:
        """Build complete log context for LLM (delegates to modal)"""
        return self.modal.build_log_context(graph_modal, task_id)

    # ========== EVENT TRACKING (coordinates with graph_modal) ==========

    def add_event(self, event_type: str, data: Dict, task_id: str, graph_modal=None) -> Dict:
        """
        Log event and detect loops (MOP: delegates to modal for detection)

        Args:
            event_type: Type of event (e.g., "tool_execution")
            data: Event data with 'tool', 'input', 'result'
            task_id: Task ID to add event to
            graph_modal: Graph modal (optional, for loop detection)

        Returns:
            {"loop_detected": bool, "count": int, "tool_name": str}
        """
        tool_name = data.get("tool", "")
        tool_input = data.get("input", {})
        result = data.get("result", {})

        # Add to graph timeline if graph_modal provided (single source of truth)
        if graph_modal and task_id in graph_modal.nodes:
            graph_modal.add_to_timeline(task_id, event_type, tool_name, tool_input, result)

        # Delegate loop detection to modal (modal has the logic)
        if graph_modal:
            return self.modal.detect_loop_in_task(graph_modal, task_id, tool_name, tool_input)

        return {"loop_detected": False}

    def add_hint(self, hint_type: str, data: Dict, task_id: str, graph_modal=None):
        """
        Add hint to task (MOP: coordinates storage in graph_modal)

        Args:
            hint_type: Type of hint ("input_validation", "output_validation", "loop_detection")
            data: Hint data with 'message', 'tool', etc.
            task_id: Task ID to add hint to
            graph_modal: Graph modal (optional, for storing hint)
        """
        hint_msg = data.get("message", "")

        # Store in task node's hints list
        if graph_modal and task_id in graph_modal.nodes:
            node = graph_modal.nodes[task_id]
            node.hints.append(hint_msg)

    # ========== PERSISTENCE ==========

    def to_dict(self, graph_modal) -> Dict:
        """Serialize logs for saving to runs/session_id/logs.json (GENERATED from graph_modal)"""
        return self.modal.render_for_json(graph_modal)

    @classmethod
    def from_dict(cls, data: Dict) -> 'LogOps':
        """Load logs from dict"""
        modal = LogModal.from_json(data)
        return cls(modal)


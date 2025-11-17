"""
MODALS - Pure state containers
OFFENSIVE & ELEGANT
"""

from .log_modal import LogModal
from .tool_modal import ToolModal, ToolEntry, ToolType, TOOL_BEHAVIOR
from .document_modal import WorkspaceModal, Document
from .task_graph_modal import TaskGraphModal, TaskNode, TaskStatus

__all__ = [
    'LogModal',
    'ToolModal', 'ToolEntry', 'ToolType', 'TOOL_BEHAVIOR',
    'WorkspaceModal', 'Document',
    'TaskGraphModal', 'TaskNode', 'TaskStatus'
]

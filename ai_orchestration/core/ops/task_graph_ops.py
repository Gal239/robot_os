"""
TASK GRAPH OPS - Business logic + persistence
Operates on TaskGraphModal
Slim - only complex workflows + DB
"""

from typing import Dict, Optional, Any, List, Union
from datetime import datetime
from pathlib import Path

# Import centralized database
from ai_orchestration.utils.global_config import agent_engine_db

# Use task_graphs collection from agent_engine_db
task_graph_db = agent_engine_db.task_graphs

from ..modals.task_graph_modal import TaskGraphModal, TaskNode, TaskStatus
from ..modals.tool_modal import ToolType


class TaskGraphOps:
    """
    Business logic and persistence for TaskGraphModal
    Modal doesn't know DB - Ops handles save/load
    Ops handles complex cross-node workflows
    """

    def __init__(self, modal: TaskGraphModal, log_ops=None):
        self.modal = modal
        self.log_ops = log_ops  # For saving logs together with graph

    # ========== PERSISTENCE ==========

    def save(self, orchestrator=None):
        """Save modal to runs/session_id/ - Saves graph + logs + agents together"""
        # Convert TaskNodes to dicts for JSON serialization
        nodes_for_db = {}
        for task_id, node in self.modal.nodes.items():
            nodes_for_db[task_id] = node.to_dict()

        # Prepare graph data
        graph_data = {
            "session_id": self.modal.session_id,
            "nodes": nodes_for_db,
            "next_id": self.modal.next_id,
            "updated_at": datetime.now().isoformat()
        }

        # Save graph to runs/session_id/graph.json
        agent_engine_db.save_run_graph(self.modal.session_id, graph_data)

        # Save snapshot to runs/session_id/snapshots/
        snapshot_data = {
            **graph_data,
            "timestamp": datetime.now().isoformat(),
            "type": "snapshot"
        }
        agent_engine_db.save_run_snapshot(self.modal.session_id, snapshot_data)

        # ALSO save logs if log_ops is available (MOP - GENERATE from graph!)
        # Save 4 separate log files instead of 1 combined file
        if self.log_ops:
            logs_data = self.log_ops.to_dict(self.modal)  # Generate all 4 logs

            # Save each log type as separate file
            agent_engine_db.save_run_log_file(self.modal.session_id, "master_log.json", logs_data["master_log"])
            agent_engine_db.save_run_log_file(self.modal.session_id, "metalog_detailed.json", logs_data["metalog_detailed"])
            agent_engine_db.save_run_log_file(self.modal.session_id, "metalog_summary.json", logs_data["metalog_summary"])
            agent_engine_db.save_run_log_file(self.modal.session_id, "task_logs.json", logs_data["task_logs"])

        # Save agents if orchestrator is provided
        if orchestrator and hasattr(orchestrator, 'agents') and orchestrator.agents:
            agents_data = {}
            for agent_id, agent in orchestrator.agents.items():
                agents_data[agent_id] = {
                    "agent_id": agent_id,
                    "description": agent.get_description(),
                    "instructions": agent.get_instructions(),
                    "tools": agent.get_tools(),
                    "force_model": agent.config["force_model"],  # MOP: Required field, no fallback
                    "max_tokens": agent.config.get("max_tokens", 4000),
                    "metadata": agent.config.get("metadata", {})
                }
            agent_engine_db.save_run_agents(self.modal.session_id, agents_data)

    @classmethod
    def load(cls, session_id: str, log_ops=None) -> 'TaskGraphOps':
        """Load modal from runs/session_id/graph.json"""
        # Load graph from runs/
        data = agent_engine_db.load_run_graph(session_id)

        # Convert dicts back to TaskNodes
        nodes = {}
        for task_id, node_data in data["nodes"].items():
            nodes[task_id] = TaskNode.from_dict(node_data)

        # Create modal
        modal = TaskGraphModal(
            nodes=nodes,
            next_id=data["next_id"],
            session_id=session_id
        )

        return cls(modal, log_ops)

    # ========== COMPLEX WORKFLOWS (Cross-node operations) ==========

    def create_node(
        self,
        agent_id: str,
        parent_task_id: Optional[str],
        tool_type: Union[str, ToolType],
        task_payload: str,
        master_agent_id: Optional[str] = None,
        documents: List[str] = None
    ) -> str:
        """Delegate to modal - modal handles auto-blocking + document passing"""
        # Convert string to enum if needed
        tool_type_enum = ToolType(tool_type) if isinstance(tool_type, str) else tool_type
        return self.modal.create_node(agent_id, parent_task_id, tool_type_enum, task_payload, master_agent_id, documents)

    def update_status(self, task_id: str, status: TaskStatus):
        """Delegate to modal - modal handles auto-unblocking"""
        self.modal.update_node_status(task_id, status)

    def complete_task(self, task_id: str, result: Dict):
        """Delegate to modal - modal handles everything"""
        self.modal.mark_node_completed(task_id, result)

    # ========== QUERIES (Delegated to Modal) ==========

    def is_ready(self, task_id: str) -> bool:
        """Check if a task is ready to run (delegated to modal)"""
        return self.modal.is_ready(task_id)

    def get_ready_tasks(self) -> List[str]:
        """Get all tasks that are ready to run (delegated to modal)"""
        return self.modal.get_ready_tasks()

    def get_pending_human_task(self) -> Optional[tuple]:
        """Find pending human task (delegated to modal)"""
        return self.modal.get_pending_human_task()

    def get_active_root_task(self) -> Optional[str]:
        """Find active ROOT task (delegated to modal)"""
        return self.modal.get_active_root_task()

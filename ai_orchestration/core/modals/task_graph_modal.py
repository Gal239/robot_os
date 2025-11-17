"""
TASK GRAPH MODAL - Pure state + self-rendering
Task graph with messages and blocker tracking
Uses TaskNode - smart architecture
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Set

# Import ToolType from tool_modal
from .tool_modal import ToolType

class TaskStatus(str, Enum):
    """Task execution states"""
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"


class TaskNode(BaseModel):
    """
    Individual task - smart data that operates on itself
    Like AssetConfig - elegant and self-contained
    """
    task_id: str
    agent_id: str
    parent_task_id: Optional[str] = None
    master_agent_id: Optional[str] = None
    master_type: str = "agent"  # "human" or "agent"
    tool_type: ToolType
    task_payload: str
    status: TaskStatus = TaskStatus.READY

    # Blocker management
    blockers: Set[str] = Field(default_factory=set)

    # Document management (paths to workspace documents)
    documents: List[str] = Field(default_factory=list)

    # TIMELINE (ordered, grows over time) - SINGLE SOURCE OF TRUTH
    tool_timeline: List[Dict[str, Any]] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)

    # Messages - ALWAYS just [system, user]
    messages: List[Dict[str, Any]] = Field(default_factory=list)

    # Result
    result: Optional[Dict[str, Any]] = None

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None

    class Config:
        # Allow sets in Pydantic
        json_encoders = {
            set: list
        }

    # ========== SMART METHODS ==========

    def is_ready(self) -> bool:
        """Check if task is ready to run (no blockers, not completed)"""
        return len(self.blockers) == 0 and self.status != TaskStatus.COMPLETED

    def add_log_event(self, event_type, tool: str, input: Dict, result: Any):
        """Add log event to timeline - SINGLE SOURCE OF TRUTH"""
        # Handle both string and enum (offensive - accepts both)
        if isinstance(event_type, ToolType):
            event_type_str = event_type.value
        else:
            event_type_str = event_type

        self.tool_timeline.append({
            "type": event_type_str,
            "tool": tool,
            "input": input,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        self.updated_at = datetime.now().isoformat()

    def add_blocker(self, blocker_task_id: str):
        """Add a blocker - this task must wait"""
        self.blockers.add(blocker_task_id)
        self.status = TaskStatus.WAITING
        self.updated_at = datetime.now().isoformat()

    def remove_blocker(self, blocker_task_id: str):
        """Remove a blocker - may become ready"""
        if blocker_task_id in self.blockers:
            self.blockers.remove(blocker_task_id)

            # If no more blockers, become ready
            if len(self.blockers) == 0 and self.status != TaskStatus.COMPLETED:
                self.status = TaskStatus.READY

            self.updated_at = datetime.now().isoformat()

    def mark_completed(self, result: Dict):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization"""
        data = self.dict()
        # Convert sets to lists for JSON
        if "blockers" in data:
            data["blockers"] = list(data["blockers"])
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskNode":
        """Create from dict (after loading from DB)"""
        # Convert lists back to sets
        if "blockers" in data and isinstance(data["blockers"], list):
            data["blockers"] = set(data["blockers"])
        return cls(**data)


@dataclass
class TaskGraphModal:
    """
    Pure task graph state - knows how to render itself
    Does NOT know how to save/load (Ops handles DB)
    Uses TaskNode for smart individual tasks
    """
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    next_id: int = 0
    session_id: str = field(default_factory=lambda: f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # ========== SELF-OPERATIONS (Modal operates on itself) ==========

    def add_to_timeline(self, task_id: str, event_type: ToolType, tool: str, input: Dict, result: Any):
        """Add typed log event to timeline - SINGLE SOURCE OF TRUTH"""
        if task_id in self.nodes:
            self.nodes[task_id].add_log_event(event_type, tool, input, result)

    def get_ready_tasks(self) -> List[str]:
        """Get all tasks ready to run"""
        return [task_id for task_id, node in self.nodes.items() if node.is_ready()]

    def is_ready(self, task_id: str) -> bool:
        """Check if task is ready"""
        return task_id in self.nodes and self.nodes[task_id].is_ready()

    def get_pending_human_task(self) -> Optional[tuple]:
        """
        Find first pending task with agent_id='human'
        MOP: Modal knows its own state

        Human tasks are "ready" (waiting for user input, no blockers)

        Returns:
            (task_id, TaskNode) or None
        """
        for task_id, node in self.nodes.items():
            # Human tasks are "ready" - waiting for user input
            if node.agent_id == "human" and node.status == "ready":
                return (task_id, node)
        return None

    def get_active_root_task(self) -> Optional[str]:
        """
        Find active (non-completed) ROOT task
        MOP: Modal knows its structure

        Returns:
            task_id or None
        """
        for task_id, node in self.nodes.items():
            if node.tool_type == ToolType.ROOT and node.status != "completed":
                return task_id
        return None

    def create_node(
        self,
        agent_id: str,
        parent_task_id: Optional[str],
        tool_type: ToolType,
        task_payload: str,
        master_agent_id: Optional[str] = None,
        documents: List[str] = None
    ) -> str:
        """
        Create node - modal reacts with auto-blocking

        State machine reaction:
        - AGENT_AS_TOOL or ASK_MASTER → auto-blocks parent
        - documents → auto-injected to child timeline
        """
        task_id = f"task_{self.next_id}"
        self.next_id += 1

        node = TaskNode(
            task_id=task_id,
            agent_id=agent_id,
            parent_task_id=parent_task_id,
            master_agent_id=master_agent_id,
            master_type="human" if master_agent_id is None else "agent",
            tool_type=tool_type,
            task_payload=task_payload,
            status=TaskStatus.READY,
            documents=documents or []
        )

        self.nodes[task_id] = node

        # Modal reacts: auto-block parent
        if tool_type in [ToolType.AGENT_AS_TOOL, ToolType.ASK_MASTER]:
            if parent_task_id and parent_task_id in self.nodes:
                self.nodes[parent_task_id].add_blocker(task_id)

        return task_id

    def update_node_status(self, task_id: str, status: TaskStatus):
        """
        Update status - modal reacts with auto-unblocking

        State machine reaction:
        - status → COMPLETED → auto-unblock all waiters
        """
        if task_id not in self.nodes:
            return

        self.nodes[task_id].status = status

        # Modal reacts: COMPLETED → unblock all waiters
        if status == TaskStatus.COMPLETED:
            for node in self.nodes.values():
                if task_id in node.blockers:
                    node.remove_blocker(task_id)

    def mark_node_completed(self, task_id: str, result: Dict):
        """
        Mark completed - triggers status update with auto-unblock + auto-inject

        State machine: sets result + completed timestamp + triggers COMPLETED status
        Modal reacts: unblocks parent + injects result to parent
        """
        if task_id in self.nodes:
            node = self.nodes[task_id]
            node.result = result
            node.completed_at = datetime.now().isoformat()

            # Update status (triggers unblocking)
            self.update_node_status(task_id, TaskStatus.COMPLETED)

            # Modal reacts: inject result to parent
            if node.parent_task_id:
                self.inject_child_result(node.parent_task_id, task_id)

    def answer_human_task(self, task_id: str, answer: Dict):
        """
        Human provides answer to ask_master task
        MOP: Automatically creates handoff event (human doesn't call handoff!)

        Args:
            task_id: The ask_master task (agent_id="human")
            answer: Raw answer from human (e.g., {"answer": "John"})

        This method:
        1. Validates task is human ask_master
        2. Creates handoff event in task's timeline (like agent would do!)
        3. Marks task completed (triggers unblock + injection)
        """
        if task_id not in self.nodes:
            raise ValueError(f"Task {task_id} not found")

        node = self.nodes[task_id]

        # Validate is human ask_master task
        if node.agent_id != "human":
            raise ValueError(f"Task {task_id} is not a human task (agent_id={node.agent_id})")

        if node.tool_type != ToolType.ASK_MASTER:
            raise ValueError(f"Task {task_id} is not ask_master (tool_type={node.tool_type})")

        # MOP: Wrap answer in handoff result format
        handoff_result = {"result": answer}

        # MOP: Add handoff to timeline (just like agent would do!)
        # This creates the handoff event that will show in visualization
        self.add_to_timeline(
            task_id,
            ToolType.HANDOFF,
            "handoff",
            {"result": answer},
            handoff_result
        )

        # MOP: Mark completed - modal automatically:
        #   1. Sets node.result
        #   2. Sets node.status = COMPLETED
        #   3. Unblocks parent
        #   4. Injects result to parent's timeline
        self.mark_node_completed(task_id, handoff_result)

    # ========== RENDERING (Modal knows how to render itself) ==========

    def render_for_llm(
        self,
        task_id: str,
        agent_config: dict,
        orchestration_preamble: str = "",
        log_context: str = None,
        workspace = None
    ) -> list:
        """
        Render messages for LLM - SINGLE SOURCE OF TRUTH
        ALWAYS returns [system, user] - user content regenerates every call

        Modal knows how to render its own data for LLM consumption
        Includes documents from orchestration tools (delegation, ask_master)
        """
        node = self.nodes[task_id]
        messages = node.messages

        # System message - create ONCE
        if not messages:
            # System prompt = agent instructions + orchestration preamble
            system_content = ""
            if agent_config.get("instructions"):
                system_content = agent_config["instructions"]
            if orchestration_preamble:
                system_content += "\n\n" + orchestration_preamble

            if system_content:
                messages.append({"role": "system", "content": system_content})

        # User message - ALWAYS REPLACE with fresh content blocks
        content_blocks = [
            {"type": "text", "text": log_context or ""}  # Fresh logs
        ]

        # Extract content blocks from timeline (load_to_context results)
        # Order preserved from timeline!
        for event in node.tool_timeline:
            if event.get("tool") == "load_to_context" and event.get("type") == ToolType.FUNCTION_TOOL.value:
                # Result is the loaded content block
                content_blocks.append(event["result"])

        # Process documents from node.documents (attached via orchestration tools)
        if workspace and node.documents:
            for doc_path in node.documents:
                doc = workspace.get_document(doc_path)
                if doc:
                    # Convert Document to content block
                    import base64
                    content = doc.render_from_json()
                    content_base64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

                    content_blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": doc.mime_type,
                            "data": content_base64
                        }
                    })

        # Process documents from orchestration tool results (child_documents)
        if workspace:
            for event in node.tool_timeline:
                if event.get("type") in [ToolType.AGENT_AS_TOOL.value, ToolType.ASK_MASTER.value]:
                    result = event.get("result", {})
                    if isinstance(result, dict):
                        child_docs = result.get("child_documents", [])
                        for doc_path in child_docs:
                            doc = workspace.get_document(doc_path)
                            if doc:
                                # Convert Document to content block
                                import base64
                                content = doc.render_from_json()
                                content_base64 = base64.b64encode(content.encode('utf-8')).decode('utf-8')

                                content_blocks.append({
                                    "type": "document",
                                    "source": {
                                        "type": "base64",
                                        "media_type": doc.mime_type,
                                        "data": content_base64
                                    }
                                })

        # Create or replace user message
        if len(messages) == 1:  # Only system exists
            messages.append({"role": "user", "content": content_blocks})
        else:  # Replace existing user message
            messages[1] = {"role": "user", "content": content_blocks}

        return messages

    def get_graph_for_viz(self) -> Dict[str, Any]:
        """
        Get graph data formatted for visualization

        Modal knows how to format itself for visualization
        """
        nodes = []
        edges = []

        for task_id, node in self.nodes.items():
            # Node data for visualization
            node_viz = {
                "id": task_id,
                "label": f"{node.agent_id}\n{node.tool_type.value}",
                "status": node.status.value,
                "color": {
                    TaskStatus.READY: "#90EE90",      # Light green
                    TaskStatus.RUNNING: "#FFD700",    # Gold
                    TaskStatus.WAITING: "#FFA500",    # Orange
                    TaskStatus.COMPLETED: "#87CEEB"   # Sky blue
                }.get(node.status, "#FFFFFF")
            }
            nodes.append(node_viz)

            # Add edges for parent relationships
            if node.parent_task_id:
                edges.append({
                    "from": node.parent_task_id,
                    "to": task_id,
                    "label": node.tool_type.value
                })

            # Add edges for blockers
            for blocker in node.blockers:
                edges.append({
                    "from": task_id,
                    "to": blocker,
                    "label": "waiting",
                    "dashes": True,
                    "color": "#FF0000"
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "session_id": self.session_id
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of task states

        Simple query on self
        """
        summary = {
            "total": len(self.nodes),
            "ready": [],
            "waiting": [],
            "completed": []
        }

        for task_id, node in self.nodes.items():
            if node.status == TaskStatus.COMPLETED:
                summary["completed"].append(task_id)
            elif node.is_ready():
                summary["ready"].append(task_id)
            else:
                blockers_list = list(node.blockers)
                summary["waiting"].append(f"{task_id} waiting for {blockers_list}")

        return summary

    def handle_tool_call(
        self,
        task_id: str,
        tool_name: str,
        tool_type: ToolType,
        tool_input: Dict,
        tool_schema: Dict,
        tool_ops: 'ToolOps',
        graph_ops: 'TaskGraphOps',
        agent_model: str,
        log_ops: Optional['LogOps'] = None
    ) -> Dict:
        """
        Modal handles tool execution based on TOOL_BEHAVIOR
        Coordinates all 3 hint types (input validation, output validation, loop detection)
        MOP: agent_model is REQUIRED parameter from agent.force_model (single source of truth)
        MOP: Validates against exact schema sent to LLM (pure MOP - single source of truth)
        Returns: {"action": "continue"|"return", "value": Any}
        """
        from .tool_modal import TOOL_BEHAVIOR

        behavior = TOOL_BEHAVIOR[tool_type]

        # ========== UNIVERSAL INPUT VALIDATION (applies to ALL tools) ==========
        # MOP: Use exact schema LLM saw - NO FALLBACK, crash if missing
        if not tool_schema:
            raise ValueError(f"VALIDATION ERROR: tool_schema not provided for {tool_name}. This is a system bug - orchestrator must always pass tool_schema.")

        required = tool_schema.get("input_schema", {}).get("required", [])
        missing = [f for f in required if f not in tool_input]

        if missing:
            input_error = {
                "error": f"Missing required field(s): {', '.join(missing)}",
                "provided_fields": list(tool_input.keys()),
                "required_fields": required,
                "hint": f"Please call {tool_name} again with all required fields"
            }
        else:
            input_error = None

        if input_error:
            # Generate hint using LogModal
            hint_msg = log_ops.modal.generate_input_validation_hint(tool_name, input_error) if log_ops else input_error.get("hint", "Input validation failed")

            # Add hint to the tool's result data (NOT a separate event!)
            result_with_hint = {**input_error, "hint": hint_msg, "tool": tool_name}
            self.add_to_timeline(task_id, tool_type, "input_validation_hint", tool_input, result_with_hint)

            # Log the hint
            if log_ops:
                log_ops.add_hint("input_validation", {"tool": tool_name, "error": input_error, "message": hint_msg}, task_id, self)

            return {"action": "continue"}  # Don't execute, let LLM retry

        # ========== CIRCULAR DELEGATION PREVENTION (ASK_MASTER specific) ==========
        node = self.nodes[task_id]
        if node.tool_type == ToolType.ASK_MASTER and tool_type == ToolType.AGENT_AS_TOOL:
            # Agent answering ask_master is trying to delegate via route_to_X
            # Check if routing back to the agent who asked the question
            target_agent = tool_name.replace("route_to_", "")

            # Find who asked (parent task's agent)
            parent_task = self.nodes.get(node.parent_task_id) if node.parent_task_id else None

            if parent_task and parent_task.agent_id == target_agent:
                # CIRCULAR DELEGATION DETECTED - routing back to asker!
                # Count previous attempts (soft block = warn once)
                circular_count = sum(
                    1 for event in node.tool_timeline
                    if event.get("type") == "circular_delegation_attempt"
                )

                if circular_count == 0:
                    # First time: BLOCK + HINT
                    hint_msg = log_ops.modal.generate_circular_delegation_hint(
                        asker_agent=target_agent,
                        master_agent=node.agent_id
                    ) if log_ops else f"Cannot route back to {target_agent} who asked you this question. Use handoff to answer."

                    # Add to timeline to track attempts
                    hint_result = {
                        "blocked": True,
                        "reason": "circular_delegation",
                        "asker": target_agent,
                        "hint": hint_msg
                    }
                    self.add_to_timeline(task_id, "circular_delegation_attempt", tool_name, tool_input, hint_result)

                    # Log the hint
                    if log_ops:
                        log_ops.add_hint("circular_delegation", {
                            "asker": target_agent,
                            "master": node.agent_id,
                            "message": hint_msg
                        }, task_id, self)

                    return {"action": "continue"}  # BLOCK - let LLM retry with hint
                else:
                    # Second time: Allow but warn
                    print(f"[WARNING] Agent {node.agent_id} routing back to asker {target_agent} despite hint")
                    # Let it through - agent insists

        # ========== TOOL TYPE SPECIFIC EXECUTION ==========
        result = None
        action = "continue"
        return_value = None

        if behavior.get("executes_function"):
            # Execute tool (validation passed) - OFFENSIVE: Let errors crash!
            # MOP: Pass agent_model to tool execution
            result = tool_ops.execute(tool_name, tool_input, task_id, agent_model=agent_model)

            # Add execution to timeline
            self.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, tool_name, tool_input, result)

        elif behavior.get("logs_input"):  # NON_FUNCTION_TOOL (think, plan, save_memory)
            # Just log the input with special formatting - no execution, instant continue
            result = {"logged": True}
            self.add_to_timeline(task_id, ToolType.NON_FUNCTION_TOOL, tool_name, tool_input, result)

        elif behavior.get("completes_task"):  # handoff
            # Extract documents from input
            documents = tool_input.get("documents", [])

            # Store documents in task result
            result = {
                **tool_input,
                "documents": documents
            }

            # Add documents to task's document list
            if task_id in self.nodes and documents:
                self.nodes[task_id].documents.extend(documents)

            # Add to timeline for MOP logging (so metalog can be generated!)
            self.add_to_timeline(task_id, ToolType.HANDOFF, tool_name, tool_input, result)

            # Modal reacts: mark task completed immediately
            self.mark_node_completed(task_id, result)
            action = "return"
            return_value = result

        elif tool_type == ToolType.AGENT_AS_TOOL:
            target_id = tool_name[9:]  # route_to_X
            from_node = self.nodes[task_id]

            # Extract documents to pass to child
            documents = tool_input.get("documents", [])

            child_id = graph_ops.create_node(
                agent_id=target_id,
                parent_task_id=task_id,
                tool_type=ToolType.AGENT_AS_TOOL,
                task_payload=tool_input.get("request", ""),
                master_agent_id=from_node.agent_id,
                documents=documents
            )

            log_msg = f"→ delegated to {target_id} (task {child_id})"
            if documents:
                log_msg += f" with {len(documents)} document(s)"

            result = log_msg
            self.add_to_timeline(task_id, ToolType.AGENT_AS_TOOL, tool_name, tool_input, result)
            action = "return"
            return_value = None

        elif tool_type == ToolType.ASK_MASTER:
            node = self.nodes[task_id]
            documents = tool_input.get("documents", [])

            if node.master_agent_id is None:
                # Root agent asking user - treat user as pseudo-agent "human"
                # Use SAME non-blocking mechanism as agent-to-agent
                question_id = graph_ops.create_node(
                    agent_id="human",  # Pseudo-agent for user
                    parent_task_id=task_id,
                    tool_type=ToolType.ASK_MASTER,
                    task_payload=tool_input.get("question"),
                    master_agent_id=node.agent_id,
                    documents=documents
                )

                log_msg = f"→ asked human (task {question_id})"
                if documents:
                    log_msg += f" with {len(documents)} document(s)"

                result = log_msg
                self.add_to_timeline(task_id, ToolType.ASK_MASTER, tool_name, tool_input, result)
                action = "return"
                return_value = None  # Pause and wait for user response
            else:
                # Ask parent agent
                master_id = node.master_agent_id
                question_id = graph_ops.create_node(
                    agent_id=master_id,
                    parent_task_id=task_id,
                    tool_type=ToolType.ASK_MASTER,
                    task_payload=tool_input.get("question"),
                    master_agent_id=node.agent_id,
                    documents=documents
                )

                log_msg = f"→ asked {master_id} (task {question_id})"
                if documents:
                    log_msg += f" with {len(documents)} document(s)"

                result = log_msg
                self.add_to_timeline(task_id, ToolType.ASK_MASTER, tool_name, tool_input, result)
                action = "return"
                return_value = None

        # ========== UNIVERSAL OUTPUT VALIDATION (applies to ALL tools with output schema) ==========
        if result and isinstance(result, dict):
            output_error = tool_ops.modal.validate_output(tool_name, result)
            if output_error:
                # Generate hint using LogModal and ADD TO RESULT
                hint_msg = log_ops.modal.generate_output_validation_hint(tool_name, output_error) if log_ops else "Output validation failed"
                result["validation_hint"] = hint_msg  # Add hint to existing result

                # Log the hint
                if log_ops:
                    log_ops.add_hint("output_validation", {"tool": tool_name, "error": output_error, "message": hint_msg}, task_id, self)

        # ========== UNIVERSAL LOOP DETECTION (applies to ALL tools) ==========
        if log_ops:
            loop_info = log_ops.modal.detect_loop_in_task(self, task_id, tool_name, tool_input)

            if loop_info.get("loop_detected"):
                # Generate loop hint and add to result
                hint_msg = log_ops.modal.generate_loop_detection_hint(
                    loop_info["tool_name"],
                    loop_info["count"]
                )

                # Add loop hint to result if it's a dict
                if result and isinstance(result, dict):
                    result["loop_hint"] = hint_msg

                # Log the hint
                log_ops.add_hint("loop_detection", {"tool": tool_name, "count": loop_info["count"], "message": hint_msg}, task_id, self)

        return {"action": action, "value": return_value}

    def inject_child_result(self, parent_task_id: str, child_task_id: str):
        """Modal handles child result injection - injects with ORIGINAL event type + documents"""
        parent = self.nodes.get(parent_task_id)
        child = self.nodes.get(child_task_id)
        if not parent or not child:
            return

        # Only inject for delegation tools
        if child.tool_type not in [ToolType.AGENT_AS_TOOL, ToolType.ASK_MASTER]:
            return

        # Inject with the ORIGINAL tool type (agent_as_tool or ask_master)
        description = f"← result from {child.agent_id}" if child.tool_type == ToolType.AGENT_AS_TOOL else f"← answer from {child.agent_id}"

        # Include child documents in result
        result_with_docs = {
            **(child.result or {}),
            "child_documents": child.documents  # Documents child created/used
        }

        self.add_to_timeline(parent_task_id, child.tool_type, description, {"completed_task": child_task_id}, result_with_docs)

"""
LOG MODAL - MOP-compliant log generation
4-level logging: ALL generated from node.tool_timeline (single source of truth)
- master_log: ALL tool calls from ALL nodes
- metalog_detailed: orchestration tools (route_to_*, handoff, ask_master) with full data
- metalog_summary: same as detailed but truncated to 150 chars
- task_log: single task's tool_timeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import uuid
from datetime import datetime


@dataclass
class LogModal:
    """
    MOP-compliant log generator
    NO stored logs - everything generated from graph_modal.nodes[].tool_timeline
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ========== LOG GENERATION (MOP - single source of truth: tool_timeline) ==========

    def generate_master_log(self, graph_modal: 'TaskGraphModal') -> List[Dict]:
        """
        Generate complete log from ALL nodes' tool_timeline
        Returns ALL tool calls across entire session, sorted by timestamp
        """
        master = []
        for node in graph_modal.nodes.values():
            for event in node.tool_timeline:
                master.append({
                    "task_id": node.task_id,
                    "agent_id": node.agent_id,
                    "timestamp": event.get("timestamp", ""),
                    "type": event.get("type", ""),
                    "tool": event.get("tool", ""),
                    "input": event.get("input", {}),
                    "result": event.get("result", {})
                })
        return sorted(master, key=lambda x: x["timestamp"])

    def generate_metalog_detailed(self, graph_modal: 'TaskGraphModal') -> List[Dict]:
        """
        Generate orchestration-only log with FULL data
        Filters for: route_to_*, handoff, ask_master
        """
        master = self.generate_master_log(graph_modal)
        orchestration_tools = ["handoff", "ask_master"]
        return [
            e for e in master
            if e["tool"].startswith("route_to_") or e["tool"] in orchestration_tools
        ]

    def generate_metalog_summary(self, graph_modal) -> List[Dict]:
        """
        Generate orchestration log with TRUNCATED content (150 chars)
        Used for LLM context to show previous conversations
        """
        detailed = self.generate_metalog_detailed(graph_modal)
        summary = []
        for event in detailed:
            truncated = event.copy()
            # Truncate input
            if isinstance(truncated["input"], str) and len(truncated["input"]) > 150:
                truncated["input"] = truncated["input"][:150] + "..."
            elif isinstance(truncated["input"], dict):
                # Truncate dict values
                truncated["input"] = {
                    k: (v[:150] + "..." if isinstance(v, str) and len(v) > 150 else v)
                    for k, v in truncated["input"].items()
                }
            # Truncate result
            if isinstance(truncated["result"], str) and len(truncated["result"]) > 150:
                truncated["result"] = truncated["result"][:150] + "..."
            elif isinstance(truncated["result"], dict):
                result_str = str(truncated["result"])
                if len(result_str) > 150:
                    truncated["result"] = result_str[:150] + "..."
            summary.append(truncated)
        return summary

    def generate_task_log(self, graph_modal, task_id: str) -> List[Dict]:
        """
        Generate log for single task
        Returns node's tool_timeline directly
        """
        if task_id not in graph_modal.nodes:
            return []
        return graph_modal.nodes[task_id].tool_timeline

    def detect_loop_in_task(self, graph_modal, task_id: str, tool_name: str, tool_input: Any) -> Dict:
        """
        Detect TWO types of loop patterns by analyzing tool_timeline
        Reads from graph_modal (MOP - no stored logs!)

        Returns:
            {"loop_detected": bool, "type": str, "count": int, "tool_name": str}

        Types:
            - "hard_loop": Same tool with IDENTICAL results (2+ times) - AGGRESSIVE hint
            - "consecutive": Same tool called 3+ times in a row - SOFT hint
        """
        if task_id not in graph_modal.nodes:
            return {"loop_detected": False}

        node = graph_modal.nodes[task_id]

        # PRIORITY 1: Check HARD LOOP (same tool with identical results)
        tool_results = []
        for event in node.tool_timeline:
            if event.get("tool") == tool_name:
                result = event.get("result")
                result_str = str(result)
                tool_results.append(result_str)

        # Hard loop: at least 2 identical consecutive results
        if len(tool_results) >= 2:
            if tool_results[-1] == tool_results[-2]:
                count = tool_results.count(tool_results[-1])
                if count >= 2:
                    return {
                        "loop_detected": True,
                        "type": "hard_loop",
                        "count": count,
                        "tool_name": tool_name
                    }

        # PRIORITY 2: Check CONSECUTIVE calls (regardless of result)
        consecutive_count = 0
        for event in reversed(node.tool_timeline):
            if event.get("tool") == tool_name:
                consecutive_count += 1
            else:
                break  # Stop counting when we hit a different tool

        # Soft reminder after 3 consecutive calls
        if consecutive_count >= 3:
            return {
                "loop_detected": True,
                "type": "consecutive",
                "count": consecutive_count,
                "tool_name": tool_name
            }

        return {"loop_detected": False}

    def render_for_display(self, graph_modal) -> str:
        """
        Human-readable log (GENERATED from graph_modal)
        """
        master = self.generate_master_log(graph_modal)
        metalog = self.generate_metalog_detailed(graph_modal)

        lines = [f"# Session {self.session_id}\n"]
        lines.append(f"Total events: {len(master)}")
        lines.append(f"Orchestration events: {len(metalog)}")
        lines.append(f"Tasks: {len(graph_modal.nodes)}\n")

        for event in metalog[-20:]:  # Last 20 meta events
            lines.append(f"[{event['timestamp']}] [{event['task_id']}] {event['tool']}")

        return "\n".join(lines)

    def render_for_json(self, graph_modal) -> Dict:
        """
        Full serialization (GENERATED from graph_modal)
        Returns all 4 log levels for saving to disk
        """
        return {
            "session_id": self.session_id,
            "master_log": self.generate_master_log(graph_modal),
            "metalog_detailed": self.generate_metalog_detailed(graph_modal),
            "metalog_summary": self.generate_metalog_summary(graph_modal),
            "task_logs": {
                task_id: self.generate_task_log(graph_modal, task_id)
                for task_id in graph_modal.nodes.keys()
            }
        }

    # ========== LOG FORMATTING (reads graph state) ==========

    def format_meta_log(self, graph_modal) -> str:
        """
        Format high-level overview + orchestration events
        INCLUDES previous conversations for multi-turn context!
        """
        from .task_graph_modal import TaskStatus

        total = len(graph_modal.nodes)
        completed = sum(1 for n in graph_modal.nodes.values() if n.status == TaskStatus.COMPLETED)
        running = sum(1 for n in graph_modal.nodes.values() if n.status == TaskStatus.READY)
        waiting = sum(1 for n in graph_modal.nodes.values() if n.status == TaskStatus.WAITING)

        log = f"""=== META LOG ===
Task Graph: {total} tasks ({completed} completed, {running} running, {waiting} waiting)
Session: {graph_modal.session_id}
"""

        # CRITICAL: Show ALL orchestration events from entire session with FULL details!
        metalog = self.generate_metalog_detailed(graph_modal)

        # PRINT: Metalog generation header
        print(f"\n{'='*60}")
        print(f"METALOG GENERATED - {len(metalog)} orchestration events")
        print(f"Session: {graph_modal.session_id}")
        print(f"{'='*60}")

        # PRINT: Detailed version (full data)
        if metalog:
            for event in metalog:
                tool = event.get("tool", "")
                task_id = event.get("task_id", "")
                result = event.get("result", "")
                input_data = event.get("input", "")
                print(f"  [Event] Task: {task_id} | Tool: {tool}")
                print(f"          Input: {input_data}")
                print(f"          Result: {result}")

        if metalog:
            log += "\nOrchestration Events (ALL - DETAILED):\n"

            # Group events by task to show question + answer together (MOP: ENABLES DREAMING)
            tasks_shown = set()
            for event in metalog:  # ALL events with full data!
                tool = event.get("tool", "")
                task_id = event.get("task_id", "")
                result = event.get("result", "")

                # Show task question first (once per task, ROOT tasks only)
                if task_id not in tasks_shown:
                    # OFFENSIVE: Crash if task_id not found (should never happen)
                    if task_id not in graph_modal.nodes:
                        raise KeyError(f"MOP violation! Event references task_id '{task_id}' not in graph")

                    node = graph_modal.nodes[task_id]
                    # ROOT tasks have parent_task_id="human" OR None (for backwards compatibility)
                    if node.parent_task_id in ["human", None]:  # ROOT tasks from human or old-style roots
                        log += f"  [{task_id}] Task: \"{node.task_payload}\"\n"
                        tasks_shown.add(task_id)

                # Show CLEAN conversation format (no duplicates, no verbose dicts)
                input_data = event.get("input", "")

                # Format based on tool type
                if isinstance(input_data, dict):
                    # For ask_master: show question + answer
                    if tool == "ask_master" and "question" in input_data:
                        answer = result.get("answer", "No answer") if isinstance(result, dict) else str(result)
                        log += f"  [{task_id}] Agent Question: {input_data['question']}\n"
                        log += f"  [{task_id}] Your Answer: {answer}\n"
                    # For handoff: ONLY show the message (no duplicate result dict!)
                    elif tool == "handoff" and "message" in input_data:
                        log += f"  [{task_id}] Echo Response: {input_data['message']}\n"
                    # For route_to_*: show delegation
                    elif tool.startswith("route_to_"):
                        delegated_to = tool.replace("route_to_", "")
                        task_desc = input_data.get("task", "")
                        log += f"  [{task_id}] Delegated to {delegated_to}: {task_desc}\n"
                    else:
                        # Other tools: show compact format
                        log += f"  [{task_id}] {tool}: {input_data}\n"
                else:
                    # Simple input (string or other)
                    log += f"  [{task_id}] {tool}: {input_data}\n"

        return log

    def format_task_log(self, graph_modal, task_id: str) -> str:
        """Format current task details from graph state"""
        node = graph_modal.nodes[task_id]
        parent_node = graph_modal.nodes.get(node.parent_task_id) if node.parent_task_id else None

        log = f"""=== TASK LOG ===
Current Task ID: {task_id}
Tool Type: {node.tool_type.value}
"""

        # SPECIAL highlighting for ASK_MASTER tasks
        from .tool_modal import ToolType
        if node.tool_type == ToolType.ASK_MASTER:
            asker = parent_node.agent_id if parent_node else "unknown"
            log += f"""
⚠️  SPECIAL TASK TYPE: ask_master
   → Agent '{asker}' asked YOU a question
   → They are BLOCKED waiting for your answer
   → Required action: handoff(answer={{...}})
   → You can delegate to others, but NOT back to '{asker}'

"""

        log += f"Task: {node.task_payload}\n"

        if parent_node:
            log += f"Parent Task: {parent_node.task_payload}\n"

        if node.tool_timeline:
            log += "\nTool Calls:\n"
            for i, tc in enumerate(node.tool_timeline, 1):
                tool_name = tc.get("tool", "unknown")
                tool_input = tc.get("input", {})
                # Show full result without truncation (removed [:150] limit)
                result = str(tc.get("result", ""))
                input_str = ", ".join([f"{k}={v}" for k, v in tool_input.items()]) if isinstance(tool_input, dict) else str(tool_input)
                log += f"  {i}. {tool_name}({input_str}) → {result}\n"

        log += f"\nBlockers: {list(node.blockers) if node.blockers else 'None'}\n"

        if node.hints:
            log += "\nHints:\n"
            for hint in node.hints:
                log += f"  - {hint}\n"

        return log

    def build_log_context(self, graph_modal, task_id: str) -> str:
        """Build complete log context for LLM"""
        return f"{self.format_meta_log(graph_modal)}\n{self.format_task_log(graph_modal, task_id)}"

    # ========== HINT GENERATION (Smart message generation) ==========

    def generate_input_validation_hint(self, tool_name: str, error_data: Dict) -> str:
        """
        Generate helpful hint message for input validation failure

        Args:
            tool_name: Name of the tool that failed validation
            error_data: Error dict with 'error', 'provided_fields', 'required_fields'

        Returns:
            Formatted hint message for LLM
        """
        missing = []
        required = error_data.get('required_fields', [])
        provided = error_data.get('provided_fields', [])

        for field in required:
            if field not in provided:
                missing.append(field)

        hint = f"[X] Tool '{tool_name}' input validation failed.\n"
        hint += f"Missing required fields: {', '.join(missing)}\n"
        hint += f"Required fields: {', '.join(required)}\n"
        hint += f"You provided: {', '.join(provided)}\n\n"
        hint += f"Action: Call '{tool_name}' again with ALL required fields."

        return hint

    def generate_output_validation_hint(self, tool_name: str, output_error: Dict) -> str:
        """
        Generate helpful hint message for output validation failure

        Args:
            tool_name: Name of the tool that produced invalid output
            output_error: Error dict with 'error', 'missing_fields', 'expected_fields'

        Returns:
            Formatted hint message for LLM
        """
        missing = output_error.get('missing_fields', [])
        expected = output_error.get('expected_fields', [])

        hint = f"[!] Tool '{tool_name}' output incomplete.\n"
        hint += f"Missing fields in output: {', '.join(missing)}\n"
        hint += f"Expected fields: {', '.join(expected)}\n\n"
        hint += f"This tool may have failed. Check the output and consider:\n"
        hint += f"1. Using a different tool\n"
        hint += f"2. Providing different input parameters\n"
        hint += f"3. Investigating why the tool didn't return complete data"

        return hint

    def generate_loop_detection_hint(self, tool_name: str, call_count: int) -> str:
        """
        Generate AGGRESSIVE hint message for HARD LOOP (same tool, same result)

        Args:
            tool_name: Name of the tool being called repeatedly
            call_count: Number of times called with same output

        Returns:
            Formatted hint message for LLM
        """
        hint = f"[LOOP] LOOP DETECTED: You called '{tool_name}' {call_count} times with the same exact output.\n\n"
        hint += f"You are stuck in a loop! This is not making progress.\n\n"
        hint += f"Action required:\n"
        hint += f"1. THINK: What is your actual goal?\n"
        hint += f"2. Use a DIFFERENT tool or DIFFERENT input parameters\n"
        hint += f"3. Remember: Your goal is to reach the 'handoff' tool when you finish the assignment\n\n"
        hint += f"Available options:\n"
        hint += f"- Try a different approach\n"
        hint += f"- Ask your master for clarification (ask_master tool)\n"
        hint += f"- If you're done, use 'handoff' to complete your task"

        return hint

    def generate_consecutive_tool_hint(self, tool_name: str, call_count: int) -> str:
        """
        Generate SOFT hint message for consecutive tool use (different results OK)

        Args:
            tool_name: Name of the tool being called consecutively
            call_count: Number of consecutive times called

        Returns:
            Formatted hint message for LLM
        """
        hint = f"[💡 HINT] You've used '{tool_name}' {call_count} times in a row.\n\n"
        hint += f"While thinking is valuable, are you sure you want to call it again?\n\n"
        hint += f"Remember your goal:\n"
        hint += f"  - Complete your assigned task\n"
        hint += f"  - Take action to make progress\n"
        hint += f"  - Use 'handoff' when you're done\n\n"
        hint += f"Consider:\n"
        hint += f"  • Is it time to take action instead of more thinking?\n"
        hint += f"  • Should you delegate to a specialist (route_to_*)?\n"
        hint += f"  • Do you have questions for your manager (ask_master)?\n"
        hint += f"  • Are you ready to complete your task (handoff)?"

        return hint

    def generate_tool_only_violation_hint(
        self, agent_id: str, available_tools: list, context_info: Dict = None
    ) -> str:
        """
        Generate helpful hint message when agent violates tool-only mode (returns no tool calls)

        Args:
            agent_id: ID of the agent that violated
            available_tools: List of tool names available to agent
            context_info: Reserved for future use (currently unused)

        Returns:
            Formatted hint message for LLM
        """
        hint = f"[X] TOOL-ONLY MODE VIOLATION: Agent '{agent_id}' returned NO tool calls.\n"
        hint += "⚠️  You MUST call a tool - you cannot respond with text.\n\n"
        hint += f"Available tools: {', '.join(available_tools)}\n\n"
        hint += "Action: Choose ONE tool to call:\n"
        for tool in available_tools:
            hint += f"  - {tool}\n"
        hint += "\nYou must call one of these tools. You cannot respond without calling a tool."

        return hint

    def generate_circular_delegation_hint(self, asker_agent: str, master_agent: str) -> str:
        """
        Generate hint when master agent tries to delegate back to the agent who asked them.

        Args:
            asker_agent: ID of agent who asked the question (waiting for answer)
            master_agent: ID of master agent who is trying to delegate back

        Returns:
            Formatted hint message explaining the circular delegation problem
        """
        hint = f"[❌ CIRCULAR DELEGATION] You are trying to delegate back to '{asker_agent}' - the same agent who asked YOU this question!\n\n"
        hint += f"This creates a circular loop. '{asker_agent}' is WAITING for your answer.\n\n"
        hint += "Your options:\n"
        hint += f"  1. ANSWER the question → Use handoff tool with your answer\n"
        hint += f"  2. Need info from a DIFFERENT agent → Use route_to_X for another agent\n"
        hint += f"  3. Need self-review → Use route_to_{master_agent} (yourself) for different perspective\n"
        hint += f"  4. Need clarification from asker → NOT POSSIBLE - they're waiting for you!\n\n"
        hint += f"❌ You CANNOT route back to '{asker_agent}'. They asked YOU and are blocked until you handoff.\n\n"
        hint += "Current task type: ask_master\n"
        hint += "Required action to complete: handoff(answer={...})"

        return hint

    @classmethod
    def from_json(cls, data: Dict) -> 'LogModal':
        """
        Load from saved logs (MOP - logs are derived, so only need session_id)
        """
        return cls(session_id=data.get("session_id", str(uuid.uuid4())))

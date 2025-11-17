"""
Orchestrator - Minimal coordinator, modal does the work
Uses centralized database for all configurations
"""
from typing import Union, Optional, Dict
from ai_orchestration.third_pary_llms.ask_llm import ask_llm

from .modals import WorkspaceModal, TaskGraphModal, TaskStatus, ToolType
from .ops import TaskGraphOps, ToolOps, LogOps, Agent, AgentBuilder, DocumentOps

# Import tools to trigger decorator registration
from .tools import llm_tools, file_tools

# Orchestration system preamble - STATIC part (always included)
PREAMBLE_STATIC = """
=== ORCHESTRATION SYSTEM ===

Multi-agent system. Tool calls ONLY - NO direct text responses.

Core tools:
- handoff(result): Complete task, return final answer (marks task as DONE)
- ask_master(question): Ask parent for clarification (waits for user response, conversation continues)

TWO WAYS TO RESPOND:
1. ask_master - Use when you need user input/clarification (creates human task, you'll get their response)
2. handoff - Use when task is COMPLETE and you have final answer (task marked done, returns to parent)

YOU MUST USE TOOLS FOR ALL INTERACTIONS.
All actions require tool calls - asking questions (ask_master), delivering answers (handoff), or any other interactions.
Choose the right tool: ask_master for questions, handoff for final answers.

"""

# DYNAMIC part - only if agent has route_to tools
PREAMBLE_DELEGATION = """
Delegation:
- You have route_to_X tools
- Delegate subtasks to agent X
- Break down complex work
"""

# DYNAMIC part - only if agent can route to itself
PREAMBLE_RECURSIVE = """
Self-Delegation:
- You can route to yourself (route_to_{agent_id})
- Use for: review, different perspectives, parallel work
"""

# DYNAMIC part - only for ASK_MASTER tasks
PREAMBLE_ASK_MASTER_RESPONSE = """
=== ANSWERING ASK_MASTER QUESTION ===

A child agent ({asker_agent}) asked YOU a question and is WAITING for your answer.

To answer: Use handoff(answer={{...}})
You CAN still delegate to OTHER agents or yourself for help
You CANNOT delegate back to {asker_agent} - they're blocked waiting for you

When ready to answer: handoff is required.
"""



class Orchestrator:
    """Minimal orchestrator - modal does the work"""

    def __init__(self, run_ready_tasks_in_parallel: bool = False, default_model: str = "claude-sonnet-4-5"):
        # MOP: Store orchestrator-level default model
        self.default_model = default_model

        # Create ops layers
        self.log_ops = LogOps.create_new()  # Create log_ops first
        self.graph_ops = TaskGraphOps(TaskGraphModal(), log_ops=self.log_ops)  # Pass log_ops
        self.tool_ops = ToolOps.create_new()

        # Agent management
        self.agents = {}
        self.agent = AgentBuilder(self, default_model=self.default_model)  # ops.agent.create() API

        # Load tools from DB
        self.tool_ops.load_from_db()

        # Workspace
        self.workspace = WorkspaceModal(session_id=self.graph_ops.modal.session_id)
        self.tool_ops.set_workspace(self.workspace)
        self.document_ops = DocumentOps(self.workspace)

        # Feature flags
        self.run_ready_tasks_in_parallel = run_ready_tasks_in_parallel

        # Custom handoff handlers for specific agents
        self.handoff_handlers = {}

    @classmethod
    def resume(cls, session_id: str, run_ready_tasks_in_parallel: bool = False):
        """
        Resume an existing session from runs/session_id/

        Loads:
        - Task graph from runs/session_id/graph.json
        - Logs from runs/session_id/logs.json
        - Workspace from runs/session_id/workspace/

        Args:
            session_id: Session ID to resume (e.g., "session_20251020_141932")
            run_ready_tasks_in_parallel: Whether to run ready tasks in parallel (default: False)

        Returns:
            Orchestrator instance with loaded state
        """
        # Create new orchestrator instance (without calling __init__)
        orchestrator = cls.__new__(cls)

        # Pure MOP: Logs are GENERATED from tool_timeline, not stored
        # LogModal only needs session_id - all logs derived from graph.nodes[].tool_timeline
        orchestrator.log_ops = LogOps.create_new()

        # Load graph (single source of truth)
        orchestrator.graph_ops = TaskGraphOps.load(session_id, log_ops=orchestrator.log_ops)

        # Create tool ops
        orchestrator.tool_ops = ToolOps.create_new()
        orchestrator.tool_ops.load_from_db()

        # Load workspace (files already exist in runs/session_id/workspace/)
        orchestrator.workspace = WorkspaceModal(session_id=session_id)
        orchestrator.workspace.load_from_disk()  # Load all files from workspace/
        orchestrator.tool_ops.set_workspace(orchestrator.workspace)

        # Agent management
        orchestrator.agents = {}
        orchestrator.agent = AgentBuilder(orchestrator)

        # Feature flags
        orchestrator.run_ready_tasks_in_parallel = run_ready_tasks_in_parallel

        print(f"[RESUME] Loaded session: {session_id}")
        print(f"[RESUME] Tasks: {len(orchestrator.graph_ops.modal.nodes)}")
        print(f"[RESUME] Events: {len(orchestrator.log_ops.modal.generate_master_log(orchestrator.graph_ops.modal))}")

        return orchestrator

    async def resume_from_human_response(self, human_task_id: str, answer: str) -> Dict:
        """
        Resume execution after human answers ask_master
        Encapsulates resume logic - MOP: orchestrator orchestrates

        Args:
            human_task_id: ID of the human task being answered
            answer: User's answer text

        Returns:
            Result dict with answer and meta_log
        """
        # Complete human task
        self.graph_ops.complete_task(
            human_task_id, result={"answer": answer, "documents": []}
        )
        self.graph_ops.save(orchestrator=self)

        # Find active ROOT task (MOP - ask the modal!)
        root_id = self.graph_ops.get_active_root_task()
        if not root_id:
            raise ValueError("No active ROOT task found to resume")

        # Run orchestration loop until no more ready tasks
        while True:
            ready = self.graph_ops.get_ready_tasks()

            # Filter out human tasks - they wait for user input, not agent execution
            agent_ready = [
                task_id
                for task_id in ready
                if self.graph_ops.modal.nodes[task_id].agent_id != "human"
            ]

            if not agent_ready:
                break

            # Execute ready tasks (parallel or sequential based on flag)
            await self._execute_ready_tasks(agent_ready)

        # Return result from ROOT task
        root_node = self.graph_ops.modal.nodes[root_id]
        return {
            "result": root_node.result or {},
            "meta_log": self.log_ops.render_for_json(self.graph_ops.modal),
        }

    async def _execute_ready_tasks(self, agent_ready: list):
        """
        Execute ready tasks either in parallel or sequentially based on flag

        Args:
            agent_ready: List of task IDs that are ready to execute
        """
        if self.run_ready_tasks_in_parallel:
            # Parallel execution using asyncio.gather
            import asyncio

            async def run_task(task_id):
                self.graph_ops.update_status(task_id, TaskStatus.RUNNING)
                self.graph_ops.save(orchestrator=self)
                await self.run_single_task(task_id, self.graph_ops.modal.nodes[task_id].agent_id)
                self.graph_ops.save(orchestrator=self)

            await asyncio.gather(*[run_task(task_id) for task_id in agent_ready])
        else:
            # Sequential execution (current behavior)
            for task_id in agent_ready:
                node = self.graph_ops.modal.nodes[task_id]
                self.graph_ops.update_status(task_id, TaskStatus.RUNNING)
                self.graph_ops.save(orchestrator=self)
                await self.run_single_task(task_id, node.agent_id)
                self.graph_ops.save(orchestrator=self)

    def build_preamble(self, agent: Agent, task_id: str = None) -> str:
        """
        Build dynamic preamble from agent's actual tools

        Args:
            agent: Agent object with config containing tools
            task_id: Optional task ID to check task type (for ASK_MASTER detection)

        Returns:
            str: Dynamic preamble tailored to this agent's capabilities
        """
        preamble = PREAMBLE_STATIC

        # Check if this is ASK_MASTER task (takes priority)
        if task_id:
            node = self.graph_ops.modal.nodes.get(task_id)
            if node and node.tool_type == ToolType.ASK_MASTER:
                # Find who asked (parent task's agent)
                parent_task = self.graph_ops.modal.nodes.get(node.parent_task_id) if node.parent_task_id else None
                if parent_task:
                    asker_agent = parent_task.agent_id
                    preamble += PREAMBLE_ASK_MASTER_RESPONSE.format(asker_agent=asker_agent)

        # Get tools from agent config (MOP - use what exists!)
        tools = agent.config.get("tools", [])
        agent_id = agent.config.get("agent_id", "")

        # Check for route_to tools
        route_tools = [t for t in tools if t.startswith("route_to_")]

        if route_tools:
            # Agent has delegation capability
            preamble += PREAMBLE_DELEGATION

            # Check if can route to self
            if f"route_to_{agent_id}" in route_tools:
                # Agent can delegate to itself!
                preamble += PREAMBLE_RECURSIVE.format(agent_id=agent_id)

        return preamble

    async def start_root_task(
        self,
        task: str,
        main_agent: Union[str, Agent],
        initiator: str = "human",
        reset_log_on_handoff: bool = False
    ):
        """
        Start execution from root task

        Args:
            task: Task description/payload
            main_agent: Agent ID or Agent object to handle the task
            initiator: Who initiated this session (default: "human", or can be agent_id)
            reset_log_on_handoff: Whether to reset logs on handoff
        """
        # Create root node with initiator as parent
        root_id = self.graph_ops.create_node(
            agent_id=main_agent,
            parent_task_id=initiator,
            tool_type=ToolType.ROOT,
            task_payload=task,
            master_agent_id=initiator
        )
        # NO log_ops.add_event() - MOP: tool_timeline is source of truth
        self.graph_ops.save(orchestrator=self)

        # Main loop
        while True:
            ready = self.graph_ops.get_ready_tasks()

            # Filter out human tasks - they wait for user input, not agent execution
            agent_ready = [
                task_id for task_id in ready
                if self.graph_ops.modal.nodes[task_id].agent_id != "human"
            ]

            if not agent_ready:
                break

            # Execute ready tasks (parallel or sequential based on flag)
            await self._execute_ready_tasks(agent_ready)

        root_node = self.graph_ops.modal.nodes[root_id]
        self.graph_ops.save(orchestrator=self)

        return {"result": root_node.result or {}, "meta_log": self.log_ops.render_for_json(self.graph_ops.modal)}

    def start_human_root_task(
        self,
        task: str,
        main_agent: Union[str, Agent],
        human_id: str = "human"
    ) -> str:
        """
        Human initiates a root task (delegation from human to agent)
        MOP: Automatically creates delegation event

        Args:
            task: Task description
            main_agent: Agent to handle the task
            human_id: Human identifier (default: "human")

        Returns:
            root_task_id: The created root task ID

        This is the "delegation" primitive for humans - they don't call it explicitly,
        system calls it when human starts work in UI/CLI.
        """
        # Create root node with human as parent
        root_id = self.graph_ops.create_node(
            agent_id=main_agent,
            parent_task_id=human_id,
            tool_type=ToolType.ROOT,
            task_payload=task,
            master_agent_id=human_id
        )

        # MOP: Add delegation event to root task's timeline
        # (This creates the delegation edge in visualization: human → root_task)
        self.graph_ops.modal.nodes[root_id].add_log_event(
            "agent_as_tool",
            "start_session",
            {"task": task},
            f"Delegated by {human_id}"
        )

        self.graph_ops.save(orchestrator=self)

        return root_id

    async def resume_from_human_answer(
        self,
        task_id: str,
        answer: Dict
    ):
        """
        Human provides answer to ask_master task, then resume execution
        MOP: Combines answer + resume in one call

        Args:
            task_id: The ask_master task (agent_id="human")
            answer: Human's answer

        Flow:
            1. Receive human answer
            2. Auto-create handoff for human task
            3. Parent gets unblocked automatically
            4. Resume orchestrator loop
        """
        # MOP: Modal handles answer → handoff conversion
        self.graph_ops.modal.answer_human_task(task_id, answer)

        # MOP: Resume execution loop (same as start_root_task)
        while True:
            ready = self.graph_ops.get_ready_tasks()

            # Filter out human tasks
            agent_ready = [
                task_id for task_id in ready
                if self.graph_ops.modal.nodes[task_id].agent_id != "human"
            ]

            if not agent_ready:
                break

            await self._execute_ready_tasks(agent_ready)

        self.graph_ops.save(orchestrator=self)

    def inject_runtime_prompt(self, agent: 'Agent', scene_path: str = None):
        """Inject live scene state into agent prompt - PURE MOP!

        Updates agent instructions to include current scene script with line numbers.
        Called before each agent turn for live scene editing.

        Args:
            agent: Agent to update
            scene_path: Path to scene file in workspace (None = auto-detect)

        Example:
            ops.inject_runtime_prompt(scene_agent, "scene.py")
        """
        # Import runtime prompt injection
        from .runtime_prompt_injection import generate_scene_editor_prompt

        # Get base instructions (stored separately for re-generation)
        base_knowledge = agent.config.get('base_instructions', agent.get_instructions())

        # Auto-detect scene file if not specified
        if scene_path is None:
            # Look for .py files in workspace
            import os
            workspace_files = os.listdir(self.workspace.workspace_path) if hasattr(self.workspace, 'workspace_path') else []
            scene_files = [f for f in workspace_files if f.endswith('.py')]
            scene_path = scene_files[0] if scene_files else None

        # Get current script from workspace
        current_script = None
        if scene_path:
            try:
                from ai_orchestration.utils.auto_db import agent_engine_db
                current_script = agent_engine_db.load_workspace_file(
                    self.workspace.session_id, scene_path
                )
            except FileNotFoundError:
                current_script = None

        # Generate updated prompt with live script state
        updated_prompt = generate_scene_editor_prompt(base_knowledge, current_script)

        # Update agent instructions
        agent.update_instructions(updated_prompt)

    async def run_single_task(self, task_id: str, agent_id: str) -> Optional[Dict]:
        """Run single task - LLM loop + modal handles tools"""
        agent = self.agents[agent_id]
        node = self.graph_ops.modal.nodes[task_id]

        while True:
            # Get model (needed for ask_llm)
            # MOP: Agent.force_model is authoritative, fallback to orchestrator default
            model = agent.config.get("force_model", self.default_model)

            # Build tools (always returns Claude format - conversion happens in ask_llm.py)
            tools = self.tool_ops.build_tools_for_agent(agent)

            # Get messages (log_ops builds log context from graph state)
            log_context = self.log_ops.build_log_context(self.graph_ops.modal, task_id)

            # BUILD DYNAMIC PREAMBLE (MOP - use agent's actual tools!)
            orchestration_preamble = self.build_preamble(agent, task_id)

            messages = self.graph_ops.modal.render_for_llm(
                task_id, agent.config,
                orchestration_preamble=orchestration_preamble,
                log_context=log_context
            )

            # Call LLM (tool_choice defaults are handled by provider wrappers)
            print(f"    [LLM] Task {task_id} calling {model}... (waiting for response)")
            response = ask_llm(
                messages=messages,
                model=model,
                tools=tools
            )

            # CHECK FOR API ERRORS FIRST!
            if "error" in response:
                print(f"\n{'=' * 80}")
                print(f"❌ LLM API ERROR")
                print(f"{'=' * 80}")
                print(f"Task ID: {task_id}")
                print(f"Agent ID: {agent_id}")
                print(f"Model: {model}")
                print(f"Error: {response.get('error')}")
                print(f"Last Error: {response.get('last_error')}")
                print(f"Attempts: {response.get('attempts')}")
                print(f"{'=' * 80}\n")
                raise RuntimeError(f"LLM API failed: {response.get('error')} - {response.get('last_error')}")

            print(f"    [LLM] Task {task_id} got response from {model} with tools: {[tc['name'] for tc in response.get('tool_calls', [])]}")

            # Extract tool call - with detailed error logging
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                # TOOL-ONLY VIOLATION - Add hint and retry (MOP pattern)
                print(f"\n{'=' * 80}")
                print("⚠️  TOOL-ONLY MODE VIOLATION: Agent returned NO tool calls")
                print(f"{'=' * 80}")
                print(f"Task ID: {task_id}")
                print(f"Agent ID: {agent_id}")

                # Generate hint using LogModal (MOP)
                hint_msg = self.log_ops.modal.generate_tool_only_violation_hint(
                    agent_id=agent_id,
                    available_tools=[t["name"] for t in tools],
                    context_info={}
                )

                # Add hint to task (MOP: stored in graph_modal)
                self.log_ops.add_hint(
                    "tool_only_violation",
                    {"message": hint_msg, "agent": agent_id},
                    task_id,
                    self.graph_ops.modal
                )

                print(f"\n📋 Hint added to task:\n{hint_msg}")
                print(f"{'=' * 80}\n")

                # Retry LLM call with hint in context
                print("  → Retrying LLM call with hint\n")
                continue  # Loop continues, hint will be in next render

            tool_call = tool_calls[0]
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]

            # NO log_ops.add_event() - MOP: tool_timeline is source of truth
            # Tool calls added to node.tool_timeline by modal.handle_tool_call()

            # Get tool type (offensive mode: crash if doesn't exist)
            tool_entry = self.tool_ops.get_tool(tool_name)
            if tool_entry is None:
                raise ValueError(f"Tool '{tool_name}' not registered! Available: {list(self.tool_ops.modal.tools.keys())}")

            # Look up the exact tool schema that was sent to LLM (for validation)
            tool_schema = next((t for t in tools if t["name"] == tool_name), None)

            # Modal handles execution (with hint coordination)
            # MOP: Pass exact schema LLM saw (pure MOP - validation uses what was sent)
            result = self.graph_ops.modal.handle_tool_call(
                task_id=task_id,
                tool_name=tool_name,
                tool_type=tool_entry.type,
                tool_input=tool_input,
                tool_schema=tool_schema,
                tool_ops=self.tool_ops,
                graph_ops=self.graph_ops,
                agent_model=model,
                log_ops=self.log_ops
            )

            # NO log_ops.add_event() - MOP: tool_timeline already updated by handle_tool_call()

            # HINT TYPE 3: Loop detection (MOP - read from graph_modal)
            loop_info = self.log_ops.modal.detect_loop_in_task(
                self.graph_ops.modal,
                task_id,
                tool_name,
                tool_input
            )

            if loop_info.get("loop_detected"):
                # Generate hint using LogModal (dispatch based on type)
                loop_type = loop_info.get("type", "hard_loop")

                if loop_type == "hard_loop":
                    # AGGRESSIVE hint for same tool + same result
                    hint_msg = self.log_ops.modal.generate_loop_detection_hint(
                        loop_info["tool_name"],
                        loop_info["count"]
                    )
                elif loop_type == "consecutive":
                    # SOFT hint for consecutive calls (different results OK)
                    hint_msg = self.log_ops.modal.generate_consecutive_tool_hint(
                        loop_info["tool_name"],
                        loop_info["count"]
                    )
                else:
                    hint_msg = None

                # Add hint to the LAST tool event in timeline
                if hint_msg:
                    node = self.graph_ops.modal.nodes[task_id]
                    if node.tool_timeline:
                        # Add hint to the last event's result
                        last_event = node.tool_timeline[-1]
                        if isinstance(last_event.get("result"), dict):
                            last_event["result"]["loop_hint"] = hint_msg

            # CUSTOM HANDOFF HANDLER EXECUTION
            # If handoff was called and agent has custom handler, execute it
            if (result["action"] == "return" and
                tool_name == "handoff" and
                agent_id in self.handoff_handlers):

                print(f"\n🔌 Executing custom handoff handler for {agent_id}...")

                handler = self.handoff_handlers[agent_id]
                handler_result = handler(
                    handoff_type=tool_input.get("handoff_type"),  # Explicit type!
                    edits=tool_input.get("edits", []),
                    message=tool_input.get("message", ""),
                    scene_name=tool_input.get("scene_name"),
                    current_script=tool_input.get("current_script", "")
                )

                # Update task result with handler output
                if handler_result.get("success"):
                    # Merge handler result into task result
                    updated_result = {**result["value"], "handoff_result": handler_result}
                    self.graph_ops.modal.nodes[task_id].result = updated_result
                    result["value"] = updated_result
                    print(f"   ✓ Handler executed successfully")
                else:
                    print(f"   ❌ Handler failed: {handler_result.get('error')}")
                    # Still return, but with error info
                    updated_result = {**result["value"], "handoff_result": handler_result}
                    self.graph_ops.modal.nodes[task_id].result = updated_result
                    result["value"] = updated_result

            # Save to DB after each tool call for real-time progress monitoring
            self.graph_ops.save(orchestrator=self)

            if result["action"] == "return":
                return result["value"]


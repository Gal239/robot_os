"""
TOOL OPS - Operations on ToolModal
Handles tool loading from DB, tool management, and execution
"""

from typing import Dict, List, Optional
from copy import deepcopy
from ai_orchestration.utils.global_config import agent_engine_db
from ..modals import ToolModal, ToolEntry
from .. import tool_decorator


class ToolOps:
    """
    Operations layer for ToolModal
    Handles persistence, complex tool operations, and execution
    """

    def __init__(self, modal: ToolModal, workspace=None):
        self.modal = modal
        self.workspace = workspace
        self.agent_model = None  # MOP: Agent's model injected per execution
        # Register orchestration tools (handoff, ask_master) into modal
        self._register_orchestration_tools()

    @classmethod
    def create_new(cls) -> 'ToolOps':
        """Create new ToolOps with empty modal"""
        return cls(ToolModal())

    def set_workspace(self, workspace):
        """Set workspace for tool execution"""
        self.workspace = workspace

    # ========== PERSISTENCE ==========

    def load_from_db(self):
        """Load all tools from database into modal"""
        for tool_id in list(agent_engine_db.tools):
            try:
                config = agent_engine_db.tools[tool_id]
                self.modal.register_tool(ToolEntry(
                    name=config.get("name", tool_id),
                    type=config.get("type", "unknown"),
                    input_schema=config.get("input_schema", {}),
                    output_schema=config.get("output_schema", {})
                ))
            except:
                pass  # Skip invalid tools

    def register_agents_as_tools(self, agents: Dict):
        """
        Auto-register route_to_X tools from agents

        Args:
            agents: Dict of agent_id -> Agent objects
        """
        for agent_id, agent in agents.items():
            tool_spec = agent.to_tool_spec()
            self.modal.register_tool(ToolEntry(
                name=tool_spec["name"],
                type=tool_spec["type"],
                input_schema=tool_spec["input_schema"],
                output_schema={}
            ))

    def _register_orchestration_tools(self):
        """Register built-in orchestration tools (handoff, ask_master) into ToolModal"""
        for tool_name, tool_spec in self.ORCHESTRATION_TOOLS.items():
            self.modal.register_tool(ToolEntry(
                name=tool_spec["name"],
                type=tool_name,  # handoff or ask_master
                input_schema=tool_spec["input_schema"],
                output_schema=tool_spec.get("output_schema", {})
            ))

    # ========== DELEGATION TO MODAL ==========

    # Built-in orchestration tools schemas (shared between get_tool and build_tools_for_agent)
    ORCHESTRATION_TOOLS = {
        "handoff": {
            "name": "handoff",
            "description": "Complete your current task and return results to the parent agent. This is the ONLY way to mark a task as complete and deliver your final answer. Use this when you have finished your work and are ready to return control. Include all relevant information in the answer object and list any workspace documents you created in the documents array.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "object",
                        "description": "Your final result as a structured object. Include all relevant information, findings, or outputs from your task. If you created files or delegated subtasks, mention them here."
                    },
                    "documents": {
                        "type": "array",
                        "description": "Array of workspace file paths that are relevant to your answer. Include files you created or modified. Return empty array if no relevant documents.",
                        "items": {"type": "string"}
                    }
                },
                "required": ["answer","documents"]
            },
            "output_schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "object"}
                },
                "required": ["answer"]
            }
        },
        "ask_master": {
            "name": "ask_master",
            "description": "Ask your parent agent for clarification, additional information, or guidance when you need help to complete your task. The parent agent will provide an answer via handoff. Use this when you're blocked, need missing information, or require a decision from your parent. Note: Your parent will use handoff to answer you, not delegation.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your question for the parent agent. Be clear and specific about what you need."
                    },
                    "documents": {
                        "type": "array",
                        "description": "Optional workspace file paths to provide context for your question.",
                        "items": {"type": "string"}
                    }
                },
                "required": ["question"]
            },
            "output_schema": {}
        }
    }

    def get_tool(self, tool_name: str) -> Optional[ToolEntry]:
        """Get tool by name (all tools now in modal)"""
        return self.modal.get_tool(tool_name)

    # ========== COMPLEX OPERATIONS ==========

    def build_tools_for_agent(self, agent) -> List[Dict]:
        """
        Build tool list for agent in INTERNAL Claude format
        Format conversion to provider-specific format happens in ask_llm.py

        Supports per-agent tool overrides via agent.config["tool_overrides"]
        If override exists, it COMPLETELY REPLACES the default schema

        Args:
            agent: Agent object with config

        Returns:
            List of tool configs in Claude format (name, description, input_schema)
        """

        tools = []
        added_names = set()

        # Get agent's tool overrides (if any)
        tool_overrides = agent.config.get("tool_overrides", {})

        # Get tools from agent config (agent config is source of truth)
        for tool_name in agent.get_tools():
            # Check if orchestration tool
            if tool_name in self.ORCHESTRATION_TOOLS:
                # Check for override - FULL REPLACEMENT
                if tool_name in tool_overrides:
                    tool_spec = deepcopy(tool_overrides[tool_name])
                    tool_spec["name"] = tool_name  # Ensure name is set correctly
                else:
                    # Use default
                    tool_spec = self.ORCHESTRATION_TOOLS[tool_name]

                tools.append(tool_spec)
                added_names.add(tool_name)
            else:
                # Regular tool from DB
                tool_entry = self.modal.get_tool(tool_name)
                if tool_entry and tool_name not in added_names:
                    # Add in Claude format
                    tools.append({
                        "name": tool_entry.name,
                        "description": tool_entry.input_schema.get("description", f"Tool: {tool_name}"),
                        "input_schema": tool_entry.input_schema
                    })
                    added_names.add(tool_name)

        return tools

    # ========== TOOL EXECUTION ==========

    def execute(self, tool_name: str, tool_input: Dict, task_id: str, agent_model: str) -> Dict:
        """
        Execute tool - offensive mode, no checks
        MOP: agent_model is REQUIRED parameter, injected from agent.force_model (single source of truth)
        No default - forces defensive programming at call site
        """
        func = tool_decorator.get_tool_function(tool_name)
        # MOP: Inject agent_model into tool execution
        return func(workspace=self.workspace, task_id=task_id, agent_model=agent_model, **tool_input)

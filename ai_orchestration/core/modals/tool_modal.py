"""
TOOL MODAL - Tool registry with schemas
Manages all tools, validates inputs/outputs, detects patterns
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum


class ToolType(str, Enum):
    """Tool types - single source of truth for routing"""
    INITIATOR = "initiator"              # Session master (user/agent who started session)
    ROOT = "root"                        # Initial user task
    FUNCTION_TOOL = "function_tool"      # Normal tools (ALL tools - ask_data, write_file, load_to_context, etc.)
    HANDOFF = "handoff"                  # Task completion
    AGENT_AS_TOOL = "agent_as_tool"      # route_to_X (agent used as tool)
    ASK_MASTER = "ask_master"            # Ask parent agent
    NON_FUNCTION_TOOL = "non_function_tool"  # Metacognition tools (think, plan, save_memory) - input logged only


# Tool behavior specifications - what each tool type does
TOOL_BEHAVIOR = {
    ToolType.INITIATOR: {
        "creates_node": False,
        "blocks_parent": False,
        "is_virtual": True  # Special node for visualization only
    },
    ToolType.ROOT: {
        "creates_node": True,
        "blocks_parent": False
    },
    ToolType.FUNCTION_TOOL: {
        "creates_node": False,
        "executes_function": True
    },
    ToolType.HANDOFF: {
        "creates_node": False,
        "completes_task": True
    },
    ToolType.AGENT_AS_TOOL: {
        "creates_node": True,
        "blocks_parent": True
    },
    ToolType.ASK_MASTER: {
        "creates_node": True,
        "blocks_parent": True
    },
    ToolType.NON_FUNCTION_TOOL: {
        "creates_node": False,
        "logs_input": True  # Input gets logged with special formatting
    }
}


@dataclass
class ToolEntry:
    """Single tool with schemas"""
    name: str
    type: str  # ToolType enum value
    input_schema: Dict
    output_schema: Dict


@dataclass
class ToolModal:
    """
    Pure tool state - registry of all tools
    Single source of truth for tool schemas and validation
    """
    tools: Dict[str, ToolEntry] = field(default_factory=dict)  # tool_name → ToolEntry

    def register_tool(self, tool_entry: ToolEntry):
        """Register a tool - OFFENSIVE"""
        self.tools[tool_entry.name] = tool_entry

    def get_tool(self, tool_name: str) -> Optional[ToolEntry]:
        """Get tool by name"""
        return self.tools.get(tool_name)

    def validate_input(self, tool_name: str, tool_input: Dict) -> Optional[Dict]:
        """
        Validate tool input against schema
        Returns None if valid, or error dict if invalid
        """
        tool = self.get_tool(tool_name)
        if not tool or not tool.input_schema:
            return None  # No schema to validate against

        schema = tool.input_schema
        required_fields = schema.get("required", [])

        # Check for missing required fields
        missing = [f for f in required_fields if f not in tool_input]

        if missing:
            return {
                "error": f"Missing required field(s): {', '.join(missing)}",
                "provided_fields": list(tool_input.keys()),
                "required_fields": required_fields,
                "hint": f"Please call {tool_name} again with all required fields"
            }

        return None  # Validation passed

    def validate_output(self, tool_name: str, tool_output: Dict) -> Optional[Dict]:
        """
        Validate tool output against schema
        Returns None if valid, or error dict if invalid
        """
        tool = self.get_tool(tool_name)
        if not tool or not tool.output_schema:
            return None  # No schema to validate against

        schema = tool.output_schema
        required_fields = schema.get("required", [])

        # Check for missing required fields in output
        missing = [f for f in required_fields if f not in tool_output]

        if missing:
            return {
                "error": f"Tool output incomplete - missing field(s): {', '.join(missing)}",
                "missing_fields": missing,
                "expected_fields": required_fields,
                "actual_fields": list(tool_output.keys())
            }

        return None  # Validation passed

    def render_for_json(self) -> Dict:
        """Full serialization"""
        return {
            "tools": {
                name: {
                    "name": tool.name,
                    "type": tool.type,
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema
                }
                for name, tool in self.tools.items()
            }
        }

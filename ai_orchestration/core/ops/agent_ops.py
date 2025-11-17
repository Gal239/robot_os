from typing import Dict, Any, Optional

from ..db_model import DBModel
from ..modals import ToolType


class Agent(DBModel):
    """Agent configuration - inherits CRUD from DBModel"""

    collection_name = "agents"

    def __init__(self, config: Dict[str, Any], agent_id: Optional[str] = None):
        # Use agent_id as the id field for DBModel
        super().__init__(config, agent_id)
        self.agent_id = self.id  # Alias for backward compatibility

    # ===== AGENT-SPECIFIC METHODS =====

    def get_description(self) -> str:
        """External description - what other agents see as tool"""
        return self.config.get('description', f'Agent {self.agent_id}')

    def get_instructions(self) -> str:
        """Internal system prompt - agent's brain"""
        return self.config.get('instructions', '')

    def update_instructions(self, new_instructions: str):
        """Update agent instructions at runtime - PURE MOP!

        Allows dynamic prompt injection for live scene editing.
        Instructions updated in-memory only (not persisted to DB).

        Args:
            new_instructions: New system prompt for this agent

        Example:
            agent.update_instructions(generate_scene_editor_prompt(base, current_script))
        """
        self.config['instructions'] = new_instructions

    def get_tools(self) -> list:
        """All available tools (functions + agents)"""
        return self.config.get('tools', [])

    def to_tool_spec(self) -> dict:
        """Convert agent to tool specification for route_to_X"""
        return {
            "name": f"route_to_{self.agent_id}",
            "type": ToolType.AGENT_AS_TOOL,
            "description": self.get_description(),
            "input_schema": {
                "type": "object",
                "properties": {
                    "request": {"type": "string", "description": "Task for agent"}
                },
                "required": ["request"]
            },
            "target_agent_id": self.agent_id
        }

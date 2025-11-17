"""
Agent Schema - Pydantic models for agent configuration
OFFENSIVE: Crashes on invalid config, no defensive checks
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, validator


class AgentConfig(BaseModel):
    """Complete agent configuration - OFFENSIVE validation"""
    agent_id: str
    description: str
    instructions: str
    tools: List[str]
    force_model: str  # MOP: Required field, must be set by AgentBuilder from orchestrator default or override
    max_tokens: Optional[int] = 4000
    metadata: Optional[Dict[str, Any]] = {}
    tool_overrides: Optional[Dict[str, Dict]] = {}

    @validator('agent_id')
    def validate_agent_id(cls, v):
        """OFFENSIVE: Crash if invalid agent_id format"""
        assert v and isinstance(v, str), "agent_id must be non-empty string"
        assert v.replace('_', '').replace('-', '').isalnum(), "agent_id should be alphanumeric with _ or -"
        return v

    @validator('description')
    def validate_description(cls, v):
        """OFFENSIVE: Crash if empty description"""
        assert v and v.strip(), "description must be non-empty"
        assert len(v) < 500, "description too long (max 500 chars)"
        return v

    @validator('instructions')
    def validate_instructions(cls, v):
        """OFFENSIVE: Instructions can be empty but must be string"""
        assert isinstance(v, str), "instructions must be string"
        # Allow empty for simple agents, warn if empty
        if not v.strip():
            print(f"WARNING: Agent has empty instructions")
        return v

    @validator('tools')
    def validate_tools(cls, v):
        """OFFENSIVE: Crash if no tools"""
        assert isinstance(v, list), "tools must be a list"
        # Allow empty for root agents, but warn
        if not v:
            print(f"WARNING: Agent has no tools - is this intentional?")
        return v

    @validator('force_model')
    def validate_model(cls, v):
        """OFFENSIVE: Crash if invalid model"""
        if v:
            allowed = ["claude-sonnet-4-5", "gpt-5", "gpt-5-mini", "gpt-5-nano"]
            assert v in allowed, f"Invalid model: {v}. Allowed: {allowed}"
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DB storage"""
        return {
            "agent_id": self.agent_id,
            "description": self.description,
            "instructions": self.instructions,
            "tools": self.tools,
            "force_model": self.force_model,
            "max_tokens": self.max_tokens,
            "metadata": self.metadata or {},
            "tool_overrides": self.tool_overrides or {}
        }

    def to_tool_spec(self) -> Dict[str, Any]:
        """Convert agent to tool specification for route_to_X"""
        return {
            "name": f"route_to_{self.agent_id}",
            "type": "agent_as_tool",
            "description": f"{self.description} Use this tool to delegate a subtask to {self.agent_id}. This creates a new task that blocks your current task until {self.agent_id} completes it with a handoff. Only use delegation for subtasks that require {self.agent_id}'s specialized capabilities. Do NOT use delegation to answer ask_master questions - use handoff directly instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "request": {
                        "type": "string",
                        "description": f"Task description for {self.agent_id}. Be specific about what you need, what constitutes completion, and what you expect in the handoff response."
                    },
                    "documents": {
                        "type": "array",
                        "description": "Optional workspace file paths to provide as reference context to the agent.",
                        "items": {"type": "string"}
                    }
                },
                "required": ["request"]
            },
            "target_agent_id": self.agent_id
        }

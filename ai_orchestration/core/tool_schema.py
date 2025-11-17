"""
Tool Schema - Pydantic models for tool configuration
OFFENSIVE: Crashes on invalid config, no defensive checks
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, validator


class ToolParameter(BaseModel):
    """Single parameter definition - OFFENSIVE validation"""
    type: str
    description: str
    required: bool
    default: Optional[Any] = None

    @validator('type')
    def validate_type(cls, v):
        """OFFENSIVE: Crash if invalid type"""
        allowed = ['string', 'integer', 'number', 'boolean', 'object', 'array']
        assert v in allowed, f"Invalid parameter type: {v}. Allowed: {allowed}"
        return v


class ToolConfig(BaseModel):
    """Complete tool configuration - OFFENSIVE validation"""
    tool_id: str
    description: str
    type: str  # ToolType enum value
    parameters: Dict[str, ToolParameter]
    returns: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None  # JSON schema for output validation

    @validator('tool_id')
    def validate_tool_id(cls, v):
        """OFFENSIVE: Crash if invalid tool_id format"""
        assert v and isinstance(v, str), "tool_id must be non-empty string"
        assert '_' in v or v.islower(), "tool_id should be snake_case"
        return v

    @validator('type')
    def validate_type(cls, v):
        """OFFENSIVE: Crash if invalid type"""
        allowed = ['function_tool', 'handoff', 'agent_as_tool', 'ask_master', 'non_function_tool']
        assert v in allowed, f"Invalid type: {v}. Allowed: {allowed}"
        return v

    @validator('description')
    def validate_description(cls, v):
        """OFFENSIVE: Crash if empty description"""
        assert v and v.strip(), "description must be non-empty"
        return v

    def to_input_schema(self) -> Dict:
        """Convert to JSON schema format for LLM API"""
        properties = {}
        required = []

        for param_name, param in self.parameters.items():
            properties[param_name] = {
                "type": param.type,
                "description": param.description
            }

            if param.required:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties
        }

        if required:
            schema["required"] = required

        return schema

"""
Tool Decorator - JSON-config based tool registration
OFFENSIVE: Crashes if config missing or invalid

@tool decorator:
- Extracts tool_id from docstring (first line)
- Loads config from JSON (crashes if missing/invalid)
- Auto-saves to agent_engine_db.tools
- Auto-registers function in global registry
"""

from typing import Callable, Dict
import inspect

# Import database
from ai_orchestration.utils.global_config import agent_engine_db

# Import config loader (OFFENSIVE: crashes if config invalid)
from ai_orchestration.core.tool_config_loader import load_tool_config


# Global function registry
_TOOL_FUNCTIONS: Dict[str, Callable] = {}


def tool(execution_type: str = "function"):
    """
    Decorator for tool functions. Loads schema from JSON config.
    OFFENSIVE: Crashes if config missing or invalid.

    Args:
        execution_type: "function" (executes Python) or "context_block" (loads to context)

    Usage:
        @tool(execution_type="function")
        def my_tool(workspace, task_id, arg1: str, arg2: int = 5) -> dict:
            '''my_tool'''  # Just tool_id, config comes from JSON
            return {"result": "..."}
    """
    def decorator(func: Callable) -> Callable:
        # Extract tool_id from docstring (first line, stripped)
        # OFFENSIVE: Crash if no docstring
        doc = inspect.getdoc(func)
        assert doc, f"Function {func.__name__} must have docstring with tool_id"

        tool_id = doc.strip().split('\n')[0].strip()
        assert tool_id, f"Function {func.__name__} docstring must have tool_id on first line"

        # OFFENSIVE: Load config from JSON (crashes if missing/invalid)
        config = load_tool_config(tool_id)

        # Build input schema from JSON config
        input_schema = config.to_input_schema()

        # Add description from config
        input_schema["description"] = config.description

        # Build tool config for database
        tool_config = {
            "name": tool_id,
            "description": config.description,
            "type": config.type,
            "input_schema": input_schema,
            "output_schema": config.output_schema  # Add output schema for validation
        }

        # Save to database (in-memory only, not persisted to disk)
        # Source of truth: /core/tools/configs/*.json
        agent_engine_db.tools[tool_id] = tool_config

        # Register function in global registry (for function and context_block types)
        if execution_type in ("function", "context_block"):
            _TOOL_FUNCTIONS[tool_id] = func

        return func

    return decorator


def get_tool_function(tool_name: str) -> Callable:
    """
    Get registered tool function by name.
    OFFENSIVE: Crashes if not found.

    Args:
        tool_name: Name of tool

    Returns:
        Function callable

    Raises:
        KeyError: If tool not registered
    """
    # OFFENSIVE: No check, just return (crash if missing)
    return _TOOL_FUNCTIONS[tool_name]


def has_tool_function(tool_name: str) -> bool:
    """Check if tool has registered function."""
    return tool_name in _TOOL_FUNCTIONS


def list_registered_tools() -> list[str]:
    """List all registered tool names."""
    return list(_TOOL_FUNCTIONS.keys())

"""
Tool Config Loader - Load and cache tool JSON configs
OFFENSIVE: Crashes if config missing or invalid, no defensive checks
"""

import json
from pathlib import Path
from typing import Dict
from ai_orchestration.core.tool_schema import ToolConfig


# Cache - load once, crash if invalid
_CONFIG_CACHE: Dict[str, ToolConfig] = {}


def load_tool_config(tool_id: str) -> ToolConfig:
    """
    Load tool configuration from JSON file
    OFFENSIVE: Crashes if file missing or JSON invalid

    Args:
        tool_id: Tool identifier (e.g., 'write_file')

    Returns:
        Validated ToolConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        pydantic.ValidationError: If config schema is invalid
    """
    # Return from cache if already loaded
    if tool_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[tool_id]

    # Build path to config file (in tools/configs/)
    config_path = Path(__file__).parent / "tools" / "configs" / f"{tool_id}.json"

    # OFFENSIVE: No file existence check, just read (crash if missing)
    data = json.loads(config_path.read_text(encoding='utf-8'))

    # OFFENSIVE: Pydantic validation (crash if schema invalid)
    config = ToolConfig(**data)

    # Cache for future use
    _CONFIG_CACHE[tool_id] = config

    return config


def load_all_configs() -> Dict[str, ToolConfig]:
    """
    Load all tool configs from /tools/configs/
    OFFENSIVE: Crashes on first invalid config

    Returns:
        Dict of tool_id → ToolConfig
    """
    configs_dir = Path(__file__).parent / "tools" / "configs"

    # OFFENSIVE: Assume directory exists
    all_configs = {}

    for config_file in configs_dir.glob("*.json"):
        tool_id = config_file.stem
        all_configs[tool_id] = load_tool_config(tool_id)

    return all_configs


def clear_cache():
    """Clear config cache - useful for testing or hot reload"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}

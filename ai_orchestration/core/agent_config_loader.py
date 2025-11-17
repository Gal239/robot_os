"""
Agent Config Loader - Load and cache agent JSON configs
OFFENSIVE: Crashes if config missing or invalid, no defensive checks
"""

import json
from pathlib import Path
from typing import Dict
from ai_orchestration.core.agent_schema import AgentConfig


# Cache - load once, crash if invalid
_CONFIG_CACHE: Dict[str, AgentConfig] = {}


def load_agent_config(agent_id: str) -> AgentConfig:
    """
    Load agent configuration from JSON file
    OFFENSIVE: Crashes if file missing or JSON invalid

    Args:
        agent_id: Agent identifier (e.g., 'robot_controller')

    Returns:
        Validated AgentConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is malformed
        pydantic.ValidationError: If config schema is invalid
    """
    # Return from cache if already loaded
    if agent_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[agent_id]

    # Build path to config file - check multiple locations
    # 1. Try /databases/agents/ (production)
    db_path = Path(__file__).parent.parent / "databases" / "agents" / f"{agent_id}.json"

    # 2. Try /core/agents/configs/ (alternative)
    core_path = Path(__file__).parent / "agents" / "configs" / f"{agent_id}.json"

    config_path = db_path if db_path.exists() else core_path

    # OFFENSIVE: No file existence check, just read (crash if missing)
    data = json.loads(config_path.read_text(encoding='utf-8'))

    # OFFENSIVE: Pydantic validation (crash if schema invalid)
    config = AgentConfig(**data)

    # Cache for future use
    _CONFIG_CACHE[agent_id] = config

    return config


def load_all_configs() -> Dict[str, AgentConfig]:
    """
    Load all agent configs from /databases/agents/
    OFFENSIVE: Crashes on first invalid config

    Returns:
        Dict of agent_id → AgentConfig
    """
    configs_dir = Path(__file__).parent.parent / "databases" / "agents"

    # OFFENSIVE: Assume directory exists, create if not
    configs_dir.mkdir(parents=True, exist_ok=True)

    all_configs = {}

    for config_file in configs_dir.glob("*.json"):
        agent_id = config_file.stem
        all_configs[agent_id] = load_agent_config(agent_id)

    return all_configs


def clear_cache():
    """Clear config cache - useful for testing or hot reload"""
    global _CONFIG_CACHE
    _CONFIG_CACHE = {}
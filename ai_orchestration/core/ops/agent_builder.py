"""
Agent Builder - ops.agent.create() API
Beautiful, sentence-like API for creating and managing agents
Supports both JSON configs and code-based creation
"""
from typing import Dict, List, Optional
from pathlib import Path
import json
from .agent_ops import Agent
from ..agent_config_loader import load_agent_config, load_all_configs
from ..agent_schema import AgentConfig


class AgentBuilder:
    """Fluent API for agent creation - supports JSON configs + code"""

    def __init__(self, orchestrator, default_model: str):
        self.orchestrator = orchestrator
        # MOP: Store orchestrator-level default model for agent inheritance
        self.default_model = default_model

    def create(
        self,
        id: str,
        describe: str,
        instructions: str = "",
        tools: List[str] = None,
        model: Optional[str] = None,
        tool_overrides: Dict[str, Dict] = None,
        save_to_json: bool = False
    ) -> Agent:
        """
        Create agent in memory (code-based)

        Args:
            id: Agent identifier
            describe: What the agent does (for routing)
            instructions: Agent's system prompt
            tools: List of tool IDs
            model: LLM model override (None = inherit orchestrator default)
            tool_overrides: Per-agent tool schema overrides (e.g., custom handoff schema)
            save_to_json: Save config to JSON file
        """
        # MOP: Agent inherits orchestrator default if no model specified
        actual_model = model if model is not None else self.default_model

        config = {
            "agent_id": id,
            "description": describe,
            "instructions": instructions,
            "tools": tools ,
            "force_model": actual_model,
            "tool_overrides": tool_overrides or {}
        }

        # Validate with Pydantic
        validated = AgentConfig(**config)

        agent = Agent(validated.to_dict(), id)

        # Add to orchestrator's agents
        self.orchestrator.agents[id] = agent

        # Auto-register as tool
        self.orchestrator.tool_ops.register_agents_as_tools({id: agent})

        # Optionally save to JSON
        if save_to_json:
            self.save_config(id, validated)

        return agent

    def from_config(self, agent_id: str) -> Agent:
        """
        Load agent from JSON config

        Example:
            agent = orchestrator.agent.from_config("robot_controller")
        """
        # Load and validate config
        config = load_agent_config(agent_id)

        agent = Agent(config.to_dict(), agent_id)

        # Add to orchestrator
        self.orchestrator.agents[agent_id] = agent
        self.orchestrator.tool_ops.register_agents_as_tools({agent_id: agent})

        return agent

    def load_all_from_configs(self):
        """Load all agents from JSON configs in /databases/agents/"""
        all_configs = load_all_configs()

        for agent_id, config in all_configs.items():
            agent = Agent(config.to_dict(), agent_id)
            self.orchestrator.agents[agent_id] = agent
            self.orchestrator.tool_ops.register_agents_as_tools({agent_id: agent})

    def save_config(self, agent_id: str, config: AgentConfig):
        """Save agent config to JSON file"""
        config_path = Path(__file__).parent.parent.parent / "databases" / "agents" / f"{agent_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        print(f"✓ Saved agent config: {config_path}")

    def load(self, agent_id: str) -> Agent:
        """Load agent from DB (legacy)"""
        agent = Agent.load(agent_id)
        self.orchestrator.agents[agent_id] = agent
        self.orchestrator.tool_ops.register_agents_as_tools({agent_id: agent})
        return agent

    def load_all(self):
        """Load all agents from DB (legacy)"""
        from ai_orchestration.utils.global_config import agent_engine_db
        for agent_id in list(agent_engine_db.agents):
            self.load(agent_id)

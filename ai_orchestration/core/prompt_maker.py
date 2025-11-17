"""
prompt_maker.py - PURE MOP Prompt Generation

Auto-generates agent system prompts from ACTUAL SOURCE DATA.
ZERO HARDCODING - everything comes from the real data archives!

MOP PRINCIPLES:
- Prompts generated from ACTUAL data (ASSETS.json, BEHAVIORS.json, etc.)
- NOT from config descriptions - from the REAL DATA
- Shows agent what's actually available
- Prompts always match current data state
"""

from typing import List, Dict, Any
from pathlib import Path
import json


class PromptMaker:
    """
    Generates agent system prompts from ACTUAL SOURCE DATA

    PURE MOP: Prompts = f(REAL DATA)
    - Loads ASSETS.json, BEHAVIORS.json, RELATIONS.json, API.json
    - Generates rich prompts showing what's actually available
    - NO hardcoded descriptions - SHOW the real data!
    """

    def __init__(self, tool_configs_dir: Path = None, data_root: Path = None):
        """
        Initialize prompt maker

        Args:
            tool_configs_dir: Path to tool configs directory
            data_root: Path to project root containing data archives
        """
        if tool_configs_dir is None:
            tool_configs_dir = Path(__file__).parent / "tools" / "configs"

        if data_root is None:
            # Navigate to simulation_center root
            data_root = Path(__file__).parent.parent.parent

        self.tool_configs_dir = tool_configs_dir
        self.data_root = data_root

        # Paths to real data archives
        self.assets_json = data_root / "core" / "modals" / "mujoco_assets" / "ASSETS.json"
        self.behaviors_json = data_root / "core" / "behaviors" / "BEHAVIORS.json"
        self.relations_json = data_root / "core" / "modals" / "RELATIONS.json"
        self.api_json = data_root / "core" / "docs" / "API.json"

    def generate_prompt(self,
                       agent_id: str,
                       tool_ids: List[str],
                       purpose: str,
                       **kwargs) -> str:
        """
        Generate complete agent prompt from ACTUAL DATA

        Args:
            agent_id: Agent identifier (e.g., "scene_maker")
            tool_ids: List of tool IDs this agent has access to
            purpose: High-level goal (e.g., "create simulation scenes")
            **kwargs: Additional customization options

        Returns:
            Complete system prompt string
        """
        # Load REAL DATA from archives
        assets_data = self._load_assets_data() if "discover_assets" in tool_ids else None
        behaviors_data = self._load_behaviors_data() if "discover_behaviors" in tool_ids else None
        relations_data = self._load_relations_data() if "discover_relations" in tool_ids else None
        api_data = self._load_api_data() if "get_api_documentation" in tool_ids else None

        # Load REAL usage examples from demo files (raw text)
        demo_examples = self._load_demo_examples() if "get_scene_examples" in tool_ids else ""

        # Load tool configs for workflow/constraints
        tools = []
        for tool_id in tool_ids:
            tool_config = self._load_tool_config(tool_id)
            if tool_config:
                tools.append(tool_config)

        # Generate each section from REAL DATA
        sections = []
        sections.append(self._generate_header(agent_id, purpose))
        sections.append(self._generate_knowledge_base(assets_data, behaviors_data, relations_data, api_data))

        # Add REAL usage examples if available
        if demo_examples:
            sections.append("\n📚 REAL USAGE EXAMPLES:\n")
            sections.append(demo_examples)

        sections.append(self._generate_workflow(tools))
        sections.append(self._generate_rules(tools))

        return "\n\n".join(sections)

    def _load_tool_config(self, tool_id: str) -> Dict[str, Any]:
        """Load tool configuration from JSON file"""
        config_path = self.tool_configs_dir / f"{tool_id}.json"

        if not config_path.exists():
            print(f"⚠️  Tool config not found: {config_path}")
            return None

        with open(config_path, 'r') as f:
            return json.load(f)

    def _load_assets_data(self) -> Dict[str, Any]:
        """Load ACTUAL assets from ASSETS.json"""
        if not self.assets_json.exists():
            return None
        with open(self.assets_json, 'r') as f:
            return json.load(f)

    def _load_behaviors_data(self) -> Dict[str, Any]:
        """Load ACTUAL behaviors from BEHAVIORS.json"""
        if not self.behaviors_json.exists():
            return None
        with open(self.behaviors_json, 'r') as f:
            return json.load(f)

    def _load_relations_data(self) -> Dict[str, Any]:
        """Load ACTUAL relations from RELATIONS.json"""
        if not self.relations_json.exists():
            return None
        with open(self.relations_json, 'r') as f:
            return json.load(f)

    def _load_api_data(self) -> Dict[str, Any]:
        """Load ACTUAL API methods from API.json"""
        if not self.api_json.exists():
            return None
        with open(self.api_json, 'r') as f:
            return json.load(f)

    def _load_demo_examples(self) -> str:
        """
        Load COMPLETE demo functions from demo files
        Show ENTIRE functions with FULL docstrings and ALL code
        """
        result = []

        # Read both demo files
        demo_files = [
            (self.data_root / "core" / "tests" / "demos" / "use_cases_demo.py", "stuff_on_table"),
            (self.data_root / "core" / "tests" / "demos" / "demo_1_ai_generated_scenes.py", "test_1_tower_stacking"),
            (self.data_root / "core" / "tests" / "demos" / "demo_1_ai_generated_scenes.py", "test_2_sorting_task")
        ]

        for demo_file, func_name in demo_files:
            if not demo_file.exists():
                continue

            content = demo_file.read_text()
            lines = content.split('\n')

            # Find and extract ENTIRE function
            in_function = False
            function_lines = []
            indent_level = None

            for i, line in enumerate(lines):
                if f'def {func_name}(' in line:
                    in_function = True
                    function_lines.append(line)
                    # Detect indent level (spaces before 'def')
                    indent_level = len(line) - len(line.lstrip())
                    continue

                if in_function:
                    # Check if we've reached the next function or end
                    if line.strip() and not line.startswith(' '):
                        # Reached module-level code
                        break
                    if line.strip().startswith('def ') and len(line) - len(line.lstrip()) == indent_level:
                        # Reached next function at same level
                        break

                    function_lines.append(line)

                    # Stop at reasonable length (first 80 lines of function)
                    if len(function_lines) > 80:
                        break

            if function_lines:
                result.append("\n" + "="*70)
                result.append(f"EXAMPLE: {func_name.replace('test_', '').replace('_', ' ').title()}")
                result.append("="*70)
                result.extend(function_lines)
                result.append("")

        return "\n".join(result)

    def _generate_header(self, agent_id: str, purpose: str) -> str:
        """Generate agent introduction"""
        agent_name = agent_id.upper().replace('_', ' ')
        return f"""=== {agent_name} AGENT ===

You {purpose}."""

    def _generate_knowledge_base(self, assets_data, behaviors_data, relations_data, api_data) -> str:
        """
        Generate RICH knowledge base from ACTUAL DATA
        Shows what's really available, not just boring descriptions!
        """
        sections = ["YOUR KNOWLEDGE BASE:"]

        # 1. ASSETS - Show what's actually available with behaviors
        if assets_data:
            sections.append("\n📦 AVAILABLE ASSETS:")
            # Categorize by 'type' field
            furniture = {k: v for k, v in assets_data.items() if v.get('type') == 'furniture'}
            objects = {k: v for k, v in assets_data.items() if v.get('type') == 'object'}

            # Show ALL furniture with behaviors
            sections.append(f"\n  🪑 FURNITURE ({len(furniture)} types):")
            for asset_name, asset_data in furniture.items():
                # Extract behaviors from components
                behaviors = set()
                for comp_name, comp_data in asset_data.get('components', {}).items():
                    for prop_name, prop_data in comp_data.items():
                        if isinstance(prop_data, dict) and 'behavior' in prop_data:
                            behaviors.add(prop_data['behavior'])
                behaviors_str = ', '.join(sorted(behaviors)) if behaviors else 'none'
                sections.append(f"    • {asset_name}: [{behaviors_str}]")

            # Show ALL objects with behaviors
            sections.append(f"\n  🍎 OBJECTS ({len(objects)} types):")
            for asset_name, asset_data in objects.items():
                # Extract behaviors from components
                behaviors = set()
                for comp_name, comp_data in asset_data.get('components', {}).items():
                    for prop_name, prop_data in comp_data.items():
                        if isinstance(prop_data, dict) and 'behavior' in prop_data:
                            behaviors.add(prop_data['behavior'])
                behaviors_str = ', '.join(sorted(behaviors)) if behaviors else 'none'
                sections.append(f"    • {asset_name}: [{behaviors_str}]")

        # 2. BEHAVIORS - Show ALL behaviors with descriptions
        if behaviors_data:
            sections.append(f"\n🎯 REWARD BEHAVIORS ({len(behaviors_data)} total):")
            # Skip _unit_types (metadata, not a real behavior)
            for behavior_name, behavior_info in behaviors_data.items():
                if behavior_name == '_unit_types':
                    continue
                desc = behavior_info.get('description', '').split('.')[0]
                # Also show example properties
                properties = behavior_info.get('properties', {})
                prop_names = list(properties.keys())[:3]  # Show first 3 properties
                prop_str = f" → Properties: {', '.join(prop_names)}" if prop_names else ""
                sections.append(f"  • {behavior_name}: {desc}{prop_str}")

        # 3. RELATIONS - Show ALL spatial placement options
        if relations_data:
            sections.append(f"\n📍 SPATIAL RELATIONS ({len(relations_data)} total):")
            for rel_name, rel_info in relations_data.items():
                desc = rel_info.get('description', '')
                # Show parameters if available
                params = rel_info.get('parameters', [])
                param_str = f" (params: {', '.join(params)})" if params else ""
                sections.append(f"  • {rel_name}: {desc}{param_str}")

        # 4. API - Show ALL methods with signatures
        if api_data:
            sections.append(f"\n🔧 EXPERIMENTOPS API ({len(api_data)} methods):")
            sections.append("\n  💡 NEW Beautiful API: ops.assets.apple.position (Pydantic modals!)\n")

            # Group methods by category for better readability
            scene_methods = ['create_scene', 'compile']
            asset_methods = [m for m in api_data.keys() if 'asset' in m.lower()]
            reward_methods = [m for m in api_data.keys() if 'reward' in m.lower()]
            robot_methods = [m for m in api_data.keys() if 'robot' in m.lower()]
            other_methods = [m for m in api_data.keys() if m not in scene_methods + asset_methods + reward_methods + robot_methods]

            if scene_methods:
                sections.append("  📋 Scene Methods:")
                for method in scene_methods:
                    if method in api_data:
                        sig = api_data[method].get('signature', '')
                        sections.append(f"    • {sig}")

            if asset_methods:
                sections.append("\n  📦 Asset Methods:")
                for method in asset_methods:
                    sig = api_data[method].get('signature', '')
                    sections.append(f"    • {sig}")

            if reward_methods:
                sections.append("\n  🎯 Reward Methods:")
                for method in reward_methods:
                    sig = api_data[method].get('signature', '')
                    sections.append(f"    • {sig}")

            if robot_methods:
                sections.append("\n  🤖 Robot Methods:")
                for method in robot_methods:
                    sig = api_data[method].get('signature', '')
                    sections.append(f"    • {sig}")

            if other_methods:
                sections.append("\n  🔧 Other Methods:")
                for method in other_methods[:10]:  # Limit other methods to 10
                    sig = api_data[method].get('signature', '')
                    sections.append(f"    • {sig}")

        return "\n".join(sections)

    def _generate_workflow(self, tools: List[Dict]) -> str:
        """
        Generate workflow steps from tool metadata
        Uses workflow_hints to determine order and when_to_use
        """
        lines = ["YOUR WORKFLOW:"]
        lines.append("\nYou have full knowledge above. Follow these steps:")

        # Sort tools by workflow stage and order
        ordered_tools = self._sort_by_workflow(tools)

        step_num = 1
        for tool in ordered_tools:
            hints = tool.get("workflow_hints", {})
            when_to_use = hints.get("when_to_use", "")
            tool_id = tool.get("tool_id", "")

            if when_to_use and "discover" in tool_id or "get_" in tool_id or "create_" in tool_id:
                lines.append(f"  {step_num}. {when_to_use} (use {tool_id})")
                step_num += 1

        return "\n".join(lines)

    def _generate_rules(self, tools: List[Dict]) -> str:
        """
        Extract constraints and best_practices from tools
        """
        all_constraints = []
        all_practices = []

        for tool in tools:
            constraints = tool.get("constraints", [])
            practices = tool.get("best_practices", [])
            all_constraints.extend(constraints)
            all_practices.extend(practices)

        if not all_constraints and not all_practices:
            return ""

        lines = ["\nIMPORTANT RULES:"]

        # Add best practices first (positive guidance)
        if all_practices:
            lines.append("\n✅ Best Practices:")
            for practice in all_practices:
                lines.append(f"  • {practice}")

        # Add constraints (restrictions)
        if all_constraints:
            lines.append("\n⚠️  Constraints:")
            for constraint in all_constraints:
                lines.append(f"  • {constraint}")

        lines.append("\nGood luck! 🚀")
        return "\n".join(lines)

    def _sort_by_workflow(self, tools: List[Dict]) -> List[Dict]:
        """
        Sort tools by workflow stage and order hints

        Stage priority: discovery → planning → action → validation → delivery
        Within stage: sort by order number
        """
        def sort_key(tool):
            hints = tool.get("workflow_hints", {})

            # Stage order
            stage_order = {
                "discovery": 1,
                "planning": 2,
                "action": 3,
                "validation": 4,
                "delivery": 5
            }
            stage = hints.get("stage", "action")
            stage_priority = stage_order.get(stage, 3)

            # Order within stage
            order = hints.get("order", 50)

            return (stage_priority, order)

        return sorted(tools, key=sort_key)


def generate_agent_prompt(agent_id: str,
                          tool_ids: List[str],
                          purpose: str,
                          **kwargs) -> str:
    """
    Generate agent prompt from tool configs

    Convenience function for prompt generation.

    Args:
        agent_id: Agent identifier (e.g., "scene_maker")
        tool_ids: List of tool IDs this agent has access to
        purpose: High-level goal (e.g., "create simulation scenes")
        **kwargs: Additional customization options

    Returns:
        Complete system prompt string

    Example:
        prompt = generate_agent_prompt(
            agent_id="scene_maker",
            tool_ids=["discover_assets", "discover_behaviors", ...],
            purpose="create Python scripts for robot simulation scenes using ExperimentOps API"
        )
    """
    maker = PromptMaker()
    return maker.generate_prompt(agent_id, tool_ids, purpose, **kwargs)

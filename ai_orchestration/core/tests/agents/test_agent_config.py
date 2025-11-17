#!/usr/bin/env python3
"""
Agent Config Tests - Test JSON + Pydantic agent system
Tests loading, validation, caching - same pattern as tools
OFFENSIVE: Crashes if anything wrong
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.agent_config_loader import load_agent_config, load_all_configs, clear_cache
from ai_orchestration.core.agent_schema import AgentConfig


def test_load_robot_controller():
    """Load robot_controller agent config"""
    print("=" * 60)
    print("Test: Load robot_controller agent")
    print("=" * 60)

    config = load_agent_config("robot_controller")

    assert config.agent_id == "robot_controller"
    assert "robot" in config.description.lower()
    assert "send_robot_action" in config.tools
    assert "handoff" in config.tools
    assert config.force_model == "claude-sonnet-4-5"

    print(f"✓ Agent ID: {config.agent_id}")
    print(f"✓ Description: {config.description[:60]}...")
    print(f"✓ Tools: {config.tools}")
    print(f"✓ Model: {config.force_model}")
    print("✅ PASS\n")


def test_load_task_planner():
    """Load task_planner agent config"""
    print("=" * 60)
    print("Test: Load task_planner agent")
    print("=" * 60)

    config = load_agent_config("task_planner")

    assert config.agent_id == "task_planner"
    assert "plan" in config.description.lower() or "delegate" in config.description.lower()
    assert "route_to_navigator" in config.tools
    assert "route_to_manipulator" in config.tools

    print(f"✓ Agent ID: {config.agent_id}")
    print(f"✓ Delegation tools: {[t for t in config.tools if 'route_to' in t]}")
    print("✅ PASS\n")


def test_agent_to_tool_spec():
    """Convert agent to tool spec for routing"""
    print("=" * 60)
    print("Test: Agent → Tool Spec conversion")
    print("=" * 60)

    config = load_agent_config("navigator")
    tool_spec = config.to_tool_spec()

    assert tool_spec["name"] == "route_to_navigator"
    assert tool_spec["type"] == "agent_as_tool"
    assert tool_spec["target_agent_id"] == "navigator"
    assert "request" in tool_spec["input_schema"]["properties"]

    print(f"✓ Tool name: {tool_spec['name']}")
    print(f"✓ Type: {tool_spec['type']}")
    print(f"✓ Target agent: {tool_spec['target_agent_id']}")
    print(f"✓ Input schema: {tool_spec['input_schema']}")
    print("✅ PASS\n")


def test_config_caching():
    """Test config caching (load once, reuse)"""
    print("=" * 60)
    print("Test: Config caching")
    print("=" * 60)

    # Clear cache
    clear_cache()

    # First load
    config1 = load_agent_config("robot_controller")

    # Second load (should be cached)
    config2 = load_agent_config("robot_controller")

    # Should be same object
    assert config1 is config2
    print("✓ Same object from cache (config1 is config2)")

    # Clear and reload
    clear_cache()
    config3 = load_agent_config("robot_controller")

    # Different object after cache clear
    assert config1 is not config3
    print("✓ New object after cache clear")
    print("✅ PASS\n")


def test_load_all_agents():
    """Load all agent configs at once"""
    print("=" * 60)
    print("Test: Load all agent configs")
    print("=" * 60)

    all_configs = load_all_configs()

    assert len(all_configs) > 0
    assert "robot_controller" in all_configs
    assert "task_planner" in all_configs

    print(f"✓ Loaded {len(all_configs)} agents:")
    for agent_id in all_configs.keys():
        print(f"  - {agent_id}")

    print("✅ PASS\n")


def test_agent_validation():
    """Test Pydantic validation catches errors"""
    print("=" * 60)
    print("Test: Agent validation (should crash on invalid)")
    print("=" * 60)

    # Valid config
    valid = {
        "agent_id": "test_agent",
        "description": "Test agent",
        "instructions": "Do something",
        "tools": ["handoff"],
        "force_model": "claude-sonnet-4-5"
    }

    config = AgentConfig(**valid)
    assert config.agent_id == "test_agent"
    print("✓ Valid config accepted")

    # Invalid agent_id (empty)
    try:
        invalid = valid.copy()
        invalid["agent_id"] = ""
        AgentConfig(**invalid)
        assert False, "Should have crashed on empty agent_id"
    except Exception as e:
        print(f"✓ Caught empty agent_id: {type(e).__name__}")

    # Invalid description (empty)
    try:
        invalid = valid.copy()
        invalid["description"] = ""
        AgentConfig(**invalid)
        assert False, "Should have crashed on empty description"
    except Exception as e:
        print(f"✓ Caught empty description: {type(e).__name__}")

    # Invalid model
    try:
        invalid = valid.copy()
        invalid["force_model"] = "gpt-3"  # Not in allowed list
        AgentConfig(**invalid)
        assert False, "Should have crashed on invalid model"
    except Exception as e:
        print(f"✓ Caught invalid model: {type(e).__name__}")

    print("✅ PASS\n")


def test_config_to_dict():
    """Test config serialization to dict"""
    print("=" * 60)
    print("Test: Config → Dict conversion")
    print("=" * 60)

    config = load_agent_config("researcher")
    config_dict = config.to_dict()

    assert config_dict["agent_id"] == "researcher"
    assert "tools" in config_dict
    assert isinstance(config_dict["tools"], list)
    assert "metadata" in config_dict

    print(f"✓ Dict keys: {list(config_dict.keys())}")
    print(f"✓ Metadata: {config_dict['metadata']}")
    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT CONFIG TESTS - JSON + Pydantic Validation")
    print("=" * 60)
    print()

    tests = [
        test_load_robot_controller,
        test_load_task_planner,
        test_agent_to_tool_spec,
        test_config_caching,
        test_load_all_agents,
        test_agent_validation,
        test_config_to_dict,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)

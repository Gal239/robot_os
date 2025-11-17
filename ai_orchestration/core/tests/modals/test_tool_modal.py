#!/usr/bin/env python3
"""
ToolModal Tests - Tool registry, schemas, ToolType enum
Tests tool registration, schemas, TOOL_BEHAVIOR specs
"""

import sys
sys.path.insert(0, '/home/gal-labs/PycharmProjects/echo_robot')

from ai_orchestration.core.modals import ToolModal, ToolEntry, ToolType, TOOL_BEHAVIOR


def test_tool_type_enum_values():
    """ToolType enum should have all expected values"""
    print("=" * 60)
    print("Test: ToolType Enum Values")
    print("=" * 60)

    # Check all enum values
    assert ToolType.ROOT == "root"
    assert ToolType.FUNCTION_TOOL == "function_tool"
    assert ToolType.HANDOFF == "handoff"
    assert ToolType.AGENT_AS_TOOL == "agent_as_tool"
    assert ToolType.ASK_MASTER == "ask_master"
    print("✓ All ToolType enum values correct")

    # Check enum is str-based
    assert isinstance(ToolType.ROOT.value, str)
    print("✓ ToolType is str-based enum")

    print("✅ PASS\n")


def test_tool_behavior_specifications():
    """TOOL_BEHAVIOR should define behavior for each tool type"""
    print("=" * 60)
    print("Test: TOOL_BEHAVIOR Specifications")
    print("=" * 60)

    # Check all types have behavior
    assert ToolType.ROOT in TOOL_BEHAVIOR
    assert ToolType.FUNCTION_TOOL in TOOL_BEHAVIOR
    assert ToolType.HANDOFF in TOOL_BEHAVIOR
    assert ToolType.AGENT_AS_TOOL in TOOL_BEHAVIOR
    assert ToolType.ASK_MASTER in TOOL_BEHAVIOR
    assert ToolType.NON_FUNCTION_TOOL in TOOL_BEHAVIOR
    print("✓ All tool types have behavior specs")

    # Check ROOT behavior
    assert TOOL_BEHAVIOR[ToolType.ROOT]["creates_node"] is True
    assert TOOL_BEHAVIOR[ToolType.ROOT]["blocks_parent"] is False
    print("✓ ROOT: creates_node=True, blocks_parent=False")

    # Check FUNCTION_TOOL behavior
    assert TOOL_BEHAVIOR[ToolType.FUNCTION_TOOL]["creates_node"] is False
    assert TOOL_BEHAVIOR[ToolType.FUNCTION_TOOL]["executes_function"] is True
    print("✓ FUNCTION_TOOL: creates_node=False, executes_function=True")

    # Check HANDOFF behavior
    assert TOOL_BEHAVIOR[ToolType.HANDOFF]["creates_node"] is False
    assert TOOL_BEHAVIOR[ToolType.HANDOFF]["completes_task"] is True
    print("✓ HANDOFF: creates_node=False, completes_task=True")

    # Check AGENT_AS_TOOL behavior
    assert TOOL_BEHAVIOR[ToolType.AGENT_AS_TOOL]["creates_node"] is True
    assert TOOL_BEHAVIOR[ToolType.AGENT_AS_TOOL]["blocks_parent"] is True
    print("✓ AGENT_AS_TOOL: creates_node=True, blocks_parent=True")

    # Check ASK_MASTER behavior
    assert TOOL_BEHAVIOR[ToolType.ASK_MASTER]["creates_node"] is True
    assert TOOL_BEHAVIOR[ToolType.ASK_MASTER]["blocks_parent"] is True
    print("✓ ASK_MASTER: creates_node=True, blocks_parent=True")

    # Check NON_FUNCTION_TOOL behavior
    assert TOOL_BEHAVIOR[ToolType.NON_FUNCTION_TOOL]["creates_node"] is False
    assert TOOL_BEHAVIOR[ToolType.NON_FUNCTION_TOOL]["logs_input"] is True
    print("✓ NON_FUNCTION_TOOL: creates_node=False, logs_input=True")

    print("✅ PASS\n")


def test_tool_entry_structure():
    """ToolEntry should have name, type, input_schema, output_schema"""
    print("=" * 60)
    print("Test: ToolEntry Structure")
    print("=" * 60)

    # Create tool entry
    entry = ToolEntry(
        name="write_file",
        type=ToolType.FUNCTION_TOOL.value,
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            }
        },
        output_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"}
            }
        }
    )

    # Check fields
    assert entry.name == "write_file"
    assert entry.type == ToolType.FUNCTION_TOOL.value
    assert "properties" in entry.input_schema
    assert "properties" in entry.output_schema
    print("✓ ToolEntry has all required fields")

    print("✅ PASS\n")


def test_tool_modal_register_and_get():
    """ToolModal register_tool and get_tool"""
    print("=" * 60)
    print("Test: Register and Get Tool")
    print("=" * 60)

    modal = ToolModal()

    # Register tool
    tool = ToolEntry(
        name="ask_data",
        type=ToolType.FUNCTION_TOOL.value,
        input_schema={"type": "object", "properties": {"file_path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"answer": {"type": "string"}}}
    )
    modal.register_tool(tool)

    assert len(modal.tools) == 1
    print("✓ Tool registered")

    # Get tool
    retrieved = modal.get_tool("ask_data")
    assert retrieved is not None
    assert retrieved.name == "ask_data"
    assert retrieved.type == ToolType.FUNCTION_TOOL.value
    print("✓ Tool retrieved")

    # Get non-existent
    missing = modal.get_tool("missing_tool")
    assert missing is None
    print("✓ Missing tool returns None")

    print("✅ PASS\n")


def test_tool_modal_multiple_tools():
    """ToolModal should handle multiple tools"""
    print("=" * 60)
    print("Test: Multiple Tools")
    print("=" * 60)

    modal = ToolModal()

    # Register multiple tools
    tools = [
        ToolEntry("write_file", ToolType.FUNCTION_TOOL.value, {}, {}),
        ToolEntry("ask_data", ToolType.FUNCTION_TOOL.value, {}, {}),
        ToolEntry("load_to_context", ToolType.FUNCTION_TOOL.value, {}, {}),
    ]

    for tool in tools:
        modal.register_tool(tool)

    assert len(modal.tools) == 3
    print(f"✓ Registered {len(modal.tools)} tools")

    # Get each tool
    for tool in tools:
        retrieved = modal.get_tool(tool.name)
        assert retrieved is not None
        assert retrieved.name == tool.name
    print("✓ All tools retrievable")

    print("✅ PASS\n")


def test_tool_modal_render_for_json():
    """ToolModal render_for_json serialization"""
    print("=" * 60)
    print("Test: Render for JSON")
    print("=" * 60)

    modal = ToolModal()

    # Register tool
    tool = ToolEntry(
        name="write_file",
        type=ToolType.FUNCTION_TOOL.value,
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"path": {"type": "string"}}}
    )
    modal.register_tool(tool)

    # Render
    data = modal.render_for_json()

    assert "tools" in data
    assert "write_file" in data["tools"]
    print("✓ JSON has tools dict")

    # Check tool structure
    tool_data = data["tools"]["write_file"]
    assert tool_data["name"] == "write_file"
    assert tool_data["type"] == ToolType.FUNCTION_TOOL.value
    assert "input_schema" in tool_data
    assert "output_schema" in tool_data
    print("✓ Tool has all fields in JSON")

    print("✅ PASS\n")


def test_tool_modal_overwrite_tool():
    """Registering same tool name should overwrite"""
    print("=" * 60)
    print("Test: Overwrite Tool")
    print("=" * 60)

    modal = ToolModal()

    # Register tool
    tool1 = ToolEntry("test_tool", ToolType.FUNCTION_TOOL.value, {"v": 1}, {})
    modal.register_tool(tool1)

    # Register same name again
    tool2 = ToolEntry("test_tool", ToolType.FUNCTION_TOOL.value, {"v": 2}, {})
    modal.register_tool(tool2)

    # Should have only 1 tool (overwritten)
    assert len(modal.tools) == 1
    retrieved = modal.get_tool("test_tool")
    assert retrieved.input_schema["v"] == 2
    print("✓ Tool overwritten (offensive mode)")

    print("✅ PASS\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TOOL MODAL TESTS")
    print("=" * 60)
    print()

    tests = [
        test_tool_type_enum_values,
        test_tool_behavior_specifications,
        test_tool_entry_structure,
        test_tool_modal_register_and_get,
        test_tool_modal_multiple_tools,
        test_tool_modal_render_for_json,
        test_tool_modal_overwrite_tool
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

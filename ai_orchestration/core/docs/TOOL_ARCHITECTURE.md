# Tool Architecture

**Offensive system**: JSON-config based tools with Pydantic validation + Document-based communication. Crashes if invalid.

---

## What Are Tools?

Tools = functions agents can call during execution. 6 types, each with different behavior:

1. **FUNCTION_TOOL** - Execute code (write_file, ask_data, search_web)
2. **NON_FUNCTION_TOOL** - Log thinking/planning (stop_and_think, plan_next_steps, save_to_*_memory)
3. **HANDOFF** - Complete task, return to parent **+ documents**
4. **AGENT_AS_TOOL** - Delegate to another agent (blocks until done) **+ documents**
5. **ASK_MASTER** - Ask parent agent/human (blocks until answered) **+ documents**
6. **ROOT** - Entry point (human-initiated task)

---

## What Are Documents?

Documents = **Document modals** stored in shared workspace with full metadata:

```python
Document(
    path="report.pdf",
    content_json={"blocks": {...}},      # Universal block structure
    created_by="task_5",                 # Which task created it
    created_at="2025-10-04T12:34:56",   # When created
    mime_type="application/pdf",         # File type
    size_bytes=15420,                    # Size
    token_count=3400,                    # Tokens
    auto_summary="Analysis of...",       # Auto-generated summary
    description="Final report",          # User description
    tags=["analysis", "Q3"],            # Searchable tags
    updated_by="task_7",                # Last modifier
    versions=[...]                       # Version history
)
```

**Key**: Documents are NOT just files - they're **rich data structures** with metadata agents can query/filter/search.

---

## Architecture: Python + JSON Separation

**Python files** = Clean logic only
```python
@tool(execution_type="function")
def write_file(workspace: WorkspaceModal, task_id: str, path: str, content: str, mime_type: str, why_created: str = "") -> Dict:
    """write_file"""  # Docstring = tool_id ONLY
    # Implementation...
    return {"path": path, "size_bytes": size}
```

**JSON files** = All metadata (validated by Pydantic)
```json
{
  "tool_id": "write_file",
  "description": "Create or overwrite file with auto-summarization.",
  "type": "function_tool",
  "parameters": {
    "path": {"type": "string", "description": "File path", "required": true},
    "content": {"type": "string", "description": "File content", "required": true},
    "mime_type": {"type": "string", "description": "MIME type", "required": true}
  },
  "returns": {
    "path": "string",
    "size_bytes": "integer",
    "summary": "string"
  }
}
```

**Why?**
- Composable - modify schemas without touching code
- Validated - Pydantic crashes if invalid
- Auto-discoverable - drop .py file → auto-registers
- Auto-documented - generate_docs.py creates TOOLS.md from JSONs

---

## ToolType Behaviors

Each ToolType has specific behavior defined in `TOOL_BEHAVIOR`:

| ToolType          | Creates Node | Blocks Parent | Executes Code | Logs Input   | Passes Documents | Agent Action                   |
|-------------------|--------------|---------------|---------------|--------------|------------------|--------------------------------|
| FUNCTION_TOOL     | No           | No            | Yes           | Input+Result | No               | Continue (with result)         |
| NON_FUNCTION_TOOL | No           | No            | No            | Input only   | No               | Continue (instant)             |
| HANDOFF           | No           | No            | No            | Input+Docs   | **YES**          | STOP + inject docs to parent   |
| AGENT_AS_TOOL     | Yes          | Yes           | No            | Input+Status | **YES**          | BLOCK → Continue (with docs)   |
| ASK_MASTER        | Yes          | Yes           | No            | Input+Answer | **YES**          | BLOCK → Continue (with docs)   |
| ROOT              | Yes          | No            | No            | Task payload | No               | Entry point                    |

**Creates Node** - Creates new task in graph (spawns agent)
**Blocks Parent** - Parent waits until child completes
**Executes Code** - Calls Python function
**Logs Input** - Records tool call in timeline
**Passes Documents** - Can attach documents from workspace
**Agent Action** - What happens to agent execution flow

### Document Flow
- **Orchestration tools** (handoff, ask_master, agent_as_tool) accept optional `documents` parameter (array of workspace paths)
- Documents are **auto-loaded** as content blocks in receiving agent's context
- Child documents are **auto-injected** to parent when child completes
- All agents share **same workspace** - documents persist across tasks

---

## How Tools Work: Agent Loop

```python
# Agent loop (simplified)
while True:
    messages = graph.render_for_llm(task_id)  # Get messages
    response = llm.generate(messages, tools)   # LLM picks tool

    if response.stop_reason == "end_turn":
        break  # No more tools

    for tool_use in response.content:
        result = graph.handle_tool_call(
            task_id=task_id,
            tool_name=tool_use.name,
            tool_type=get_tool_type(tool_use.name),
            tool_input=tool_use.input,
            tool_ops=tool_ops,
            graph_ops=graph
        )

        if result["action"] == "return":
            return result["value"]  # STOP (handoff/delegation)
        # else: continue loop
```

**Key**: `handle_tool_call` routes based on `TOOL_BEHAVIOR`:

```python
def handle_tool_call(self, task_id, tool_name, tool_type, tool_input, tool_ops, graph_ops):
    behavior = TOOL_BEHAVIOR[tool_type]

    # FUNCTION_TOOL - execute and continue
    if behavior.get("executes_function"):
        result = tool_ops.execute(tool_name, tool_input, task_id)
        self.add_to_timeline(task_id, ToolType.FUNCTION_TOOL, tool_name, tool_input, result)
        return {"action": "continue"}

    # NON_FUNCTION_TOOL - log and continue (instant)
    if behavior.get("logs_input"):
        self.add_to_timeline(task_id, ToolType.NON_FUNCTION_TOOL, tool_name, tool_input, {"logged": True})
        return {"action": "continue"}

    # HANDOFF - complete task, STOP
    if behavior.get("completes_task"):
        self.mark_node_completed(task_id, tool_input)
        return {"action": "return", "value": tool_input}

    # AGENT_AS_TOOL - create child node, BLOCK parent
    if tool_type == ToolType.AGENT_AS_TOOL:
        child_id = graph_ops.create_node(...)  # Auto-blocks parent
        self.add_to_timeline(task_id, ToolType.AGENT_AS_TOOL, tool_name, tool_input, f"→ delegated (task {child_id})")
        return {"action": "return", "value": None}  # Parent BLOCKS

    # ASK_MASTER - create question node, BLOCK parent
    if tool_type == ToolType.ASK_MASTER:
        question_id = graph_ops.create_node(...)  # Auto-blocks parent
        self.add_to_timeline(task_id, ToolType.ASK_MASTER, tool_name, tool_input, f"→ asked (task {question_id})")
        return {"action": "return", "value": None}  # Parent BLOCKS
```

---

## How to Add a Tool

**3 steps:**

1. **Create JSON config** in `/tools/configs/{tool_id}.json`
```json
{
  "tool_id": "my_tool",
  "description": "What it does",
  "type": "function_tool",
  "parameters": {
    "param1": {"type": "string", "description": "...", "required": true}
  },
  "returns": {"result": "string"}
}
```

2. **Create Python function** in `/tools/{category}_tools.py`
```python
@tool(execution_type="function")
def my_tool(workspace: WorkspaceModal, task_id: str, param1: str) -> Dict:
    """my_tool"""  # Docstring = tool_id ONLY
    # Implementation
    return {"result": "value"}
```

3. **Regenerate docs** (optional)
```bash
python3 ai_orchestration/core/tools/generate_docs.py
```

**That's it.** Tool auto-registers on import.

---

## File Structure

```
ai_orchestration/core/
├── tool_schema.py              # Pydantic models (ToolParameter, ToolConfig)
├── tool_config_loader.py       # JSON loader with caching
├── tool_decorator.py           # @tool decorator (loads JSON)
├── modals/
│   ├── tool_modal.py           # ToolType enum, TOOL_BEHAVIOR specs
│   └── task_graph_modal.py     # handle_tool_call implementation
├── tools/
│   ├── __init__.py             # Auto-imports all .py files
│   ├── file_tools.py           # write_file, load_to_context, etc.
│   ├── llm_tools.py            # ask_gpt, ask_claude, ask_data, etc.
│   ├── configs/
│   │   ├── write_file.json
│   │   ├── ask_data.json
│   │   ├── handoff.json
│   │   ├── stop_and_think.json
│   │   └── ... (14 total)
│   └── generate_docs.py        # Auto-generates TOOLS.md
└── docs/
    ├── TOOLS.md                # Auto-generated (DO NOT EDIT)
    └── TOOL_ARCHITECTURE.md    # This file
```

---

## How Logging Works: Timeline

All tool calls logged to **timeline** (single source of truth):

```python
# Timeline entry structure
{
    "type": "function_tool",           # ToolType enum value
    "tool": "write_file",              # Tool name
    "input": {"path": "...", ...},     # Tool input
    "result": {"path": "...", ...},    # Tool output
    "timestamp": "2025-10-04T12:34:56" # ISO format
}
```

**Different logging per ToolType:**

- **FUNCTION_TOOL** - Logs input + result (code output)
- **NON_FUNCTION_TOOL** - Logs input only (thinking/planning)
- **HANDOFF** - Logs input (final result)
- **AGENT_AS_TOOL** - Logs input + status ("→ delegated to X"), later injects result ("← result from X")
- **ASK_MASTER** - Logs input + status ("→ asked X"), later injects answer ("← answer from X")

Timeline reconstructs full execution history. Used by `render_for_llm` to build messages.

---

## How to Change/Modify

### Change tool behavior
Edit JSON config in `/tools/configs/{tool_id}.json`, regenerate docs.

### Add new parameter
Add to JSON `parameters`, add to Python function signature.

### Change ToolType behavior
Edit `TOOL_BEHAVIOR` in `tool_modal.py`, update `handle_tool_call` in `task_graph_modal.py`.

### Add new ToolType
1. Add to `ToolType` enum
2. Add to `TOOL_BEHAVIOR` dict
3. Add handler in `handle_tool_call`
4. Update validator in `tool_schema.py`

### Modify logging format
Edit `add_log_event` in `task_graph_modal.py:TaskNode`.

---

## Examples

### FUNCTION_TOOL (Text-only)
```python
# Agent creates file
write_file(path="report.pdf", content="...", mime_type="application/pdf")
# → Executes Python → logs input+result → continues
```

### NON_FUNCTION_TOOL (Metacognition)
```python
# Agent thinks
stop_and_think(thoughts="I should analyze the data before proceeding...")
# → Logs input only → continues instantly (no execution)
```

### HANDOFF (With Documents)
```python
# Agent creates files
write_file(path="report.pdf", content="...")
write_file(path="data.csv", content="...")

# Agent completes with documents
handoff(
    result={"status": "Analysis complete", "findings": 3},
    documents=["report.pdf", "data.csv"]
)
# → Logs input+docs → task completes → STOPS
# → Parent receives: result + documents auto-loaded as content blocks
```

### AGENT_AS_TOOL (Document-Based Delegation)
```python
# Agent creates requirements
write_file(path="requirements.md", content="Build a dashboard with...")

# Agent delegates with documents
route_to_developer(
    request="Build this feature",
    documents=["requirements.md"]
)
# → Creates child task → blocks parent
# → Child auto-loads requirements.md as content block
# → Child executes, creates output.py
# → Child completes with handoff(documents=["output.py"])
# → Parent unblocks, gets output.py auto-loaded
# → Parent continues
```

### ASK_MASTER (Document-Based Question)
```python
# Agent creates draft
write_file(path="draft_report.md", content="...")

# Agent asks with document
ask_master(
    question="Is this analysis correct?",
    documents=["draft_report.md"]
)
# → Creates question task → blocks self
# → Master auto-loads draft_report.md
# → Master answers (as text)
# → Answer injected to child
# → Child unblocks → continues
```

### Complete Flow Example
```python
# ROOT AGENT (task_0)
route_to_analyst(
    request="Analyze Q3 sales",
    documents=["sales_raw.csv"]  # Attach existing data
)
# → BLOCKS, spawns task_1

# ANALYST AGENT (task_1) - receives sales_raw.csv in context
write_file(path="analysis.md", content="Q3 sales up 15%...")
route_to_visualizer(
    request="Create charts",
    documents=["sales_raw.csv", "analysis.md"]  # Pass both files
)
# → BLOCKS, spawns task_2

# VISUALIZER AGENT (task_2) - receives both files in context
write_file(path="charts.png", content=...)
handoff(
    result={"charts_created": 1},
    documents=["charts.png"]
)
# → task_2 COMPLETES
# → charts.png injected to task_1 (analyst)
# → task_1 UNBLOCKS

# ANALYST AGENT (task_1) - now has charts.png in context
handoff(
    result={"analysis": "complete"},
    documents=["analysis.md", "charts.png"]
)
# → task_1 COMPLETES
# → analysis.md + charts.png injected to task_0 (root)
# → task_0 UNBLOCKS

# ROOT AGENT (task_0) - now has all documents
# Can review analysis.md and charts.png
```

---

## Key Principles

1. **Offensive** - Crash if invalid JSON, no defensive checks
2. **Separation** - Python = logic, JSON = metadata
3. **Modal-driven** - State reacts to changes (auto-block/unblock)
4. **Timeline = Truth** - All events in ordered log
5. **Type-driven behavior** - ToolType determines execution flow
6. **Auto-everything** - Discovery, validation, documentation

---

**Generated from**: JSON configs in `/tools/configs/*.json`
**Auto-docs**: Run `python3 tools/generate_docs.py` to regenerate TOOLS.md

# Tool Catalog (Auto-Generated)

**DO NOT EDIT MANUALLY** - Generated from `/tools/configs/*.json`

Run `python3 tools/generate_docs.py` to regenerate.

---

## Function Tools

### ask_claude

**Description**: Ask Claude a question. Use this when you need Claude's perspective or want to cross-check reasoning.

**Type**: `function_tool`

**Parameters**:

- `question` (string, **required**): The question to ask Claude
- `context` (string, optional (default: ``)): Optional context for the question
- `model` (string, optional (default: `claude-sonnet-4-5-20250929`)): Claude model to use (default: claude-sonnet-4-5-20250929)
- `max_tokens` (integer, optional (default: `4000`)): Max tokens in response

**Returns**:

```json
{
  "answer": "string",
  "model": "string",
  "error": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The question to ask Claude"
    },
    "context": {
      "type": "string",
      "description": "Optional context for the question"
    },
    "model": {
      "type": "string",
      "description": "Claude model to use (default: claude-sonnet-4-5-20250929)"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Max tokens in response"
    }
  },
  "required": [
    "question"
  ]
}
```

</details>

---

### ask_data

**Description**: Ask questions about files to extract data, summarize, analyze, or get structure. BE CREATIVE - you can extract specific data (prices, names, dates), get sections (conclusion, intro), summarize (50-page PDF in 3 bullets), convert formats (Excel to JSON), analyze structure (function names in code), or ask anything. Both your QUESTION and LLM's ANSWER load to context (efficient - not full file). Works with images (vision), PDFs, Excel, code, text. Supports multiple files for comparison.

**Type**: `function_tool`

**Parameters**:

- `file_path` (string, optional): Single file path (for backward compatibility)
- `file_paths` (array, optional): List of file paths (for multi-file queries)
- `question` (string, optional (default: ``)): Question about the file(s) - be creative, extract anything
- `model` (string, optional (default: `claude-sonnet-4-5-20250929`)): LLM model to use (should support vision for images)
- `max_tokens` (integer, optional (default: `4000`)): Max tokens in response

**Returns**:

```json
{
  "answer": "string",
  "file_paths": "array",
  "file_types": "array",
  "file_path": "string",
  "file_type": "string",
  "error": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "file_path": {
      "type": "string",
      "description": "Single file path (for backward compatibility)"
    },
    "file_paths": {
      "type": "array",
      "description": "List of file paths (for multi-file queries)"
    },
    "question": {
      "type": "string",
      "description": "Question about the file(s) - be creative, extract anything"
    },
    "model": {
      "type": "string",
      "description": "LLM model to use (should support vision for images)"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Max tokens in response"
    }
  }
}
```

</details>

---

### ask_gpt

**Description**: Ask GPT a question. Use this when you need GPT's perspective or want to cross-check with another model.

**Type**: `function_tool`

**Parameters**:

- `question` (string, **required**): The question to ask GPT
- `context` (string, optional (default: ``)): Optional context for the question
- `model` (string, optional (default: `gpt-5`)): GPT model to use (default: gpt-5)
- `max_tokens` (integer, optional (default: `4000`)): Max tokens in response

**Returns**:

```json
{
  "answer": "string",
  "model": "string",
  "error": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The question to ask GPT"
    },
    "context": {
      "type": "string",
      "description": "Optional context for the question"
    },
    "model": {
      "type": "string",
      "description": "GPT model to use (default: gpt-5)"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Max tokens in response"
    }
  },
  "required": [
    "question"
  ]
}
```

</details>

---

### edit_file_block

**Description**: Edit file blocks with multiple operations (update, add, replace, delete).

**Type**: `function_tool`

**Parameters**:

- `path` (string, **required**): File path in workspace
- `operation` (string, **required**): Operation to perform: update, add, replace, or delete
- `block_id` (string, optional): Block ID to edit/delete/replace (required for update/delete/replace)
- `block_type` (string, optional): Block type (required for add)
- `new_data` (object, optional): New data for block (required for update/add/replace)

**Returns**:

```json
{
  "success": "boolean",
  "operation": "string",
  "block_id": "string",
  "error": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "File path in workspace"
    },
    "operation": {
      "type": "string",
      "description": "Operation to perform: update, add, replace, or delete"
    },
    "block_id": {
      "type": "string",
      "description": "Block ID to edit/delete/replace (required for update/delete/replace)"
    },
    "block_type": {
      "type": "string",
      "description": "Block type (required for add)"
    },
    "new_data": {
      "type": "object",
      "description": "New data for block (required for update/add/replace)"
    }
  },
  "required": [
    "path",
    "operation"
  ]
}
```

</details>

---

### list_files

**Description**: List files in workspace with optional glob pattern.

**Type**: `function_tool`

**Parameters**:

- `pattern` (string, optional (default: `*`)): Glob pattern (e.g., *.md for markdown files)

**Returns**:

```json
{
  "files": "array"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Glob pattern (e.g., *.md for markdown files)"
    }
  }
}
```

</details>

---

### load_to_context

**Description**: Load FULL file into context permanently. Wasteful - only use for files you need always available. For extracting info or querying files, use ask_data instead (more efficient).

**Type**: `function_tool`

**Parameters**:

- `path` (string, **required**): File path in workspace

**Returns**:

```json
{
  "type": "string",
  "error": "string",
  "path": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "File path in workspace"
    }
  },
  "required": [
    "path"
  ]
}
```

</details>

---

### search_web

**Description**: Search the web and get summarized results. Use this to find current information, facts, or research a topic.

**Type**: `function_tool`

**Parameters**:

- `query` (string, **required**): Search query
- `use_llm` (boolean, optional (default: `True`)): If True, use LLM to summarize results (default: true)
- `model` (string, optional (default: `claude-sonnet-4-5-20250929`)): LLM model for summarization
- `max_tokens` (integer, optional (default: `4000`)): Max tokens in response

**Returns**:

```json
{
  "results": "string",
  "query": "string",
  "summarized": "boolean",
  "model": "string",
  "error": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Search query"
    },
    "use_llm": {
      "type": "boolean",
      "description": "If True, use LLM to summarize results (default: true)"
    },
    "model": {
      "type": "string",
      "description": "LLM model for summarization"
    },
    "max_tokens": {
      "type": "integer",
      "description": "Max tokens in response"
    }
  },
  "required": [
    "query"
  ]
}
```

</details>

---

### write_file

**Description**: Create or overwrite file with auto-summarization. File summary auto-generated using Claude and stored in document metadata with exact token counts.

**Type**: `function_tool`

**Parameters**:

- `path` (string, **required**): File path in workspace
- `content` (string, **required**): File content in actual format
- `mime_type` (string, **required**): MIME type (e.g., text/markdown, text/csv)
- `why_created` (string, optional (default: ``)): Optional reason for creating file

**Returns**:

```json
{
  "path": "string",
  "size_bytes": "integer",
  "token_count": "integer",
  "blocks_count": "integer",
  "summary": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "File path in workspace"
    },
    "content": {
      "type": "string",
      "description": "File content in actual format"
    },
    "mime_type": {
      "type": "string",
      "description": "MIME type (e.g., text/markdown, text/csv)"
    },
    "why_created": {
      "type": "string",
      "description": "Optional reason for creating file"
    }
  },
  "required": [
    "path",
    "content",
    "mime_type"
  ]
}
```

</details>

---

## Metacognition Tools

### plan_next_steps

**Description**: Plan your approach before executing. Your plan is logged to track your strategy and decision-making process.

**Type**: `non_function_tool`

**Parameters**:

- `plan` (string, **required**): Your step-by-step plan or strategy for completing the task

**Returns**:

```json
{
  "logged": "boolean"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "plan": {
      "type": "string",
      "description": "Your step-by-step plan or strategy for completing the task"
    }
  },
  "required": [
    "plan"
  ]
}
```

</details>

---

### save_to_long_term_memory

**Description**: Save important information to long-term memory across sessions. Use for patterns, lessons learned, or knowledge you'll need in future tasks.

**Type**: `non_function_tool`

**Parameters**:

- `memory` (string, **required**): Information to remember permanently across sessions
- `category` (string, optional (default: `general`)): Category for organizing long-term memories (e.g., 'patterns', 'errors', 'insights')

**Returns**:

```json
{
  "logged": "boolean"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "memory": {
      "type": "string",
      "description": "Information to remember permanently across sessions"
    },
    "category": {
      "type": "string",
      "description": "Category for organizing long-term memories (e.g., 'patterns', 'errors', 'insights')"
    }
  },
  "required": [
    "memory"
  ]
}
```

</details>

---

### save_to_short_term_memory

**Description**: Save important information to short-term memory for this task. Use for temporary facts, intermediate results, or context you'll need soon.

**Type**: `non_function_tool`

**Parameters**:

- `memory` (string, **required**): Information to remember for the duration of this task
- `key` (string, optional (default: ``)): Optional key to identify this memory item

**Returns**:

```json
{
  "logged": "boolean"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "memory": {
      "type": "string",
      "description": "Information to remember for the duration of this task"
    },
    "key": {
      "type": "string",
      "description": "Optional key to identify this memory item"
    }
  },
  "required": [
    "memory"
  ]
}
```

</details>

---

### stop_and_think

**Description**: Pause and think deeply about the problem. Your thoughts are logged to help you reason through complex situations.

**Type**: `non_function_tool`

**Parameters**:

- `thoughts` (string, **required**): Your internal reasoning, analysis, or considerations about the current task

**Returns**:

```json
{
  "logged": "boolean"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "thoughts": {
      "type": "string",
      "description": "Your internal reasoning, analysis, or considerations about the current task"
    }
  },
  "required": [
    "thoughts"
  ]
}
```

</details>

---

## Orchestration: Handoff

### handoff

**Description**: Complete your task and return result to parent. Use this when your work is done to hand back control.

**Type**: `handoff`

**Parameters**:

- `result` (object, **required**): The final result of your task (can be any structured data)

**Returns**:

```json
{
  "status": "string",
  "result": "object"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "result": {
      "type": "object",
      "description": "The final result of your task (can be any structured data)"
    }
  },
  "required": [
    "result"
  ]
}
```

</details>

---

## Orchestration: Ask Master

### ask_master

**Description**: Ask your parent agent for clarification or additional information. Use when you need guidance or missing context.

**Type**: `ask_master`

**Parameters**:

- `question` (string, **required**): The question to ask your parent agent

**Returns**:

```json
{
  "answer": "string",
  "from_agent": "string"
}
```

<details>
<summary>View Full JSON Schema</summary>

```json
{
  "type": "object",
  "properties": {
    "question": {
      "type": "string",
      "description": "The question to ask your parent agent"
    }
  },
  "required": [
    "question"
  ]
}
```

</details>

---

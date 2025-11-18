"""
Unified LLM interface - maps models to clients and routes appropriately
"""

from typing import Dict, List, Any, Optional
from .ask_claude import ClaudeClient
from .ask_openai import OpenAIClient


MODEL_MAPPING = {
    # Claude Sonnet 4.5 ONLY
    "claude-sonnet-4-5-20250929": "anthropic",
    "claude-sonnet-4-5": "anthropic",

    # GPT-5 series ONLY
    "gpt-5": "openai",
    "gpt-5.1": "openai",  # Newest version
    "gpt-5-mini": "openai",
    "gpt-5-nano": "openai",
}



def ask_llm(
    messages: List[Dict],
    model: str,
    tools: Optional[List[Dict]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Route to appropriate LLM based on model mapping
    Uses OpenAI-style interface: system message is first message with role="system"
    """
    # Get client type from mapping
    if model in MODEL_MAPPING:
        client_type = MODEL_MAPPING[model]
    else:
        return {"error": f"Unknown model: {model}. Available: {list(MODEL_MAPPING.keys())}"}
    
    # Route to appropriate client
    if client_type == "anthropic":
        # Convert OpenAI-style to Anthropic-style
        system = None
        filtered_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                # Extract system message for Anthropic
                system = msg.get("content", "")
            else:
                # Keep user/assistant messages
                filtered_messages.append(msg)

        # Map generic name to specific version
        if model == "claude-sonnet-4-5":
            model = "claude-sonnet-4-5-20250929"

        client = ClaudeClient()
        result = client.ask(
            messages=filtered_messages,
            model=model,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

        # Normalize: Claude returns "response", convert to "content" format
        if "response" in result and isinstance(result["response"], str):
            result["content"] = [{"type": "text", "text": result["response"]}]

        return result

    elif client_type == "openai":
        # Convert tools from Claude format to OpenAI format
        openai_tools = None
        if tools:
            openai_tools = []
            for tool in tools:
                # Claude format: {name, description, input_schema}
                # OpenAI format: {type: "function", function: {name, description, parameters}}
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", f"Tool: {tool['name']}"),
                        "parameters": tool.get("input_schema", {})
                    }
                })

        # OpenAI accepts system messages in the messages list directly
        client = OpenAIClient()
        return client.ask(
            messages=messages,
            model=model,
            tools=openai_tools,
            tool_choice=tool_choice,
            **kwargs
        )

    else:
        return {"error": f"Unknown client type: {client_type}"}
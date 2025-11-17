"""
Simple OpenAI wrapper - no lies, just works
NOW WITH PROPER TOOL CALLING FLOW
AND VISION SUPPORT
"""

import json
import base64
from openai import OpenAI
from typing import Dict, List, Any, Optional, Union
import sys
from pathlib import Path

# Import global config with clients
from ai_orchestration.utils.global_config import openai_client


class OpenAIClient:
    """Simple OpenAI client that actually works"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.client = OpenAI(api_key=api_key)
        elif openai_client:
            self.client = openai_client
        else:
            raise ValueError("No OpenAI client. Pass api_key or set up global_config.")
    
    def ask(
        self, 
        messages: List[Dict],
        model: str = "gpt-5",
        response_format: Dict = None,
        tools: List[Dict] = None,
        tool_choice: Any = None,
        strict_schema: bool = False,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Just ask OpenAI and get response"""
        
        # Build params
        params = {
            "model": model,
            "messages": messages,
        }
        
        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            # DEFAULT: Force tool use when tools are provided
            if tool_choice is None:
                params["tool_choice"] = "required"
            else:
                params["tool_choice"] = tool_choice
        elif tool_choice:
            params["tool_choice"] = tool_choice
        
        # Try to get response
        attempt = 0
        last_error = None
        
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(**params)
                
                # Get the content
                content = response.choices[0].message.content
                
                # If we need JSON and strict_schema is on, validate it
                if strict_schema and response_format and response_format.get("type") == "json_object":
                    try:
                        json_content = json.loads(content)
                        content = json_content
                    except json.JSONDecodeError as e:
                        if attempt < max_retries - 1:
                            # Tell it what went wrong
                            messages.append({"role": "assistant", "content": content})
                            messages.append({"role": "user", "content": f"Invalid JSON. Error: {str(e)}. Please return valid JSON."})
                            attempt += 1
                            continue
                        else:
                            return {
                                "error": f"Invalid JSON after {max_retries} attempts",
                                "last_response": content,
                                "attempts": attempt + 1
                            }
                
                # Build result
                result = {
                    "content": content,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
                
                # Add tool calls if any
                if response.choices[0].message.tool_calls:
                    result["tool_calls"] = []
                    for tc in response.choices[0].message.tool_calls:
                        result["tool_calls"].append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "input": json.loads(tc.function.arguments)  # Use "input" to match Claude format
                        })
                
                return result
                
            except Exception as e:
                last_error = str(e)
                attempt += 1
                if attempt < max_retries:
                    continue
        
        return {
            "error": f"Failed after {max_retries} attempts",
            "last_error": last_error,
            "attempts": max_retries
        }




# Simple function to use
def ask_gpt(messages, **kwargs):
    """Simple function to ask GPT"""
    client = OpenAIClient()
    return client.ask(messages, **kwargs)


# Helper functions for tool calling
def format_tool_message(tool_call_id: str, result: Any) -> Dict[str, Any]:
    """Format tool result for conversation"""
    return {
        "role": "tool",
        "content": json.dumps(result) if not isinstance(result, str) else result,
        "tool_call_id": tool_call_id
    }


def format_assistant_with_tools(content: str, tool_calls: List[Dict]) -> Dict[str, Any]:
    """Format assistant message with tool calls"""
    msg = {
        "role": "assistant",
        "content": content
    }
    
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc['id'],
                "type": "function",
                "function": {
                    "name": tc['name'],
                    "arguments": json.dumps(tc['arguments']) if isinstance(tc['arguments'], dict) else tc['arguments']
                }
            }
            for tc in tool_calls
        ]
    
    return msg


def ask_gpt_with_validation(messages, expected_format="json", max_retries=3, **kwargs):
    """
    Ask GPT with automatic validation and retry
    
    Args:
        messages: Messages to send
        expected_format: "json" or "text" 
        max_retries: Number of retries if validation fails
        **kwargs: Other params for ask_gpt
    """
    client = OpenAIClient()
    conversation = messages.copy()
    
    for attempt in range(max_retries):
        # Force JSON format if expected
        if expected_format == "json" and "response_format" not in kwargs:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["strict_schema"] = True
        
        result = client.ask(conversation, **kwargs)
        
        # If no error and we got what we expected, return
        if "error" not in result:
            if expected_format == "json" and isinstance(result.get('content'), dict):
                return result
            elif expected_format == "text":
                return result
        
        # Validation failed - retry with repair
        if attempt < max_retries - 1:
            # Add failed response to conversation
            if result.get('last_response'):
                conversation.append({
                    "role": "assistant",
                    "content": str(result['last_response'])
                })
            
            # Add repair prompt
            if expected_format == "json":
                repair = "Your response was not valid JSON. Please return a properly formatted JSON object."
            else:
                repair = "Please try again with a clear response."
            
            conversation.append({
                "role": "user",
                "content": repair
            })
    
    # All retries failed
    return {
        "error": f"Failed validation after {max_retries} attempts",
        "attempts": max_retries
    }


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """Encode image to base64 for vision API"""
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_gpt_with_image(
    text_prompt: str,
    image_path: Union[str, Path],
    model: str = "gpt-4.1-mini",
    system_prompt: str = None,
    detail: str = "high",
    response_format: Dict = None,
    **kwargs  # Can include run_id, agent_name, etc.
) -> Dict[str, Any]:
    """
    Ask GPT with both text and image input

    Args:
        text_prompt: The question/instruction about the image
        image_path: Path to image file
        model: Model to use (default: gpt-4.1-mini)
        system_prompt: System message for behavior/role
        detail: Image detail level (low/high/auto, default: high)
        response_format: OpenAI response format (e.g. {"type": "json_object"})
        **kwargs: Other OpenAI parameters

    Returns:
        Dict with content, model, usage, etc
    """
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Determine MIME type
    suffix = Path(image_path).suffix.lower()
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(suffix, 'image/jpeg')
    
    # Build messages
    messages = []
    
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add user message with text and image
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": text_prompt
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}",
                    "detail": detail
                }
            }
        ]
    })
    
    # Call OpenAI - filter out context params
    api_kwargs = {
        'model': model
    }
    if response_format:
        api_kwargs['response_format'] = response_format
    
    # Add any other OpenAI-specific kwargs
    for k, v in kwargs.items():
        if k not in ['run_id', 'agent_name', 'tool_name']:
            api_kwargs[k] = v
    
    return ask_gpt(
        messages=messages,
        **api_kwargs
    )


def parse_json_response(response: Union[str, Dict]) -> Dict[str, Any]:
    """
    Parse JSON from API response, handling markdown formatting
    
    Args:
        response: Either raw string or dict with 'content' key
        
    Returns:
        Parsed JSON object or error dict
    """
    # Extract content if dict
    if isinstance(response, dict):
        content = response.get('content', '')
        # Check if we have an error response
        if 'error' in response:
            return response  # Return error as-is
    else:
        content = response
    
    # Check for empty content
    if not content or not content.strip():
        return {
            "error": "Empty response from API",
            "raw": ""
        }
    
    # Clean markdown formatting if present
    if content.strip().startswith('```'):
        lines = content.strip().split('\n')
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines[-1].startswith('```'):
            lines = lines[:-1]
        content = '\n'.join(lines)
    
    # Parse JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse error: {str(e)}",
            "raw": content[:500] + "..." if len(content) > 500 else content
        }


def validate_against_schema(data: Dict, schema: Dict) -> Dict[str, Any]:
    """
    Simple schema validation (can be extended later)
    
    Args:
        data: Parsed data to validate
        schema: Expected schema structure
        
    Returns:
        Dict with validation result
    """
    # Basic validation - just check if all schema keys exist
    missing_keys = []
    
    def check_keys(schema_part, data_part, path=""):
        if isinstance(schema_part, dict):
            if not isinstance(data_part, dict):
                missing_keys.append(f"{path} should be dict")
                return
            for key in schema_part:
                if key not in data_part:
                    missing_keys.append(f"{path}.{key}" if path else key)
                else:
                    check_keys(schema_part[key], data_part[key], f"{path}.{key}" if path else key)
    
    check_keys(schema, data)
    
    if missing_keys:
        return {
            "valid": False,
            "missing": missing_keys,
            "data": data
        }
    
    return {
        "valid": True,
        "data": data
    }
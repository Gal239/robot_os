"""
Clean Claude API wrapper - Class-based, simple, no hidden errors
"""

import anthropic
import base64
import asyncio
from functools import partial
from typing import Optional, Dict, List, Any
from pathlib import Path
import sys

# Import global config with clients
from ai_orchestration.utils.global_config import anthropic_client



class ClaudeClient:
    """Clean wrapper for Claude API"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.client = anthropic.Anthropic(api_key=api_key)
        elif anthropic_client:
            self.client = anthropic_client
        else:
            raise ValueError("No anthropic client available. Pass api_key or configure global.css client.")
    
    
    def ask(
        self,
        messages: List[Dict],
        model: str = "claude-sonnet-4-20250514",
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[Dict] = None,
        web_search_config: Optional[Dict] = None,
        thinking: Optional[Dict] = None,
        stream: bool = False,
        max_tokens: int = 8192,
    ) -> Dict[str, Any]:
        """
        Call Claude API with messages.
        
        Returns:
            Dict with response, tool calls, and metadata
        """
        
        # Process messages for files
        messages = self._process_files(messages.copy())
        
        # Handle empty user messages
        for msg in messages:
            if msg.get("role") == "user" and not msg.get("content"):
                msg["content"] = "Hello! How can I help you today?"
        

        # Build API parameters
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": 8192  # Default max tokens (required by Anthropic API)
        }

        # Add system parameter if provided
        if system:
            params["system"] = system
        
        # Add tools
        all_tools = tools or []
        if web_search_config:
            # Build agent_orc_web search tool with config
            web_tool = {"type": "web_search_20250305"}
            if "name" in web_search_config:
                web_tool["name"] = web_search_config["name"]
            else:
                web_tool["name"] = "web_search"
            
            # Add optional parameters if provided
            if "max_uses" in web_search_config:
                web_tool["max_uses"] = web_search_config["max_uses"]
            if "allowed_domains" in web_search_config:
                web_tool["allowed_domains"] = web_search_config["allowed_domains"]
            if "blocked_domains" in web_search_config:
                web_tool["blocked_domains"] = web_search_config["blocked_domains"]
            if "user_location" in web_search_config:
                web_tool["user_location"] = web_search_config["user_location"]
                
            all_tools.append(web_tool)
            # Web search requires specific model
            params["model"] = "claude-opus-4-20250514"
        
        # Format tools for Claude API
        # Per Claude API docs: tools only need name, description, input_schema (NO "type" field!)
        if all_tools:
            formatted_tools = []
            for tool in all_tools:
                # Special handling for web_search tool (has type field)
                if tool.get("type") == "web_search_20250305":
                    formatted_tools.append(tool)
                # Regular tools: ensure correct format (name, description, input_schema)
                elif "name" in tool:
                    formatted_tools.append({
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("input_schema", {})
                    })
                else:
                    # Unknown format, pass through
                    formatted_tools.append(tool)
            params["tools"] = formatted_tools
            # DEFAULT: Force tool use when tools are provided
            if tool_choice is None:
                params["tool_choice"] = {"type": "any"}
            else:
                params["tool_choice"] = tool_choice
        elif tool_choice:
            params["tool_choice"] = tool_choice
        
        # Add thinking mode
        if thinking:
            params["thinking"] = thinking
        
        # Make API call
        if web_search_config or thinking:
            response = self.client.beta.messages.create(**params)
        else:
            response = self.client.messages.create(**params)

        
        # Check if response has tool use
        has_tool_use = hasattr(response, "content") and any(
            hasattr(block, "type") and block.type == "tool_use" 
            for block in response.content
        )
        
        # Build comprehensive result dict
        result = {
            "response": None,  # Will be set below
            "raw": response,  # Keep raw for compatibility
            "messages": messages,
            "usage": response.usage if hasattr(response, 'usage') else None,
            "model": response.model if hasattr(response, 'model') else model,
            "has_tool_use": has_tool_use,
            "tool_calls": []
        }
        
        # Extract tool calls if present
        if has_tool_use:
            for block in response.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    result["tool_calls"].append({
                        "id": block.id if hasattr(block, 'id') else None,
                        "name": block.name,
                        "input": block.input
                    })
            # For tool use, response is the full object
            result["response"] = response
        else:
            # Extract text response
            text_response = self._extract_text(response)
            result["response"] = text_response
        
        return result
    
    def _process_files(self, messages: List[Dict]) -> List[Dict]:
        """Process file references in messages"""
        for msg in messages:
            if not isinstance(msg.get("content"), list):
                continue
                
            new_content = []
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "file":
                    file_content = self._read_file(item["path"])
                    new_content.append(file_content)
                else:
                    new_content.append(item)
            
            msg["content"] = new_content
        
        return messages
    
    def _read_file(self, file_path: str) -> Dict:
        """Read file and return content block"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Text files
        text_exts = {'.txt', '.py', '.js', '.json', '.md', '.html', '.svg', '.xml', '.csv'}
        if path.suffix.lower() in text_exts:
            content = path.read_text(encoding='utf-8')
            return {"type": "text", "text": f"File: {path.name}\n\n{content}"}
        
        # PDF
        if path.suffix.lower() == '.pdf':
            data = base64.b64encode(path.read_bytes()).decode()
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data}
            }
        
        # Images
        img_exts = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
        if path.suffix.lower() in img_exts:
            mime_type = f"image/{path.suffix[1:]}"
            data = base64.b64encode(path.read_bytes()).decode()
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": data}
            }
        
        # Try as text
        content = path.read_text(encoding='utf-8')
        return {"type": "text", "text": f"File: {path.name}\n\n{content}"}
    

    
    def _extract_text(self, response) -> str:
        """Extract text from API response"""
        if not hasattr(response, "content"):
            return str(response)
        
        parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        
        return " ".join(parts).strip()


    


# Create default client lazily
_default_client = None

# Backwards compatible function for tests
def ask_claude(messages: List[Dict], **kwargs) -> Any:
    """Function wrapper for backwards compatibility"""
    global _default_client
    if _default_client is None:
        if anthropic_client:
            _default_client = ClaudeClient()
        else:
            # Try to create with rooms variable
            import os
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                _default_client = ClaudeClient(api_key=api_key)
            else:
                raise ValueError("No anthropic client available. Set ANTHROPIC_API_KEY or configure global.css client.")
    
    return _default_client.ask(messages, **kwargs)
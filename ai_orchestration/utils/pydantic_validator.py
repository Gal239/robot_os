"""
Pydantic Validator for Tool Inputs
Provides validation for tool inputs and outputs using dynamic Pydantic models
"""
from typing import Dict, Any, List, Tuple, Optional, Union
from pydantic import BaseModel, ValidationError, create_model, Field
import re


class PydanticValidator:
    """
    Dynamic Pydantic validator for tool schemas
    Creates Pydantic models on the fly from JSON schemas
    """
    
    @staticmethod
    def create_model_from_schema(name: str, schema: Dict) -> Optional[BaseModel]:
        """
        Create a Pydantic model dynamically from a JSON schema
        
        Args:
            name: Name for the model (e.g., "TikTokSearchInput")
            schema: JSON schema definition
            
        Returns:
            Pydantic model class or None if schema is invalid
        """
        if not schema or "properties" not in schema:
            return None
            
        properties = schema["properties"]
        required = schema.get("required", [])
        
        # Build Pydantic field definitions
        fields = {}
        for field_name, field_def in properties.items():
            field_type, field_kwargs = PydanticValidator._parse_field_definition(
                field_def, field_name in required
            )
            
            if field_kwargs:
                fields[field_name] = (field_type, Field(**field_kwargs))
            else:
                fields[field_name] = (field_type, ... if field_name in required else None)
        
        # Create model dynamically
        return create_model(name, **fields)
    
    @staticmethod
    def _parse_field_definition(field_def: Dict, is_required: bool) -> Tuple[type, Dict]:
        """
        Parse JSON schema field definition into Pydantic type and constraints
        
        Returns:
            (python_type, field_kwargs_dict)
        """
        field_type = str  # Default
        field_kwargs = {}
        
        # Determine base type
        json_type = field_def.get("type", "string")
        
        if json_type == "array":
            field_type = list
            # Handle array constraints
            if "minItems" in field_def:
                field_kwargs["min_length"] = field_def["minItems"]
            if "maxItems" in field_def:
                field_kwargs["max_length"] = field_def["maxItems"]
                
        elif json_type == "integer":
            field_type = int
            # Handle number constraints
            if "minimum" in field_def:
                field_kwargs["ge"] = field_def["minimum"]
            if "maximum" in field_def:
                field_kwargs["le"] = field_def["maximum"]
                
        elif json_type == "boolean":
            field_type = bool
            
        elif json_type == "object":
            field_type = dict
            
        else:  # string
            field_type = str
            # Handle string constraints
            if "minLength" in field_def:
                field_kwargs["min_length"] = field_def["minLength"]
            if "maxLength" in field_def:
                field_kwargs["max_length"] = field_def["maxLength"]
            if "pattern" in field_def:
                field_kwargs["pattern"] = field_def["pattern"]
            if "enum" in field_def:
                # For enums, we'll validate in post-processing
                pass
            
            # For required strings, ensure not empty
            if is_required and "min_length" not in field_kwargs:
                field_kwargs["min_length"] = 1
        
        # Add description if available
        if "description" in field_def:
            field_kwargs["description"] = field_def["description"]
            
        # Set default based on required status
        if is_required:
            field_kwargs["default"] = ...
        else:
            field_kwargs["default"] = None
            
        return field_type, field_kwargs
    
    @staticmethod
    def validate_tool_input(tool_name: str, tool_input: Dict, schema: Dict) -> Tuple[bool, Optional[Dict], List[str]]:
        """
        Validate tool input against schema
        
        Args:
            tool_name: Name of the tool
            tool_input: Input parameters to validate
            schema: JSON schema for validation
            
        Returns:
            (is_valid, fixed_input, error_messages)
            - is_valid: Whether validation passed
            - fixed_input: Input with automatic fixes applied (if any)
            - error_messages: List of validation errors
        """
        # Create validator model
        validator_model = PydanticValidator.create_model_from_schema(
            f"{tool_name}Input", 
            schema
        )
        
        if not validator_model:
            return True, tool_input, []  # No schema, no validation
        
        # Try to validate
        try:
            # First attempt: validate as-is
            validated = validator_model(**tool_input)
            return True, validated.model_dump(), []
            
        except ValidationError as e:
            # Try to auto-fix common issues
            fixed_input, fixes = PydanticValidator._attempt_auto_fix(
                tool_input, e, schema
            )
            
            if fixes:
                # Try validation again with fixes
                try:
                    validated = validator_model(**fixed_input)
                    fix_messages = [f"Auto-fixed: {fix}" for fix in fixes]
                    return True, validated.model_dump(), fix_messages
                    
                except ValidationError as e2:
                    # Still failed after fixes
                    errors = PydanticValidator._format_validation_errors(e2, schema)
                    return False, None, errors
            else:
                # No fixes possible
                errors = PydanticValidator._format_validation_errors(e, schema)
                return False, None, errors
    
    @staticmethod
    def _attempt_auto_fix(tool_input: Dict, error: ValidationError, schema: Dict) -> Tuple[Dict, List[str]]:
        """
        Attempt to automatically fix common validation errors
        
        Returns:
            (fixed_input, list_of_fixes_applied)
        """
        fixed_input = tool_input.copy()
        fixes = []
        
        for err in error.errors():
            field = err["loc"][0] if err["loc"] else None
            error_type = err["type"]
            
            if not field:
                continue
                
            # Auto-fix: String to int conversion
            if error_type == "int_parsing" and field in fixed_input:
                try:
                    fixed_input[field] = int(fixed_input[field])
                    fixes.append(f"Converted {field} from string to integer")
                except (ValueError, TypeError):
                    pass
                    
            # Auto-fix: String to list conversion
            elif error_type == "list_type" and field in fixed_input:
                value = fixed_input[field]
                if isinstance(value, str):
                    # Try to parse as Python literal
                    try:
                        import ast
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            fixed_input[field] = parsed
                            fixes.append(f"Converted {field} from string to list")
                    except:
                        # Try single item to list
                        fixed_input[field] = [value]
                        fixes.append(f"Converted {field} to single-item list")
                        
            # Auto-fix: None to empty string for required strings
            elif error_type == "missing" and field in schema.get("properties", {}):
                field_schema = schema["properties"][field]
                if field_schema.get("type") == "string":
                    fixed_input[field] = ""
                    fixes.append(f"Set missing {field} to empty string")
                elif field_schema.get("type") == "array":
                    fixed_input[field] = []
                    fixes.append(f"Set missing {field} to empty array")
                    
        return fixed_input, fixes
    
    @staticmethod
    def _format_validation_errors(error: ValidationError, schema: Dict) -> List[str]:
        """
        Format Pydantic validation errors into helpful messages
        """
        errors = []
        properties = schema.get("properties", {})
        
        for err in error.errors():
            field = err["loc"][0] if err["loc"] else "unknown"
            error_type = err["type"]
            msg = err["msg"]
            
            # Create helpful error message
            if error_type == "missing":
                errors.append(f"{field}: Required field is missing")
                
            elif error_type == "string_too_short":
                min_length = properties.get(field, {}).get("minLength", 1)
                errors.append(f"{field}: Must be at least {min_length} characters")
                
            elif error_type == "value_error":
                # Check for enum constraint
                if field in properties and "enum" in properties[field]:
                    allowed = properties[field]["enum"]
                    errors.append(f"{field}: Must be one of {allowed}")
                else:
                    errors.append(f"{field}: {msg}")
                    
            else:
                # Generic format
                errors.append(f"{field}: {msg}")
                
        return errors
    
    @staticmethod
    def generate_example_input(schema: Dict) -> Dict:
        """
        Generate example input based on schema
        """
        example = {}
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string")
            
            if field_type == "string":
                if "enum" in field_def:
                    example[field_name] = field_def["enum"][0]
                elif "format" in field_def:
                    if field_def["format"] == "email":
                        example[field_name] = "user@example.com"
                    elif field_def["format"] == "url":
                        example[field_name] = "https://example.com"
                    elif field_def["format"] == "date":
                        example[field_name] = "2024-01-15"
                    else:
                        example[field_name] = "example_string"
                else:
                    example[field_name] = field_def.get("description", "example")
                    
            elif field_type == "integer":
                minimum = field_def.get("minimum", 0)
                maximum = field_def.get("maximum", 100)
                example[field_name] = min(minimum + 10, maximum)
                
            elif field_type == "boolean":
                example[field_name] = True
                
            elif field_type == "array":
                item_type = field_def.get("items", {}).get("type", "string")
                min_items = field_def.get("minItems", 1)
                if item_type == "string":
                    example[field_name] = [f"item{i+1}" for i in range(min_items)]
                else:
                    example[field_name] = [{"example": "data"}]
                    
            elif field_type == "object":
                example[field_name] = {"key": "value"}
                
        # Only include required fields in minimal example
        minimal_example = {k: v for k, v in example.items() if k in required}
        
        return minimal_example if minimal_example else example
    
    @staticmethod
    def fix_tool_input(tool_name: str, tool_input: dict) -> Tuple[dict, List[str]]:
        """
        Try to fix common input issues. Returns (fixed_input, log_messages)
        """
        fixes = []
        fixed_input = tool_input.copy()
        
        if tool_name == "tiktok_universal_search":
            # Fix: queries passed as string instead of array
            if isinstance(fixed_input.get("queries"), str):
                try:
                    import ast
                    queries = ast.literal_eval(fixed_input["queries"])
                    fixed_input["queries"] = queries
                    fixes.append(f"FIXED: Converted 'queries' from string to array ({len(queries)} items)")
                except Exception as e:
                    fixes.append("ERROR: 'queries' is a string but couldn't parse it. HINT: Pass the search_terms array directly from plan_search_strategy, don't JSON.stringify it")
                    
        elif tool_name == "youtube_search":
            # Fix: searches passed as string
            if isinstance(fixed_input.get("searches"), str):
                try:
                    import ast
                    searches = ast.literal_eval(fixed_input["searches"])
                    fixed_input["searches"] = searches
                    fixes.append(f"FIXED: Converted 'searches' from string to array ({len(searches)} items)")
                except Exception as e:
                    fixes.append("ERROR: 'searches' must be an array. HINT: Use the search_terms from plan_search_terms directly as searches")
                    
        elif tool_name == "twitter_universal_search":
            # Fix: queries passed as string
            if isinstance(fixed_input.get("queries"), str):
                try:
                    import ast
                    queries = ast.literal_eval(fixed_input["queries"])
                    fixed_input["queries"] = queries
                    fixes.append(f"FIXED: Converted 'queries' from string to array ({len(queries)} items)")
                except Exception as e:
                    fixes.append("ERROR: 'queries' is a string. HINT: Pass the array directly, not as a JSON string")
                    
        elif tool_name == "twitter_user_deep_dive":
            # Fix: usernames passed as string instead of array
            if isinstance(fixed_input.get("usernames"), str):
                username = fixed_input["usernames"]
                fixed_input["usernames"] = [username]
                fixes.append(f"FIXED: Converted 'usernames' from string '{username}' to array ['{username}']")
        
        return fixed_input, fixes
    
    @staticmethod
    def generate_validation_hint(tool_name: str, errors: List[str], tool_input: Dict, schema: Dict) -> str:
        """Generate helpful hint for validation errors"""
        
        # Check for missing required fields FIRST
        required_fields = set(schema.get("required", []))
        provided_fields = set(tool_input.keys())
        missing_required = required_fields - provided_fields
        
        if missing_required:
            # MISSING REQUIRED FIELDS - Special message
            missing_list = ", ".join(sorted(missing_required))
            hint = f"FAILED: Missing required fields: {missing_list}\n\n"
            hint += f"You must call {tool_name} with ALL required fields:\n"
            
            # Show the full required schema
            properties = schema.get("properties", {})
            for field in sorted(required_fields):
                field_info = properties.get(field, {})
                field_type = field_info.get("type", "string")
                desc = field_info.get("description", "")
                hint += f"  - {field} ({field_type}): {desc}\n"
            
            hint += f"\nCall {tool_name} again with complete input including all required fields."
            return hint
        
        # Otherwise handle type/validation errors
        hints = []
        
        for error in errors:
            if "String should have at least 1 character" in error:
                field = error.split(":")[0]
                hints.append(f"• {field}: Cannot be empty string - provide actual content")
                
            elif "Input should be a valid integer" in error:
                field = error.split(":")[0]
                current_value = tool_input.get(field, "missing")
                hints.append(f"• {field}: Wrong type - must be a number, not '{current_value}'")
                
            elif "Input should be a valid list" in error:
                field = error.split(":")[0]
                current_value = tool_input.get(field, "missing")
                hints.append(f"• {field}: Wrong type - must be an array like ['item1', 'item2'], not '{current_value}'")
                
            elif "should have at least" in error and "item" in error:
                field = error.split(":")[0]
                # Extract number from error message
                import re
                match = re.search(r'at least (\d+) item', error)
                if match:
                    min_items = match.group(1)
                    hints.append(f"• {field}: Array too short - need at least {min_items} items")
                    
            elif "should have at most" in error and "item" in error:
                field = error.split(":")[0]
                # Extract number from error message
                import re
                match = re.search(r'at most (\d+) item', error)
                if match:
                    max_items = match.group(1)
                    hints.append(f"• {field}: Array too long - maximum {max_items} items allowed")
                    
            elif "Required field is missing" in error:
                field = error.split(":")[0]
                hints.append(f"• {field}: Required field is missing")
                
            else:
                # Generic hint
                hints.append(f"• {error}")
        
        # Add examples if available
        properties = schema.get("properties", {})
        for field, error in [(e.split(":")[0], e) for e in errors if ":" in e]:
            if field in properties:
                field_schema = properties[field]
                if "enum" in field_schema:
                    hints.append(f"  Allowed values: {field_schema['enum']}")
                elif field_schema.get("type") == "array" and "items" in field_schema:
                    item_type = field_schema["items"].get("type", "any")
                    hints.append(f"  Example: [{item_type}_1, {item_type}_2]")
        
        if hints:
            return "FAILED: Validation errors:\n" + "\n".join(hints) + f"\n\nFix these issues and call {tool_name} again."
        else:
            return f"FAILED: Check the {tool_name} parameters"
    
    @staticmethod
    def validate_tool_output(tool_name: str, output: Any, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate tool output against schema
        
        Args:
            tool_name: Name of the tool
            output: Output to validate
            schema: JSON schema for output validation
            
        Returns:
            (is_valid, error_messages)
        """
        # For now, just check basic structure
        # This can be expanded based on needs
        if not schema:
            return True, []
            
        errors = []
        
        # Basic type checking
        expected_type = schema.get("type")
        if expected_type:
            if expected_type == "object" and not isinstance(output, dict):
                errors.append(f"Expected object/dict, got {type(output).__name__}")
            elif expected_type == "array" and not isinstance(output, list):
                errors.append(f"Expected array/list, got {type(output).__name__}")
            elif expected_type == "string" and not isinstance(output, str):
                errors.append(f"Expected string, got {type(output).__name__}")
                
        return len(errors) == 0, errors
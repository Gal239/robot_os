"""
Tools - Auto-discovers all tool modules in this directory
OFFENSIVE: Just add ANY .py file with @tool decorators and it auto-loads
Drop a .py file → tools auto-register → ready to use
"""

import importlib
from pathlib import Path

# Get current directory
tools_dir = Path(__file__).parent

# Auto-import all Python files (except __init__.py, infrastructure files)
# OFFENSIVE: No try/except - crash if import fails
for file in tools_dir.glob("*.py"):
    # Skip infrastructure files
    if file.name in ["__init__.py", "generate_docs.py"]:
        continue

    module_name = file.stem  # e.g., "file_tools"

    # OFFENSIVE: Import (crash if module has errors)
    importlib.import_module(f".{module_name}", package=__package__)

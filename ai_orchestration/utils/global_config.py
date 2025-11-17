#!/usr/bin/env python3
"""
GLOBAL CONFIG - All configuration in ONE place
"""
import json
import os
import sys
from pathlib import Path
import anthropic
from openai import OpenAI

# Base directory (project root)
BASE_DIR = Path(__file__).parent.parent

# Import AutoDB from local-only module (no cloud dependencies)
from ai_orchestration.utils.auto_db import AutoDB


database_path = BASE_DIR / "databases"

# Initialize Agent Engine DB (purely local, no cloud)
agent_engine_db = AutoDB(local_path=str(database_path))
# Load secrets from local secrets.json
SECRETS_FILE = Path(__file__).parent / "secrets.json"
with open(SECRETS_FILE) as f:
    secrets = json.load(f)

# API Tokens
TOKENS = {
    "anthropic": secrets.get("anthropic", {}).get("api_key"),
    "openai": secrets.get("openai", {}).get("api_key"),
}

# Set environment variables for LLM clients
os.environ["ANTHROPIC_API_KEY"] = TOKENS["anthropic"]
os.environ["OPENAI_API_KEY"] = TOKENS["openai"]

# Initialize API Clients
anthropic_client = anthropic.Anthropic(api_key=TOKENS["anthropic"])
openai_client = OpenAI(api_key=TOKENS["openai"])


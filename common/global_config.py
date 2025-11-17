import anthropic
import json
import os
from pathlib import Path
from openai import OpenAI

# Base directories
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent  # dual_agent_system directory

# Load secrets and scrapers config
secret_path = BASE_DIR / "secrets.json"

secrets = json.load(open(secret_path))

# API Clients
anthropic_client = anthropic.Anthropic(api_key=secrets["anthropic_api_key"])
openai_client = OpenAI(api_key=secrets["openai_api_key"])

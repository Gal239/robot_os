#!/usr/bin/env python3
"""
Test Scene Maker Agent with Auto-Generated Prompt
==================================================
Single agent with FULL knowledge prompt + handoff tool only
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from ai_orchestration.core.orchestrator import Orchestrator
from ai_orchestration.core.prompt_maker import generate_agent_prompt


async def test_scene_maker_agent():
    """
    Test scene maker agent with auto-generated prompt

    Flow:
    1. User request: "Create a kitchen scene with apple on table"
    2. Agent uses full prompt knowledge to generate script
    3. Agent calls handoff with python_script
    4. Compiler wraps + runs the script
    5. Returns: videos, screenshots, data
    """
    print("\n" + "="*70)
    print("TESTING SCENE MAKER AGENT WITH AUTO-GENERATED PROMPT")
    print("="*70)

    # Initialize orchestrator
    ops = Orchestrator()

    # Generate FULL prompt with all knowledge + examples
    print("\n📝 Generating auto-prompt from source data...")
    scene_maker_prompt = generate_agent_prompt(
        agent_id="scene_maker",
        tool_ids=[
            "discover_assets",
            "discover_behaviors",
            "discover_relations",
            "get_api_documentation",
            "get_scene_examples"
        ],
        purpose="create Python scripts for robot simulation scenes using ExperimentOps API"
    )

    print(f"✅ Prompt generated: {len(scene_maker_prompt)} characters")
    print(f"   - 71 assets with behaviors")
    print(f"   - 18 behaviors with properties")
    print(f"   - 8 relations with parameters")
    print(f"   - 48 API methods")
    print(f"   - 3 complete working examples")

    # Create scene maker agent with ONLY handoff tool
    print("\n🤖 Creating scene_maker agent...")
    scene_agent = ops.agent.create(
        id="scene_maker",
        describe="Creates simulation scene scripts using full knowledge of assets, behaviors, relations, and API",
        instructions=scene_maker_prompt,  # FULL auto-generated prompt as instructions
        tools=["handoff"],  # ONLY handoff tool
        model="claude-sonnet-4-5",  # Use Sonnet for code generation
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": "Submit complete scene script. Include ops.create_scene() through ops.compile() with all setup, rewards, cameras, and validation like you see in examples. Function content ONLY - no imports or function definition needed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "scene_name": {
                            "type": "string",
                            "description": "Descriptive name for the scene (e.g., 'kitchen_breakfast', 'tower_stacking')"
                        },
                        "scene_description": {
                            "type": "string",
                            "description": "Brief description of what the scene contains and demonstrates"
                        },
                        "python_script": {
                            "type": "string",
                            "description": "The complete scene code from ops.create_scene() to ops.compile(). Include all setup, assets, rewards, cameras. Function content ONLY - no imports or def statements."
                        }
                    },
                    "required": ["scene_name", "scene_description", "python_script"]
                }
            }
        }
    )

    print(f"✅ Agent created: scene_maker")
    print(f"   Tools: handoff only")
    print(f"   Prompt size: {len(scene_maker_prompt)} chars")

    # Test with NEW request - not in examples!
    print("\n📋 Testing with request NOT in examples...")
    user_request = "Create a warehouse scene with a shelf, put a bowl on the shelf, then stack 3 blocks on top of each other inside the bowl. Track if the blocks stay stacked."

    print(f"\n👤 User Request (NEW - not in examples): {user_request}")

    result = await ops.start_root_task(
        task=user_request,
        main_agent="scene_maker"
    )

    print("\n" + "="*70)
    print("RESULT:")
    print("="*70)
    print(result)

    return True


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_scene_maker_agent())
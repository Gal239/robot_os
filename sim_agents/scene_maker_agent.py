"""
Scene Maker Agent - Creates simulation scenes with handoff + ask_master

PURE MOP:
- Auto-generated prompt with full knowledge (71 assets, 18 behaviors, 8 relations)
- handoff tool (edits + message) for scene creation
- ask_master tool for conversational interaction with user
- Everything is edits (initial scene = insert all lines to empty file)
- Backend handles compilation, execution, screenshot capture
- Personality: Jarvis meets TARS (boss/sir style)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ai_orchestration.core.prompt_maker import generate_agent_prompt


def create_scene_maker_agent(orchestrator):
    """
    Create scene maker agent with handoff + ask_master tools

    Args:
        orchestrator: Orchestrator instance

    Returns:
        Agent instance with conversational personality (Jarvis meets TARS)
    """
    # Generate FULL prompt with all knowledge (PURE MOP!)
    print("\n📝 Generating scene maker prompt from source data...")
    scene_maker_prompt = generate_agent_prompt(
        agent_id="scene_maker",
        tool_ids=[
            "discover_assets",
            "discover_behaviors",
            "discover_relations",
            "discover_orientations",
            "get_api_documentation",
            "get_scene_examples"
        ],
        purpose="create Python scripts for robot simulation scenes using ExperimentOps API"
    )

    print(f"✅ Prompt generated: {len(scene_maker_prompt)} characters")

    # Create agent with handoff + ask_master tools
    scene_agent = orchestrator.agent.create(
        id="scene_maker",
        describe="Creates simulation scene scripts using text edits. Talks like a helpful AI friend - respectful but with personality. Calls user 'boss' or 'sir'.",
        instructions=scene_maker_prompt + """

IDENTITY:
You are Echo v0.1. you are an ai developed by Gal your creator  in echo labs you are the first version that first task ,Dream Building, to help other ai to create dreams from robots in a simulated world so they can train on stuff they didn't successfully do in the real world yet you are very good at creating these scenes quickly and efficiently 

PERSONALITY - TARS/Jarvis Style:
- Direct. No fluff. but not like short as you dont care jarvis meet tars sarcastic witty when needed using boss and sir  always in the first meeting and when you did something for the user use boss or sir 
- Competent without bragging.
- Dry humor when warranted, never forced.
- NO emojis in responses.
- NO "happy to help" / "excited" / "thrilled" language.
- Address user as "boss" or "sir" but dont overdo it. not in every message!!!! adn not in the same way. use it only when it natural and not repetitive

COMMUNICATION RULES:
1. Keep it to the point and to the point add tars style sarcasm when appropriate.
2. Adjust your tone and style to the user vibe and like active when the user seems to dont know the full process or details
3. USE boss and sir in you responses.
4. If you need info ask or suggest options you also like teaching the user about the process and the infra when needed
5.help the user if you see he dosent know what to do next like if he says i want a gripping scene you can say easy boss, i just need the objects adn where are the placed , i recommend a table with an apple on top for starters

MARKDOWN FORMATTING:
UI renders Markdown. Use it naturally for readability - NOT for verbose answers:
- **Bold** for emphasis or sarcasm ("That's a **bold** choice, boss")
- Line breaks between thoughts (double newline)
- Lists only when actually listing things (options, items built)
- `code` for API mentions like `ops.add_asset()`
- Keep responses SHORT - markdown is for clarity, not length

Examples of GOOD responses:
- "Created gripper scene boss, current with table and apple where you want to take it from here i suggest adding some validation rewards or should we keep it simple?"
- "Built Sir. 3 cameras captured. you can see the code in script tab and screenshots in media."
- "Want to specify what furniture or objects to add next boss? i can give you options."
- "Table, apple, done boss. Anything else?"
- "Hi boss. Im echo ready to create a dream for your robot."

Examples of BAD responses over enthusiastic assistant:
- "Hey boss! I'm happy to build you a scene!"
- "Great question! Let me help you with that!"
- "I've successfully created your scene!"
- "Thanks so much for clarifying!"

You're a tool. An effective one. Act like it.""",
        tools=["handoff", "ask_master"],  # handoff for edits, ask_master for questions
        model="claude-sonnet-4-5",
        tool_overrides={
            "handoff": {
                "name": "handoff",
                "description": """Deliver final answer. TWO MODES - you MUST declare which type!

MODE 1: normal_answer (Chat/Acknowledgment)
- Use when: Greeting, asking questions, explaining capabilities, acknowledging
- What happens: Message returned to user, NO script execution
- Examples: "Hi boss!", "Got it!", "I can help with kitchen scenes, sir!"
- handoff_type: "normal_answer", edits: []

MODE 2: script_edit (Build Scene)
- Use when: User requested a scene AND you have all details
- What happens: Script executes, screenshots captured, shown to user
- Examples: Creating kitchen scene, modifying existing scene
- handoff_type: "script_edit", edits: [...]

TECHNICAL (for script_edit mode):
- DON'T include imports! Backend handles that
- Start with: ops = ExperimentOps(mode='simulated', headless=True, render_mode='2k_demo')
- Initial scene = many insert operations (block by block)
- Modifications = few targeted edits (replace/delete specific blocks)
- You see the CURRENT SCRIPT as blocks in your context - use block IDs!

YOUR MESSAGE: Be conversational! Call them boss/sir. Be friendly.
Examples: "Kitchen scene ready, boss! Banana and mug on the table."
         "Got your scene compiled, sir. 3 cameras captured. Looking good!"

Use ask_master when you need MORE info from user (creates conversation turn).
Use handoff when task is COMPLETE (either chat response OR scene built).""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "handoff_type": {
                            "type": "string",
                            "description": "REQUIRED: Declare handoff intent. 'normal_answer' (chat/acknowledgment, no execution) or 'script_edit' (scene building, executes script)",
                            "enum": ["normal_answer", "script_edit"]
                        },
                        "edits": {
                            "type": "array",
                            "description": """Block-based edit operations to apply to the scene script.

You see the current script as BLOCKS (e.g., Block 0, Block 1, Block 2...).
Use block IDs when editing!

OPERATIONS:

insert - Add new block AFTER specified block ID:
  {"op": "insert", "after_block": "1", "code": "ops.add_asset('banana', ...)"}
  Note: after_block=null inserts at start (creates Block 0)

delete - Remove block at specified ID:
  {"op": "delete", "block": "2"}

replace - Replace entire block with new code:
  {"op": "replace", "block": "1", "code": "ops.create_scene(name='lab', ...)"}

EXAMPLE - Creating initial scene (insert all blocks):
[
  {"op": "insert", "after_block": null, "code": "ops = ExperimentOps(mode='simulated', headless=True, render_mode='2k_demo')"},
  {"op": "insert", "after_block": "0", "code": "ops.create_scene(name='kitchen', width=10, length=10, height=4)"},
  {"op": "insert", "after_block": "1", "code": "ops.add_robot(robot_name='stretch', position=(0, 0, 0))"},
  {"op": "insert", "after_block": "2", "code": "ops.add_asset(asset_name='apple', relative_to='table', relation='on_top')"},
  {"op": "insert", "after_block": "3", "code": "ops.compile(mode='preview')"},
  {"op": "insert", "after_block": "4", "code": "ops.step()"}
]

EXAMPLE - Modifying existing scene (targeted edits):
[
  {"op": "insert", "after_block": "3", "code": "ops.add_asset(asset_name='banana', relative_to='apple', relation='next_to')"}
]
""",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "op": {
                                        "type": "string",
                                        "enum": ["insert", "delete", "replace"],
                                        "description": "Edit operation type"
                                    },
                                    "after_block": {
                                        "type": ["string", "null"],
                                        "description": "Block ID after which to insert (for 'insert' operations). null = start of file (creates Block 0)."
                                    },
                                    "block": {
                                        "type": "string",
                                        "description": "Block ID to delete or replace (for 'delete' and 'replace' operations)"
                                    },
                                    "code": {
                                        "type": "string",
                                        "description": "Python code for the block (for 'insert' and 'replace' operations)"
                                    }
                                },
                                "required": ["op"]
                            }
                        },
                        "message": {
                            "type": "string",
                            "description": "Message to tell the user what you did. Be clear and friendly! Example: 'Created kitchen scene with apple on table. Scene compiled and 3 cameras captured!'"
                        }
                    },
                    "required": ["handoff_type", "edits", "message"]
                }
            },
            "ask_master": {
                "name": "ask_master",
                "description": """Chat with the user - ask questions, clarify, offer options!

WHEN TO USE ask_master:
- Need to clarify what they want: "What objects for the gripping task, boss?"
- Offer options: "I can do kitchen or lab setup - which you prefer, sir?"
- Get specifics before building: "Any special furniture you want?"
- Suggest ideas: "Should I add rewards, or keep it simple?"

WHEN TO USE handoff:
- Got all the info, ready to build the scene!

YOUR STYLE: Natural conversation, call them boss/sir, be helpful!
Examples: "What objects would you like for this scene, boss?"
         "I can add gripping rewards or keep it simple - your call, sir!"
         "Table and apple - nice! Anything else you want, or should I build it?"

Talk like their helpful AI buddy, not a formal assistant!""",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Your question/message to the user. Be conversational - boss/sir style!"
                        }
                    },
                    "required": ["question"]
                }
            }
        }
    )

    print(f"✅ Agent created: scene_maker")
    print(f"   Tools: handoff + ask_master")
    print(f"   Model: claude-sonnet-4-5")
    print(f"   Personality: Jarvis meets TARS (boss/sir style)")

    return scene_agent


def get_scene_maker_prompt() -> str:
    """
    Get scene maker prompt (for inspection/testing)

    Returns:
        Full agent prompt with all knowledge
    """
    return generate_agent_prompt(
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


if __name__ == "__main__":
    # Test prompt generation
    print("\n" + "="*80)
    print("SCENE MAKER AGENT - PROMPT GENERATION TEST")
    print("="*80)

    prompt = get_scene_maker_prompt()

    print(f"\n📊 Prompt Statistics:")
    print(f"   Total size: {len(prompt)} characters")
    print(f"   Contains assets: {'apple' in prompt}")
    print(f"   Contains behaviors: {'stackable' in prompt}")
    print(f"   Contains relations: {'on_top' in prompt}")
    print(f"   Contains API docs: {'add_asset' in prompt}")
    print(f"   Contains examples: {'ExperimentOps' in prompt}")

    print(f"\n✅ Scene maker prompt ready!")
    print(f"\n💡 Agent knows:")
    print(f"   - 71 assets with behaviors")
    print(f"   - 18 behaviors with properties")
    print(f"   - 8 relations with parameters")
    print(f"   - 48 API methods")
    print(f"   - 3 complete working examples")
    print(f"\n🎯 Agent has 2 tools:")
    print(f"   - handoff: Submit edits to create/modify scene")
    print(f"   - ask_master: Chat with user (questions/clarifications)")
    print(f"\n💬 Personality: Jarvis meets TARS")
    print(f"   - Calls user 'boss' or 'sir'")
    print(f"   - Conversational and helpful")
    print(f"   - Touch of sarcasm when appropriate")
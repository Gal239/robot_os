"""
Echo Conversation Manager - Thin Layer for Dream Factory
Connects UI to Orchestrator using MOP pattern
"""
import sys
import os
from pathlib import Path

# Ensure paths are set up
# echo_ops.py is at: simulation_center/sim_agents/ui/backend/echo_ops.py
simulation_center = Path(__file__).parent.parent.parent.parent
if str(simulation_center) not in sys.path:
    sys.path.insert(0, str(simulation_center))

# CRITICAL: Set custom database path BEFORE importing orchestrator
# This ensures all session data goes to sim_agents/database/
custom_db_path = simulation_center / "sim_agents" / "database"
custom_db_path.mkdir(parents=True, exist_ok=True)

# Override global database path
import ai_orchestration.utils.global_config as global_config
from ai_orchestration.utils.auto_db import AutoDB
global_config.database_path = custom_db_path
global_config.agent_engine_db = AutoDB(local_path=str(custom_db_path))

print(f"[Echo] Using custom database path: {custom_db_path}")

import asyncio
from ai_orchestration.core.orchestrator import Orchestrator
from sim_agents.scene_maker_agent import create_scene_maker_agent
from sim_agents.scene_maker_handoff_handler import handle_scene_maker_handoff, format_script_as_blocks, render_script_with_imports


class EchoConversationManager:
    """Thin layer connecting UI to Orchestrator"""

    def __init__(self):
        """Create orchestrator - no agent yet"""
        self.ops = Orchestrator()
        self.agent = None

        # UI state (quick access to latest data)
        self.current_script = ""
        self.latest_screenshots = {}
        self.latest_scene_data = {}
        self.status = "not_started"

    def start_session(self):
        """
        Explicit session start:
        1. Create scene_maker agent (Dream Factory!)
        2. Register with ops
        3. Create Document for scene script
        4. Register handoff handler
        5. Return welcome message
        """
        print("\n[Echo] Starting Dream Factory session...")
        self.status = "initializing"

        # Create and register agent with ops
        self.agent = create_scene_maker_agent(self.ops)
        print(f"[Echo] Agent registered: {self.agent.agent_id}")

        # Create Document for scene script (empty initially)
        self.ops.document_ops.create_document(
            path="scene_script.py",
            content="",
            created_by="scene_maker"
        )
        print(f"[Echo] Scene script document created")

        # Register handoff handler (updates UI state)
        def handoff_wrapper(handoff_type, edits, message, scene_name=None, current_script=""):
            """
            Wrapper that updates UI state after handoff
            OFFENSIVE: Crashes if result structure is wrong!
            """
            result = handle_scene_maker_handoff(
                handoff_type=handoff_type,  # Pass through explicit type
                edits=edits,
                message=message,
                scene_name=scene_name,
                document_ops=self.ops.document_ops,  # Pass document_ops for edit tracking
                execute_scene=False  # DISABLED by default - just create script, don't compile
            )

            # Update UI state - CRASH if keys missing!
            if result["success"]:  # No .get() - CRASH if missing!
                self.current_script = result["script"]  # CRASH if missing!
                self.latest_screenshots = result["screenshots"]  # CRASH if missing!
                self.latest_scene_data = result["ui_data"]  # CRASH if missing!
                print(f"[Echo] Handoff complete: {len(self.latest_screenshots)} cameras")
            else:
                # Handoff failed - CRASH to see why!
                raise RuntimeError(f"Handoff failed: {result}")

            return result

        self.ops.handoff_handlers['scene_maker'] = handoff_wrapper
        print("[Echo] Handoff handler registered")

        self.status = "ready"

        # Dream Factory welcome message!
        return {
            "success": True,
            "session_id": self.ops.graph_ops.modal.session_id,
            "welcome_message": "Hi boss! Welcome to the Dream Factory. What scene do we want to build today?"
        }

    async def send_message(self, user_message: str):
        """
        Send message to Echo
        EXACT COPY of test_8 pattern - uses start_root_task()!
        """
        print(f"\n[User] {user_message}")
        self.status = "thinking"

        # Clear previous edit stream before processing new message
        # This ensures we only get delta edits for THIS turn
        self.ops.document_ops.clear_edit_stream()

        # Use test_8 pattern - start_root_task handles execution automatically!
        result = await self.ops.start_root_task(
            task=user_message,
            main_agent="scene_maker"
        )

        # Result structure: {"result": {...}, "meta_log": ...}
        task_result = result["result"]  # CRASH if missing!

        # Check for handoff first (most complete response)
        if "handoff_result" in task_result:
            handoff_result = task_result["handoff_result"]

            if handoff_result["success"]:  # CRASH if missing!
                echo_response = handoff_result["message"]  # CRASH if missing!
                print(f"[Echo handoff] {echo_response}")

                self.status = "ready"
                return {
                    "success": True,
                    "echo_response": echo_response,
                    "response_type": "handoff",
                    "screenshots": self.latest_screenshots,
                    "scene_data": self.latest_scene_data
                }
            else:
                # Handoff failed!
                raise RuntimeError(f"Handoff failed: {handoff_result}")

        # Check for ask_master (agent asking question)
        # When agent uses ask_master, it creates a human task that's pending
        pending = self.ops.graph_ops.modal.get_pending_human_task()
        if pending:
            human_task_id, human_node = pending
            echo_response = human_node.task_payload  # The question!
            print(f"[Echo ask_master] {echo_response}")

            self.status = "ready"
            return {
                "success": True,
                "echo_response": echo_response,
                "response_type": "ask_master"
            }

        # No recognized response type!
        self.status = "ready"
        raise RuntimeError(f"Unknown response! task_result: {task_result}, pending: {pending}")

    # Simple getters (UI state) - OFFENSIVE: return exactly what we have!
    def get_status(self):
        return self.status

    def get_screenshots(self):
        return self.latest_screenshots

    def get_scene_script(self):
        return self.current_script

    def get_scene_script_from_workspace(self):
        """Get live script from workspace WITH imports (for UI display)"""
        return render_script_with_imports(self.ops.document_ops)

    def get_script_blocks(self):
        """Get script as blocks for agent context view"""
        return format_script_as_blocks(self.ops.document_ops)

    def get_edit_stream(self):
        """Get live edit stream for UI animations"""
        return self.ops.document_ops.get_edit_stream()

    def clear_edit_stream(self):
        """Clear edit stream after UI consumed it"""
        self.ops.document_ops.clear_edit_stream()

    def get_workspace_info(self):
        """Get workspace info for debugging"""
        workspace = self.ops.document_ops.workspace
        return {
            "session_id": workspace.session_id,
            "documents": list(workspace.documents.keys()),
            "document_count": len(workspace.documents)
        }

    def get_scene_data(self):
        return self.latest_scene_data  # NO defensive `or {}` - return what we have!

    def get_metalog(self):
        """Get metalog from graph"""
        roots = [
            nid for nid, node in self.ops.graph_ops.modal.nodes.items()
            if node.parent_task_id is None
        ]

        if roots:
            return self.ops.log_ops.build_log_context(
                self.ops.graph_ops.modal,
                roots[-1]
            )

        return ""

    def get_conversation(self):
        """
        Get conversation history from graph
        OFFENSIVE: Crashes if data structure is wrong - no silent failures!
        """
        print(f"[get_conversation] Total nodes in graph: {len(self.ops.graph_ops.modal.nodes)}", flush=True)
        print(f"[get_conversation] All node IDs: {list(self.ops.graph_ops.modal.nodes.keys())}", flush=True)

        # Root nodes created by start_root_task have parent_task_id="human" (the initiator)
        roots = [
            nid for nid, node in self.ops.graph_ops.modal.nodes.items()
            if node.parent_task_id == "human"
        ]

        print(f"[get_conversation] Root nodes found: {len(roots)}", flush=True)
        for rid in roots:
            n = self.ops.graph_ops.modal.nodes[rid]
            print(f"  Root {rid}: agent={n.agent_id}, status={n.status.value}, payload={n.task_payload[:50]}", flush=True)

        conversation = []

        for root_id in roots:
            node = self.ops.graph_ops.modal.nodes[root_id]

            # User message - CRASH if missing!
            conversation.append({
                "role": "user",
                "message": node.task_payload,  # No .get() - crash if missing!
                "timestamp": node.created_at
            })

            # Echo response - OFFENSIVE: expect exact structure!
            handoffs = [e for e in node.tool_timeline if e["type"] == "handoff"]  # CRASH if no "type" key!
            print(f"  Root {root_id}: {len(handoffs)} handoffs in timeline", flush=True)
            if handoffs:
                last_handoff = handoffs[-1]
                result = last_handoff["result"]  # CRASH if no "result"!

                conversation.append({
                    "role": "echo",
                    "message": result["message"],  # CRASH if no "message"!
                    "timestamp": last_handoff["timestamp"]  # CRASH if no "timestamp"!
                })

        print(f"[get_conversation] Returning {len(conversation)} messages", flush=True)
        return conversation

"""
Scene Maker Database - Extends DatabaseOps (PURE MOP!)

Reuses ALL ExperimentOps infrastructure:
- Timeline data (cameras, sensors, actions, actuators)
- Physics snapshots (scene_state/)
- UI snapshots (ui_db/hot_compile_N/)
- GOD MODAL (experiment.json)

Adds ONLY Scene Maker specific data:
- Conversation history
- Agent state evolution
- Edit operations
- Knowledge base snapshot
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class SceneMakerDB:
    """
    Database for Scene Maker - EXTENDS DatabaseOps (not replaces!)

    Uses existing ExperimentOps database structure and adds scene-specific data.
    No duplication - references existing timeline/sensor/camera data!
    """

    def __init__(self, root_path: str = "database"):
        self.root_path = Path(root_path)
        self.db_ops = None  # Set when session is created
        self.session_id = None
        self.scene_maker_dir = None

    def create_session(self, scene_name: str, experiment_ops) -> str:
        """
        Create new session - REUSES ExperimentOps!

        Args:
            scene_name: User-friendly scene name
            experiment_ops: ExperimentOps instance (has DatabaseOps!)

        Returns:
            session_id: experiment_id from ExperimentOps
        """
        # Get experiment_id from ExperimentOps (already has timestamp!)
        self.session_id = experiment_ops.experiment_id
        self.db_ops = experiment_ops.db_ops

        # Create scene_maker subdirectory in existing experiment
        self.scene_maker_dir = Path(experiment_ops.experiment_dir) / "scene_maker"
        self.scene_maker_dir.mkdir(exist_ok=True, parents=True)

        # Initialize Scene Maker specific files
        self._init_scene_maker_files(scene_name)

        print(f"📁 Scene Maker session created: {self.session_id}")
        print(f"   Scene name: {scene_name}")
        print(f"   Directory: {self.scene_maker_dir}")

        return self.session_id

    def _init_scene_maker_files(self, scene_name: str):
        """Initialize Scene Maker specific files"""

        # conversation.json - All turns with user
        conversation_file = self.scene_maker_dir / "conversation.json"
        conversation_file.write_text(json.dumps({
            "scene_name": scene_name,
            "created_at": datetime.now().isoformat(),
            "turns": []
        }, indent=2))

        # agent_state.json - Agent prompt evolution
        agent_file = self.scene_maker_dir / "agent_state.json"
        agent_file.write_text(json.dumps({
            "agent_id": "scene_editor",
            "base_instructions": "",
            "prompt_evolution": []
        }, indent=2))

        # edits_history.json - All edit operations
        edits_file = self.scene_maker_dir / "edits_history.json"
        edits_file.write_text(json.dumps({
            "edits": []
        }, indent=2))

        print(f"   ✓ Initialized scene_maker files")

    # === CONVERSATION ===

    def save_conversation_turn(self, turn: Dict):
        """
        Save conversation turn with references to existing data

        Args:
            turn: {
                "turn_number": int,
                "timestamp": str,
                "user_message": str,
                "agent_action": str,
                "edits_applied": List[Dict],
                "ui_snapshot_ref": str,  # Reference to ui_db/
                "timeline_frame": int,   # Reference to timeline/frames/
                "all_cameras": Dict      # ALL camera screenshot paths
            }
        """
        if not self.scene_maker_dir:
            raise ValueError("Session not created! Call create_session() first")

        conv_file = self.scene_maker_dir / "conversation.json"
        data = json.loads(conv_file.read_text())

        # Add turn
        data["turns"].append(turn)

        # Save
        conv_file.write_text(json.dumps(data, indent=2))

    def load_conversation(self) -> Dict:
        """Load all conversation turns"""
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        conv_file = self.scene_maker_dir / "conversation.json"
        return json.loads(conv_file.read_text())

    # === EDITS HISTORY ===

    def save_edit(self, edit: Dict):
        """
        Save edit operation

        Args:
            edit: {
                "edit_number": int,
                "turn_number": int,
                "timestamp": str,
                "operation": str,  # insert, delete, replace
                "line": int,
                "code": str,
                "success": bool
            }
        """
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        edits_file = self.scene_maker_dir / "edits_history.json"
        data = json.loads(edits_file.read_text())

        # Add edit
        data["edits"].append(edit)

        # Save
        edits_file.write_text(json.dumps(data, indent=2))

    def load_edits_history(self) -> Dict:
        """Load all edit operations"""
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        edits_file = self.scene_maker_dir / "edits_history.json"
        return json.loads(edits_file.read_text())

    # === AGENT STATE ===

    def save_agent_prompt_snapshot(self, prompt: str, turn: int):
        """
        Save agent prompt evolution

        Args:
            prompt: Current agent prompt (full text)
            turn: Turn number
        """
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        agent_file = self.scene_maker_dir / "agent_state.json"
        data = json.loads(agent_file.read_text())

        # Add snapshot
        data["prompt_evolution"].append({
            "turn": turn,
            "prompt_size": len(prompt),
            "has_live_script": "Line 1:" in prompt,
            "timestamp": datetime.now().isoformat()
        })

        # Store base instructions on first turn
        if turn == 1 and not data["base_instructions"]:
            data["base_instructions"] = prompt

        # Save
        agent_file.write_text(json.dumps(data, indent=2))

    def load_agent_state(self) -> Dict:
        """Load agent state"""
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        agent_file = self.scene_maker_dir / "agent_state.json"
        return json.loads(agent_file.read_text())

    # === KNOWLEDGE BASE (AUTO-DISCOVERY!) ===

    def generate_knowledge_base(self) -> Dict:
        """
        Auto-generate knowledge base from source files - PURE MOP!

        Returns:
            {
                "assets": Dict,      # From ASSETS.json (71 items)
                "behaviors": Dict,   # From BEHAVIORS.json (18 items)
                "relations": Dict,   # From RELATIONS.json (8 items)
                "robots": List,      # Available robots
                "generated_at": str
            }
        """
        # Load from source files (PURE MOP!)
        base_path = Path(__file__).parent.parent / "data"

        assets_json = base_path / "ASSETS.json"
        behaviors_json = base_path / "BEHAVIORS.json"
        relations_json = base_path / "RELATIONS.json"

        knowledge = {
            "assets": json.loads(assets_json.read_text()) if assets_json.exists() else {},
            "behaviors": json.loads(behaviors_json.read_text()) if behaviors_json.exists() else {},
            "relations": json.loads(relations_json.read_text()) if relations_json.exists() else {},
            "robots": ["stretch"],  # Could auto-discover from robot configs
            "generated_at": datetime.now().isoformat()
        }

        # Save to scene_maker directory
        if self.scene_maker_dir:
            kb_file = self.scene_maker_dir / "knowledge_base.json"
            kb_file.write_text(json.dumps(knowledge, indent=2))

            print(f"📚 Knowledge base generated:")
            print(f"   Assets: {len(knowledge['assets'])}")
            print(f"   Behaviors: {len(knowledge['behaviors'])}")
            print(f"   Relations: {len(knowledge['relations'])}")

        return knowledge

    def load_knowledge_base(self) -> Dict:
        """Load knowledge base snapshot"""
        if not self.scene_maker_dir:
            raise ValueError("Session not created!")

        kb_file = self.scene_maker_dir / "knowledge_base.json"
        if kb_file.exists():
            return json.loads(kb_file.read_text())
        else:
            # Generate if not exists
            return self.generate_knowledge_base()

    # === UI EXPORT (REFERENCES EXISTING DATA!) ===

    def export_for_ui(self) -> Dict:
        """
        Export all data for UI - REFERENCES existing data (not copies!)

        Returns:
            Complete data structure with:
            - Scene Maker specific data (conversation, agent, edits)
            - References to ALL existing DatabaseOps data:
              - experiment.json (GOD MODAL)
              - timeline/ (cameras, sensors, actions, assets)
              - scene_state/ (physics snapshots)
              - ui_db/ (hot compile snapshots)
        """
        if not self.db_ops:
            raise ValueError("Session not created!")

        exp_dir = Path(self.db_ops.experiment_dir)

        # Load Scene Maker data
        conversation = self.load_conversation()
        agent_state = self.load_agent_state()
        edits_history = self.load_edits_history()
        knowledge_base = self.load_knowledge_base()

        return {
            # Session info
            "session_id": self.session_id,
            "scene_name": conversation["scene_name"],
            "created_at": conversation["created_at"],
            "total_turns": len(conversation["turns"]),

            # Scene Maker specific data
            "conversation": conversation,
            "agent_state": agent_state,
            "edits_history": edits_history,
            "knowledge_base": knowledge_base,

            # === REFERENCES TO EXISTING DATABASE_OPS DATA ===

            # GOD MODAL
            "experiment_json": str(exp_dir / "experiment.json"),
            "scene_xml": str(exp_dir / "scene.xml"),

            # Timeline root
            "timeline_root": str(exp_dir / "timeline"),

            # ALL CAMERAS (10+ with videos and screenshots)
            "cameras": self._get_camera_refs(exp_dir),

            # Sensor data (IMU, LiDAR, odometry, gripper force)
            "sensors": {
                "imu": str(exp_dir / "timeline/sensors/imu.csv"),
                "lidar": str(exp_dir / "timeline/sensors/lidar.csv"),
                "odometry": str(exp_dir / "timeline/sensors/odometry.csv"),
                "gripper_force": str(exp_dir / "timeline/sensors/gripper_force.csv"),
                "respeaker": str(exp_dir / "timeline/sensors/respeaker.csv")
            },

            # Action/Actuator data (joint states, velocities, commands)
            "actions": str(exp_dir / "timeline/actions/action_queue.json"),
            "actuators": str(exp_dir / "timeline/actuators"),

            # Asset tracking (per-asset position, velocity, contacts)
            "assets": self._get_asset_refs(exp_dir),

            # Rewards
            "rewards": str(exp_dir / "timeline/rewards/rewards_view.json"),

            # Physics snapshots (per-frame)
            "physics_snapshots": str(exp_dir / "scene_state"),

            # UI snapshots (hot compiles)
            "ui_snapshots": self._get_ui_snapshot_refs(exp_dir),

            # Room view
            "room": str(exp_dir / "timeline/room_view.json")
        }

    def _get_camera_refs(self, exp_dir: Path) -> Dict[str, Dict[str, str]]:
        """Get references to ALL camera data"""
        cameras_dir = exp_dir / "timeline/cameras"

        refs = {}
        if cameras_dir.exists():
            for camera_dir in cameras_dir.iterdir():
                if camera_dir.is_dir():
                    camera_name = camera_dir.name
                    refs[camera_name] = {
                        "video": str(camera_dir / f"{camera_name}_rgb.avi"),
                        "screenshots_dir": str(camera_dir),
                        "name": camera_name
                    }

        return refs

    def _get_asset_refs(self, exp_dir: Path) -> Dict[str, str]:
        """Get references to all asset tracking data"""
        timeline_dir = exp_dir / "timeline"

        refs = {}
        if timeline_dir.exists():
            for asset_file in timeline_dir.glob("asset_*_view.json"):
                # Extract asset name from filename
                # asset_apple_view.json -> apple
                asset_name = asset_file.stem.replace("asset_", "").replace("_view", "")
                refs[asset_name] = str(asset_file)

        return refs

    def _get_ui_snapshot_refs(self, exp_dir: Path) -> List[str]:
        """Get references to all UI snapshots (hot compiles)"""
        ui_db_dir = exp_dir / "ui_db"

        refs = []
        if ui_db_dir.exists():
            for snapshot_dir in sorted(ui_db_dir.iterdir()):
                if snapshot_dir.is_dir() and snapshot_dir.name.startswith("hot_compile"):
                    refs.append(str(snapshot_dir))

        return refs

    # === UTILITIES ===

    def get_session_summary(self) -> Dict:
        """Get summary for UI display"""
        if not self.db_ops:
            raise ValueError("Session not created!")

        conversation = self.load_conversation()
        edits = self.load_edits_history()

        exp_dir = Path(self.db_ops.experiment_dir)

        # Get asset list
        assets = self._get_asset_refs(exp_dir)

        # Get last screenshot (if any)
        last_screenshot = None
        if conversation["turns"]:
            last_turn = conversation["turns"][-1]
            if "all_cameras" in last_turn:
                # Get first available camera screenshot
                cameras = last_turn["all_cameras"]
                if cameras:
                    last_screenshot = list(cameras.values())[0] if isinstance(cameras, dict) else None

        return {
            "session_id": self.session_id,
            "scene_name": conversation["scene_name"],
            "turns": len(conversation["turns"]),
            "edits": len(edits["edits"]),
            "assets": list(assets.keys()),
            "last_screenshot": last_screenshot,
            "created_at": conversation["created_at"],
            "experiment_dir": str(exp_dir)
        }

    def list_all_cameras(self) -> List[str]:
        """List all available cameras in timeline"""
        if not self.db_ops:
            raise ValueError("Session not created!")

        cameras = self._get_camera_refs(Path(self.db_ops.experiment_dir))
        return list(cameras.keys())

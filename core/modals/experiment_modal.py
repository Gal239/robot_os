"""
EXPERIMENT MODAL - GOD MODAL that owns ALL experiment configuration
PURE MOP: Self-saving, self-loading, compilable, timeline-aware

THE PATTERN: Master container for complete experiment state
- Owns: Scene, agents, config, snapshots (CONFIGURATION ONLY!)
- Does NOT own: Execution history (ExperimentArtifactModal owns that!)
- Separation: Configuration (us) vs Execution (artifact modal)

KEY FEATURES:
1. SAVE/LOAD: Complete experiment to/from JSON
2. RL STATE: All agent training state, progress, checkpoints
3. TIMELINE: Save snapshots at any point, restore to any snapshot
4. COMPILABLE: Load and resume from any saved state
5. PURE MOP: Self-describing, self-executing, LEGO composition

Usage:
    # Create experiment
    ops = ExperimentOps(headless=True)
    ops.create_scene("kitchen", width=5, length=5, height=3)
    ops.add_robot("stretch")
    ops.compile()

    # Export to GOD MODAL
    experiment = ops.to_experiment_modal()
    experiment.description = "Level 2A: Base rotation training"

    # Save complete state (configuration only!)
    experiment.save()

    # Train for 50 episodes...
    # Create timeline snapshot
    experiment.create_snapshot("checkpoint_episode_50", "After 50 episodes")
    experiment.save()

    # Later: Load and resume - MODAL COMPILES ITSELF!
    experiment = ExperimentModal.load("experiments/level_2a/experiment.json")

    # Restore to specific snapshot
    experiment.restore_snapshot("checkpoint_episode_50")

    # MODAL COMPILES ITSELF - PURE MOP!
    ops = experiment.compile()  # ExperimentModal knows how to run itself!
    ops.compile()  # Now compile MuJoCo
    # Continue training!

Directory Structure:
    experiment_dir/
    ├── experiment.json          # GOD MODAL - configuration (us!)
    └── experiment_artifact.json # Artifact Modal - execution history (them!)

Clean separation, zero redundancy!
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time
from datetime import datetime
import copy


@dataclass
class Snapshot:
    """Point-in-time configuration snapshot - TIMELINE FEATURE!

    Captures complete experiment state at a specific moment.
    Allows time-travel: restore to any saved point!

    PURE MOP: Self-saving, self-loading
    """
    timestamp: float
    name: str
    description: str = ""

    # Complete configuration at this point in time
    scene_data: Dict[str, Any] = field(default_factory=dict)
    config_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    step_number: Optional[int] = None  # Training step when created
    episode_number: Optional[int] = None  # Training episode when created

    def to_json(self) -> dict:
        """I know how to serialize myself - OFFENSIVE"""
        return {
            "timestamp": self.timestamp,
            "name": self.name,
            "description": self.description,
            "scene_data": self.scene_data,
            "config_data": self.config_data,
            "step_number": self.step_number,
            "episode_number": self.episode_number,
            "created_at": datetime.fromtimestamp(self.timestamp).isoformat()
        }

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself - OFFENSIVE"""
        return cls(
            timestamp=data["timestamp"],
            name=data["name"],
            description=data["description"],  # OFFENSIVE - crash if missing!
            scene_data=data["scene_data"],
            config_data=data["config_data"],
            step_number=data["step_number"],  # OFFENSIVE - crash if missing!
            episode_number=data["episode_number"]  # OFFENSIVE - crash if missing!
        )


@dataclass
class ExperimentModal:
    """GOD MODAL - Master container for experiment CONFIGURATION

    PURE MOP: Self-saving, self-loading, compilable

    OWNS (Configuration Only):
    - Scene (complete scene with robots, agents, assets, rewards)
    - Config (experiment parameters, mode, viewer settings)
    - Snapshots (timeline feature - save state at any point!)
    - Metadata (experiment_id, description, created_at)

    DOES NOT OWN (Execution History):
    - Step-by-step execution trace (ExperimentArtifactModal owns this!)
    - Actions/state/rewards per step (ExperimentArtifactModal owns this!)

    THE SILK THREAD:
    1. AUTO-DISCOVERY: Scene discovers all modals
    2. SELF-GENERATION: to_json() knows complete state
    3. SELF-RENDERING: Renders to experiment.json
    4. LEGO COMPOSITION: Scene holds agents, robots, assets
    5. MODAL-TO-MODAL: ExperimentModal + ArtifactModal = complete experiment
    """

    # Core configuration (LEGO composition!)
    scene: Any  # Scene modal with all modals inside
    config: Dict[str, Any] = field(default_factory=dict)

    # MuJoCo compiled state (CRITICAL for exact reproduction!)
    compiled_xml: Optional[str] = None  # Generated MuJoCo XML after compilation
    mujoco_state: Optional[Dict[str, Any]] = None  # qpos, qvel, etc. from MuJoCo

    # Timeline feature (save state at any point!)
    snapshots: Dict[str, Snapshot] = field(default_factory=dict)

    # Metadata
    experiment_id: str = ""
    description: str = ""
    created_at: str = ""
    experiment_dir: Path = field(default_factory=lambda: Path("."))

    # Version tracking
    version: str = "1.0"

    def __post_init__(self):
        """Auto-initialize metadata"""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

        if not isinstance(self.experiment_dir, Path):
            self.experiment_dir = Path(self.experiment_dir)

    # === SAVE/LOAD (PURE MOP!) ===

    def save(self, path: Optional[str] = None) -> Path:
        """Save complete experiment configuration to JSON - PURE MOP!

        Args:
            path: Path to save JSON (default: experiment_dir/experiment.json)

        Returns:
            Path to saved file

        Saves CONFIGURATION ONLY (not execution history!)
        ExperimentArtifactModal saves execution history separately.
        """
        if path is None:
            path = self.experiment_dir / "experiment.json"
        else:
            path = Path(path)

        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build complete configuration
        data = self.to_json()

        # Write JSON
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Quiet mode - no save prints

        return path

    @classmethod
    def load(cls, path: str) -> 'ExperimentModal':
        """Load experiment configuration from JSON - PURE MOP!

        Args:
            path: Path to experiment.json

        Returns:
            ExperimentModal instance

        Loads CONFIGURATION ONLY (not execution history!)
        Use ExperimentArtifactModal.load() for execution history.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Experiment not found: {path}")

        with open(path) as f:
            data = json.load(f)

        experiment = cls.from_json(data)
        experiment.experiment_dir = path.parent

        # Quiet mode - no load prints

        return experiment

    # === TIMELINE FEATURE (Save/Restore Snapshots) ===

    def create_snapshot(
        self,
        name: str,
        description: str = "",
        step_number: Optional[int] = None,
        episode_number: Optional[int] = None
    ):
        """Create snapshot of current configuration - TIMELINE FEATURE!

        Args:
            name: Snapshot name (e.g., "checkpoint_episode_50")
            description: Human-readable description
            step_number: Training step when created
            episode_number: Training episode when created

        Allows time-travel: Save state at any point, restore later!
        """
        # Deep copy current state (so future changes don't affect snapshot)
        snapshot = Snapshot(
            timestamp=time.time(),
            name=name,
            description=description,
            scene_data=self.scene.to_json() if hasattr(self.scene, 'to_json') else {},
            config_data=copy.deepcopy(self.config),
            step_number=step_number,
            episode_number=episode_number
        )

        self.snapshots[name] = snapshot
        print(f"✓ Created snapshot: {name}")
        if description:
            print(f"  {description}")

    def restore_snapshot(self, name: str):
        """Restore experiment to saved snapshot - TIME TRAVEL!

        Args:
            name: Snapshot name to restore

        OFFENSIVE: Crashes if snapshot not found
        """
        if name not in self.snapshots:
            available = list(self.snapshots.keys())
            raise ValueError(f"Snapshot '{name}' not found! Available: {available}")

        snapshot = self.snapshots[name]

        # Restore scene from snapshot
        from .scene_modal import Scene
        self.scene = Scene.from_json(snapshot.scene_data)

        # Restore config
        self.config = copy.deepcopy(snapshot.config_data)

        print(f"✓ Restored to snapshot: {name}")
        print(f"  Timestamp: {datetime.fromtimestamp(snapshot.timestamp).isoformat()}")
        if snapshot.description:
            print(f"  {snapshot.description}")

    def list_snapshots(self) -> list:
        """List all available snapshots

        Returns:
            List of snapshot names sorted by timestamp
        """
        return sorted(
            self.snapshots.keys(),
            key=lambda k: self.snapshots[k].timestamp
        )

    # === SERIALIZATION (PURE MOP!) ===

    def to_json(self) -> dict:
        """I know how to serialize myself - OFFENSIVE & COMPLETE

        Returns complete configuration (not execution history!)
        """
        return {
            "_meta": {
                "version": self.version,
                "pattern": "GOD-MODAL",
                "description": "Experiment configuration (not execution history)",
                "note": "Execution history saved by ExperimentArtifactModal separately",
                "created": self.created_at
            },
            "experiment_info": {
                "id": self.experiment_id,
                "description": self.description,
                "created_at": self.created_at,
                "experiment_dir": str(self.experiment_dir)
            },
            "scene": self.scene.to_json() if hasattr(self.scene, 'to_json') else {},
            "config": self.config,
            "compiled_xml": self.compiled_xml,  # MuJoCo XML (CRITICAL for exact reproduction!)
            "mujoco_state": self.mujoco_state,  # MuJoCo state (qpos, qvel, etc.)
            "snapshots": {
                name: snapshot.to_json()
                for name, snapshot in self.snapshots.items()
            }
        }

    @classmethod
    def from_json(cls, data: dict) -> 'ExperimentModal':
        """I know how to deserialize myself - OFFENSIVE"""
        # Import Scene here to avoid circular import
        from .scene_modal import Scene

        # Restore scene
        scene_data = data["scene"]  # OFFENSIVE - crash if missing!
        scene = Scene.from_json(scene_data) if scene_data else None

        # Restore snapshots
        snapshots = {}
        for name, snapshot_data in data["snapshots"].items():  # OFFENSIVE - crash if missing!
            snapshots[name] = Snapshot.from_json(snapshot_data)

        # Create experiment
        experiment_info = data["experiment_info"]  # OFFENSIVE - crash if missing!
        return cls(
            scene=scene,
            config=data["config"],  # OFFENSIVE - crash if missing!
            compiled_xml=data.get("compiled_xml"),  # LEGITIMATE - optional (GOD MODAL restoration)
            mujoco_state=data.get("mujoco_state"),  # LEGITIMATE - optional (GOD MODAL restoration)
            snapshots=snapshots,
            experiment_id=experiment_info["id"],  # OFFENSIVE - crash if missing!
            description=experiment_info["description"],  # OFFENSIVE - crash if missing!
            created_at=experiment_info["created_at"],  # OFFENSIVE - crash if missing!
            experiment_dir=Path(experiment_info["experiment_dir"]),  # OFFENSIVE - crash if missing!
            version=data.get("_meta", {}).get("version", "1.0")  # LEGITIMATE - version has default
        )

    # === COMPILE - ExperimentModal CAN CREATE ExperimentOps! ===

    def compile(self):
        """Compile experiment to runnable ExperimentOps - OFFENSIVE!

        ExperimentModal CAN DO THINGS! It creates ExperimentOps from itself.
        This is PURE MOP - modal knows how to execute itself!

        If compiled_xml is available, restores MuJoCo backend from saved state!

        Returns:
            ExperimentOps ready to run

        Usage:
            experiment = ExperimentModal.load("experiment.json")
            ops = experiment.compile()  # Modal compiles itself!
            # Backend is restored from saved XML (skips regeneration!)
            ops.step()  # Ready to run!
        """
        # Import here to avoid circular import
        from ..main.experiment_ops_unified import ExperimentOps
        from ..runtime.mujoco_backend import MuJoCoBackend

        # Create ExperimentOps from this modal
        ops = ExperimentOps(
            mode=self.config["mode"],  # OFFENSIVE - crash if missing!
            headless=self.config["headless"],  # OFFENSIVE - crash if missing!
            viewer_config=self.config["viewer_config"],  # OFFENSIVE - crash if missing!
            experiment_id=self.experiment_id
        )

        # Set scene and experiment dir
        ops.scene = self.scene
        ops.experiment_dir = str(self.experiment_dir)

        # If we have saved MuJoCo state, restore backend from it!
        if self.compiled_xml:
            print(f"✓ Restoring MuJoCo backend from saved state ({len(self.compiled_xml)} chars)")
            ops.backend = MuJoCoBackend.from_json({
                "compiled_xml": self.compiled_xml,
                "state": self.mujoco_state,
                "enable_viewer": not self.config["headless"],  # OFFENSIVE - crash if missing!
                "is_headless": self.config["headless"]  # OFFENSIVE - crash if missing!
            })
            print("✓ Backend restored (skipped XML regeneration!)")

        print(f"✓ Compiled experiment: {self.experiment_id}")
        print(f"  Ready to run!")

        return ops

    # === INTEGRATION WITH ExperimentOps ===

    def get_data(self) -> Dict[str, Any]:
        """I know my complete state - OFFENSIVE & MODAL-ORIENTED

        Returns:
            Dict with all experiment data
        """
        return {
            "experiment_id": self.experiment_id,
            "description": self.description,
            "created_at": self.created_at,
            "scene": self.scene.get_data() if hasattr(self.scene, 'get_data') else {},
            "config": self.config,
            "snapshots": {
                name: snapshot.to_json()
                for name, snapshot in self.snapshots.items()
            },
            "num_agents": len(self.scene.agents) if hasattr(self.scene, 'agents') else 0,
            "num_snapshots": len(self.snapshots)
        }


# === USAGE EXAMPLE ===

if __name__ == "__main__":
    print("=" * 80)
    print("EXPERIMENT MODAL (GOD MODAL) - Example")
    print("=" * 80)

    # This would normally come from ExperimentOps.to_experiment_modal()
    # For testing, create simple mock scene
    from .scene_modal import Scene
    from .room_modal import RoomModal

    # Create simple scene
    room = RoomModal(
        name="test_room",
        width=5.0,
        length=5.0,
        height=3.0
    )

    scene = Scene(room=room)
    scene.name = "test_scene"

    print("\n=== Creating Experiment ===")
    experiment = ExperimentModal(
        scene=scene,
        config={
            "mode": "simulated",
            "headless": True,
            "rl_config": {
                "policy": "PPO",
                "total_timesteps": 10000,
                "learning_rate": 0.0003
            }
        },
        experiment_id="test_exp_001",
        description="Testing GOD MODAL pattern",
        experiment_dir=Path("/tmp/test_experiment")
    )

    print(f"✓ Created experiment: {experiment.experiment_id}")
    print(f"  Scene: {experiment.scene.name}")

    # Create snapshot (timeline feature!)
    print("\n=== Creating Snapshot ===")
    experiment.create_snapshot(
        "checkpoint_1",
        description="Initial configuration",
        episode_number=0
    )

    # Simulate training progress...
    experiment.config["rl_config"]["total_timesteps"] = 20000

    # Create another snapshot
    experiment.create_snapshot(
        "checkpoint_2",
        description="After doubling timesteps",
        episode_number=50
    )

    print(f"✓ Created {len(experiment.snapshots)} snapshots")

    # Save experiment
    print("\n=== Saving Experiment ===")
    save_path = experiment.save()

    # Load experiment
    print("\n=== Loading Experiment ===")
    loaded = ExperimentModal.load(save_path)

    print(f"✓ Loaded: {loaded.experiment_id}")
    print(f"  Snapshots: {loaded.list_snapshots()}")

    # Restore to snapshot
    print("\n=== Restoring Snapshot ===")
    loaded.restore_snapshot("checkpoint_1")
    print(f"  Timesteps: {loaded.config['rl_config']['total_timesteps']}")

    # Get complete data
    print("\n=== Get Complete Data ===")
    data = loaded.get_data()
    print(f"  Keys: {list(data.keys())}")
    print(f"  Agents: {data['num_agents']}")
    print(f"  Snapshots: {data['num_snapshots']}")

    print("\n✅ GOD MODAL working perfectly!")
    print("\nNext: ExperimentArtifactModal saves execution history separately")
    print("Both live in same directory:")
    print(f"  {experiment.experiment_dir}/experiment.json          (configuration)")
    print(f"  {experiment.experiment_dir}/experiment_artifact.json (execution)")
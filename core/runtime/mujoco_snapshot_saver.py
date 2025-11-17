"""
MUJOCO SNAPSHOT SAVER - Saves MuJoCo state snapshots for perfect replay

Saves qpos, qvel, qacc at FPS rate to timeline/mujoco/ directory.
Enables:
- Perfect replay from any frame
- State debugging (why did X fail at step Y?)
- Restart simulation from any point

Pattern: ViewConsumer that saves MuJoCo state snapshots
"""

from pathlib import Path
from typing import Dict, Any
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class MuJoCoSnapshotSaver:
    """Saves MuJoCo state snapshots at FPS rate - OFFENSIVE!

    Saves to: timeline/mujoco/state_XXXXXX.json

    Each snapshot contains:
    - qpos: Joint positions
    - qvel: Joint velocities
    - qacc: Joint accelerations
    - ctrl: Control inputs
    - time: Simulation time
    - step: Step number

    Usage:
        saver = MuJoCoSnapshotSaver(timeline_dir, backend, fps=10.0)
        # Called by ViewAggregator automatically
        saver.save_frame(views)  # Consumer interface
    """

    def __init__(self, timeline_dir: Path, backend: Any, global_fps: float):
        """Initialize snapshot saver

        Args:
            timeline_dir: Path to timeline directory
            backend: MuJoCo backend (for accessing model/data)
            global_fps: Snapshots per second
        """
        self.timeline_dir = Path(timeline_dir)
        self.mujoco_dir = self.timeline_dir / "mujoco"
        self.mujoco_dir.mkdir(exist_ok=True)

        self.backend = backend
        self.global_fps = global_fps
        self.step = 0
        self.sim_time = 0.0
        self.frames_per_snapshot = 1  # Save every frame at FPS rate

        print(f"  ✓ MuJoCo snapshots: {self.mujoco_dir}")

    def save_frame(self, views: Dict[str, Any]):
        """Save MuJoCo state snapshot - ViewConsumer interface

        Args:
            views: Views dict (not used, but required by consumer interface)
        """
        # Get model/data from backend
        model = self.backend.model
        data = self.backend.data

        if model is None or data is None:
            return  # Backend not compiled yet

        # Build snapshot
        snapshot = {
            "step": self.step,
            "time": data.time,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "qacc": data.qacc.copy(),
            "ctrl": data.ctrl.copy(),
        }

        # Save to file
        snapshot_path = self.mujoco_dir / f"state_{self.step:06d}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(snapshot, f, indent=2, cls=NumpyEncoder)

        self.step += 1
        self.sim_time = data.time

    def close(self):
        """Finalize snapshots - called when experiment ends"""
        # Save manifest with all snapshot files
        snapshots = sorted(self.mujoco_dir.glob("state_*.json"))
        manifest = {
            "total_snapshots": len(snapshots),
            "fps": self.global_fps,
            "final_step": self.step - 1,
            "final_time": self.sim_time,
            "snapshots": [s.name for s in snapshots]
        }

        manifest_path = self.mujoco_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"  ✓ MuJoCo snapshots: {len(snapshots)} states saved")

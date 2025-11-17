"""
FRAME SNAPSHOT SAVER - Complete per-frame snapshots for TRUE TIME TRAVELER

PURE MOP: Each frame has just 4 files!
- experiment.json: God Modal (contains robot, scene, queue, reward nested!)
- scene.xml: Compiled MuJoCo model (XML, so separate)
- mujoco.json: Physics runtime state (runtime, so separate)
- video_refs.json: Video frame references

Pattern: ViewConsumer that saves complete frame snapshots
The God Modal (ExperimentModal) contains EVERYTHING - robot, scene, queue, reward!
Each modal saves itself via to_json(), nested in the container modal.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import numpy as np
from .async_writer import AsyncWriter


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


class FrameSnapshotSaver:
    """Saves complete frame snapshots at FPS rate - TRUE TIME TRAVELER!

    Saves to: timeline/frames/frame_XXXXXX/

    Each frame contains EVERYTHING needed to recreate that exact moment:
    - All modal states (JSONs)
    - Compiled MuJoCo model (scene.xml)
    - Physics state (qpos/qvel/qacc/ctrl)
    - Video frame references

    Usage:
        saver = FrameSnapshotSaver(
            timeline_dir=timeline_dir,
            backend=backend,
            experiment=experiment,
            robot=robot,
            scene=scene,
            action_queue=queue,
            reward_computer=reward,
            fps=10.0
        )
        # Called by ViewAggregator automatically
        saver.save_frame(views)  # Consumer interface
    """

    def __init__(
        self,
        timeline_dir: Path,
        backend: Any,
        experiment: Any,
        robot: Optional[Any] = None,
        scene: Optional[Any] = None,
        action_queue: Optional[Any] = None,
        reward_computer: Optional[Any] = None,
        global_fps: float = 10.0
    ):
        """Initialize frame snapshot saver

        Args:
            timeline_dir: Path to timeline directory
            backend: MuJoCo backend (for accessing model/data/xml)
            experiment: ExperimentModal instance
            robot: Robot modal instance (optional)
            scene: Scene modal instance (optional)
            action_queue: ActionQueue modal instance (optional)
            reward_computer: RewardComputer instance (optional)
            global_fps: Snapshots per second
        """
        self.timeline_dir = Path(timeline_dir)
        self.frames_dir = self.timeline_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        self.backend = backend
        self.experiment = experiment
        self.robot = robot
        self.scene = scene
        self.action_queue = action_queue
        self.reward_computer = reward_computer

        self.global_fps = global_fps
        self.frame_count = 0  # Frames saved counter
        self.sim_time = 0.0

        # ASYNC I/O: Background writer for non-blocking file writes
        self.async_writer = AsyncWriter(maxsize=200, name="FrameSnapshotSaver")

        # Quiet mode - no initialization prints

    def save_frame(self, views: Dict[str, Any]):
        """Save complete frame snapshot - ASYNC! ViewConsumer interface

        Args:
            views: Views dict (contains video frame references)
        """
        # Get model/data from backend
        model = self.backend.model
        data = self.backend.data

        if model is None or data is None:
            return  # Backend not compiled yet

        # NOTE: FPS throttling handled by RuntimeEngine!
        # RuntimeEngine only calls save_frame() at FPS rate, so we save every time we're called

        # MAIN THREAD: Capture all data that needs to be saved (must be synchronous!)
        # This is fast (~1-2ms) - just copies arrays and JSONs
        frame_num = self.frame_count
        frame_dir_path = self.frames_dir / f"frame_{frame_num:06d}"

        # Capture experiment JSON (on main thread - modal access not thread-safe)
        experiment_json = None
        if self.experiment and hasattr(self.experiment, 'to_json'):
            experiment_json = self.experiment.to_json()

        # Capture scene XML (on main thread - backend access not thread-safe)
        compiled_xml = None
        if self.backend.modal and hasattr(self.backend.modal, 'compiled_xml'):
            compiled_xml = self.backend.modal.compiled_xml

        # Capture MuJoCo state (copy arrays on main thread!)
        mujoco_state = {
            "frame": frame_num,
            "time": data.time,
            "qpos": data.qpos.copy(),  # Copy arrays!
            "qvel": data.qvel.copy(),
            "qacc": data.qacc.copy(),
            "ctrl": data.ctrl.copy(),
        }

        # Capture video refs (on main thread - view dict access not thread-safe)
        video_refs = {}
        for view_name, view_data in views.items():
            meta = view_data["__meta__"]  # OFFENSIVE - crash if missing!
            view_type = meta["view_type"]  # OFFENSIVE - crash if missing!

            # Only track video views (cameras + video_and_data)
            if view_type in ["video", "video_and_data"]:
                # Clean name matches video file name
                clean_name = view_name.replace("_view", "").replace("_sensor", "").replace("_actuator", "")
                video_refs[clean_name] = frame_num  # Frame number in the video

        # Update counters (on main thread)
        self.frame_count += 1
        self.sim_time = data.time

        # BACKGROUND THREAD: Write all files (slow I/O operations!)
        def save_frame_async():
            """Background thread: Write frame files without blocking simulation"""
            try:
                # Create frame directory
                frame_dir_path.mkdir(exist_ok=True)

                # 1. Save experiment.json (GOD MODAL - contains EVERYTHING!)
                if experiment_json is not None:
                    with open(frame_dir_path / "experiment.json", 'w') as f:
                        json.dump(experiment_json, f, indent=2, cls=NumpyEncoder)

                # 2. Save scene.xml (Compiled MuJoCo model - separate because it's XML)
                if compiled_xml is not None:
                    with open(frame_dir_path / "scene.xml", 'w') as f:
                        f.write(compiled_xml)

                # 3. Save mujoco.json (Physics runtime state - separate because it's runtime)
                with open(frame_dir_path / "mujoco.json", 'w') as f:
                    json.dump(mujoco_state, f, indent=2, cls=NumpyEncoder)

                # 4. Save video_refs.json (Video frame references)
                with open(frame_dir_path / "video_refs.json", 'w') as f:
                    json.dump(video_refs, f, indent=2)

            except Exception as e:
                print(f"⚠️ Failed to save frame snapshot {frame_num}: {e}")
                # Don't crash simulation - frame saving is optional

        # Submit to background thread - returns immediately!
        self.async_writer.submit(save_frame_async)

    def close(self):
        """Finalize snapshots - called when experiment ends"""
        # Close async writer first to finish all pending writes
        if hasattr(self, 'async_writer'):
            self.async_writer.close()

        # Defensive check: Only create manifest if frames directory exists
        if not self.frames_dir.exists():
            return  # No frames saved, nothing to finalize

        # Save manifest with all frame directories
        frames = sorted(self.frames_dir.glob("frame_*"))

        # Only create manifest if we actually saved frames
        if len(frames) > 0:
            manifest = {
                "total_frames": len(frames),
                "fps": self.global_fps,
                "final_frame": self.frame_count - 1,
                "final_time": self.sim_time,
                "frames": [f.name for f in frames]
            }

            manifest_path = self.frames_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

            # Quiet mode - no closing prints

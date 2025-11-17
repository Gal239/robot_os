"""
Database Operations - Experiment persistence and filesystem management

Handles all database I/O for experiments:
- Experiment folder creation and management
- experiment_id generation
- Saving experiment.json (GOD MODAL)
- Saving scene.xml
- UI snapshot management (hot_compile)
- Timeline directory setup

OFFENSIVE & CLEAN: Pure database operations, no business logic
"""

import os
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
from datetime import datetime
import uuid


class DatabaseOps:
    """Database operations manager - STATEFUL helper for experiment persistence

    Pattern: Stateful helper class (tracks experiment_dir, snapshot count)
    Separation: Database I/O separated from experiment orchestration
    """

    def __init__(self, experiment_id: str = None):
        """Initialize database ops with optional experiment ID

        Args:
            experiment_id: Optional experiment ID (auto-generated if None)
        """
        self.experiment_id = experiment_id or self._generate_experiment_id()
        self.experiment_dir = self._create_experiment_dir()
        self.snapshot_count = 0  # Track hot_compile snapshots

    # =========================================================================
    # CORE DATABASE OPERATIONS
    # =========================================================================

    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID - OFFENSIVE

        Format: exp_YYYYMMDD_HHMMSS_uuid8

        Returns:
            Experiment ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        experiment_id = f"exp_{timestamp}_{short_uuid}"
        return experiment_id

    def _create_experiment_dir(self) -> str:
        """Create experiment directory structure - OFFENSIVE

        Creates:
        - database/{experiment_id}/
        - database/{experiment_id}/views/
        - database/{experiment_id}/logs/
        - database/{experiment_id}/snapshots/
        - database/{experiment_id}/timeline/
        - database/{experiment_id}/ui_db/
        - database/{experiment_id}/mujoco_package/
        - database/{experiment_id}/scene_state/

        Returns:
            Path to experiment directory (string)
        """
        # Get simulation_center directory (3 levels up from this file)
        simulation_center_dir = Path(__file__).parent.parent.parent
        runs_dir = simulation_center_dir / "database" / self.experiment_id

        # Create directory structure
        runs_dir.mkdir(parents=True, exist_ok=True)
        (runs_dir / "views").mkdir(exist_ok=True)
        (runs_dir / "logs").mkdir(exist_ok=True)
        (runs_dir / "snapshots").mkdir(exist_ok=True)
        (runs_dir / "timeline").mkdir(exist_ok=True)
        (runs_dir / "ui_db").mkdir(exist_ok=True)  # UI snapshots
        (runs_dir / "mujoco_package").mkdir(exist_ok=True)  # NEW: Portable asset package
        (runs_dir / "scene_state").mkdir(exist_ok=True)  # NEW: Per-frame state snapshots

        return str(runs_dir)

    # =========================================================================
    # PATH UTILITIES
    # =========================================================================

    def get_experiment_path(self, subdir: str = None) -> Path:
        """Get path to experiment directory or subdirectory

        Args:
            subdir: Optional subdirectory name (e.g., "timeline", "ui_db")

        Returns:
            Path object to experiment dir or subdir
        """
        base_path = Path(self.experiment_dir)
        if subdir:
            return base_path / subdir
        return base_path

    def get_database_root(self) -> Path:
        """Get root database directory

        Returns:
            Path to simulation_center/database/
        """
        return Path(self.experiment_dir).parent

    # =========================================================================
    # GOD MODAL PERSISTENCE
    # =========================================================================

    def save_experiment_modal(self, experiment_modal) -> Path:
        """Save experiment.json (GOD MODAL) - PURE MOP!

        Args:
            experiment_modal: ExperimentModal instance

        Returns:
            Path to saved experiment.json
        """
        saved_path = experiment_modal.save()
        return Path(saved_path)

    def save_scene_xml(self, xml_string: str) -> Path:
        """Save compiled scene.xml

        Args:
            xml_string: MuJoCo XML string

        Returns:
            Path to saved scene.xml
        """
        xml_path = Path(self.experiment_dir) / "scene.xml"
        with open(xml_path, 'w') as f:
            f.write(xml_string)
        return xml_path

    def load_experiment_modal(self, path: str = None):
        """Load experiment.json

        Args:
            path: Optional path to experiment.json (defaults to experiment_dir/experiment.json)

        Returns:
            ExperimentModal instance
        """
        if path is None:
            path = Path(self.experiment_dir) / "experiment.json"

        from ..modals.experiment_modal import ExperimentModal
        return ExperimentModal.load(path)

    # =========================================================================
    # UI SNAPSHOT MANAGEMENT (hot_compile)
    # =========================================================================

    def save_ui_snapshot(self,
                        views: Dict[str, Any],
                        camera_images: Dict[str, np.ndarray],
                        scene_state: Dict[str, Any],
                        script: str = "") -> Path:
        """Save hot_compile snapshot to ui_db/ - PURE I/O (no view logic!)

        Creates:
        - ui_db/hot_compile_N/
        - ui_db/hot_compile_N/views.json (all view data)
        - ui_db/hot_compile_N/cameras/*.jpg (camera images)
        - ui_db/hot_compile_N/scene_state.json (objects list, robot details)
        - ui_db/hot_compile_N/script.txt (code executed)

        Args:
            views: Complete views dict from ViewAggregator (for serialization)
            camera_images: Camera RGB images extracted by ViewAggregator (runtime layer)
            scene_state: Scene state dict (objects, robot, etc.)
            script: Python code executed for this compile

        Returns:
            Path to snapshot directory

        MOP: Database layer does pure I/O, runtime layer classifies views!
        """
        # Create snapshot directory
        ui_db_dir = self.get_experiment_path("ui_db")
        snapshot_dir = ui_db_dir / f"hot_compile_{self.snapshot_count}"
        snapshot_dir.mkdir(exist_ok=True)
        (snapshot_dir / "cameras").mkdir(exist_ok=True)

        # Save camera images (runtime layer extracted them for us!)
        saved_count = 0
        for view_name, rgb_data in camera_images.items():
            img_path = snapshot_dir / "cameras" / f"{view_name}.jpg"
            # Convert RGB to BGR for OpenCV (only view knowledge: cv2 requires BGR)
            bgr_image = cv2.cvtColor(rgb_data.astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(img_path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

        if saved_count > 0:
            print(f"  💾 Saved {saved_count} camera images to ui_db")

        # Serialize views to JSON
        views_serializable = self._make_serializable(views)
        with open(snapshot_dir / "views.json", "w") as f:
            json.dump(views_serializable, f, indent=2)

        # Save scene_state.json
        with open(snapshot_dir / "scene_state.json", "w") as f:
            json.dump(scene_state, f, indent=2)

        # Save script.txt
        with open(snapshot_dir / "script.txt", "w") as f:
            f.write(script)

        # Increment counter
        self.snapshot_count += 1

        return snapshot_dir

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _make_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format

        Args:
            data: Any data structure

        Returns:
            JSON-serializable version
        """
        # Handle basic JSON types
        if data is None or isinstance(data, (bool, int, float, str)):
            return data

        # Handle dicts
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                # Skip modal references (they're not serializable)
                if k in ['__modal__', 'modal_ref']:
                    result[k] = f"<{type(v).__name__}>"
                    continue
                result[k] = self._make_serializable(v)
            return result

        # Handle lists/tuples
        if isinstance(data, (list, tuple)):
            return [self._make_serializable(item) for item in data]

        # Handle numpy arrays
        if isinstance(data, np.ndarray):
            # Don't serialize large arrays (images)
            if data.size > 1000:
                return f"<ndarray shape={data.shape} dtype={data.dtype}>"
            return data.tolist()

        # Handle numpy types
        if isinstance(data, (np.integer, np.floating)):
            return float(data)

        # Handle Path objects
        if isinstance(data, Path):
            return str(data)

        # Handle objects with modal-like names (skip them)
        if 'Modal' in type(data).__name__:
            return f"<{type(data).__name__}>"

        # Default: Try converting to string
        try:
            return str(data)
        except:
            return f"<{type(data).__name__}>"

    def list_ui_snapshots(self) -> list:
        """List all UI snapshots in ui_db/

        Returns:
            List of snapshot directory names
        """
        ui_db_dir = self.get_experiment_path("ui_db")
        if not ui_db_dir.exists():
            return []

        snapshots = sorted([d.name for d in ui_db_dir.iterdir() if d.is_dir()])
        return snapshots

    # =========================================================================
    # MUJOCO PACKAGE & SCENE STATE OPERATIONS (NEW)
    # =========================================================================

    def save_asset_package(self, package_modal):
        """Save mujoco_package/ - THIN LAYER (delegates to modal)

        Args:
            package_modal: AssetPackageModal instance

        The modal saves itself to mujoco_package/ directory.
        DatabaseOps just provides the path (Pure MOP delegation).
        """
        package_dir = self.get_experiment_path("mujoco_package")
        package_modal.save(package_dir)

    def save_scene_state(self, state_modal):
        """Save scene_state/frame_XXXX.xml - THIN LAYER (delegates to modal)

        Args:
            state_modal: SceneStateModal instance

        The modal saves itself to scene_state/ directory.
        DatabaseOps just provides the path (Pure MOP delegation).
        """
        scene_state_dir = self.get_experiment_path("scene_state")
        state_modal.save(scene_state_dir)

    def get_available_frames(self, scene_state_dir: Path = None) -> List[int]:
        """Get list of available frame numbers - THIN LAYER

        Args:
            scene_state_dir: Optional path to scene_state/ (defaults to experiment's scene_state/)

        Returns:
            Sorted list of frame numbers [0, 1, 2, ..., N]

        Example:
            frames = db_ops.get_available_frames()
            # [0, 1, 2, ..., 99]
        """
        if scene_state_dir is None:
            scene_state_dir = self.get_experiment_path("scene_state")

        if not scene_state_dir.exists():
            return []

        frames = []
        for frame_file in scene_state_dir.glob("frame_*.xml"):
            # Extract number from "frame_0042.xml" → 42
            frame_num = int(frame_file.stem.split('_')[1])
            frames.append(frame_num)

        return sorted(frames)

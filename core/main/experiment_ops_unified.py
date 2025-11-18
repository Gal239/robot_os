"""
EXPERIMENT OPS - THE unified ops layer for experiments
OFFENSIVE & SELF-EXPLANATORY: Scene + Robot + Compile + Run + View + Rewards

This is THE ops layer - everything you need for an experiment
"""

import os
from typing import Optional, Tuple, Any, Dict, List, Union

# ============================================================================
# CRITICAL: SET MUJOCO BACKEND BEFORE IMPORTS!
# MuJoCo's rendering backend MUST be set BEFORE MuJoCo is imported!
# ============================================================================
if 'MUJOCO_GL' not in os.environ:
    # Auto-detect GPU and set optimal rendering backend
    import subprocess

    # Try 1: Check for NVIDIA GPU (best performance)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=1)
        has_nvidia = (result.returncode == 0)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_nvidia = False

    if has_nvidia:
        os.environ['MUJOCO_GL'] = 'egl'
        print("🎮 MuJoCo Rendering: NVIDIA GPU (EGL) - FAST ✅")
    else:
        # Try 2: Check for any other GPU
        try:
            result = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=1)
            has_gpu = ('OpenGL' in result.stdout)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            has_gpu = False

        if has_gpu:
            os.environ['MUJOCO_GL'] = 'glfw'
            print("🎮 MuJoCo Rendering: GPU (GLFW) - FAST ✅")
        else:
            # ASSUME NVIDIA GPU exists even if nvidia-smi fails!
            # nvidia-smi can fail in Docker/WSL but GPU still works
            os.environ['MUJOCO_GL'] = 'egl'
            print("⚠️  MuJoCo Rendering: Assuming NVIDIA GPU (EGL) - nvidia-smi failed but trying GPU anyway")
            print("    If rendering is slow, you may be on CPU. Set MUJOCO_GL=osmesa explicitly for CPU rendering.")
else:
    # User explicitly set MUJOCO_GL
    print(f"🎮 MuJoCo Rendering: User-specified ({os.environ['MUJOCO_GL'].upper()})")

# NOW import MuJoCo-dependent modules
from ..runtime.runtime_engine import RuntimeEngine
from .state_ops import StateOps
from .action_ops import ActionOps
from ..runtime.mujoco_backend import MuJoCoBackend
from ..modals.room_modal import RoomModal
from ..modals.scene_modal import Scene
from .robot_ops import create_robot
from .database_ops import DatabaseOps
from dataclasses import dataclass

# Smart GPU detection - DEFERRED until we know if cameras are needed!
# We'll set MUJOCO_GL in compile() based on camera usage
# User can override by setting MUJOCO_GL before importing
_AUTO_DETECT_GPU = ('MUJOCO_GL' not in os.environ)


@dataclass
class Experiment:
    """Experiment container"""
    scene: Any
    robot: Any = None


# ============================================================================
# LOAD EXPERIMENT - MOP TIME MACHINE (Factory Function)
# ============================================================================

def load_experiment(path: str,
                   frame: int = None,
                   headless: bool = False) -> 'ExperimentOps':
    """Load saved experiment for playback - PURE MOP TIME MACHINE!

    Loads experiment EXACTLY as saved (read-only).
    Perfect for viewing/replaying past experiments.

    Args:
        path: Path to experiment directory (e.g., "database/exp_123/")
        frame: Frame number to start at (None = frame 0)
        headless: False = show viewer, True = headless

    Returns:
        ExperimentOps instance (read-only - for playback only!)

    What Works (from saved XML):
        ✅ All cameras that were compiled (nav_camera, free cameras)
        ✅ All camera resolutions (640x480, 1920x1080, etc.)
        ✅ All sensors (lidar, odometry, depth)
        ✅ All assets (tables, objects, robot)
        ✅ Viewer toggle (headless override)
        ✅ Frame playback
        ✅ Physics stepping

    What Doesn't Work (read-only):
        ❌ Can't add NEW cameras
        ❌ Can't add NEW assets
        ❌ Can't hot_reload()
        ❌ Can't modify scene

    Examples:
        # Load with viewer
        ops = load_experiment("database/exp_123/", headless=False)

        # Jump to specific frame
        ops = load_experiment("database/exp_123/", frame=100, headless=False)

        # Replay full timeline
        ops = load_experiment("database/exp_123/", headless=False)
        ops.replay_frames()
    """
    from pathlib import Path

    # Validate path
    exp_dir = Path(path)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    package_dir = exp_dir / "mujoco_package"
    if not package_dir.exists():
        raise FileNotFoundError(
            f"mujoco_package not found in {exp_dir}\n"
            f"This experiment may not have been compiled yet."
        )

    scene_xml_path = package_dir / "scene.xml"
    if not scene_xml_path.exists():
        raise FileNotFoundError(f"scene.xml not found in {package_dir}")

    print(f"📂 Loading experiment: {exp_dir.name}")

    # Create ExperimentOps (minimal config)
    ops = ExperimentOps(
        headless=headless,
        render_mode="slow",  # Doesn't matter - XML defines everything
        experiment_id=exp_dir.name  # Reuse same ID
    )

    # Read scene.xml as string
    with open(scene_xml_path, 'r') as f:
        xml_string = f.read()

    # Create backend (BEFORE compile!)
    ops.backend = MuJoCoBackend(
        enable_viewer=(not headless),
        headless=headless
    )

    # PROPER MOP: Use backend.compile_xml() with working directory for asset loading
    original_cwd = os.getcwd()
    try:
        os.chdir(package_dir)
        # This is the official interface - handles model, data, and viewer automatically!
        ops.backend.compile_xml(xml_string)
    finally:
        os.chdir(original_cwd)

    # Create minimal RuntimeEngine (for viewer only, no timeline)
    ops.engine = RuntimeEngine(
        backend=ops.backend,
        camera_fps=0,  # No recording
        sensor_hz=0,   # No sensors
        timeline_fps=0,  # No timeline
        state_extraction_hz=0
    )

    # Load frame if specified
    if frame is not None:
        scene_state_dir = exp_dir / "scene_state"
        frame_path = scene_state_dir / f"frame_{frame:04d}.xml"

        if not frame_path.exists():
            print(f"  ⚠️  Frame {frame} not found, starting at frame 0")
        else:
            from ..modals.scene_state_modal import SceneStateModal
            state = SceneStateModal.load(frame_path)
            state.apply(ops.backend.model, ops.backend.data)
            print(f"  ✓ Loaded frame {frame} (t={state.time:.3f}s)")

    print(f"  ✓ Loaded mujoco_package from {package_dir.name}")
    print(f"  ✓ Viewer: {'ON' if not headless else 'OFF'}")

    # Mark as read-only (informational)
    ops._read_only = True

    return ops


class ExperimentOps:
    """THE ops layer for experiments - OFFENSIVE & SELF-EXPLANATORY

    Does EVERYTHING: scene, robot, compile, run, view, rewards, state

    Usage:
        ops = ExperimentOps()
        ops.create_scene("kitchen", width=5, length=5, height=3)
        ops.add_asset("table", relative_to=(2, 2, 0))
        ops.add_robot("stretch")
        model, data = ops.compile()
        ops.step()
        state = ops.get_state()
        reward = ops.get_reward()
    """

    def __init__(self, mode: str = "simulated", headless: bool = True, viewer_config: Dict = None,
                 experiment_id: str = None, save_fps: float = 30.0, render_mode: str = "rl_core", fast_mode: bool = None,
                 camera_shadows: bool = True, camera_reflections: bool = True):
        """Initialize experiment ops

        Args:
            mode: "simulated" (MuJoCo sim) or "real" (real robot hardware)
            headless: Run headless (no display). When False, viewer is automatically enabled.
            viewer_config: MuJoCo viewer configuration (only used when headless=False)
                - 'disable_lidar_rays': bool - Hide lidar visualization
                - 'camera': str - Set viewer to specific camera (e.g., 'd405_rgb')
                - 'flags': dict - MuJoCo visualization flags
            experiment_id: Optional experiment ID (auto-generated if None)
            save_fps: Frames per second to save timeline (default: 30 FPS unified rate for humanoids!)
            render_mode: Rendering mode (default: "rl_core")
                - "rl_core": Fast, no cameras (for RL training)
                - "vision_rl": Fast WITH cameras (640x480, 10fps) for vision-based RL
                - "demo": Fast WITH high-res cameras (1280x720, 30fps) for professional demos
                - "mujoco_demo": Full-rate HD cameras (1280x720, 200Hz) for MuJoCo-quality streaming
                - "slow": Full rendering (640x480, full rate) for debugging, visualization
            fast_mode: DEPRECATED - use render_mode instead
            camera_shadows: Enable shadows in camera rendering (default True, disable for 2x faster!)
            camera_reflections: Enable reflections in camera rendering (default True, disable for faster!)
        """
        assert mode in ["simulated", "real"], f"mode must be 'simulated' or 'real', got '{mode}'"

        # BACKWARD COMPATIBILITY: Map fast_mode to render_mode
        if fast_mode is not None:
            import warnings
            warnings.warn(
                "fast_mode is deprecated. Use render_mode instead:\n"
                "  fast_mode=True  → render_mode='rl_core'\n"
                "  fast_mode=False → render_mode='slow'",
                DeprecationWarning,
                stacklevel=2
            )
            render_mode = "rl_core" if fast_mode else "slow"

        # Validate render_mode
        valid_modes = ["rl_core", "rl_core_no_timeline", "vision_rl", "demo", "2k_demo", "4k_demo", "mujoco_demo", "slow"]
        assert render_mode in valid_modes, \
            f"render_mode must be one of {valid_modes}, got '{render_mode}'"

        # VALIDATION: Prevent EGL conflicts
        if not headless and render_mode in ["vision_rl", "demo", "mujoco_demo", "2k_demo", "4k_demo"]:
            raise ValueError(
                f"Cannot use headless=False with render_mode='{render_mode}' (EGL context conflict).\n"
                "Solutions:\n"
                "  1. Use headless=True (recommended)\n"
                "  2. Use render_mode='slow' (if you need viewer + cameras)"
            )

        self.mode = mode
        self.save_fps = save_fps
        self.render_mode = render_mode
        self.camera_shadows = camera_shadows
        self.camera_reflections = camera_reflections

        # Use VideoConfig for ALL video settings! - MOP
        # Single source of truth - no more duplicate resolution/FPS logic
        from ..video import VideoConfig
        self.video_config = VideoConfig.from_render_mode(render_mode, save_fps)

        # Keep direct access for backward compatibility
        self.camera_width = self.video_config.camera_width
        self.camera_height = self.video_config.camera_height

        # NOTE: Backend (EGL/OSMesa/GLFW) already set at module level!
        # MuJoCo must be configured BEFORE import, so it happens at top of file.
        # Default: EGL (good for vision). Fast-mode users: export MUJOCO_GL=osmesa

        # Database operations (THIN LAYER)
        self.db_ops = DatabaseOps(experiment_id=experiment_id)
        self.experiment_id = self.db_ops.experiment_id
        self.experiment_dir = self.db_ops.experiment_dir

        self.scene = None
        self.robot = None
        self.backend = None
        self.engine = None
        self.actions = None  # ActionOps - created after compile
        self.last_result = None
        self.last_state = None  # State from sync_from_mujoco()
        self.headless = headless
        self.viewer_config = viewer_config or {}
        self._robot_initial_state = None  # Store for application after compile

        # TIME TRAVELER: Track save state
        self._step_counter = 0
        self._last_save_step = -1

        # MUJOCO PACKAGE: Lazy creation flag (created on first frame save)
        self._package_created = False

        # PREVIEW MODE: Track last screenshot paths for get_last_screenshots()
        self._last_screenshot_paths = {}

        # Keep only essential prints
        print(f"📁 Experiment: {self.experiment_id}")
        print(f"📂 Directory: {self.experiment_dir}")

    # === SCENE CREATION ===

    def create_scene(self, name: str, width: float, length: float, height: float,
                    floor_texture: str = "wood_floor", wall_texture: str = "gray_wall",
                    ceiling_texture: str = "ceiling_tiles") -> Scene:
        """Create scene with room - NICE DEFAULTS (wood parquet floor!)"""
        room = RoomModal(
            name=name,
            width=width,
            length=length,
            height=height,
            floor_texture=floor_texture,
            wall_texture=wall_texture,
            ceiling_texture=ceiling_texture
        )
        self.scene = Scene(room=room)
        return self.scene

    def create(self, name: str, width: float, length: float, height: float, **kwargs) -> Scene:
        """Alias for create_scene (backward compatibility)"""
        return self.create_scene(name, width, length, height, **kwargs)

    # === ADDING STUFF ===

    def add_asset(self, asset_name: str, asset_id: str = None, relative_to=None, relation: str = None,
                  distance: float = None, surface_position: str = None,
                  offset: tuple = None, orientation: Union[Tuple[float, float, float, float], str] = None,
                  initial_state: Dict = None, is_tracked: bool = False):
        """Add asset to scene - SELF-EXPLANATORY + PURE MOP

        Args:
            asset_name: Asset type to add (e.g., "table", "wood_block")
            asset_id: Unique ID for this asset (e.g., "block_red", "block_blue") - MUST BE UNIQUE!
                     If not provided, uses asset_name as ID (only one instance allowed).
            relative_to: Position (x,y,z) or asset name
            relation: Relation like "on_top", "front", "left", "inside"
            distance: Distance for relative placement (optional - auto-extracted if None)
            surface_position: Semantic surface position ("top_left", "center", etc.) - NEW
            offset: Manual (x, y) offset in meters - NEW
            orientation: Quaternion (w,x,y,z) or preset like "upright" - NEW for stable stacking
            initial_state: Initial joint states
            is_tracked: Track this asset for state extraction (default False) - PERFORMANCE OPTIMIZATION!
                       Set to True to extract state for this asset even without rewards.
                       Automatically set to True when add_reward() is called.

        PURE MOP: If distance=None, dimensions extracted from MuJoCo model.
        """
        assert self.scene is not None, "Call create_scene() first"
        return self.scene.add(asset_name, asset_id=asset_id, relative_to=relative_to,
                             relation=relation, distance=distance,
                             surface_position=surface_position, offset=offset,
                             orientation=orientation,
                             initial_state=initial_state, is_tracked=is_tracked)

    def add_object(self, asset_type: str = None, name: str = None, position: tuple = None,
                   on_top: str = None, relative_to: str = None, offset: tuple = None,
                   color: str = None, **kwargs):
        """Add object to scene - backward compatibility wrapper

        Supports both old and new API:
        - Old: add_object("cube", name="target", position=(1,2,3))
        - New: add_asset("cube", relative_to=(1,2,3))
        - Dict: add_object("apple", position={"relative_to": "table", "relation": "on_top"})

        Args:
            asset_type: Asset type from registry (e.g., "cube", "table")
            name: Custom name for the asset (IGNORED - uses asset_type for now)
            position: Absolute position (x,y,z) OR dict with relative_to/relation/distance
            on_top: Asset to place this on top of
            relative_to: Asset or position to place relative to
            offset: Offset from relative_to position
            color: Color (not yet implemented)

        Note: Custom names not fully supported yet - uses asset_type as name
        """
        # Use asset type for registry lookup
        # TODO: Support custom instance names (e.g., "target1:cube")
        asset_name = asset_type

        # Determine positioning - handle dict syntax!
        if position is not None:
            # Check if position is a dict with relative_to/relation
            if isinstance(position, dict):
                rel_to = position['relative_to']  # OFFENSIVE - crash if missing!
                kwargs['relation'] = position['relation']  # OFFENSIVE - crash if missing!
                kwargs['distance'] = position.get('distance', 0)  # LEGITIMATE - has default
            else:
                # Absolute position tuple - validate boundaries!
                rel_to = position
                if isinstance(position, tuple) and len(position) == 3:
                    self._validate_position(position, asset_name=f"Object '{asset_type}'")
        elif on_top is not None:
            rel_to = on_top
            kwargs['relation'] = 'on_top'
        elif relative_to is not None:
            rel_to = relative_to
            # TODO: Handle offset parameter
        else:
            rel_to = None

        return self.add_asset(asset_name, relative_to=rel_to, **kwargs)

    def _validate_position(self, position: tuple, asset_name: str = "asset"):
        """Validate position is within room boundaries (origin-centered coordinate system)

        COORDINATE SYSTEM:
        - Origin (0,0,0) is at ROOM CENTER, floor level
        - X-axis: -width/2 to +width/2
        - Y-axis: -length/2 to +length/2
        - Z-axis: 0 (floor) to height

        Args:
            position: (x, y, z) tuple in meters
            asset_name: Name for error message

        Raises:
            ValueError: If position is outside room boundaries with safety margin

        Examples:
            For 10x10x3 room:
            - (0, 0, 0): Valid - room center ✓
            - (4, 4, 0): Valid - inside boundaries ✓
            - (5, 5, 0): Invalid - at wall (needs margin) ✗
            - (6, 0, 0): Invalid - outside room ✗
        """
        assert self.scene is not None, "Scene not created"
        assert self.scene.room is not None, "Room not created"

        x, y, z = position
        room = self.scene.room

        # Calculate boundaries (origin-centered)
        x_min, x_max = -room.width / 2, room.width / 2
        y_min, y_max = -room.length / 2, room.length / 2
        z_min, z_max = 0, room.height

        # Add safety margin (robot/object has physical size ~0.5m radius)
        margin = 0.5

        if not (x_min + margin <= x <= x_max - margin):
            raise ValueError(
                f"{asset_name} position x={x:.2f}m outside room boundaries "
                f"[{x_min + margin:.2f}, {x_max - margin:.2f}]m "
                f"(room width={room.width}m, origin-centered coordinate system)"
            )

        if not (y_min + margin <= y <= y_max - margin):
            raise ValueError(
                f"{asset_name} position y={y:.2f}m outside room boundaries "
                f"[{y_min + margin:.2f}, {y_max - margin:.2f}]m "
                f"(room length={room.length}m, origin-centered coordinate system)"
            )

        if not (z_min <= z <= z_max):
            raise ValueError(
                f"{asset_name} position z={z:.2f}m outside room height "
                f"[{z_min:.2f}, {z_max:.2f}]m"
            )

    def add_robot(self, robot_name: str, robot_id: str = None, position=None, orientation=None, sensors=None, task_hint=None, initial_state=None):
        """Add robot to scene with optional sensor configuration

        Args:
            robot_name: Robot type (e.g., "stretch")
            robot_id: Optional robot ID
            position: Robot position (x, y, z) tuple. Default: (0, 0, 0) - room center
            orientation: Robot orientation (optional):
                - None: Identity quaternion (1, 0, 0, 0) - faces default direction (+Y north)
                - Preset string: "north", "south", "east", "west", "upright", "sideways", "inverted"
                - Relational: "facing_table", "facing_apple", "facing_origin" (auto-calculated)
                - Manual quaternion: (w, x, y, z) tuple
            sensors: Sensor configuration:
                - None: All sensors (default)
                - List[str]: Only these sensors (e.g., ["nav_camera", "lidar"])
                - Dict[str, bool]: Enable/disable specific sensors
            task_hint: Auto-configure for task (e.g., "manipulation", "navigation")
            initial_state: Initial actuator positions dict (e.g., {"arm": 0.3, "lift": 0.5})
                - Overrides default positions from XML keyframe
                - Keys: actuator names (arm, lift, gripper, head_pan, head_tilt, wrist_yaw, wrist_pitch, wrist_roll)
                - Values: positions in actuator's unit (meters for arm/lift, radians for joints)

        Examples:
            ops.add_robot("stretch")  # All sensors, center position, default orientation
            ops.add_robot("stretch", position=(2.5, 2.5, 0))  # Custom position
            ops.add_robot("stretch", position=(2.5, 2.5, 0), orientation="east")  # Face east
            ops.add_robot("stretch", position=(2.5, 2.5, 0), orientation="facing_table")  # Face table
            ops.add_robot("stretch", sensors=["nav_camera", "odometry"])  # Only these
            ops.add_robot("stretch", task_hint="manipulation")  # Auto-config
            ops.add_robot("stretch", initial_state={"arm": 0.3, "lift": 0.5})  # Custom initial pose
        """
        assert self.scene is not None, "Call create_scene() first"
        robot_id = robot_id or robot_name
        self.robot = create_robot(robot_name, robot_id, sensors=sensors, task_hint=task_hint)
        pos = position if position is not None else (0, 0, 0)

        # MOP FIX: Update robot.initial_position to match placement!
        # Robot modal's initial_position is used for state["stretch"]["position"]
        # Must sync it with placement position for correct state reporting
        self.robot.initial_position = pos

        # MOP FIX 2: Update robot.initial_orientation to match placement!
        # Robot modal's initial_orientation is used for state["stretch"]["quaternion"]
        # Must sync it with placement orientation for correct state reporting
        if orientation is not None:
            # Resolve string orientation (preset or relational) to quaternion
            if isinstance(orientation, str):
                # Use Placement.get_quat() to handle presets and relational patterns
                from ..modals.scene_modal import Placement
                temp_placement = Placement(
                    asset=robot_name,
                    position=pos,
                    orientation=orientation
                )
                # Pass runtime_state for relational orientation (model needed for get_xyz)
                runtime_state = {'model': self.backend.model} if hasattr(self, 'backend') and hasattr(self.backend, 'model') else None
                quat = temp_placement.get_quat(self.scene, runtime_state)
            else:
                # Direct quaternion tuple
                quat = orientation
            self.robot.initial_orientation = quat
            print(f"  [MOP] Synced robot.initial_orientation: {quat}")

        # Validate position is within room boundaries
        self._validate_position(pos, asset_name=f"Robot '{robot_name}'")

        # Store initial_state for application during compile() - PURE MOP!
        # (Applied AFTER physics settling but BEFORE keyframe save)
        if initial_state:
            # Validate actuator names NOW (fail fast!)
            for actuator_name in initial_state.keys():
                if actuator_name not in self.robot.actuators:
                    available = list(self.robot.actuators.keys())
                    raise KeyError(
                        f"Actuator '{actuator_name}' not found in robot '{robot_name}'\n"
                        f"Available actuators: {available}"
                    )

            # Convert percentage strings to numeric values - MOP delegation!
            # Supports: "100%", "50%", "0%" alongside numeric values 0.3, 0.5
            from ..modals.robot_modal import convert_initial_state_percentages
            self._robot_initial_state = convert_initial_state_percentages(initial_state, self.robot)
        else:
            self._robot_initial_state = None

        # OFFENSIVE: Remove ALL cameras when render_mode="rl_core" (no cameras)
        # "rl_core" mode is for pure RL training without vision - cameras slow down sim!
        if self.render_mode == "rl_core":
            # Remove robot sensor cameras (nav_camera, d405_camera)
            cameras_to_remove = [name for name in ['nav_camera', 'd405_camera']
                                if name in self.robot.sensors]
            if cameras_to_remove:
                self.robot.remove_sensors(cameras_to_remove)

            # ALSO remove scene free cameras (for viewer compatibility!)
            # Viewer mode (headless=False) needs NO cameras to avoid GLFW/EGL conflict
            if hasattr(self.scene, 'cameras') and self.scene.cameras:
                num_free_cams = len(self.scene.cameras)
                self.scene.cameras.clear()
                print(f"  ⚠️  Removed {num_free_cams} free camera(s) (render_mode='rl_core' = no cameras)")

        # Apply camera resolution based on render_mode
        # Update nav_camera and d405_camera if they exist in robot sensors
        for sensor_name in ['nav_camera', 'd405_camera']:
            if sensor_name in self.robot.sensors:
                sensor = self.robot.sensors[sensor_name]
                # NavCamera has width/height attributes
                if hasattr(sensor, 'width') and hasattr(sensor, 'height'):
                    sensor.width = self.camera_width
                    sensor.height = self.camera_height

        self.scene.add_robot(self.robot, relative_to=pos, orientation=orientation)
        return self.robot

    def _fast_compile_for_solver(self):
        """Fast compile ONLY for dimension extraction - TEMPORARY!

        This creates a temporary MuJoCo model to extract asset positions/dimensions,
        then DISCARDS it. Used by scene solver to get dimensions before adding robot.

        Returns:
            MuJoCo model with dimensions extracted

        Note:
            - NO robot included (just scene assets)
            - NO timeline saving
            - NO viewer
            - Minimal settling (10 steps)
            - Model is temporary - will be discarded after solver uses it
        """
        print("  🔧 Fast compile for dimension extraction (temporary)...")

        from ..runtime.mujoco_backend import MuJoCoBackend
        from ..runtime.runtime_engine import RuntimeEngine

        # Create temporary backend+engine (headless, no timeline)
        temp_backend = MuJoCoBackend(enable_viewer=False, headless=True)
        temp_engine = RuntimeEngine(
            backend=temp_backend,
            camera_fps=0,
            sensor_hz=0,
            timeline_fps=0,  # NO timeline!
            state_extraction_hz=0,
            step_rate=30,
            camera_width=640,
            camera_height=480,
            camera_shadows=False,
            camera_reflections=False
        )

        # Compile scene WITHOUT robot
        experiment = Experiment(scene=self.scene, robot=None)
        temp_engine.load_experiment(experiment, experiment_dir=self.experiment_dir)

        # Minimal settling for dimension extraction
        for _ in range(10):
            temp_engine.step()

        model = temp_backend.model
        print("  ✅ Dimensions extracted (model will be discarded after solver)")

        # Store for cleanup later
        self._temp_solver_engine = temp_engine
        self._temp_solver_backend = temp_backend

        return model

    def _cleanup_solver_compile(self):
        """Clean up temporary compilation from _fast_compile_for_solver()"""
        if hasattr(self, '_temp_solver_engine') and self._temp_solver_engine:
            print("  🗑️  Discarding temporary solver compile...")
            # Just delete - destructor will handle cleanup
            del self._temp_solver_engine
            del self._temp_solver_backend

    def solve_robot_placement(
        self,
        robot_id: str,
        task: str,
        target_asset: str,
        **kwargs
    ) -> Dict:
        """Calculate optimal robot placement for task - PURE MOP!

        Automatically calculates robot position, orientation, and joint positions
        to optimally perform the specified task on the target asset.

        NO HARDCODING: All dimensions and capabilities auto-discovered from:
        - Robot actuators (range, behaviors)
        - Asset dimensions (MuJoCo model or XML)
        - Task requirements (grasp, inspect, manipulate)

        Args:
            robot_id: Robot identifier (currently self.robot.name)
            task: Task type:
                - "grasp": Position robot to grasp target object
                - "inspect": Position camera to visually inspect target
                - "manipulate": Position for general manipulation
            target_asset: Target asset name in scene
            **kwargs: Additional solver parameters

        Returns:
            {
                'position': (x, y, z),  # Base position
                'orientation': (w, x, y, z) or preset string,  # Base orientation
                'initial_state': {'joint_name': value}  # Initial joint state (matches add_robot!)
            }

        Example:
            # Create scene and add assets first
            ops = ExperimentOps()
            ops.create_scene("kitchen", width=5, length=5, height=3)
            ops.add_asset("table", relative_to=(2, 0, 0))
            ops.add_asset("apple", relative_to="table", relation="on_top")

            # Calculate optimal placement for grasping apple
            placement = ops.solve_robot_placement(
                robot_id="stretch",
                task="grasp",
                target_asset="apple"
            )

            # Add robot with calculated placement (HYBRID API - two ways!)

            # Way 1: Unpack placement dict (CLEAN!)
            ops.add_robot("stretch", **placement)

            # Way 2: Explicit parameters (VERBOSE)
            ops.add_robot(
                "stretch",
                position=placement['position'],
                orientation=placement['orientation'],
                initial_state=placement['initial_state']
            )

            ops.compile()

        Note:
            - Call this BEFORE add_robot() to calculate placement
            - Or use after compilation to check if current placement is optimal
            - Requires scene and target asset to exist first
        """
        # OFFENSIVE: Validate scene exists
        assert self.scene is not None, (
            "Call create_scene() first!\n"
            "💡 ops.create_scene('room_name', width=5, length=5, height=3)"
        )

        # OFFENSIVE: Validate robot exists (if calculating for existing robot)
        # If robot doesn't exist yet, that's fine - user will call add_robot() after
        robot = self.robot if self.robot else None

        # If no robot exists, create a temporary one for capability discovery
        if robot is None:
            from ..main.robot_ops import create_robot
            robot = create_robot(robot_id, robot_id)
            print(f"  ℹ️  Created temporary robot '{robot_id}' for capability discovery")

        # Get model if compiled (for runtime dimension extraction)
        model = self.backend.model if hasattr(self, 'backend') and hasattr(self.backend, 'model') else None

        # Delegate to scene solver (MODAL-TO-MODAL!)
        placement = self.scene.solve_robot_placement(
            robot=robot,
            task=task,
            target_asset_name=target_asset,
            model=model,
            **kwargs
        )

        return placement

    def add_robot_for_task(
        self,
        robot_name: str,
        task: str,
        target_asset: str,
        robot_id: str = None,
        sensors: Any = None,
        task_hint: str = None,
        **kwargs
    ):
        """Add robot with auto-calculated placement for task - ONE-STEP API!

        HYBRID API: This is the ONE-STEP method that does everything:
        1. Calculate optimal placement (solve_robot_placement)
        2. Add robot with calculated placement (add_robot)

        For TWO-STEP control, use solve_robot_placement() separately.

        Args:
            robot_name: Robot type (e.g., "stretch")
            task: Task type ("grasp", "inspect", "manipulate")
            target_asset: Target asset name in scene
            robot_id: Optional robot ID (default: robot_name)
            sensors: Optional sensor configuration (see add_robot())
            task_hint: Optional task hint for auto-config
            **kwargs: Additional solver parameters

        Returns:
            Robot modal instance (same as add_robot())

        Example - ONE-STEP (simple):
            ops = ExperimentOps()
            ops.create_scene("kitchen", width=5, length=5, height=3)
            ops.add_asset("table", relative_to=(2, 0, 0))
            ops.add_asset("apple", relative_to="table", relation="on_top")

            # ONE call does everything!
            ops.add_robot_for_task("stretch", task="grasp", target_asset="apple")
            ops.compile()

        Example - TWO-STEP (detailed control):
            # Calculate placement first (inspect before applying)
            placement = ops.solve_robot_placement("stretch", "grasp", "apple")
            print(f"Position: {placement['position']}")  # Check it

            # Then apply
            ops.add_robot("stretch", **placement)
            ops.compile()
        """
        # Step 1: Calculate optimal placement (MODAL-TO-MODAL!)
        placement = self.solve_robot_placement(
            robot_id=robot_name,
            task=task,
            target_asset=target_asset,
            **kwargs
        )

        print(f"\n🤖 Adding robot '{robot_name}' for task '{task}' on '{target_asset}'")
        print(f"   Auto-calculated position: {placement['position']}")
        print(f"   Auto-calculated orientation: {placement['orientation']}")
        print(f"   Auto-calculated joints: {len(placement['initial_state'])} joints")

        # Step 2: Add robot with calculated placement
        robot = self.add_robot(
            robot_name=robot_name,
            robot_id=robot_id,
            position=placement['position'],
            orientation=placement['orientation'],
            initial_state=placement['initial_state'],
            sensors=sensors,
            task_hint=task_hint
        )

        return robot

    def add_free_camera(self,
                       camera_id: str = "birds_eye",
                       lookat: Tuple[float, float, float] = (0, 0, 0.5),
                       distance: float = 5.0,
                       azimuth: float = 90.0,
                       elevation: float = -30.0,
                       width: int = None,
                       height: int = None,
                       track_target: str = None,
                       track_offset: Tuple[float, float, float] = (0, 0, 0)) -> Any:
        """Add virtual free camera (SIM ONLY!) - INTERACTIVE with tracking support

        VIRTUAL: Not a real sensor - simulation only! Cannot be used with real hardware.
        The camera automatically saves to timeline like other cameras.

        NEW: Cameras now managed at scene level (scene.cameras) with asset tracking support!

        Args:
            camera_id: Camera name (e.g., 'birds_eye', 'side_view', 'top_down')
            lookat: [x, y, z] point to look at (default: center of scene)
            distance: Distance from lookat point in meters (default: 5.0)
            azimuth: Horizontal rotation angle in degrees
                     0=forward, 90=right, 180=back, 270=left (default: 90)
            elevation: Vertical rotation angle in degrees
                       -90=top-down, 0=horizontal, 90=bottom-up (default: -30)
            width: Image width in pixels (default: based on render_mode - 1280 for demo, 640 for others)
            height: Image height in pixels (default: based on render_mode - 720 for demo, 480 for others)
            track_target: Optional - asset or robot name to track (NEW!)
            track_offset: Offset from tracked target position (NEW!)

        Returns:
            CameraModal instance

        Examples:
            # Bird's eye view (default)
            ops.add_free_camera('birds_eye')

            # Top-down view over table
            ops.add_free_camera('top_down',
                              lookat=(2, 2, 0.5),
                              distance=3.0,
                              elevation=-90)

            # NEW: Track robot!
            ops.add_free_camera('follow_cam',
                              track_target='stretch',
                              track_offset=(0, 0, 2.0))

            # NEW: Track asset!
            ops.add_free_camera('apple_cam',
                              track_target='apple',
                              track_offset=(0, 0, 0.5))

        Usage:
            ops.add_free_camera('birds_eye')
            ops.compile()
            ops.step()

            # Change angle during simulation!
            ops.set_camera_angle('birds_eye', azimuth=180, elevation=-60)
            ops.step()

        Camera automatically saved to: timeline/cameras/{camera_id}_rgb.mp4
        """
        assert self.mode == "simulated", \
            "Free cameras are SIMULATION ONLY! Cannot be used with real hardware (mode='real')"
        assert self.scene is not None, \
            "Create scene first with create_scene() before adding cameras"
        assert self.robot is not None, \
            "Add robot first with add_robot() before adding cameras"

        # Use render_mode resolution if not specified
        if width is None:
            width = self.camera_width
        if height is None:
            height = self.camera_height

        # NEW: Delegate to scene.add_camera() (proper architecture!)
        camera = self.scene.add_camera(
            camera_id=camera_id,
            lookat=lookat,
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            width=width,
            height=height,
            track_target=track_target,
            track_offset=track_offset
        )

        # MOP FIX: DON'T add to robot.sensors - cameras are in scene.cameras!
        # FREE CAMERAS ARE NOT ROBOT SENSORS - they're virtual scene cameras!
        # This was breaking viewer mode (duplicated storage = MOP violation)
        # OLD (BROKEN): self.robot.sensors[camera_id] = camera

        return camera

    def add_overhead_camera(self,
                           camera_id: str = "overhead_cam",
                           width: int = None,
                           height: int = None) -> Any:
        """Add overhead camera with MuJoCo viewer defaults - AUTO-CALCULATES based on room!

        This is a convenience method that automatically calculates optimal overhead
        camera position using MuJoCo's mjv_defaultFreeCamera algorithm:
        - lookat = scene center (0, 0, room_height/2)
        - distance = 1.5 * scene_extent (max of room width/length)
        - azimuth = 90° (MuJoCo default)
        - elevation = -45° (MuJoCo default - angled, not straight down!)

        This gives a professional viewer-like perspective that shows the whole scene.

        Args:
            camera_id: Camera name (default: 'overhead_cam')
            width: Image width (default: based on render_mode)
            height: Image height (default: based on render_mode)

        Returns:
            CameraModal instance

        Example:
            ops.create_scene("kitchen", width=8, length=8, height=3)
            ops.add_robot("stretch")
            ops.add_overhead_camera()  # Auto-calculates: lookat=(0,0,1.5), distance=12m
            ops.compile()

        Note: Camera name must contain 'overhead', 'viewer', or 'birds_eye' to skip
              room bounds validation (since it's positioned outside for full view).
        """
        assert self.scene is not None, "Create scene first with create_scene()"
        assert self.scene.room is not None, "Scene must have room for overhead camera"

        # FIXED: MuJoCo viewer defaults - simple and reliable!
        room = self.scene.room

        # MuJoCo defaults (like mjv_defaultFreeCamera)
        lookat_x = 0.0  # Room center X
        lookat_y = 0.0  # Room center Y
        lookat_z = room.height / 2.0  # Room mid-height

        extent = max(room.width, room.length)
        distance = 1.5 * extent  # MuJoCo default distance

        azimuth = 90.0  # MuJoCo default (looking from side)
        elevation = -45.0  # MuJoCo default (angled view)
        # Use add_free_camera with calculated values
        return self.add_free_camera(
            camera_id=camera_id,
            lookat=(lookat_x, lookat_y, lookat_z),
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            width=width,
            height=height
        )

    def set_camera_angle(self,
                        camera_id: str,
                        lookat: Tuple[float, float, float] = None,
                        distance: float = None,
                        azimuth: float = None,
                        elevation: float = None):
        """Change free camera angle during simulation - INTERACTIVE!

        Update camera parameters to orbit, zoom, or change viewing angle.
        Changes take effect on the next step().

        NEW: Cameras now in scene.cameras (with backward compat for robot.sensors)

        Args:
            camera_id: Camera to update (e.g., 'birds_eye')
            lookat: New lookat point [x, y, z] (optional)
            distance: New distance in meters (optional)
            azimuth: New horizontal angle in degrees (optional)
            elevation: New vertical angle in degrees (optional)

        Examples:
            # Orbit around scene
            for angle in range(0, 360, 10):
                ops.set_camera_angle('birds_eye', azimuth=angle)
                ops.step()

            # Zoom in/out
            ops.set_camera_angle('birds_eye', distance=3.0)   # Closer
            ops.set_camera_angle('birds_eye', distance=10.0)  # Farther

            # Change elevation (top-down vs angled)
            ops.set_camera_angle('birds_eye', elevation=-90)  # Pure top-down
            ops.set_camera_angle('birds_eye', elevation=-30)  # Angled view

            # Look at robot
            ops.set_camera_angle('birds_eye', lookat=(0, 0, 0.5))

            # Multiple parameters at once
            ops.set_camera_angle('birds_eye',
                                azimuth=180,
                                elevation=-45,
                                distance=8.0)

        Note: Only changes specified parameters. Others remain unchanged.
        """
        # NEW: Look for camera in scene.cameras first (proper location)
        camera = None
        if self.scene and hasattr(self.scene, 'cameras') and camera_id in self.scene.cameras:
            camera = self.scene.cameras[camera_id]
        # BACKWARD COMPAT: Also check robot.sensors
        elif self.robot and camera_id in self.robot.sensors:
            camera = self.robot.sensors[camera_id]

        assert camera is not None, \
            f"Camera '{camera_id}' not found. " \
            f"Available cameras: {list(self.scene.cameras.keys()) if self.scene else []}"

        # Verify it's a free camera (has azimuth parameter)
        assert hasattr(camera, 'azimuth'), \
            f"'{camera_id}' is not a free camera! Only free cameras support set_camera_angle()"

        # Update parameters (only if provided)
        if lookat is not None:
            camera.lookat = list(lookat)
        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation

    def add_reward(self, tracked_asset: str, behavior: str, target=None,
                  reward: float = None, mode: str = None, id: str = None,
                  requires=None, within=None, after=None, after_event=None,
                  speed_bonus=None, spatial_target=None,
                  tolerance_override: float = None, natural_range_override: float = None):
        """Add reward condition - UNIFIED SYNTAX - TRUE MOP!

        Args:
            tracked_asset: Asset.property to track (e.g., "stretch.arm")
            behavior: Behavior property (e.g., "extension", "closed", "holding")
            target: Target value (e.g., 90.0 for degrees, True for boolean, 0.5 for meters)
            reward: Reward points
            mode: "discrete" (default), "convergent" (partial + penalties), "achievement" (partial, forgiving)
            id: Condition ID (REQUIRED for sequential dependencies)
            requires: Condition ID or list - must be met first (sequential!)
            within: Seconds - condition must be met WITHIN this time
            after: Seconds - only check AFTER this time
            after_event: Condition ID that starts the timer
            speed_bonus: Extra points for faster completion
            spatial_target: Target asset for spatial relations (e.g., "apple" for "holding")
            tolerance_override: Override discovered tolerance (optional)
            natural_range_override: Override discovered natural range (optional)

        Examples:
            # Discrete (binary 0/100pts)
            ops.add_reward("stretch.base", "rotation", target=90.0, reward=100, id="turn_left")

            # Convergent (partial credit + penalize overshooting)
            ops.add_reward("stretch.base", "rotation", target=90.0, reward=100,
                          mode="convergent", id="precise_turn")

            # Achievement (partial credit, forgiving)
            ops.add_reward("stretch.base", "rotation", target=90.0, reward=100,
                          mode="achievement", id="reach_angle")

            # Sequential dependency
            ops.add_reward("stretch.lift", "height", target=0.8, reward=30,
                          requires="precise_turn", id="lift_up")

            # Spatial relation
            ops.add_reward("stretch.gripper", "holding", target=True,
                          spatial_target="apple", reward=50, id="grab_apple")

            # With speed bonus
            ops.add_reward("stretch.base", "rotation", target=0.0, reward=100,
                          mode="convergent", requires="lift_up", speed_bonus=50,
                          within=30.0, id="return_home")
        """
        assert self.scene is not None, "Call create_scene() first"
        return self.scene.add_reward(
            tracked_asset=tracked_asset,
            behavior=behavior,
            target=target,
            reward=reward,
            mode=mode,
            id=id,
            requires=requires,
            within=within,
            after=after,
            after_event=after_event,
            speed_bonus=speed_bonus,
            spatial_target=spatial_target,
            tolerance_override=tolerance_override,
            natural_range_override=natural_range_override
        )

    def add_reward_composite(self, operator: str, conditions, reward: float,
                            id: str, mode: str = None, requires=None,
                            within=None, after=None, speed_bonus=None):
        """Add composite reward (AND/OR/NOT) - SELF-DOCUMENTING API!

        Args:
            operator: "AND", "OR", or "NOT"
            conditions: List of condition IDs (or single ID for NOT)
            reward: Reward points
            id: Condition ID (REQUIRED)
            mode: "discrete" (default) or "smooth" (partial credit for AND/OR)
            requires: Condition ID or list - must be met first
            within: Seconds - composite must be met WITHIN this time
            after: Seconds - only check AFTER this time
            speed_bonus: Extra points for faster completion

        Examples:
            # AND composite
            ops.add_reward_composite("AND", ["gripper_closed", "bottle_held"],
                                    reward=100, id="grasped")

            # OR composite with smooth mode
            ops.add_reward_composite("OR", ["door_open", "window_open"],
                                    reward=50, mode="smooth", id="ventilation")

            # NOT composite (penalty)
            ops.add_reward_composite("NOT", "door_open", reward=-50,
                                    after=60.0, id="timeout_penalty")
        """
        assert self.scene is not None, "Call create_scene() first"
        return self.scene.add_reward_composite(
            operator=operator,
            conditions=conditions,
            reward=reward,
            id=id,
            mode=mode,
            requires=requires,
            within=within,
            after=after,
            speed_bonus=speed_bonus
        )

    # === COMPILATION ===

    def compile(self, enable_timeline: bool = True, settling_steps: int = None, mode: str = "full") -> Tuple[Any, Any]:
        """Compile scene to physics engine - SELF-EXPLANATORY

        Args:
            enable_timeline: Enable timeline recording (default True).
                            Set False for lightweight scene building (hot_compile).
            settling_steps: Physics settling steps (default: 100 for full, 10 for hot)
            mode: Compilation mode - "full" (default) or "preview" (fast iteration)
                  "preview" automatically sets enable_timeline=False and settling_steps=10

        Returns:
            (model, data) from MuJoCo
        """
        # Preview mode: fast iteration for scene editing
        if mode == "preview":
            enable_timeline = False
            if settling_steps is None:
                settling_steps = 10
        assert self.scene is not None, "Call create_scene() first"

        # DEFENSIVE: Don't recompile if already compiled (prevents double RuntimeEngine creation!)
        if self.engine is not None:
            return self.engine.backend.model, self.engine.backend.data

        # Use VideoConfig for camera FPS! - MOP
        # Single source of truth - read from video_config (MOP!)
        camera_fps_value = getattr(self, 'camera_fps', self.video_config.camera_fps)

        # Read sensor_hz, state_hz, step_rate from video_config
        from ..video.video_config import VideoConfig
        mode_config = {
            "rl_core": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "rl_core_no_timeline": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},  # BENCHMARK!
            "vision_rl": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "demo": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "2k_demo": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "4k_demo": {"sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "mujoco_demo": {"sensor_hz": 200, "state_hz": 200, "step_rate": 200},
            "slow": {"sensor_hz": 200, "state_hz": 200, "step_rate": 200},
        }[self.render_mode]

        sensor_hz_value = getattr(self, 'sensor_hz', mode_config["sensor_hz"])
        state_extraction_hz = mode_config["state_hz"]
        step_rate_value = mode_config["step_rate"]

        # PERFORMANCE: Auto-disable timeline for benchmark mode
        if self.render_mode == "rl_core_no_timeline" and enable_timeline:
            print(f"  ⚡ BENCHMARK MODE: Disabling timeline (rl_core_no_timeline)")
            enable_timeline = False

        # OFFENSIVE: Remove ALL cameras when render_mode="rl_core" (no cameras!)
        # This happens HERE (not in add_robot) because free cameras are added AFTER add_robot()
        if self.render_mode in ["rl_core", "rl_core_no_timeline"]:
            # Remove robot sensor cameras (nav_camera, d405_camera)
            if self.robot:
                cameras_to_remove = [name for name in ['nav_camera', 'd405_camera']
                                    if name in self.robot.sensors]
                if cameras_to_remove:
                    self.robot.remove_sensors(cameras_to_remove)
                    print(f"  ⚠️  Removed {len(cameras_to_remove)} robot camera(s) (render_mode='rl_core' = no cameras)")

            # Remove scene free cameras (for viewer compatibility!)
            if hasattr(self.scene, 'cameras') and self.scene.cameras:
                num_free_cams = len(self.scene.cameras)
                self.scene.cameras.clear()
                print(f"  ⚠️  Removed {num_free_cams} free camera(s) (render_mode='rl_core' = no cameras)")

        # Backend selection already done in __init__() based on render_mode!
        # Create backend and engine (headless=False automatically enables viewer)
        self.backend = MuJoCoBackend(
            enable_viewer=(not self.headless),
            headless=self.headless
        )

        # Conditional timeline recording
        timeline_fps_value = self.save_fps if enable_timeline else 0

        self.engine = RuntimeEngine(
            backend=self.backend,
            camera_fps=camera_fps_value,
            sensor_hz=sensor_hz_value,
            timeline_fps=timeline_fps_value,  # TIME TRAVELER! (0 = disabled for hot_compile)
            state_extraction_hz=state_extraction_hz,  # Controlled by render_mode
            step_rate=step_rate_value,  # RL agent observation + action frequency (sim-to-real!)
            camera_width=self.camera_width,  # Resolution from render_mode
            camera_height=self.camera_height,  # Resolution from render_mode
            camera_shadows=self.camera_shadows,  # Shadow rendering quality
            camera_reflections=self.camera_reflections  # Reflection rendering quality
        )

        # Initialize StateOps (clean API for state access - MOP!)
        self.state_ops = StateOps(self.backend)

        # Initialize ActionOps (clean API for action access - MOP!)
        self.actions = ActionOps(self.engine)
        self.actions.robot = self.robot

        # Load experiment (with experiment_dir for artifact recording)
        experiment = Experiment(scene=self.scene, robot=self.robot)
        self.engine.load_experiment(
            experiment,
            experiment_dir=self.experiment_dir,
            db_ops=self.db_ops,  # NEW: For mujoco package saving
            experiment_ops=self  # NEW: For package creation flag
        )

        # MOP FIX: Load 'initial' keyframe (with correct base position + actuators)!
        # The 'initial' keyframe is modal-generated with CORRECT position from placement
        import mujoco
        initial_keyframe_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_KEY, 'initial')
        if initial_keyframe_id >= 0 and not self._robot_initial_state:
            print("  [Loading 'initial' keyframe with correct position...]")
            # Use MuJoCo's keyframe loader (loads qpos + ctrl + everything!)
            mujoco.mj_resetDataKeyframe(self.backend.model, self.backend.data, initial_keyframe_id)
            # Zero velocities (robot is stationary)
            self.backend.data.qvel[:] = 0
            # Apply forward kinematics
            mujoco.mj_forward(self.backend.model, self.backend.data)

        # Apply viewer configuration if viewer is enabled
        if not self.headless and hasattr(self.backend, 'viewer') and self.backend.viewer:
            self._apply_viewer_config()

        # MuJoCo best practice: Let physics settle after spawning
        # Resolves penetrations, establishes contacts, settles gravity
        # Configurable steps: 100 for full compile, 10 for hot_compile (faster iteration)
        if settling_steps is None:
            settling_steps = 100 if enable_timeline else 10

        # PURE MOP: Robot applies its own initial_state BEFORE settling!
        # Robot modal knows how to handle joint vs tendon actuators
        # Apply initial_state FIRST so robot settles in the correct configuration
        if self._robot_initial_state and self.robot:
            self.robot.apply_initial_state(
                self.backend.model,
                self.backend.data,
                self._robot_initial_state
            )

        # MOP: Backend knows how to settle physics!
        # For MuJoCoBackend: Settles objects while freezing robot
        # For RealBackend: No settling needed (real world is settled!)
        self.backend.settle_physics(robot=self.robot, steps=settling_steps)

        # MOP FIX: Re-sync actuators from backend after settling!
        # Settling preserves qpos, but actuator modals need to be updated
        if self.robot:
            self.backend.sync_actuators_from_backend(self.robot)

        # Bootstrap relational properties (stacked_on_X, supporting_X)
        # Make components self-aware of neighbors, then extract ALL properties
        print("  [Bootstrapping relational properties...]")
        from ..modals.behavior_extractors import extract_component_state, _get_body_position, build_extraction_cache

        # Build extraction cache ONCE (MOP: Performance optimization!)
        extraction_cache = build_extraction_cache(self.backend.model, self.backend.data)

        # CRITICAL: Include ROOM in all_assets so floor can be detected!
        # MOP: Room is an AssetModal with components (floor, walls, ceiling)
        all_assets_with_room = dict(self.scene.assets)  # Copy regular assets
        if self.scene.room:
            all_assets_with_room["floor"] = self.scene.room  # Add room as "floor" asset

        for asset_name, asset in self.scene.assets.items():
            # Self-awareness: Give each component reference to all assets + room
            for component in asset.components.values():
                component._scene_assets = all_assets_with_room

                # Extract ALL properties (including dynamic relational ones!)
                all_props = extract_component_state(
                    self.backend.model,
                    self.backend.data,
                    component,
                    all_assets=all_assets_with_room,
                    contact_cache=extraction_cache  # PERFORMANCE: Pass cache!
                )

                # Debug: Show ALL objects and their relational properties + positions
                stacked_props = [p for p in all_props.keys() if 'stacked_on_' in p or 'supporting_' in p]

                # Get position for debugging
                if component.geom_names:
                    pos = _get_body_position(self.backend.model, self.backend.data, component.geom_names[0], extraction_cache)
                    pos_str = f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                else:
                    pos_str = "no geom"

                if stacked_props:
                    print(f"    ✓ {asset_name}.{component.name}: {stacked_props} {pos_str}")
                else:
                    # Show objects WITHOUT relational properties too (debugging!)
                    if 'stackable' in str(component.behaviors):
                        print(f"    ✗ {asset_name}.{component.name}: NO CONTACTS (stackable but not stacked) {pos_str}")

        # KEYFRAME IS TRUTH: Save settled state as 'initial' keyframe!
        # This ensures reset() restores to SETTLED state, not XML default at (0,0)
        import mujoco
        keyframe_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_KEY, 'initial')
        if keyframe_id >= 0:
            # Overwrite 'initial' keyframe with current settled state
            mujoco.mj_setKeyframe(self.backend.model, self.backend.data, keyframe_id)

        # Reset sensors after settling so they track from settled position
        if self.robot:
            self.backend.sync_sensors_from_backend(self.robot, camera_backends=self.engine.camera_backends)
        # Quiet mode - no settling print

        # CAMERA BACKENDS: RuntimeEngine handles all camera backend creation and connection automatically!
        # Old code removed - cameras now autodiscovered and connected in runtime_engine.py

        # GOD MODAL: Auto-save experiment configuration + compiled XML
        self._save_god_modal()

        return self.backend.model, self.backend.data

    def _save_god_modal(self):
        """Auto-save God Modal (experiment.json + scene.xml) - delegates to DatabaseOps

        Saves:
        1. experiment.json - Complete configuration (scene, robot, config)
        2. scene.xml - Compiled MuJoCo model (for exact replay)

        Called automatically after compile() for full reproducibility.
        """
        # Create God Modal with current configuration
        experiment_modal = self.to_experiment_modal()

        # Save experiment.json (db_ops handles I/O)
        self.db_ops.save_experiment_modal(experiment_modal)

        # Save scene.xml (db_ops handles I/O)
        if self.backend and hasattr(self.backend, 'modal') and hasattr(self.backend.modal, 'compiled_xml'):
            compiled_xml = self.backend.modal.compiled_xml
            self.db_ops.save_scene_xml(compiled_xml)
            # Quiet mode - no save prints

    def hot_compile(self, script: str = "") -> Tuple[Any, Any]:
        """Lightweight compile for scene building (no timeline recording)

        Differences from compile():
        - No timeline recording (timeline_fps=0)
        - 10 settling steps (vs 100)
        - Saves snapshot to ui_db/
        - Same experiment folder
        - Forces recompilation (resets engine)

        Use cases:
        - Scene building/testing in UI
        - Quick iterations
        - Incremental scene construction (add robot → compile → add asset → compile)

        Args:
            script: Python code executed for this compile (saved to ui_db/)

        Returns:
            (model, data) from MuJoCo
        """
        # Force recompilation (like hot_reload does)
        # This ensures new objects/cameras get compiled in
        self.engine = None

        # Lightweight compile (no timeline, fast settling)
        result = self.compile(enable_timeline=False, settling_steps=10)

        # Run one step to trigger camera rendering (needed for snapshot)
        # Cameras need at least one render cycle to have valid images
        if self.engine:
            self.engine.step()

        # Save UI snapshot for display
        self._save_ui_snapshot(script)

        return result

    def _save_ui_snapshot(self, script: str = ""):
        """Save current scene state to ui_db for UI display - delegates to DatabaseOps

        Args:
            script: Python code executed for this compile
        """
        # Get all views from ViewAggregator (RUNTIME OPERATION!)
        if self.engine and hasattr(self.engine, 'view_aggregator'):
            views = self.engine.view_aggregator.create_views(
                robot=self.robot,
                scene=self.scene,
                model=self.backend.model,
                data=self.backend.data,
                update_cameras=True,
                update_sensors=True
            )

            # Extract camera images (RUNTIME OPERATION - ViewAggregator classifies!)
            camera_images = self.engine.view_aggregator.extract_camera_images(views)
        else:
            views = {}
            camera_images = {}

        # Build scene state
        scene_state = {
            "objects": [],
            "robot": None
        }

        if self.scene:
            # List all assets
            for asset_name, asset in self.scene.assets.items():
                # Skip room components
                if asset_name in {'wall_north', 'wall_south', 'wall_east', 'wall_west',
                                'floor', 'ceiling'}:
                    continue

                scene_state["objects"].append({
                    "name": asset_name,
                    "type": asset.asset_type if hasattr(asset, "asset_type") else "unknown",
                    "position": list(asset.position) if hasattr(asset, "position") else None
                })

            # Robot details
            if self.robot:
                scene_state["robot"] = {
                    "name": self.robot.name,
                    "position": list(self.robot.position) if hasattr(self.robot, "position") else None,
                    "actuators": list(self.robot.actuators.keys()) if hasattr(self.robot, "actuators") else [],
                    "sensors": list(self.robot.sensors.keys()) if hasattr(self.robot, "sensors") else []
                }

        # Save snapshot via db_ops (delegates I/O, not view classification!)
        self.db_ops.save_ui_snapshot(views, camera_images, scene_state, script)

    def _apply_viewer_config(self):
        """Apply viewer configuration settings"""
        import mujoco
        viewer = self.backend.viewer
        model = self.backend.model

        # Disable lidar rays
        if self.viewer_config.get('disable_lidar_rays', False):  # LEGITIMATE - optional config
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = 0

        # Set camera
        if 'camera' in self.viewer_config:
            cam_name = self.viewer_config['camera']
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id

        # Apply custom flags
        if 'flags' in self.viewer_config:
            for flag_name, value in self.viewer_config['flags'].items():
                if hasattr(mujoco.mjtVisFlag, flag_name):
                    flag = getattr(mujoco.mjtVisFlag, flag_name)
                    viewer.opt.flags[flag] = value

    # === SIMULATION ===

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step - SELF-EXPLANATORY

        SEAMLESS: Auto-compiles if needed (simulated mode only)
e        TIME TRAVELER: Auto-saves timeline at specified FPS!

        Returns:
            Dict with step_count, reward, state, elapsed_time
        """
        # Auto-compile if not yet compiled (simulated mode only)
        if self.engine is None:
            if self.mode == "simulated":
                print("  [Auto-compiling scene...]")
                self.compile()
            else:
                raise RuntimeError("Real mode not yet implemented. Use mode='simulated'")

        self.last_result = self.engine.step()
        self._step_counter += 1

        # TIME TRAVELER: Auto-save timeline at FPS rate
        # TODO: Implement _maybe_save_timeline_step() method
        # if self.save_fps > 0:
        #     self._maybe_save_timeline_step()

        return self.last_result

    def load_frame(self, frame_num: int):
        """Load and apply specific frame state - TIME MACHINE!

        Args:
            frame_num: Frame number to load (0-indexed)

        Example:
            ops = load_experiment("database/exp_123/", headless=False)
            ops.load_frame(42)  # Jump to frame 42
        """
        if not hasattr(self, '_read_only') or not self._read_only:
            print("  ⚠️  load_frame() is intended for loaded experiments")

        assert self.backend is not None, "No backend - call compile() or load_experiment() first"

        # Load frame
        from pathlib import Path
        scene_state_dir = Path(self.experiment_dir) / "scene_state"
        frame_path = scene_state_dir / f"frame_{frame_num:04d}.xml"

        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {frame_num} not found at {frame_path}")

        from ..modals.scene_state_modal import SceneStateModal
        state = SceneStateModal.load(frame_path)
        state.apply(self.backend.model, self.backend.data)

        print(f"  ✓ Loaded frame {frame_num} (t={state.time:.3f}s)")

    def get_frame_count(self) -> int:
        """Get number of available frames - TIME MACHINE!

        Returns:
            Number of frames saved in scene_state/ (0 if none)

        Example:
            ops = load_experiment("database/exp_123/")
            print(f"Total frames: {ops.get_frame_count()}")
        """
        from pathlib import Path
        scene_state_dir = Path(self.experiment_dir) / "scene_state"

        if not scene_state_dir.exists():
            return 0

        frame_files = list(scene_state_dir.glob("frame_*.xml"))
        return len(frame_files)

    def replay_frames(self, start_frame: int = 0, end_frame: int = None,
                     fps: float = 30, loop: bool = False):
        """Replay saved frames in viewer - TIME MACHINE!

        Args:
            start_frame: Starting frame number (default 0)
            end_frame: Ending frame number (None = last frame)
            fps: Playback FPS (default 30)
            loop: Loop playback forever (default False)

        Examples:
            # Replay all frames
            ops = load_experiment("database/exp_123/", headless=False)
            ops.replay_frames()

            # Replay subset
            ops.replay_frames(start_frame=50, end_frame=150, fps=60)

            # Loop forever (Ctrl+C to stop)
            ops.replay_frames(loop=True)
        """
        import time
        from pathlib import Path

        assert self.backend is not None, "No backend loaded"

        scene_state_dir = Path(self.experiment_dir) / "scene_state"
        if not scene_state_dir.exists():
            print("  ⚠️  No scene_state/ directory - no frames to replay")
            return

        # Get available frames
        frame_files = sorted(scene_state_dir.glob("frame_*.xml"))
        if not frame_files:
            print("  ⚠️  No frames found in scene_state/")
            return

        # Extract frame numbers
        frame_nums = [int(f.stem.split('_')[1]) for f in frame_files]

        if end_frame is None:
            end_frame = max(frame_nums)

        frame_dt = 1.0 / fps

        print(f"  ▶️  Replaying frames {start_frame} to {end_frame} at {fps} FPS")
        if loop:
            print("  🔁 Loop mode enabled (Ctrl+C to stop)")

        try:
            while True:
                for frame_num in range(start_frame, end_frame + 1):
                    if frame_num not in frame_nums:
                        continue

                    start_time = time.time()

                    # Load and apply frame (quiet mode for replay)
                    frame_path = scene_state_dir / f"frame_{frame_num:04d}.xml"
                    from ..modals.scene_state_modal import SceneStateModal
                    state = SceneStateModal.load(frame_path)
                    state.apply(self.backend.model, self.backend.data)

                    # Viewer updates automatically

                    # Sleep to maintain FPS
                    elapsed = time.time() - start_time
                    sleep_time = max(0, frame_dt - elapsed)
                    time.sleep(sleep_time)

                if not loop:
                    break

            print("  ✓ Replay complete")

        except KeyboardInterrupt:
            print("\n  ⏸️  Replay stopped by user")

    def sync_from_mujoco(self):
        """Sync all state from MuJoCo - EDUCATIONAL API

        Updates all modals (robot + assets) from physics state.
        Use this after direct mj_step() or when you need explicit sync.

        This is the MIDDLE-LEVEL API:
        - step() = physics + sync + rewards (high-level, most common)
        - sync_from_mujoco() = just sync (mid-level, for custom loops)
        - backend.sync_*() = low-level (experts only)

        Example:
            # Custom physics loop
            for i in range(100):
                mujoco.mj_step(ops.model, ops.data)
                ops.sync_from_mujoco()  # Sync modals
                reward = ops.get_reward()
        """
        assert self.engine is not None, "Call compile() first"

        # Sync robot (actuators + sensors)
        if self.robot:
            self.backend.sync_actuators_from_backend(self.robot)
            self.backend.sync_sensors_from_backend(self.robot, camera_backends=self.engine.camera_backends)

        # Sync all assets (objects, furniture, walls)
        self.backend.sync_assets_from_backend(self.scene)

        # Extract state for rewards (so get_total_reward() works after sync!)
        if self.engine and self.scene:
            self.last_state = self.engine.state_extractor.extract(self.scene, self.robot)

    def run(self, steps: int = None):
        """Run simulation for N steps - SELF-EXPLANATORY

        Args:
            steps: Number of steps (None = infinite)
        """
        assert self.engine is not None, "Call compile() first"
        self.engine.run(steps=steps)

    # === DYNAMIC SCENE MODIFICATION ===

    def hot_reload(self):
        """Recompile scene while preserving robot state - MOP STYLE!

        Use case: Add objects/cameras during simulation without losing robot state.

        NOTE: Only existing state is preserved. New objects will settle naturally.

        Example:
            ops.compile()
            ops.run(100)

            # Add new object dynamically
            ops.add_asset("table", relative_to=(3, 3, 0))
            ops.hot_reload()  # Recompile + preserve robot state

            ops.run(100)  # Continue with new object
        """
        assert self.backend is not None, "Call compile() first"

        # Save ENTIRE current state (all qpos/qvel before adding new objects)
        old_qpos_size = len(self.backend.data.qpos)
        old_qvel_size = len(self.backend.data.qvel)

        saved_qpos = self.backend.data.qpos.copy()
        saved_qvel = self.backend.data.qvel.copy()
        saved_time = self.backend.data.time

        # Force recompile (clear engine to bypass protection)
        self.engine = None

        # Recompile with new scene contents
        self.compile()

        # Restore state (only restore what existed before)
        # New objects get default positions
        restore_size_qpos = min(old_qpos_size, len(self.backend.data.qpos))
        restore_size_qvel = min(old_qvel_size, len(self.backend.data.qvel))

        self.backend.data.qpos[:restore_size_qpos] = saved_qpos[:restore_size_qpos]
        self.backend.data.qvel[:restore_size_qvel] = saved_qvel[:restore_size_qvel]
        self.backend.data.time = saved_time

        # Forward kinematics to update derived quantities
        import mujoco
        mujoco.mj_forward(self.backend.model, self.backend.data)

        print(f"  ✓ Hot reload complete - preserved {restore_size_qpos}/{len(self.backend.data.qpos)} DOFs")

    def teleport_object(self, name: str, position: tuple):
        """Teleport object to new position - CLEAN API!

        Args:
            name: Object/body name (e.g., 'apple', 'table')
            position: (x, y, z) coordinates

        Example:
            ops.teleport_object('apple', position=(2, 2, 1))
        """
        assert self.backend is not None, "Call compile() first"
        import mujoco

        # Find body ID
        body_id = mujoco.mj_name2id(self.backend.model, mujoco.mjtObj.mjOBJ_BODY, name)

        if body_id < 0:
            available = self.get_all_bodies()
            raise ValueError(
                f"Body '{name}' not found in simulation\n"
                f"Available bodies: {list(available.keys())}"
            )

        # Teleport (freejoint qpos: x, y, z, qw, qx, qy, qz)
        qpos_addr = self.backend.model.jnt_qposadr[body_id] if body_id < self.backend.model.njnt else None

        if qpos_addr is not None:
            self.backend.data.qpos[qpos_addr:qpos_addr+3] = position  # x, y, z
        else:
            # No joint - probably fixed body
            raise ValueError(f"Body '{name}' has no freejoint - cannot teleport")

        # Update forward kinematics
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def get_all_bodies(self) -> Dict[str, int]:
        """Get all bodies in simulation with their IDs - MOP STYLE!

        Returns:
            Dict mapping body names to IDs

        Example:
            bodies = ops.get_all_bodies()
            # {'apple': 5, 'table': 6, 'stretch': 2, 'world': 0, ...}
        """
        assert self.backend is not None, "Call compile() first"
        import mujoco

        bodies = {}
        for i in range(self.backend.model.nbody):
            name = mujoco.mj_id2name(self.backend.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                bodies[name] = i

        return bodies

    def add_debug_camera(self, camera_id: str, track_target: str = None, **kwargs):
        """Add camera that auto-connects to running simulation - CONVENIENCE!

        Like add_free_camera() but auto-connects if simulation already running.

        Args:
            camera_id: Camera name
            track_target: Object/robot to track
            **kwargs: Other camera parameters (lookat, distance, azimuth, etc.)

        Returns:
            CameraModal instance

        Example:
            ops.compile()
            ops.run(100)

            # Add camera during simulation - auto-connects!
            ops.add_debug_camera('side_view', track_target='apple', azimuth=0)

            ops.run(100)  # Camera renders immediately
        """
        # Create camera (scene-level)
        camera = self.add_free_camera(camera_id, track_target=track_target, **kwargs)

        # RuntimeEngine handles camera backend creation and connection automatically!
        # Backend will be created on next compile() or if already running
        print(f"  ✓ Debug camera '{camera_id}' added (backend auto-connected by RuntimeEngine)")

        return camera

    def submit_block(self, action_block):
        """Submit action block for execution - SELF-EXPLANATORY

        SEAMLESS: Auto-compiles if needed (simulated mode only)

        Args:
            action_block: ActionBlock to execute

        Returns:
            Block ID
        """
        # Auto-compile if not yet compiled (simulated mode only)
        if self.engine is None:
            if self.mode == "simulated":
                print("  [Auto-compiling scene...]")
                self.compile()
            else:
                raise RuntimeError("Real mode not yet implemented. Use mode='simulated'")

        return self.engine.submit_block(action_block)

    def submit_action(self, action):
        """Submit single action for execution - CONVENIENCE METHOD

        Wraps action in ActionBlock and submits to engine.

        Args:
            action: Action instance (e.g., ArmMoveTo, BaseRotateBy, etc.)

        Returns:
            Block ID
        """
        from ..modals.stretch.action_modals import ActionBlock

        block = ActionBlock(
            id=f"single_action_{action.id}",
            description=f"Single action: {action.__class__.__name__}",
            execution_mode="sequential",
            actions=[action]
        )
        return self.submit_block(block)

    # === SKILLS SYSTEM - RL LEGO COMPOSITION ===

    def load_skills(self) -> Dict[str, Any]:
        """Load all skills from SKILLS_REGISTRY.json - PURE MOP

        Returns:
            Dict[skill_id -> Skill]

        Example:
            skills = ops.load_skills()
            nav_skill = skills['navigate_to_table_rl']
        """
        from ..modals.stretch.action_blocks_registry import load_skills_from_json
        return load_skills_from_json()




    def compose_manual(self, block_ids: list, name: str = None, description: str = ""):
        """Compose skill manually from ActionBlock IDs - LEGO COMPOSITION

        Args:
            block_ids: List of ActionBlock IDs
            name: Skill name (auto-generated if None)
            description: Skill description

        Returns:
            Skill object

        Example:
            skill = ops.compose_manual(
                ['investigate', 'robot_dance', 'moonwalk'],
                name='Party Investigation'
            )
            ops.execute_skill(skill)
        """
        from ..rl.composers import ManualComposer
        composer = ManualComposer()
        return composer.compose(block_ids, name=name, description=description)

    def compose_text(self, text: str, available_blocks: list = None):
        """Compose skill from text using AI Orchestrator - LEGO COMPOSITION

        Args:
            text: Natural language description (e.g., "investigate the room then celebrate")
            available_blocks: List of available ActionBlock IDs (None = all from registry)

        Returns:
            Skill object

        Example:
            skill = ops.compose_text("navigate to the table and pick up the cup")
            ops.execute_skill(skill)
        """
        import asyncio
        from ..rl.composers import TextComposer

        # Get available blocks if not provided
        if available_blocks is None:
            from ..modals.stretch.action_blocks_registry import get_all_action_blocks
            available_blocks = list(get_all_action_blocks().keys())

        composer = TextComposer()

        # Run async composition
        loop = asyncio.get_event_loop()
        skill = loop.run_until_complete(composer.compose(text, available_blocks))

        return skill

    def compose_vision(self, image, goal: str):
        """Compose skill from vision using VLM - LEGO COMPOSITION (STUB)

        Args:
            image: Image data (numpy array, PIL, or path)
            goal: Goal description

        Returns:
            Skill object

        Example:
            skill = ops.compose_vision(camera_image, "pick up the red cup")
            ops.execute_skill(skill)
        """
        from ..rl.composers import VisionComposer
        composer = VisionComposer()
        return composer.compose(image, goal)

    def compose_behavior_tree(self, tree_spec: dict):
        """Compose skill from behavior tree spec - LEGO COMPOSITION (STUB)

        Args:
            tree_spec: BT specification dict

        Returns:
            Skill object

        Example:
            spec = {
                'root': 'sequence',
                'children': [
                    {'condition': 'at_table'},
                    {'action': 'pick_object_rl'}
                ]
            }
            skill = ops.compose_behavior_tree(spec)
        """
        from ..rl.composers import BehaviorTreeComposer
        composer = BehaviorTreeComposer()
        return composer.compose(tree_spec)

    # === STATE & REWARD ===

    def get_state(self) -> Dict[str, Any]:
        """Get current state - SELF-SYNCING (MOP Principle #2)

        MOP VIOLATION FIX: State was NOT self-syncing before first step()!
        Now automatically triggers step() if state is empty.

        Educational: This ensures state is ALWAYS available when requested,
        following Modal-Oriented Programming principle of SELF-SYNCING modals.
        """
        # SELF-SYNCING: Auto-step if no state yet (first call after compile)
        if not self.last_result:
            print("🔄 AUTO-SYNC: get_state() called before step() - auto-stepping to populate state (MOP self-syncing)")
            self.step()

        if self.last_result:
            return self.last_result['state']  # OFFENSIVE - crash if missing!

        # Should never reach here after auto-step, but defensive
        return {}

    def get_reward(self) -> float:
        """Get current reward delta (per-step) - SELF-EXPLANATORY"""
        if self.last_result:
            return self.last_result['reward_step']  # OFFENSIVE - crash if missing!
        return 0.0


    def get_reward_timeline(self, format: str = "dict"):
        """Get reward timeline - MOP delegation to RewardModal!
        
        Convenience method that delegates to RewardModal.get_reward_timeline().
        Following MOP: RewardModal knows its timeline, ExperimentOps just asks.
        
        Args:
            format: "dict" (JSON-serializable) or "numpy" (structured array)
        
        Returns:
            Reward timeline data (see RewardModal.get_reward_timeline for format details)
        """
        return self.scene.reward_modal.get_reward_timeline(format=format)

    def is_facing(self, object1: str, object2: str, threshold: float = 0.7) -> dict:
        """Check if object1 is facing object2 - MOP UTILITY!

        Convenience wrapper that automatically uses current runtime state.
        Delegates to Scene.is_facing() (MOP delegation pattern).

        Args:
            object1: Asset or component name (e.g., "stretch.arm", "apple")
            object2: Asset or component name (e.g., "table", "apple")
            threshold: Dot product threshold for "facing" (default 0.7 = ~45°)

        Returns:
            dict with:
                - facing: bool (True if dot > threshold)
                - dot: float (how much facing: 1.0 = directly facing, 0.0 = perpendicular, -1.0 = opposite)
                - dot_class: str (category: "directly_facing", "facing", "partially_facing", "perpendicular", "partially_away", "facing_away", "directly_opposite")
                - dot_explain: str (human-readable explanation with angle estimate)
                - distance: float (distance between objects in meters)
                - object1_direction: list [dx, dy, dz]
                - object2_position: list [x, y, z]

        Example:
            result = ops.is_facing("stretch.arm", "apple")
            print(result["dot_explain"])  # "stretch.arm is perpendicular to apple (side-by-side, ~90°)"
            if result["facing"]:
                print(f"Arm is facing apple! (dot={result['dot']:.2f})")
        """
        # SELF-SYNCING: Auto-step if no state yet
        if not self.last_result:
            print("🔄 AUTO-SYNC: is_facing() called before step() - auto-stepping to populate state")
            self.step()

        # Delegate to Scene modal (MOP: Scene knows its spatial relationships)
        runtime_state = {"extracted_state": self.last_result["state"]}
        return self.scene.is_facing(object1, object2, runtime_state, threshold)

    def get_distance(self, object1: str, object2: str) -> dict:
        """Get distance between two objects - MOP UTILITY!"""
        if not self.last_result:
            print("🔄 AUTO-SYNC: get_distance() called before step() - auto-stepping")
            self.step()

        runtime_state = {"extracted_state": self.last_result["state"]}
        return self.scene.get_distance(object1, object2, runtime_state)

    def get_offset(self, object1: str, object2: str) -> dict:
        """Get offset from object1 to object2 - MOP UTILITY!"""
        if not self.last_result:
            print("🔄 AUTO-SYNC: get_offset() called before step() - auto-stepping")
            self.step()

        runtime_state = {"extracted_state": self.last_result["state"]}
        return self.scene.get_offset(object1, object2, runtime_state)

    def get_asset_info(self, asset_name: str) -> dict:
        """Get asset information - MOP! Asset knows itself!

        Can be called BEFORE or AFTER compile:
        - Before compile: Returns height from XML definition
        - After compile: Returns actual runtime position + height

        Wrapper that delegates to Asset modal.
        """
        # Check if asset exists in scene
        if asset_name not in self.scene.assets:
            return {"exists": False, "error": f"Asset '{asset_name}' not found in scene"}

        asset = self.scene.assets[asset_name]

        # If compiled, get runtime info
        if self.last_result:
            runtime_state = {"extracted_state": self.last_result["state"]}
            return self.scene.get_asset_info(asset_name, runtime_state)

        # Before compile - extract from XML definition
        from ..modals.xml_resolver import extract_dimensions_from_xml, XMLResolver
        from ..modals.registry import get_asset_xml_path
        import xml.etree.ElementTree as ET
        from pathlib import Path

        try:
            # Load asset XML
            xml_file = asset.config.get("xml_file")
            if not xml_file:
                return {"exists": True, "error": "Asset has no XML file (virtual asset)"}

            # Get asset type (furniture, objects, robots, etc.)
            asset_type = asset.config.get("type", "furniture")

            # MOP: Delegate to centralized path resolver (SINGLE SOURCE OF TRUTH!)
            full_path = get_asset_xml_path(asset_name, asset_type, xml_file)

            tree = ET.parse(full_path)
            root = tree.getroot()

            # MOP FIX: Resolve XML includes BEFORE extracting dimensions!
            # Table XML uses <include> tags, so we need to resolve them first
            base_dir = full_path.parent
            XMLResolver._resolve_includes(root, base_dir)

            # Extract dimensions (now includes are resolved!)
            dims = extract_dimensions_from_xml(root, asset_name)

            # Return surface_z as height (where objects sit on furniture)
            # For furniture with "surface" behavior, this is the top surface position
            height = dims.get("surface_z", dims.get("height", 0.0))

            return {
                "exists": True,
                "height": height,
                "z": height,
                "width": dims.get("width", 0.0),
                "depth": dims.get("depth", 0.0),
                "from_xml": True  # Indicates pre-compile extraction
            }

        except Exception as e:
            return {"exists": True, "error": f"Failed to extract info from XML: {e}"}

    def get_robot_info(self, robot_type: str = "stretch") -> dict:
        """Get robot specifications - MOP! Robot knows itself!

        Returns robot actuator specs + geometry for dynamic positioning calculations.
        Enables calculating robot positions (no hardcoding!)
        Works BEFORE or AFTER adding robot to scene!

        Args:
            robot_type: Type of robot (e.g., "stretch") - default "stretch"

        Returns:
            dict with:
                - robot_type: str
                - actuators: {name: {min_position, max_position, unit, type}}
                - geometry: {gripper_length, base_to_arm_offset, base_height}
                - margins: {reach_safety, placement_safety, grasp_threshold}
                - comfortable_pct: {arm_reach, lift_height}

        Example:
            ```python
            robot_info = ops.get_robot_info("stretch")
            arm_max = robot_info['actuators']['arm']['max_position']  # 0.52m
            gripper_len = robot_info['geometry']['gripper_length']  # 0.144m (from XML!)
            comfortable_pct = robot_info['comfortable_pct']['arm_reach']  # 0.7
            ```

        Raises:
            ValueError: If robot not found in scene
        """
        # OFFENSIVE: Scene MUST be built!
        if self.scene is None:
            raise ValueError(
                "Cannot get robot info - scene not created yet!\n"
                "Call ops.create_scene() first"
            )

        # Delegate to scene modal - MOP!
        return self.scene.get_robot_info(robot_type)

    def get_view(self, view_name: str):
        """Get view - THIN MOP WRAPPER (delegates to ViewAggregator)"""
        return self.engine.last_views[view_name]

    # === COMPATIBILITY WITH OLD SceneOps API ===

    def get_current_state_for_rewards(self) -> Dict[str, Any]:
        """Get state (old API compatibility)"""
        return self.get_state()

    def get_total_reward(self, current_time: float = None) -> float:
        """Get reward - OFFENSIVE & ACTUALLY COMPUTES!

        This is the MIDDLE-LEVEL API for custom physics loops.
        Works after sync_from_mujoco() OR step().

        Returns:
            Total reward from all conditions
        """
        assert self.engine is not None, "Call compile() first"
        assert self.scene is not None, "No scene loaded"
        assert self.scene.reward_modal is not None, "No reward_modal in scene"

        # Get state (from sync or step)
        state = self.last_state if self.last_state else self.get_state()

        assert state, "No state available! Call sync_from_mujoco() or step() first"

        # Compute rewards using reward_computer (same as step() does!) - OFFENSIVE: crashes reveal bugs!
        import time
        # Use engine's actual start_time, not hardcoded 0.0!
        if current_time is None and self.engine.start_time:
            current_time = time.time() - self.engine.start_time

        return self.engine.reward_computer.compute(
            state,
            self.scene.reward_modal,
            current_time or 0.0,
            self.scene.reward_modal.start_time
        )

    # === PURE MOP RL API (Modal-to-Modal Communication!) ===

    def reset(self):
        """Reset episode without recompile - PURE MOP!

        Modals reset themselves:
        - Backend resets to 'initial' keyframe (10ms vs 300ms+ recompile!)
        - Robot modals reset
        - Reward modals reset
        - Sync state from MuJoCo

        This is 10-30x FASTER than compile()!

        Example:
            ops.compile()  # Once at start
            for episode in range(1000):
                ops.reset()  # Fast keyframe reset!
                for step in range(max_steps):
                    ops.apply_action(action)
                    ops.step()
        """
        assert self.engine is not None, "Call compile() first!"
        assert self.backend is not None, "No backend!"

        # Reset MuJoCo to 'initial' keyframe (FAST!)
        # This restores robot state automatically!
        self.backend.reset_to_keyframe('initial')

        # Reward modals reset themselves
        if self.scene and self.scene.reward_modal:
            self.scene.reward_modal.reset()

        # Sync state from MuJoCo (robot state comes from keyframe)
        self.sync_from_mujoco()

    def apply_action(self, action: 'np.ndarray', actuators_active: Optional[List[str]] = None):
        """Modals execute actions - PURE MOP (NO MANUAL MAPPING!)

        Args:
            action: Numpy array of continuous actions
            actuators_active: None (all actuators) or List[str] (focused subset)

        Example:
            # Full action space (12D)
            ops.apply_action(action)

            # Focused action space (1D - just base rotation!)
            ops.apply_action(action, actuators_active=['left_wheel_vel'])
        """
        assert self.engine is not None, "Call compile() first!"
        assert self.robot is not None, "No robot!"
        assert self.backend is not None, "No backend!"

        # Get active actuators (focused or all)
        if actuators_active is not None:
            active_actuators = {
                name: actuator
                for name, actuator in self.robot.actuators.items()
                if name in actuators_active
            }
        else:
            active_actuators = self.robot.actuators

        # Modals execute actions and return commands
        commands = {}
        for i, (name, actuator) in enumerate(active_actuators.items()):
            if i >= len(action):
                break

            # Skip virtual actuators (no MuJoCo joints)
            if not actuator.joint_names:
                continue

            # Modal executes action
            target = float(action[i])
            clipped = actuator.move_to(target)  # Returns clipped value
            commands[name] = clipped

        # Send commands to MuJoCo
        if commands:
            self.backend.set_controls(commands)

    def get_action_space(self, actuators_group: str = "full"):
        """Get action space - PURE MOP! Delegates to modals (no reconstruction!)

        Registry is the SINGLE SOURCE OF TRUTH for atomic actions.
        ExperimentOps just asks the registry - NO transformation, NO dtype loss!

        Args:
            actuators_group:
                - "atomic_rotation", "atomic_movement", etc. → Single atomic action
                - "atomic_all" → Dict of all atomic actions
                - "full", "movement", "manipulation", etc. → Raw actuator groups (legacy)

        Returns:
            gym.spaces.Box for single action, or Dict[str, Box] for "atomic_all"

        Example:
            # Atomic actions (preserves dtype!)
            space = ops.get_action_space("atomic_rotation")  # Box(-360, 360, int32)
            all_atomic = ops.get_action_space("atomic_all")  # Dict of all atomic

            # Raw actuators (legacy)
            space = ops.get_action_space("full")  # Box(10,) all actuators
        """
        import gymnasium as gym
        import numpy as np

        # PURE MOP: Import action space resolver from registry (SINGLE SOURCE OF TRUTH!)
        from ..modals.stretch.action_blocks_registry import (
            get_action_space as get_atomic_space,
            list_atomic_actions
        )

        # Case 1: "atomic_all" - return dict of all atomic actions
        if actuators_group == "atomic_all":
            return {name: get_atomic_space(name, self.robot)
                    for name in list_atomic_actions()}

        # Case 2: Single atomic action - delegate to registry (PRESERVES DTYPE!)
        elif actuators_group.startswith("atomic_"):
            return get_atomic_space(actuators_group, self.robot)

        # Case 3: Raw actuator groups (existing functionality for backward compat)
        else:
            assert self.robot is not None, "No robot! Call add_robot() first"

            # Actuator groups (kept for legacy compatibility)
            GROUPS = {
                "full": None,  # All actuators
                "movement": ["left_wheel_vel", "right_wheel_vel"],
                "manipulation": ["arm", "lift", "gripper", "wrist_yaw", "wrist_pitch", "wrist_roll"],
                "arm": ["arm", "gripper"],
            }

            # Get actuator list
            if isinstance(actuators_group, list):
                actuator_names = actuators_group
            elif actuators_group in GROUPS:
                if GROUPS[actuators_group] is None:
                    # Full - all non-virtual actuators
                    actuator_names = [name for name, act in self.robot.actuators.items() if act.joint_names]
                else:
                    actuator_names = GROUPS[actuators_group]
            else:
                available = list(GROUPS.keys()) + list_atomic_actions() + ["atomic_all"]
                raise ValueError(f"Unknown group '{actuators_group}'. Available: {available}")

            # Build action space bounds from actuator modals (PURE MOP!)
            lows = []
            highs = []

            for name in actuator_names:
                if name not in self.robot.actuators:
                    raise ValueError(f"Actuator '{name}' not found in robot")

                actuator = self.robot.actuators[name]

                # Skip virtual actuators (no MuJoCo joints)
                if not actuator.joint_names:
                    continue

                # PURE MOP: Actuator modal knows its own range!
                low, high = actuator.range
                lows.append(low)
                highs.append(high)

            return gym.spaces.Box(
                low=np.array(lows, dtype=np.float32),
                high=np.array(highs, dtype=np.float32),
                dtype=np.float32
            )

    def evaluate_rewards(self) -> Tuple[float, Dict[str, Any]]:
        """RewardModals compute their own rewards - PURE MOP!

        Returns:
            (total_reward, reward_info)

        Example:
            reward, info = ops.evaluate_rewards()
            # info = {"rotation_reward": {"value": 50.0, "progress": 50%}, ...}
        """
        assert self.engine is not None, "Call compile() first!"
        assert self.scene is not None, "No scene!"
        assert self.scene.reward_modal is not None, "No rewards!"

        # Get current state
        state = self.last_state if self.last_state else self.get_state()
        assert state, "No state! Call step() or sync_from_mujoco() first"

        # RewardModals compute their own rewards
        import time
        current_time = time.time() - self.engine.start_time if self.engine.start_time else 0.0

        reward_dict = self.engine.reward_computer.compute(
            state,
            self.scene.reward_modal,
            current_time,
            self.scene.reward_modal.start_time
        )

        # Return total and full dict with breakdown
        return reward_dict.get("total", 0.0), reward_dict

    def check_termination(self) -> bool:
        """RewardModals know when task complete - PURE MOP!

        Returns:
            True if any reward goal achieved, False otherwise

        Example:
            if ops.check_termination():
                print("Task complete!")
                break
        """
        assert self.engine is not None, "Call compile() first!"
        assert self.scene is not None, "No scene!"

        if not self.scene.reward_modal:
            return False

        # Get current state
        state = self.last_state if self.last_state else self.get_state()
        if not state:
            return False

        # RewardModals check if goals achieved
        for condition in self.scene.reward_modal.conditions:
            if hasattr(condition, 'is_complete') and condition.is_complete(state):
                return True

        return False

    # === DIRECT ACCESS ===

    @property
    def model(self):
        """Access MuJoCo model directly"""
        return self.backend.model if self.backend else None

    @property
    def data(self):
        """Access MuJoCo data directly"""
        return self.backend.data if self.backend else None

    @property
    def assets(self):
        """Beautiful asset access - PURE MOP!

        Returns AssetsCollection with property access to all assets.
        No more dict parsing! IDE autocomplete works!

        Returns:
            AssetsCollection with property-based asset access

        Example:
            pos = ops.assets.apple.position  # ✨ Beautiful!
            behaviors = ops.assets.apple.behaviors
            is_on_table = ops.assets.apple.is_stacked_on('table')
            distance = ops.assets.apple.distance_to('table')
        """
        from ..modals.assets_collection_modal import AssetsCollection

        # Get current state (auto-syncs if needed)
        state = self.get_state()

        # Return beautiful collection
        return AssetsCollection(self.scene, state)

    # === GOD MODAL INTEGRATION (PURE MOP!) ===

    def to_experiment_modal(self):
        """Export to GOD MODAL - PURE MOP!

        Backend saves itself! OFFENSIVE - crashes if backend not compiled!

        Returns:
            ExperimentModal with scene, config, and MuJoCo state

        Usage:
            ops = ExperimentOps(headless=True)
            ops.create_scene("kitchen", width=5, length=5, height=3)
            ops.add_robot("stretch")
            ops.compile()

            # Export to GOD MODAL (backend saves itself!)
            experiment = ops.to_experiment_modal()
            experiment.description = "Level 2A training"
            experiment.save()

            # Later: Load and resume
            experiment = ExperimentModal.load("path/experiment.json")
            ops = experiment.compile()  # Backend loads from saved XML!
        """
        from ..modals.experiment_modal import ExperimentModal
        from pathlib import Path

        return ExperimentModal(
            scene=self.scene,
            config={
                "mode": self.mode,
                "headless": self.headless,
                "viewer_config": self.viewer_config,
            },
            compiled_xml=self.backend.to_xml(),      # Backend saves itself!
            mujoco_state=self.backend.to_state(),    # Backend saves its state!
            experiment_id=self.db_ops.experiment_id,  # From DatabaseOps
            experiment_dir=self.db_ops.get_experiment_path()  # From DatabaseOps
        )

    # ============================================================================
    # Resource Management - Close & Context Manager Support
    # ============================================================================

    def validate_videos(self, timeout: float = 180):
        """Validate that ALL videos were converted successfully - MOP!

        Args:
            timeout: Maximum seconds to wait for async conversion (default 3 minutes)

        Raises:
            RuntimeError: If any videos failed to convert or timeout occurs

        MOP: Tests MUST call this after close() to validate video conversion!
              Silent failures are NOT acceptable - we crash explicitly!

        Example:
            ops = ExperimentOps(headless=True, render_mode="2k_demo")
            ops.create_scene(...)
            ops.compile()
            for _ in range(300):
                ops.step()
            ops.close()  # Start async video conversion
            ops.validate_videos()  # Wait and validate!
        """
        if not hasattr(self, 'engine') or self.engine is None:
            return  # No engine - nothing to validate

        # Access timeline saver through runtime engine
        if hasattr(self.engine, 'timeline_saver') and self.engine.timeline_saver is not None:
            self.engine.timeline_saver.validate_videos(timeout=timeout)

    # ============================================================================
    # Test Validation Methods - Clean MOP API
    # ============================================================================

    def get_positions(self, objects: List[str], print_summary: bool = True) -> Dict[str, Tuple[float, float, float]]:
        """Get positions of multiple objects - MOP!

        Args:
            objects: List of object names (e.g., ['apple', 'banana', 'mug'])
            print_summary: Print formatted summary (default: True)

        Returns:
            Dict[object_name -> (x, y, z) position]

        Example:
            positions = ops.get_positions(['apple', 'banana', 'mug'])
            # Prints:
            # 📍 Object Positions:
            #   apple  : (2.1, 0.3, 0.75)
            #   banana : (2.0, 0.0, 0.73)
        """
        positions = {}

        if print_summary:
            print("\n📍 Object Positions:")

        for obj_name in objects:
            try:
                asset = getattr(self.assets, obj_name)
                pos = asset.position
                positions[obj_name] = pos

                if print_summary:
                    behaviors = asset.behaviors
                    # Check contact status
                    is_on_table = asset.is_stacked_on('table')
                    is_on_floor = asset.is_stacked_on('floor')
                    status = "✅ ON TABLE" if is_on_table else ("✅ ON FLOOR" if is_on_floor else "⚠️  FALLING")
                    print(f"  {obj_name:8s}: pos={pos}  behaviors={behaviors}  {status}")
            except AttributeError as e:
                if print_summary:
                    print(f"  {obj_name:8s}: ❌ NOT FOUND - {e}")
                positions[obj_name] = None

        return positions

    def validate_positions(
        self,
        expected: Dict[str, Union[Tuple[float, float, float], Dict[str, Any]]],
        tolerance: float = 0.01,
        print_summary: bool = True
    ) -> Dict[str, Any]:
        """Validate positions against expected values - MOP!

        Supports absolute positions OR relative positioning!

        Args:
            expected: Dict of object_name -> expected_position where position can be:
                - Tuple: (x, y, z) absolute position
                - Dict: {"relative_to": "obj", "offset": (dx, dy, dz)} relative position
            tolerance: Maximum allowed difference per axis (default: 0.01m = 1cm)
            print_summary: Print formatted summary (default: True)

        Returns:
            Dict with:
            {
                "valid": bool,
                "errors": List[str],
                "actual": Dict[obj -> position],
                "expected_resolved": Dict[obj -> position]
            }

        Examples:
            # Absolute positions
            result = ops.validate_positions({
                "apple": (1.6, 0.25, 0.76),
                "banana": (2.0, 0.0, 0.76)
            })

            # Relative positioning - Beautiful MOP!
            result = ops.validate_positions({
                "apple": {"relative_to": "table", "offset": (-0.4, 0.25, 0.76)},
                "banana": {"relative_to": "table", "offset": (0.0, 0.0, 0.76)}
            })
        """
        # Get actual positions
        actual = self.get_positions(list(expected.keys()), print_summary=False)

        # Resolve expected positions (handle relative positioning)
        expected_resolved = {}
        for obj_name, exp_pos in expected.items():
            if isinstance(exp_pos, dict):
                # Relative positioning
                base_obj = exp_pos["relative_to"]
                offset = exp_pos["offset"]
                base_pos = getattr(self.assets, base_obj).position
                expected_resolved[obj_name] = (
                    base_pos[0] + offset[0],
                    base_pos[1] + offset[1],
                    base_pos[2] + offset[2]
                )
            else:
                # Absolute position
                expected_resolved[obj_name] = exp_pos

        # Validate
        errors = []
        if print_summary:
            print("\n📍 Position Validation:")

        for obj_name, exp_pos in expected_resolved.items():
            act_pos = actual.get(obj_name)
            if act_pos is None:
                error = f"{obj_name}: NOT FOUND"
                errors.append(error)
                if print_summary:
                    print(f"  ❌ {obj_name}: NOT FOUND")
                continue

            # Check each axis
            diff = [abs(a - e) for a, e in zip(act_pos, exp_pos)]
            max_diff = max(diff)

            if max_diff > tolerance:
                error = f"{obj_name}: expected {exp_pos}, got {act_pos}, diff={max_diff:.4f}m"
                errors.append(error)
                if print_summary:
                    print(f"  ❌ {obj_name}: expected {exp_pos}, got {act_pos}")
                    print(f"     Difference: ({diff[0]:.4f}, {diff[1]:.4f}, {diff[2]:.4f})m > tolerance {tolerance}m")
            else:
                if print_summary:
                    print(f"  ✅ {obj_name}: {act_pos} (within {tolerance}m tolerance)")

        valid = len(errors) == 0
        if print_summary:
            if valid:
                print(f"\n✅ All {len(expected)} positions valid (tolerance: {tolerance}m)")
            else:
                print(f"\n❌ {len(errors)} position(s) invalid:")
                for error in errors:
                    print(f"   - {error}")

        return {
            "valid": valid,
            "errors": errors,
            "actual": actual,
            "expected_resolved": expected_resolved
        }

    def validate_semantics(
        self,
        expected_on_table: List[str] = None,
        expected_on_floor: List[str] = None,
        expected_stacked: Union[Dict[str, str], List[Dict[str, str]]] = None,
        expected_towers: List[List[str]] = None,
        print_summary: bool = False
    ) -> Dict[str, Any]:
        """Validate semantic relationships (stacking, contact) - PURE MOP!

        Physics has gravity - objects are either in contact OR falling.
        No "floating" concept - if no contact, object is falling.

        Args:
            expected_on_table: Objects that should be on table (None = don't check)
            expected_on_floor: Objects that should be on floor (None = don't check)
            expected_stacked: Stacking relationships - EXPLICIT & COMPOSABLE!
                             List format (NEW - more explicit):
                                [{'stacked_object': 'block_red', 'stacked_on': 'table'},
                                 {'stacked_object': 'block_blue', 'stacked_on': 'block_red'}]
                             Dict format (old - still supported):
                                {'block_red': 'table', 'block_blue': 'block_red'}
            expected_towers: List of tower structures (bottom to top)
            print_summary: Print formatted summary (default: True)

        Returns:
            Dict with:
            {
                "on_table": List[str],
                "on_floor": List[str],
                "falling": List[str],  # No contact yet (still settling)
                "stacking": Dict[str, bool],  # Stacking validation results
                "towers": Dict[int, bool],  # Tower validation results
                "valid": bool,
                "errors": List[str]
            }

        Examples:
            # NEW: Explicit list format (COMPOSABLE!)
            result = ops.validate_semantics(
                expected_stacked=[
                    {'stacked_object': 'block_red', 'stacked_on': 'table'},
                    {'stacked_object': 'block_blue', 'stacked_on': 'block_red'},
                    {'stacked_object': 'block_green', 'stacked_on': 'block_blue'}
                ]
            )

            # OLD: Dict format (still works)
            result = ops.validate_semantics(
                expected_stacked={
                    'block_red': 'table',
                    'block_blue': 'block_red',
                    'block_green': 'block_blue'
                }
            )

            # Tower validation
            result = ops.validate_semantics(
                expected_towers=[
                    ['block_red', 'block_blue', 'block_green', 'block_yellow']
                ]
            )
        """
        # Convert list format to dict format for internal processing
        if isinstance(expected_stacked, list):
            expected_stacked = {
                item['stacked_object']: item['stacked_on']
                for item in expected_stacked
            }

        state = self.get_state()

        # Determine which objects to check
        objects = set()
        if expected_on_table:
            objects.update(expected_on_table)
        if expected_on_floor:
            objects.update(expected_on_floor)
        if not objects:
            # Check all objects in state
            objects = set(state.keys())

        on_table = []
        on_floor = []
        falling = []
        errors = []

        if print_summary:
            print("\n" + "="*70)
            print("SEMANTIC VALIDATION (CONTACT STATUS)")
            print("="*70)
            print("\n📦 Object Contact Status:")

        for obj in objects:
            if obj not in state:
                errors.append(f"{obj}: NOT IN STATE")
                if print_summary:
                    print(f"   ❌ {obj}: NOT IN STATE!")
                continue

            obj_state = state[obj]
            is_on_table = obj_state.get("stacked_on_table", False)
            is_on_floor = obj_state.get("stacked_on_floor", False)

            if is_on_table:
                on_table.append(obj)
                if print_summary:
                    print(f"   ✅ {obj}: ON TABLE")
            elif is_on_floor:
                on_floor.append(obj)
                if print_summary:
                    print(f"   ✅ {obj}: ON FLOOR")
            else:
                falling.append(obj)
                if print_summary:
                    print(f"   ⚠️  {obj}: FALLING (no contact - physics settling)")

        # Validate expectations
        valid = True
        if expected_on_table:
            for obj in expected_on_table:
                if obj not in on_table:
                    valid = False
                    errors.append(f"{obj}: Expected on table but found {('on floor' if obj in on_floor else 'falling')}")

        if expected_on_floor:
            for obj in expected_on_floor:
                if obj not in on_floor:
                    valid = False
                    errors.append(f"{obj}: Expected on floor but found {('on table' if obj in on_table else 'falling')}")

        # MOP: Dynamic stacking validation - ANY object on ANY target!
        stacking_results = {}
        if expected_stacked:
            if print_summary:
                print(f"\n🔗 Stacking Relationships:")

            for obj, target in expected_stacked.items():
                if obj not in state:
                    errors.append(f"{obj}: NOT IN STATE")
                    stacking_results[f"{obj}_on_{target}"] = False
                    valid = False
                    if print_summary:
                        print(f"   ❌ {obj} on {target}: OBJECT NOT IN STATE!")
                    continue

                # Check dynamic property: stacked_on_{target}
                property_name = f"stacked_on_{target}"
                is_stacked = state[obj].get(property_name, False)
                stacking_results[f"{obj}_on_{target}"] = is_stacked

                if is_stacked:
                    if print_summary:
                        print(f"   ✅ {obj} on {target}")
                else:
                    valid = False
                    errors.append(f"{obj}: Expected on {target} but property '{property_name}' is False")
                    if print_summary:
                        print(f"   ❌ {obj} on {target}: NOT STACKED!")

        # Tower validation - validate vertical stacking chains
        tower_results = {}
        if expected_towers:
            if print_summary:
                print(f"\n🏗️  Tower Validation:")

            for tower_idx, tower_blocks in enumerate(expected_towers):
                tower_valid = True
                tower_name = f"tower_{tower_idx}"

                if print_summary:
                    print(f"\n   Tower {tower_idx + 1}: {' → '.join(tower_blocks)}")

                # Validate bottom block
                if len(tower_blocks) > 0:
                    bottom_block = tower_blocks[0]
                    # Assume bottom is on table by default
                    bottom_on_table = state.get(bottom_block, {}).get('stacked_on_table', False)

                    if bottom_on_table:
                        if print_summary:
                            print(f"      ✅ {bottom_block} on table")
                    else:
                        tower_valid = False
                        valid = False
                        errors.append(f"Tower {tower_idx}: {bottom_block} not on table")
                        if print_summary:
                            print(f"      ❌ {bottom_block} NOT on table!")

                # Validate each level stacked on the one below
                for i in range(1, len(tower_blocks)):
                    current_block = tower_blocks[i]
                    below_block = tower_blocks[i-1]

                    property_name = f"stacked_on_{below_block}"
                    is_stacked = state.get(current_block, {}).get(property_name, False)

                    if is_stacked:
                        if print_summary:
                            print(f"      ✅ {current_block} on {below_block}")
                    else:
                        tower_valid = False
                        valid = False
                        errors.append(f"Tower {tower_idx}: {current_block} not on {below_block}")
                        if print_summary:
                            print(f"      ❌ {current_block} NOT on {below_block}!")

                # Check no blocks fell to floor
                for block in tower_blocks:
                    if state.get(block, {}).get('stacked_on_floor', False):
                        tower_valid = False
                        valid = False
                        errors.append(f"Tower {tower_idx}: {block} fell to floor!")
                        if print_summary:
                            print(f"      ❌ {block} FELL TO FLOOR!")

                tower_results[tower_name] = tower_valid

        if print_summary:
            print(f"\n📊 Object Distribution:")
            print(f"   On Table: {len(on_table)} objects {on_table}")
            print(f"   On Floor: {len(on_floor)} objects {on_floor}")
            if falling:
                print(f"   Falling:  {len(falling)} objects {falling}")

            print(f"\n🔍 Validation:")
            if valid and not falling:
                total = len(on_table) + len(on_floor)
                print(f"   ✅ PASSED: All {total} objects in expected locations")
                print(f"   ✅ {len(on_table)} on table, {len(on_floor)} on floor")
            elif falling:
                print(f"   ⚠️  WARNING: {len(falling)} object(s) still falling (need more settling time)")
            else:
                print(f"   ❌ FAILED: {len(errors)} validation error(s)")
                for error in errors:
                    print(f"      - {error}")

        return {
            "on_table": on_table,
            "on_floor": on_floor,
            "falling": falling,
            "stacking": stacking_results,  # MOP: Dynamic stacking validation results
            "towers": tower_results,  # MOP: Tower validation results
            "valid": valid and not falling,
            "errors": errors
        }

    def save_all_screenshots(
        self,
        cameras: List[str] = None,
        frame: int = None,
        subdir: str = "screenshots",
        print_summary: bool = True
    ) -> Dict[str, str]:
        """Save screenshots from all/specified cameras - MOP!

        Args:
            cameras: Camera IDs (None = all cameras)
            frame: Frame number (None = last frame from timeline)
            subdir: Subdirectory name in experiment_dir (default: "screenshots")
            print_summary: Print formatted summary (default: True)

        Returns:
            Dict[camera_id -> screenshot_path]

        Example:
            paths = ops.save_all_screenshots(frame=1999)
        """
        from pathlib import Path

        # Determine cameras to use
        if cameras is None:
            cameras = list(self.scene.cameras.keys())

        # Create output directory
        output_dir = Path(self.experiment_dir) / subdir
        output_dir.mkdir(exist_ok=True, parents=True)

        # Determine frame
        if frame is None and hasattr(self, 'engine') and hasattr(self.engine, 'timeline_saver'):
            # Use last frame from timeline
            frame = len(self.engine.timeline_saver.timeline) - 1

        if print_summary:
            print(f"\n📸 Saving {len(cameras)} camera screenshot(s) to {subdir}/:")

        paths = {}
        for cam_id in cameras:
            if cam_id not in self.scene.cameras:
                if print_summary:
                    print(f"   ❌ {cam_id}: Camera not found")
                continue

            camera = self.scene.cameras[cam_id]
            path = camera.screenshot(frame, str(output_dir))
            paths[cam_id] = path

            if print_summary:
                print(f"   ✓ {cam_id}: {Path(path).name}")

        if print_summary:
            print(f"\n📂 Screenshots saved to: {output_dir}")

        # Track last screenshot paths for get_last_screenshots()
        self._last_screenshot_paths = paths

        return paths

    def get_last_screenshots(self) -> Dict[str, str]:
        """Get paths to last saved screenshots - MOP!

        Returns paths from the most recent save_all_screenshots() call.
        Useful for preview mode iterations.

        Returns:
            Dict[camera_id -> screenshot_path]

        Example:
            ops.save_all_screenshots(frame=0, subdir="preview")
            paths = ops.get_last_screenshots()
            # {"overhead_cam": "/path/to/overhead_cam_frame_0000.jpg", ...}
        """
        return self._last_screenshot_paths.copy()

    def validate_video_files(
        self,
        cameras: List[str] = None,
        print_summary: bool = True
    ) -> Dict[str, Any]:
        """Validate all video files exist and are valid - MOP!

        Automatically checks MP4 files, thumbnails, and uses VideoOps for validation.

        Args:
            cameras: List of camera names to validate (None = all cameras)
            print_summary: Print formatted validation summary (default: True)

        Returns:
            Dict with:
            {
                "valid": bool,
                "errors": List[str],
                "videos": Dict[str, Dict],  # cam_name -> {mp4_path, thumb_path, size_mb, valid}
                "total_count": int,
                "valid_count": int
            }

        Example:
            # Wait for conversion first
            ops.validate_videos(timeout=180)

            # Validate all videos
            result = ops.validate_video_files()

            # Check if all valid
            if not result['valid']:
                return False
        """
        from pathlib import Path

        # Import VideoOps for validation
        try:
            from core.video import VideoOps
        except ImportError:
            error = "VideoOps not available - cannot validate video files"
            if print_summary:
                print(f"\n❌ {error}")
            return {"valid": False, "errors": [error], "videos": {}, "total_count": 0, "valid_count": 0}

        # Auto-detect cameras if not specified
        if cameras is None:
            cameras = list(self.scene.cameras.keys())

        videos_dir = Path(self.experiment_dir) / "timeline" / "cameras"

        if print_summary:
            print(f"\n📹 Validating {len(cameras)} camera video(s):")

        errors = []
        videos = {}
        valid_count = 0

        for cam_name in cameras:
            mp4_path = videos_dir / cam_name / f"{cam_name}_rgb.mp4"
            thumb_path = videos_dir / cam_name / f"{cam_name}_rgb_thumbnail.jpg"

            video_info = {
                "mp4_path": str(mp4_path),
                "thumb_path": str(thumb_path),
                "size_mb": None,
                "valid": False,
                "error": None
            }

            # Check MP4 exists
            if not mp4_path.exists():
                error = f"{cam_name}: Video file not found"
                errors.append(error)
                video_info["error"] = "Video file not found"
                videos[cam_name] = video_info
                if print_summary:
                    print(f"   ❌ {cam_name}: Video not found!")
                continue

            # Validate MP4 with VideoOps
            validation_info = VideoOps.validate_video_file(mp4_path)
            if not validation_info['valid']:
                error = f"{cam_name}: MP4 invalid - {validation_info['error']}"
                errors.append(error)
                video_info["error"] = validation_info['error']
                videos[cam_name] = video_info
                if print_summary:
                    print(f"   ❌ {cam_name} MP4 invalid: {validation_info['error']}")
                continue

            # Check thumbnail exists
            if not thumb_path.exists():
                error = f"{cam_name}: Thumbnail missing"
                errors.append(error)
                video_info["error"] = "Thumbnail missing"
                videos[cam_name] = video_info
                if print_summary:
                    print(f"   ❌ {cam_name} thumbnail missing!")
                continue

            # All good!
            size_mb = mp4_path.stat().st_size / (1024 * 1024)
            video_info["size_mb"] = size_mb
            video_info["valid"] = True
            videos[cam_name] = video_info
            valid_count += 1

            if print_summary:
                print(f"   ✓ {cam_name}: {mp4_path.name} ({size_mb:.2f}MB) + thumbnail")

        # Summary
        all_valid = len(errors) == 0
        if print_summary:
            if all_valid:
                print(f"\n   ✅ All {len(cameras)} videos and thumbnails validated")
            else:
                print(f"\n   ❌ {len(errors)} validation error(s):")
                for error in errors:
                    print(f"      - {error}")

        return {
            "valid": all_valid,
            "errors": errors,
            "videos": videos,
            "total_count": len(cameras),
            "valid_count": valid_count
        }

    def close(self):
        """Close ExperimentOps and finalize all resources (videos, files, etc.)

        CRITICAL: Call this to ensure camera videos are properly written!
        Without close(), video files will be incomplete (44 bytes - header only).

        Usage:
            # Manual close
            ops = ExperimentOps(headless=True, fast_mode=False)
            ops.create_scene(...)
            ops.compile()
            for _ in range(1000):
                ops.step()
            ops.close()  # ← Finalizes videos!

            # Or use context manager (auto-closes)
            with ExperimentOps(headless=True, fast_mode=False) as ops:
                ops.create_scene(...)
                ops.compile()
                for _ in range(1000):
                    ops.step()
            # Videos automatically finalized when exiting 'with' block
        """
        if hasattr(self, '_closed') and self._closed:
            return  # Already closed

        # Close runtime engine (finalizes timeline saver, videos, etc.)
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                if hasattr(self.engine, '__del__'):
                    self.engine.__del__()
            except Exception as e:
                print(f"Warning: Error during engine cleanup: {e}")

        self._closed = True

    def __enter__(self):
        """Context manager entry - allows 'with ExperimentOps() as ops:'"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - automatically closes and finalizes videos"""
        self.close()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor - attempt to close if not already closed"""
        try:
            if not hasattr(self, '_closed') or not self._closed:
                self.close()
        except:
            pass  # Silent fail in destructor

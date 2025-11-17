"""
CAMERA MODAL - Virtual camera for simulation viewpoints
Modal-Oriented Programming: Self-managing camera that can track assets
OFFENSIVE & ELEGANT
"""

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Any, Dict
import numpy as np


class CameraModal(BaseModel):
    """Virtual free camera for custom viewpoints - SIMULATION ONLY!

    Modal-Oriented Programming:
    - Self-syncs from MuJoCo via sync_from_mujoco()
    - Self-reports state via get_data()
    - Can track assets/robots by connecting to scene
    - Interactive parameters (lookat, distance, azimuth, elevation)

    Connection to other modals:
    - SceneModal: via _scene_ref for asset tracking
    - AssetModal: tracks assets by querying scene.assets
    - RobotModal: tracks robots by querying scene.robots

    Example:
        # Static camera
        camera = CameraModal(camera_id='birds_eye', lookat=[5, 5, 0.5])

        # Tracking camera
        camera = CameraModal(
            camera_id='follow_cam',
            track_target='stretch',
            track_offset=(0, 0, 2.0)
        )
    """

    # Identity
    camera_id: str = 'birds_eye'
    camera_type: str = 'free'  # Duck typing: marks as camera for view system

    # MOP: Cameras are visualization-only, not trackable for rewards
    trackable_behaviors: List[str] = Field(default_factory=list)  # Empty = not trackable
    behaviors: List[str] = Field(default_factory=lambda: ['vision'])  # For view system compatibility
    geom_names: List[str] = Field(default_factory=list)  # No geoms (virtual)
    joint_names: List[str] = Field(default_factory=list)  # No joints (virtual)
    site_names: List[str] = Field(default_factory=list)  # No sites (virtual)

    # Interactive parameters (mutable - user can change during simulation!)
    # DEFAULTS: Match MuJoCo's actual MjvCamera defaults!
    lookat: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    distance: float = 2.0
    azimuth: float = 90.0  # Horizontal angle (degrees)
    elevation: float = -45.0  # Vertical angle (degrees)

    # Resolution
    width: int = 640
    height: int = 480

    # Render output
    rgb_image: Any = None  # np.ndarray (H, W, 3) uint8

    # Tracking (NEW!)
    track_target: Optional[str] = None  # Asset/robot name to track
    track_offset: Tuple[float, float, float] = (0, 0, 0)  # Offset from target

    # Backend connection (private) - NO DEFAULTS!
    _backend: Any = None  # CameraBackend instance (MuJoCo or real hardware)
    _scene_ref: Any = None  # Reference to SceneModal for asset tracking
    _model: Any = None  # Model reference (for tracking logic only)
    _data: Any = None  # Data reference (for tracking logic only)

    class Config:
        arbitrary_types_allowed = True

    def connect(self, model, data, scene=None, camera_backend=None):
        """Connect to simulation/hardware and scene

        Args:
            model: Model reference (for tracking logic)
            data: Data reference (for tracking logic)
            scene: SceneModal instance (for asset tracking)
            camera_backend: CameraBackend instance (REQUIRED! No defaults!)
        """
        self._model = model
        self._data = data
        self._scene_ref = scene

        # OFFENSIVE: Backend is REQUIRED, no defaults!
        if camera_backend is None:
            raise RuntimeError(
                f"❌ CameraModal.connect() requires camera_backend parameter!\n"
                f"   Backend must be explicitly provided - NO DEFAULTS!\n"
                f"   FIX: Pass camera_backend=MuJoCoRenderingBackend.create_free_camera(...)\n"
                f"   REASON: Modal-Oriented Programming requires explicit backend selection."
            )
        self._backend = camera_backend

        # MOP: Ask Room to validate my position! (Room owns "inside/outside" concept)
        # EXCEPTION: Skip validation for viewer-style cameras (overhead, viewer, etc.)
        # that intentionally position outside room for full scene visibility
        skip_validation = any(keyword in self.camera_id.lower() for keyword in ['overhead', 'viewer', 'birds_eye'])
        if scene and hasattr(scene, 'room') and scene.room and not skip_validation:
            camera_x, camera_y, camera_z = self._calculate_camera_position()
            scene.room.validate_point_inside(
                camera_x, camera_y, camera_z,
                entity_name=self.camera_id,
                entity_type="camera",
                extra_context={
                    "distance": self.distance,
                    "elevation": self.elevation,
                    "azimuth": self.azimuth,
                    "lookat": self.lookat
                }
            )

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Camera rendering is EXPENSIVE.

        Args:
            update_cameras: Whether cameras should render this step
            update_sensors: Whether sensors should update this step

        Returns:
            True if camera should render, False to skip
        """
        return update_cameras  # Only sync when camera update requested

    def _calculate_camera_position(self) -> tuple:
        """MOP: Pure calculation - Camera knows how to compute its own position

        Calculates camera position from spherical coordinates relative to lookat.
        Does NOT validate - that's Room's job!

        Returns:
            (camera_x, camera_y, camera_z): Absolute camera position in world coordinates
        """
        import math

        # Calculate camera position from spherical coordinates
        # elevation: -90 (above) to +90 (below)
        # azimuth: 0 (forward) to 360
        elevation_rad = math.radians(self.elevation)
        azimuth_rad = math.radians(self.azimuth)

        # Camera position relative to lookat
        # Z offset: negative elevation means camera is ABOVE lookat
        z_offset = self.distance * math.sin(-elevation_rad)
        horizontal_distance = self.distance * math.cos(elevation_rad)
        x_offset = horizontal_distance * math.cos(azimuth_rad)
        y_offset = horizontal_distance * math.sin(azimuth_rad)

        # Absolute camera position
        camera_x = self.lookat[0] + x_offset
        camera_y = self.lookat[1] + y_offset
        camera_z = self.lookat[2] + z_offset

        return (camera_x, camera_y, camera_z)

    def sync_from_mujoco(self, model, data, robot=None, **kwargs):
        """MOP: Self-sync - render camera with tracking support

        Modal-Oriented: Camera knows how to render itself using backend.

        Process:
        1. If tracking target, auto-update lookat to follow target
        2. Update backend with current camera parameters
        3. Read RGB image from backend
        4. Store rgb_image for view system

        Args:
            model: Model reference (for tracking logic)
            data: Data reference (for tracking logic)
            robot: Robot modal (not used for cameras, but required by sensor interface)
            **kwargs: Additional parameters
        """
        # OFFENSIVE: Backend must be set up before sync!
        if self._backend is None:
            raise RuntimeError(
                f"❌ Camera '{self.camera_id}' has no backend!\n"
                f"   Backend must be set via connect() before syncing.\n"
                f"   FIX: Call camera.connect(model, data, scene, camera_backend=...)\n"
                f"   REASON: Modal needs explicit backend - NO DEFAULTS!"
            )

        # Update connection references (for tracking logic)
        self._model = model
        self._data = data

        # Auto-update lookat if tracking enabled
        if self.track_target and self._scene_ref:
            target_pos = self._get_target_position(self.track_target)
            if target_pos:
                # Apply offset to target position
                self.lookat = [
                    target_pos[0] + self.track_offset[0],
                    target_pos[1] + self.track_offset[1],
                    target_pos[2] + self.track_offset[2]
                ]

                # MOP: Ask Room to validate my NEW position after tracking update!
                # Tracking can move camera dynamically, potentially outside room
                skip_validation = any(keyword in self.camera_id.lower()
                                    for keyword in ['overhead', 'viewer', 'birds_eye'])
                if self._scene_ref and hasattr(self._scene_ref, 'room') and self._scene_ref.room and not skip_validation:
                    camera_x, camera_y, camera_z = self._calculate_camera_position()
                    self._scene_ref.room.validate_point_inside(
                        camera_x, camera_y, camera_z,
                        entity_name=self.camera_id,
                        entity_type="camera",
                        extra_context={
                            "distance": self.distance,
                            "elevation": self.elevation,
                            "azimuth": self.azimuth,
                            "lookat": self.lookat,
                            "tracked_target": self.track_target,
                            "target_position": target_pos
                        }
                    )

        # Update backend with current camera parameters (for free cameras)
        # NOTE: Backend handles all rendering quality settings internally!
        self._backend.update_free_camera(
            lookat=tuple(self.lookat),
            distance=self.distance,
            azimuth=self.azimuth,
            elevation=self.elevation
        )

        # PERFORMANCE FIX: Use read_rgb_and_depth() to share update_scene() call!
        # Free cameras have enable_depth=False, so depth will be None (that's OK, we discard it)
        # This cuts update_scene() calls by 50% (same optimization as sensors_modals.py)
        self.rgb_image, _ = self._backend.read_rgb_and_depth()

    def _get_target_position(self, target_name: str) -> Optional[Tuple[float, float, float]]:
        """MOP: Get position from target modal (SINGLE SOURCE OF TRUTH!)

        Queries scene.assets or scene.robots and asks the MODAL for its position.
        Does NOT query MuJoCo directly - that would violate MOP!

        Modal-Oriented Programming:
        - AssetModal is authoritative source for asset positions
        - RobotModal is authoritative source for robot positions
        - Camera asks modals, never bypasses them

        Args:
            target_name: Name of asset or robot to track

        Returns:
            (x, y, z) position tuple, or None if not found
        """
        if not self._scene_ref:
            return None

        # MOP: Ask AssetModal for position (authoritative source!)
        if target_name in self._scene_ref.assets:
            asset = self._scene_ref.assets[target_name]

            # Get state from asset
            state = asset.get_data()

            # Check if body.position exists (may not during first compile before bootstrapping)
            if 'body.position' not in state:
                return None  # Skip tracking until asset is bootstrapped

            pos = state['body.position']  # Exact key - no guessing!

            # If position is None (asset hasn't synced yet), return None
            # sync_from_mujoco() will skip tracking update on first frame
            if pos is None:
                return None

            return tuple(float(x) for x in pos)

        # MOP: Ask RobotModal for position (authoritative source!)
        if target_name in self._scene_ref.robots:
            robot_info = self._scene_ref.robots[target_name]
            robot_modal = robot_info['robot_modal']

            # Get base position from robot's state
            try:
                state = robot_modal.get_data()
                if "base_position" in state:
                    pos = state["base_position"]
                    if hasattr(pos, '__iter__'):
                        return tuple(float(x) for x in pos)
            except Exception as e:
                # Fall back to direct query if robot doesn't expose position via state
                # (for backwards compatibility)
                import mujoco
                try:
                    joint_name = f"{robot_modal.name}_base_freejoint"
                    joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        qpos_addr = self._model.jnt_qposadr[joint_id]
                        return (
                            self._data.qpos[qpos_addr],
                            self._data.qpos[qpos_addr + 1],
                            self._data.qpos[qpos_addr + 2]
                        )
                except:
                    pass

        return None

    def track_asset(self, asset_name: str, offset: Tuple[float, float, float] = (0, 0, 2.0)):
        """Enable asset tracking - camera will follow asset

        Args:
            asset_name: Name of asset in scene.assets
            offset: (x, y, z) offset from asset position

        Example:
            camera.track_asset('apple', offset=(0, 0, 0.5))
            # Camera will look at point 0.5m above apple
        """
        self.track_target = asset_name
        self.track_offset = offset

    def track_robot(self, robot_name: str, offset: Tuple[float, float, float] = (0, 0, 2.0)):
        """Enable robot tracking - camera will follow robot

        Args:
            robot_name: Name of robot in scene.robots
            offset: (x, y, z) offset from robot base position

        Example:
            camera.track_robot('stretch', offset=(0, 0, 1.5))
            # Camera will look at point 1.5m above robot base
        """
        self.track_target = robot_name
        self.track_offset = offset

    def stop_tracking(self):
        """Disable tracking - camera will stay at current lookat"""
        self.track_target = None

    def screenshot(self, frame_number: int, save_dir: str) -> str:
        """MOP: Camera knows how to save its own screenshots for debugging

        Takes current rgb_image and saves with frame number.
        Useful for frame-by-frame debugging of tracking cameras.

        OFFENSIVE: Crashes if rgb_image is None (must sync/render first!)

        Args:
            frame_number: Current simulation frame number
            save_dir: Directory to save screenshot (creates if needed)

        Returns:
            Path to saved screenshot file

        Example:
            for i in range(400):
                ops.step()
                if i % 50 == 0:
                    path = camera.screenshot(i, "debug_frames")
                    print(f"Frame {i} saved: {path}")
        """
        from pathlib import Path
        import cv2

        # OFFENSIVE: Must have rendered image!
        if self.rgb_image is None:
            raise RuntimeError(
                f"❌ Cannot screenshot camera '{self.camera_id}' - no image data!\n"
                f"   Camera must sync/render before taking screenshot.\n"
                f"   FIX: Call ops.step() before camera.screenshot()\n"
                f"   REASON: screenshot() saves self.rgb_image which is set by sync_from_mujoco()"
            )

        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate filename: {camera_id}_frame_{frame:04d}.jpg
        filename = f"{self.camera_id}_frame_{frame_number:04d}.jpg"
        full_path = save_path / filename

        # Save RGB image (convert from RGB to BGR for OpenCV)
        img_bgr = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(full_path), img_bgr)

        return str(full_path)

    def get_data(self) -> Dict:
        """MOP: Self-report state for view system

        Returns dict with camera parameters and image data.
        Used by AtomicView to create camera views.

        NOTE: Uses key 'rgb' (not 'camera_id_rgb') for timeline compatibility.
        TimelineSaver expects 'rgb' and 'depth' keys for camera videos.

        OFFENSIVE: Crashes if camera has no image data (rgb or depth required!)
        """
        data = {
            "camera_id": self.camera_id,
            "camera_type": self.camera_type,
            "lookat": self.lookat,
            "distance": self.distance,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "width": self.width,
            "height": self.height,
            "track_target": self.track_target,
            "track_offset": self.track_offset
        }

        # Add RGB image data with key 'rgb' (timeline saver expects this!)
        # NOTE: May be None if camera hasn't rendered yet (first call during compile)
        if self.rgb_image is not None:
            data['rgb'] = self.rgb_image
        # If no rgb yet, timeline saver will skip video creation for this frame

        return data

    def get_rl(self) -> np.ndarray:
        """MOP: RL representation (normalized camera parameters)

        Not typically used for cameras, but provides complete modal interface.
        Returns normalized parameters: [azimuth/360, elevation/180, distance/max]
        """
        return np.array([
            self.azimuth / 360.0,
            (self.elevation + 90.0) / 180.0,  # Normalize from [-90, 90] to [0, 1]
            min(self.distance / 20.0, 1.0)  # Assume max distance of 20m
        ])

    def render_visualization(self):
        """MOP: Provide visualization data for timeline saving

        TimelineSaver calls this to get camera images for video encoding.

        OFFENSIVE: Crashes if camera has no image data (rgb or depth required!)

        Returns:
            Dict with RGB image data (key='rgb' for timeline compatibility)
        """
        # OFFENSIVE: Camera MUST have image data for timeline!
        if self.rgb_image is None:
            raise RuntimeError(
                f"❌ Camera '{self.camera_id}' has no image data for timeline!\n"
                f"   Camera must render at least once before timeline saving.\n"
                f"   FIX: Ensure camera.should_sync() returns True and sync_from_mujoco() is called.\n"
                f"   REASON: Cameras need either 'rgb' or 'depth' data for video encoding."
            )

        viz = {'rgb': self.rgb_image}
        return viz

    def __repr__(self):
        tracking = f", tracking={self.track_target}" if self.track_target else ""
        return f"CameraModal('{self.camera_id}', lookat={self.lookat}, distance={self.distance:.1f}m{tracking})"

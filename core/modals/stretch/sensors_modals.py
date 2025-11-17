"""
SENSORS MODALS - Sensor definitions and configuration
"""
from typing import Dict, List, Tuple, Optional, Literal
from pydantic import BaseModel, Field, PrivateAttr
from collections import deque

# Pydantic models for sensor outputs (what data each sensor produces)
import numpy as np
from typing import Any, Union


# === CAMERA RESOLUTION PRESETS ===
CAMERA_RESOLUTIONS = {
    # For high-quality research, debugging, visualization
    'high': {
        'nav_camera': (800, 600),     # 480K pixels - best quality
        'd405_camera': (480, 270),    # 130K pixels
        'render_time_ms': 308,        # Slow but beautiful
        'use_case': 'Research, debugging, visualization, recording demos'
    },

    # For balanced quality and speed
    'medium': {
        'nav_camera': (640, 480),     # 307K pixels
        'd405_camera': (320, 240),    # 77K pixels
        'render_time_ms': 195,        # ~2x faster than high
        'use_case': 'Development, testing, moderate quality recording'
    },

    # For fast iteration
    'low': {
        'nav_camera': (320, 240),     # 77K pixels
        'd405_camera': (160, 120),    # 19K pixels
        'render_time_ms': 49,         # ~6x faster than high
        'use_case': 'Fast iteration, quick testing'
    },

    # For RL agent training (good environment, not too slow)
    # RL agents need GOOD visual environment to learn well!
    # Too low resolution = agent can't see important details = bad policy
    'rl_core': {
        'nav_camera': (480, 360),     # 173K pixels - good for vision-based RL
        'd405_camera': (240, 180),    # 43K pixels
        'render_time_ms': 110,        # ~3x faster than high, still good quality
        'use_case': 'RL training - good visual environment for agent learning'
    },

    # For multimodal LLM fine-tuning (VLM training)
    # LLMs need high-quality images to understand scenes!
    'llm': {
        'nav_camera': (640, 480),     # 307K pixels - VLMs expect decent resolution
        'd405_camera': (320, 240),    # 77K pixels
        'render_time_ms': 195,        # Same as medium
        'use_case': 'Fine-tuning multimodal LLMs (vision-language models)'
    }
}

# Default resolution preset
DEFAULT_CAMERA_PRESET = 'low'  # 6X faster than 'high' (49ms vs 308ms), super fast viewer
 
class NavCamera(BaseModel):
    """Navigation camera output data (D435i on head)"""
    sensor_id: Literal['nav'] = 'nav'
    rgb_image: Any  # np.ndarray shape (600, 800, 3)
    depth_image: Any  # np.ndarray shape (600, 800)
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["vision", "robot_head_spatial"]
    trackable_behaviors: List[str] = ["vision"]  # robot_head_spatial is just data for head actuator
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["nav_camera_site"]

    # Backend abstraction (NO MuJoCo specifics!)
    _camera_backend: Any = PrivateAttr(default=None)

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Camera rendering is EXPENSIVE."""
        return update_cameras  # Only sync when camera update requested

    def get_data(self, include_base64=False):
        """Get raw camera data - both RGB and depth

        Args:
            include_base64: If True, include base64-encoded images for WebSocket streaming

        Returns:
            Dict with rgb/depth arrays and optionally base64-encoded images
        """
        data = {
            "rgb": self.rgb_image,
            "depth": self.depth_image,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id
        }

        # Add base64-encoded images for WebSocket streaming
        if include_base64:
            import base64
            import cv2

            if self.rgb_image is not None:
                # Encode RGB as JPEG (good compression for photos)
                bgr = cv2.cvtColor(self.rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                data["rgb_base64"] = base64.b64encode(buffer).decode('utf-8')
                data["rgb_format"] = "jpeg"

            if self.depth_image is not None:
                # Encode depth as PNG (lossless, supports 16-bit)
                depth_normalized = (self.depth_image * 1000).astype(np.uint16)  # meters to mm
                _, buffer = cv2.imencode('.png', depth_normalized)
                data["depth_base64"] = base64.b64encode(buffer).decode('utf-8')
                data["depth_format"] = "png"
                data["depth_scale"] = 1000  # mm scale

        return data

    def get_rl(self):
        """Get normalized vector for RL - downsampled RGB+depth"""
        vectors = []
        if self.rgb_image is not None:
            # Downsample RGB
            rgb_down = self.rgb_image[::10, ::10, :]
            vectors.append(rgb_down.flatten() / 255.0)
        if self.depth_image is not None:
            # Downsample depth
            depth_down = self.depth_image[::10, ::10]
            vectors.append(depth_down.flatten() / 10.0)
        return np.concatenate(vectors) if vectors else np.zeros(60 * 80 * 4)

    def render_visualization(self):
        """Render camera visualization for timeline video - OFFENSIVE!

        NavCamera renders itself: return RGB image or CRASH!

        Returns:
            RGB image (H, W, 3) uint8

        Raises:
            RuntimeError if no image available (OFFENSIVE - reveals camera sync failure!)
        """
        if self.rgb_image is None:
            raise RuntimeError(
                "❌ NavCamera.render_visualization() called but rgb_image is None!\n"
                "   This means camera sync failed or camera not found in MuJoCo model.\n"
                "   FIX: Check if 'nav_camera_rgb' exists in your robot XML.\n"
                "   FIX: Check warnings from sync_from_mujoco() for camera failures."
            )

        return self.rgb_image.astype(np.uint8)

    def sync_from_mujoco(self, model, data, robot, camera_backend=None, **kwargs):
        """Render camera images via backend - backend-agnostic!

        Args:
            model: Model reference (for timestamps)
            data: Data reference (for timestamps)
            robot: Robot modal
            camera_backend: CameraBackend instance (REQUIRED! No defaults!)

        OFFENSIVE: Requires camera backend! No MuJoCo specifics!
        Backend abstracts MuJoCo rendering OR real camera hardware.
        """
        import warnings

        # OFFENSIVE: Require camera backend! NO DEFAULTS!
        if camera_backend is None:
            raise RuntimeError(
                "❌ NavCamera.sync_from_mujoco() called without camera_backend!\n"
                "\n"
                "   Backend must be explicitly provided - NO DEFAULTS!\n"
                "   FIX: Pass camera_backend=MuJoCoRenderingBackend.create_sensor_camera(...)\n"
                "   REASON: Modal-Oriented Programming requires explicit backend selection.\n"
                "\n"
                "   The camera backend is created in RuntimeEngine.load_experiment()."
            )

        # Cache backend on first call
        if self._camera_backend is None:
            self._camera_backend = camera_backend

        try:
            # OPTIMIZED: Read RGB and depth with SINGLE scene update!
            # This cuts camera rendering time by avoiding duplicate scene preparation
            self.rgb_image, self.depth_image = self._camera_backend.read_rgb_and_depth()

            # Update timestamp
            self.timestamp = data.time
            self.frame_id = int(data.time / model.opt.timestep)

        except Exception as e:
            # Catch ANY rendering errors (EGL not available, memory issues, etc.)
            warnings.warn(f"NavCamera rendering failed: {e}. Camera disabled.")
            self.rgb_image = None
            self.depth_image = None



class D405Camera(BaseModel):
    """D405 wrist camera output data"""
    sensor_id: Literal['d405'] = 'd405'
    rgb_image: Any  # np.ndarray shape (270, 480, 3)
    depth_image: Any  # np.ndarray shape (270, 480)
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["vision", "robot_gripper_spatial"]
    trackable_behaviors: List[str] = ["vision"]  # robot_gripper_spatial is just data for gripper actuator
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["d405_camera_site"]

    # Backend abstraction (NO MuJoCo specifics!)
    _camera_backend: Any = PrivateAttr(default=None)

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Camera rendering is EXPENSIVE."""
        return update_cameras  # Only sync when camera update requested

    def get_data(self, include_base64=False):
        """Get raw camera data - both RGB and depth

        Args:
            include_base64: If True, include base64-encoded images for WebSocket streaming

        Returns:
            Dict with rgb/depth arrays and optionally base64-encoded images
        """
        data = {
            "rgb": self.rgb_image,
            "depth": self.depth_image,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id
        }

        # Add base64-encoded images for WebSocket streaming
        if include_base64:
            import base64
            import cv2

            if self.rgb_image is not None:
                # Encode RGB as JPEG (good compression for photos)
                bgr = cv2.cvtColor(self.rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                data["rgb_base64"] = base64.b64encode(buffer).decode('utf-8')
                data["rgb_format"] = "jpeg"

            if self.depth_image is not None:
                # Encode depth as PNG (lossless, supports 16-bit)
                depth_normalized = (self.depth_image * 1000).astype(np.uint16)  # meters to mm
                _, buffer = cv2.imencode('.png', depth_normalized)
                data["depth_base64"] = base64.b64encode(buffer).decode('utf-8')
                data["depth_format"] = "png"
                data["depth_scale"] = 1000  # mm scale

        return data

    def get_rl(self):
        """Get normalized vector for RL - downsampled RGB+depth"""
        vectors = []
        if self.rgb_image is not None:
            # Downsample RGB
            rgb_down = self.rgb_image[::9, ::8, :]
            vectors.append(rgb_down.flatten() / 255.0)
        if self.depth_image is not None:
            # Downsample depth
            depth_down = self.depth_image[::9, ::8]
            vectors.append(depth_down.flatten() / 10.0)
        return np.concatenate(vectors) if vectors else np.zeros(30 * 60 * 4)

    def should_visualize(self) -> bool:
        """Don't visualize if camera is disabled (prevents spam!)

        Returns False if camera not found in model (rgb_image stays None).
        This prevents render_visualization() from being called.
        """
        return self.rgb_image is not None and self.depth_image is not None

    def render_visualization(self):
        """Render D405 camera visualization - OFFENSIVE!

        Shows RGB + Depth heatmap side-by-side OR CRASH!

        Returns:
            RGB image (H, W*2, 3) uint8 - RGB on left, depth heatmap on right

        Raises:
            RuntimeError if no image available (OFFENSIVE - reveals camera failure!)
        """
        import cv2

        if self.rgb_image is None:
            raise RuntimeError(
                "❌ D405Camera.render_visualization() called but rgb_image is None!\n"
                "   This means camera sync failed or 'd405_camera' not found in MuJoCo model.\n"
                "   FIX: Check if 'd405_camera' exists in your robot XML.\n"
                "   FIX: Check warnings from sync_from_mujoco() for camera failures."
            )

        # RGB on left side
        rgb = self.rgb_image.astype(np.uint8)

        # Depth heatmap on right side - OFFENSIVE: CRASH if missing!
        if self.depth_image is None:
            raise RuntimeError(
                "❌ D405Camera has RGB but depth_image is None!\n"
                "   D405 is a DEPTH camera - it MUST have depth data!\n"
                "   FIX: Check depth rendering in sync_from_mujoco()."
            )

        # Normalize depth to 0-255
        depth_norm = (self.depth_image / 10.0 * 255).clip(0, 255).astype(np.uint8)
        # Apply colormap (hot colors = close, cool colors = far)
        depth_viz = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        # Convert BGR to RGB
        depth_viz = cv2.cvtColor(depth_viz, cv2.COLOR_BGR2RGB)
        # Side-by-side
        return np.hstack([rgb, depth_viz])

    def sync_from_mujoco(self, model, data, robot, camera_backend=None, **kwargs):
        """Render wrist camera images via backend - backend-agnostic!

        Args:
            model: Model reference (for timestamps)
            data: Data reference (for timestamps)
            robot: Robot modal
            camera_backend: CameraBackend instance (REQUIRED! No defaults!)

        OFFENSIVE: Requires camera backend! No MuJoCo specifics!
        Backend abstracts MuJoCo rendering OR real camera hardware.
        """
        import warnings

        # OFFENSIVE: Require camera backend! NO DEFAULTS!
        if camera_backend is None:
            raise RuntimeError(
                "❌ D405Camera.sync_from_mujoco() called without camera_backend!\n"
                "\n"
                "   Backend must be explicitly provided - NO DEFAULTS!\n"
                "   FIX: Pass camera_backend=MuJoCoRenderingBackend.create_sensor_camera(...)\n"
                "   REASON: Modal-Oriented Programming requires explicit backend selection.\n"
                "\n"
                "   The camera backend is created in RuntimeEngine.load_experiment()."
            )

        # Cache backend on first call
        if self._camera_backend is None:
            self._camera_backend = camera_backend

        try:
            # OPTIMIZED: Read RGB and depth with SINGLE scene update!
            # This cuts camera rendering time by avoiding duplicate scene preparation
            self.rgb_image, self.depth_image = self._camera_backend.read_rgb_and_depth()

            # Update timestamp
            self.timestamp = data.time
            self.frame_id = int(data.time / model.opt.timestep)

        except Exception as e:
            # Catch ANY rendering errors (EGL not available, memory issues, etc.)
            warnings.warn(f"D405Camera rendering failed: {e}. Camera disabled.")
            self.rgb_image = None
            self.depth_image = None


class FreeCameraSensor(BaseModel):
    """Virtual free camera for bird's eye/custom views - SIM ONLY!

    VIRTUAL: Not a real sensor - only exists in simulation!
    INTERACTIVE: User can change camera angle during simulation!

    This camera renders from any position/angle you specify.
    Perfect for bird's eye views, side views, or any custom perspective.
    """
    sensor_id: str = 'birds_eye'
    camera_type: str = 'free'  # Duck typing: marks this as a camera

    # Camera OUTPUT
    rgb_image: Any = None  # np.ndarray (H, W, 3) uint8

    # Camera PARAMETERS (mutable - user can change during simulation!)
    lookat: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.5])  # Look at this point [x, y, z]
    distance: float = 5.0                   # Distance from lookat point (meters)
    azimuth: float = 90.0                   # Horizontal angle: 0=forward, 90=right, 180=back, 270=left
    elevation: float = -30.0                # Vertical angle: -90=top-down, 0=horizontal, 90=bottom-up

    # Resolution
    width: int = 640
    height: int = 480

    # Metadata for MOP system
    behaviors: List[str] = Field(default_factory=lambda: ["vision"])
    trackable_behaviors: List[str] = Field(default_factory=list)  # Not trackable (virtual sensor!)
    geom_names: List[str] = Field(default_factory=list)
    joint_names: List[str] = Field(default_factory=list)
    site_names: List[str] = Field(default_factory=list)

    # Backend abstraction (NO MuJoCo specifics!)
    _camera_backend: Any = PrivateAttr(default=None)

    def sync_from_mujoco(self, model, data, robot, camera_backend=None, **kwargs):
        """Render from free camera position via backend - backend-agnostic!

        Virtual camera that renders from arbitrary viewpoints.
        User can change lookat/distance/azimuth/elevation between steps!

        Args:
            model: Model reference
            data: Data reference
            robot: Robot modal (not used, but required by interface)
            camera_backend: CameraBackend instance (REQUIRED! No defaults!)
        """
        import warnings

        # OFFENSIVE: Require camera backend! NO DEFAULTS!
        if camera_backend is None:
            raise RuntimeError(
                "❌ FreeCameraSensor.sync_from_mujoco() called without camera_backend!\n"
                "\n"
                "   Backend must be explicitly provided - NO DEFAULTS!\n"
                "   FIX: Pass camera_backend=MuJoCoRenderingBackend.create_free_camera(...)\n"
                "   REASON: Modal-Oriented Programming requires explicit backend selection."
            )

        # Cache backend on first call
        if self._camera_backend is None:
            self._camera_backend = camera_backend

        try:
            # Update backend with current camera parameters (allows interactive camera control!)
            self._camera_backend.update_free_camera(
                lookat=tuple(self.lookat),
                distance=self.distance,
                azimuth=self.azimuth,
                elevation=self.elevation
            )

            # Read RGB from backend (backend handles all rendering!)
            self.rgb_image = self._camera_backend.read_rgb()

        except Exception as e:
            # Catch rendering errors (EGL not available, etc.)
            warnings.warn(f"FreeCameraSensor rendering failed: {e}. Camera disabled.")
            self.rgb_image = None

    def get_data(self) -> Dict:
        """Return camera data - MODAL-ORIENTED

        Returns both the RGB image and current camera parameters,
        so you can see what angle the image was rendered from.
        """
        return {
            f'{self.sensor_id}_rgb': self.rgb_image,
            'lookat': self.lookat,
            'distance': self.distance,
            'azimuth': self.azimuth,
            'elevation': self.elevation,
            'width': self.width,
            'height': self.height
        }

    def get_rl(self):
        """Get normalized vector for RL - downsampled RGB"""
        if self.rgb_image is not None:
            # Downsample heavily for RL (640x480 → ~40x30)
            rgb_down = self.rgb_image[::16, ::16, :]
            return rgb_down.flatten() / 255.0
        return np.zeros(40 * 30 * 3)

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """Camera rendering is EXPENSIVE - only sync when cameras requested"""
        return update_cameras



class Lidar2D(BaseModel):
    """2D Lidar scanner output data"""
    sensor_id: Literal['lidar'] = 'lidar'
    ranges: List[float] = Field(..., description="Distance measurements in meters")
    angles: List[float] = Field(..., description="Angle for each measurement in radians")
    intensities: Optional[List[float]] = Field(None, description="Return intensity values")
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["distance_sensing"]
    trackable_behaviors: List[str] = ["distance_sensing"]  # All behaviors trackable
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["lidar_site"]

    # PERFORMANCE: Cache sensor ID array (360 lookups → cached once!)
    _sensor_ids: Optional[List[int]] = PrivateAttr(default=None)

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Medium cost (360 raycasts)."""
        return update_sensors  # Sync at 100Hz (good balance)

    def sync_from_mujoco(self, model, data, robot):
        """Sync lidar from MuJoCo rangefinder sensors - OFFENSIVE"""
        import mujoco

        # Read from 360 rangefinder sensors (base_lidar000 through base_lidar359)
        # Each sensor is 1° apart
        num_rays = 360

        # PERFORMANCE FIX: Cache all 360 sensor IDs on first call (eliminates 36,000 lookups/sec @ 100Hz!)
        if self._sensor_ids is None:
            self._sensor_ids = []
            for i in range(num_rays):
                sensor_name = f'base_lidar{i:03d}'
                sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                if sensor_id < 0:
                    raise ValueError(f"Lidar2D: '{sensor_name}' sensor not found in model")
                self._sensor_ids.append(sensor_id)

        ranges = []
        angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)

        for sensor_id in self._sensor_ids:
            # Read range from sensor (ID already cached)
            sensor_adr = model.sensor_adr[sensor_id]
            range_val = data.sensordata[sensor_adr]
            ranges.append(float(range_val))

        self.ranges = ranges
        self.angles = angles.tolist()
        self.intensities = None  # Not simulated
        self.timestamp = data.time
        self.frame_id = int(data.time / model.opt.timestep)

    def get_data(self):
        """Get raw lidar data"""
        return {"ranges": self.ranges, "angles": self.angles, "intensities": self.intensities}

    def get_rl(self):
        """Get normalized vector for RL - subsample ranges"""
        if self.ranges:
            # Take every 3rd point to reduce from 1080 to 360
            subsampled = np.array(self.ranges[::3])
            return subsampled / 10.0  # Normalize to [0, 1] assuming max range 10m
        return np.zeros(360)

    def render_visualization(self):
        """Render lidar visualization - OFFENSIVE!

        Top-down point cloud plot OR CRASH!

        Returns:
            RGB image (H, W, 3) uint8
        """
        import matplotlib.pyplot as plt
        from ..visualization_protocol import fig_to_numpy

        if not self.ranges or len(self.ranges) == 0:
            raise RuntimeError(
                "❌ Lidar2D.render_visualization() called but no range data!\n"
                "   FIX: Call sync_from_mujoco() first."
            )

        # Convert polar to cartesian
        angles = np.array(self.angles)
        ranges = np.array(self.ranges)
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x, y, c=ranges, cmap='viridis', s=2, alpha=0.8)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.set_title('2D Lidar Scan (top-down)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(scatter, label='Distance (m)', ax=ax)
        ax.grid(True, alpha=0.3)

        # Mark robot position
        ax.plot(0, 0, 'ro', markersize=8, label='Robot')
        ax.legend()

        plt.tight_layout()
        return fig_to_numpy(fig)


class IMU(BaseModel):
    """IMU sensor output data"""
    sensor_id: Literal['imu'] = 'imu'
    linear_acceleration: List[float] = Field(..., description="Linear acceleration [ax, ay, az] in m/s^2")
    angular_velocity: List[float] = Field(..., description="Angular velocity [gx, gy, gz] in rad/s")
    orientation: List[float] = Field(..., description="Orientation as quaternion [x, y, z, w]")
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["motion_sensing"]
    trackable_behaviors: List[str] = ["motion_sensing"]  # All behaviors trackable
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["imu_site"]

    # Rolling history for visualization (PERFORMANCE: deque for O(1) operations)
    _gyro_history: deque = PrivateAttr(default_factory=lambda: deque(maxlen=100))
    _accel_history: deque = PrivateAttr(default_factory=lambda: deque(maxlen=100))

    # PERFORMANCE: Cache MuJoCo ID lookups (called 200x/sec!)
    _imu_body_id: Optional[int] = PrivateAttr(default=None)
    _accel_sensor_id: Optional[int] = PrivateAttr(default=None)

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! I'm critical - sync EVERY step."""
        return True  # Always sync (cheap, critical for wheel actions)

    def sync_from_mujoco(self, model, data, robot):
        """Sync IMU from robot base - OFFENSIVE"""
        import mujoco

        # PERFORMANCE FIX: Cache IMU body ID on first call (eliminates 200 lookups/sec)
        if self._imu_body_id is None:
            self._imu_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_imu')
            if self._imu_body_id < 0:
                raise ValueError("IMU: 'base_imu' body not found in model")

        # Get orientation (quaternion [w, x, y, z])
        quat = data.xquat[self._imu_body_id]
        self.orientation = [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]  # [x,y,z,w]

        # Get angular velocity (rad/s) from body velocity
        # data.cvel[body_id] = [angular_vel (3), linear_vel (3)]
        cvel = data.cvel[self._imu_body_id]
        self.angular_velocity = [float(cvel[0]), float(cvel[1]), float(cvel[2])]

        # PERFORMANCE FIX: Cache accel sensor ID on first call (eliminates 200 lookups/sec)
        if self._accel_sensor_id is None:
            self._accel_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, 'base_accel')
            if self._accel_sensor_id < 0:
                raise ValueError("IMU: 'base_accel' sensor not found in model")

        # Read accelerometer data
        sensor_adr = model.sensor_adr[self._accel_sensor_id]
        self.linear_acceleration = [
            float(data.sensordata[sensor_adr]),
            float(data.sensordata[sensor_adr + 1]),
            float(data.sensordata[sensor_adr + 2])
        ]

        # Update timestamp
        self.timestamp = data.time
        self.frame_id = int(data.time / model.opt.timestep)

        # Update rolling history (deque with maxlen auto-truncates, no manual pop needed)
        self._gyro_history.append(self.angular_velocity)
        self._accel_history.append(self.linear_acceleration)

    def render_visualization(self):
        """Render IMU visualization - OFFENSIVE!

        Shows gyro and accel time-series plots OR CRASH!

        Returns:
            RGB image (H, W, 3) uint8 - plot of IMU data

        Raises:
            RuntimeError if no data available
        """
        import matplotlib.pyplot as plt
        from ..visualization_protocol import fig_to_numpy

        if len(self._gyro_history) == 0 or len(self._accel_history) == 0:
            raise RuntimeError(
                "❌ IMU.render_visualization() called but no history data!\n"
                "   This means IMU has never been synced.\n"
                "   FIX: Call sync_from_mujoco() at least once before rendering."
            )

        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

        # Plot angular velocity (gyro)
        gyro_arr = np.array(self._gyro_history)
        ax1.plot(gyro_arr[:, 0], label='gx', color='r', linewidth=1.5)
        ax1.plot(gyro_arr[:, 1], label='gy', color='g', linewidth=1.5)
        ax1.plot(gyro_arr[:, 2], label='gz', color='b', linewidth=1.5)
        ax1.set_title('Angular Velocity (rad/s)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('rad/s')

        # Plot linear acceleration
        accel_arr = np.array(self._accel_history)
        ax2.plot(accel_arr[:, 0], label='ax', color='r', linewidth=1.5)
        ax2.plot(accel_arr[:, 1], label='ay', color='g', linewidth=1.5)
        ax2.plot(accel_arr[:, 2], label='az', color='b', linewidth=1.5)
        ax2.set_title('Linear Acceleration (m/s²)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('m/s²')
        ax2.set_xlabel('Sample')

        plt.tight_layout()
        return fig_to_numpy(fig)

    def get_data(self):
        """Get raw IMU data"""
        return {
            "linear_acceleration": self.linear_acceleration,
            "angular_velocity": self.angular_velocity,
            "orientation": self.orientation
        }

    def get_rl(self):
        """Get normalized vector for RL"""
        return np.concatenate([
            np.array(self.linear_acceleration) / 16.0,  # Normalize accel
            np.array(self.angular_velocity) / 35.0,     # Normalize gyro
            np.array(self.orientation)                  # Quaternion already normalized
        ])


class GripperForceSensor(BaseModel):
    """Gripper force sensor output data"""
    sensor_id: Literal['gripper_force'] = 'gripper_force'
    force_left: float = Field(..., description="Left finger force in Newtons")
    force_right: float = Field(..., description="Right finger force in Newtons")
    contact_left: bool = Field(..., description="Left finger contact detected")
    contact_right: bool = Field(..., description="Right finger contact detected")
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["tactile", "robot_gripper"]
    trackable_behaviors: List[str] = ["tactile"]  # robot_gripper is just data for gripper actuator
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = []

    # Event tracking (ALWAYS-ON!)
    _previous_data: Optional[Dict] = PrivateAttr(default=None)
    _force_history: deque = PrivateAttr(default_factory=lambda: deque(maxlen=50))  # Rolling history (PERFORMANCE: deque for O(1))

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Low cost but not critical."""
        return update_sensors  # Sync at 100Hz

    def get_data(self):
        """Get raw force sensor data"""
        return {
            "force_left": self.force_left,
            "force_right": self.force_right,
            "contact_left": self.contact_left,
            "contact_right": self.contact_right
        }

    def get_rl(self):
        """Get normalized vector for RL"""
        return np.array([
            self.force_left / 10.0,   # Normalize forces
            self.force_right / 10.0,
            float(self.contact_left),
            float(self.contact_right)
        ])

    def sync_from_mujoco(self, model, data, robot, event_log=None):
        """Sync gripper force sensor from MuJoCo contacts - STRETCH-SPECIFIC

        Args:
            event_log: EventLog to track changes (ALWAYS-ON!)
        """
        import mujoco

        # Get gripper actuator to find geom names
        gripper = robot.actuators.get('gripper')
        if not gripper or not hasattr(gripper, 'geom_names'):
            return

        # Calculate contact forces for left and right fingers
        force_left = 0.0
        force_right = 0.0

        # Check all contacts in simulation
        for i in range(data.ncon):
            contact = data.contact[i]

            # Get geom names
            geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)

            # Check if this is a gripper contact
            if geom1_name and geom2_name:
                # Left finger
                if 'left' in geom1_name.lower() or 'left' in geom2_name.lower():
                    c_array = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, c_array)
                    force_left += float(np.linalg.norm(c_array[:3]))

                # Right finger
                if 'right' in geom1_name.lower() or 'right' in geom2_name.lower():
                    c_array = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, c_array)
                    force_right += float(np.linalg.norm(c_array[:3]))

        # Store old values for event tracking
        old_data = self.get_data() if event_log and self._previous_data is not None else None

        # Update sensor values
        self.force_left = force_left
        self.force_right = force_right
        self.contact_left = force_left > 0.1
        self.contact_right = force_right > 0.1
        self.timestamp = data.time
        self.frame_id = int(data.time / model.opt.timestep)

        # Emit change events (if event_log provided and values changed)
        if event_log and old_data:
            new_data = self.get_data()
            for field in ['force_left', 'force_right', 'contact_left', 'contact_right']:
                old_val = old_data.get(field)
                new_val = new_data.get(field)
                # Only emit if value changed significantly (avoid noise)
                if field.startswith('force'):
                    # For force, only emit if change > 0.1N (avoid sensor noise)
                    if abs(new_val - old_val) > 0.1:
                        from ..event_log_modal import create_sensor_change_event
                        event = create_sensor_change_event(
                            timestamp=data.time,
                            sensor_id=self.sensor_id,
                            field=field,
                            old_value=old_val,
                            new_value=new_val,
                            step=self.frame_id
                        )
                        event_log.add_event(event)
                elif old_val != new_val:  # Contact is boolean, emit on any change
                    from ..event_log_modal import create_sensor_change_event
                    event = create_sensor_change_event(
                        timestamp=data.time,
                        sensor_id=self.sensor_id,
                        field=field,
                        old_value=old_val,
                        new_value=new_val,
                        step=self.frame_id
                    )
                    event_log.add_event(event)

        # Update previous state for next comparison
        self._previous_data = self.get_data()

        # Update rolling history (deque with maxlen auto-truncates)
        self._force_history.append([self.force_left, self.force_right])

    def render_visualization(self):
        """Render gripper force visualization - OFFENSIVE!

        Bar chart OR CRASH!
        """
        import matplotlib.pyplot as plt
        from ..visualization_protocol import fig_to_numpy

        if len(self._force_history) == 0:
            raise RuntimeError(
                "❌ GripperForceSensor.render_visualization() called but no history!\n"
                "   FIX: Call sync_from_mujoco() first."
            )

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Current forces (bar chart)
        ax1.bar(['Left', 'Right'], [self.force_left, self.force_right], color=['blue', 'green'])
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Current Gripper Force', fontweight='bold')
        ax1.set_ylim(0, max(10, max(self.force_left, self.force_right) * 1.2))
        ax1.grid(True, alpha=0.3, axis='y')

        # Force history (time series)
        history_arr = np.array(self._force_history)
        ax2.plot(history_arr[:, 0], label='Left', color='blue', linewidth=2)
        ax2.plot(history_arr[:, 1], label='Right', color='green', linewidth=2)
        ax2.set_ylabel('Force (N)')
        ax2.set_xlabel('Sample')
        ax2.set_title('Force History', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_numpy(fig)


class ReSpeakerArray(BaseModel):
    """ReSpeaker 4-mic array sensor output data"""
    sensor_id: Literal['respeaker'] = 'respeaker'
    audio_channels: List[Any] = Field(..., description="4-channel audio from MEMS microphones")
    sample_rate: int = Field(16000, description="Audio sample rate in Hz")
    direction_of_arrival: float = Field(..., description="Estimated sound source direction in radians relative to head")
    sound_intensity: float = Field(..., description="Sound intensity level in dB")
    voice_activity: bool = Field(..., description="Voice activity detected")
    detected_speech: Optional[str] = Field(None, description="Transcribed speech if available")
    led_pattern: Optional[List[int]] = Field(None, description="Current LED pattern (12 RGB LEDs)")
    timestamp: float
    frame_id: int

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["audio_sensing"]
    trackable_behaviors: List[str] = ["audio_sensing"]  # All behaviors trackable
    geom_names: List[str] = []
    joint_names: List[str] = []
    site_names: List[str] = ["respeaker_site"]

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! Audio is not simulated yet."""
        return update_sensors  # Sync at 100Hz when implemented

    def get_data(self):
        """Get raw audio data"""
        return {
            "audio_channels": self.audio_channels,
            "sample_rate": self.sample_rate,
            "direction_of_arrival": self.direction_of_arrival,
            "sound_intensity": self.sound_intensity,
            "voice_activity": self.voice_activity,
            "detected_speech": self.detected_speech
        }

    def get_rl(self):
        """Get normalized vector for RL - just key features, not raw audio"""
        return np.array([
            self.direction_of_arrival / 3.14159,  # Normalize direction
            self.sound_intensity / 100.0,         # Normalize dB
            float(self.voice_activity),           # Binary flag
            1.0 if self.detected_speech else 0.0  # Has speech flag
        ])


class Odometry(BaseModel):
    """Odometry sensor - tracks position from wheel encoders"""
    sensor_id: Literal['odometry'] = 'odometry'
    x: float = Field(0.0, description="X position in meters")
    y: float = Field(0.0, description="Y position in meters")
    theta: float = Field(0.0, description="Heading angle in radians")
    vx: float = Field(0.0, description="X velocity in m/s")
    vy: float = Field(0.0, description="Y velocity in m/s")
    vtheta: float = Field(0.0, description="Angular velocity in rad/s")
    timestamp: float = 0.0
    frame_id: int = 0

    # Modal-oriented: Self-declare trackable behaviors for rewards
    behaviors: List[str] = ["robot_base"]
    trackable_behaviors: List[str] = ["robot_base"]  # robot_base IS trackable - creates stretch.base asset!
    geom_names: List[str] = ["base_link"]  # References robot base body for position/rotation tracking
    joint_names: List[str] = []
    site_names: List[str] = []

    # Rolling history for visualization (PERFORMANCE: deque for O(1) append/pop vs O(n) list)
    _trajectory: deque = PrivateAttr(default_factory=lambda: deque(maxlen=200))

    # PERFORMANCE: Cache body ID lookup (called 200x/sec!)
    _robot_body_id: Optional[int] = PrivateAttr(default=None)

    # CUMULATIVE ROTATION TRACKING (MOP: Single source of truth for rotation!)
    _prev_theta: Optional[float] = PrivateAttr(default=None)  # Previous instant theta for unwrapping
    _cumulative_theta: float = PrivateAttr(default=0.0)  # Unbounded cumulative rotation

    def should_sync(self, update_cameras: bool = True, update_sensors: bool = True) -> bool:
        """MOP: I decide when to sync myself! I'm critical - sync EVERY step."""
        return True  # Always sync (cheap, critical for base movement)

    def sync_from_mujoco(self, model, data, robot):
        """Sync odometry from robot base position/velocity - OFFENSIVE"""
        import mujoco

        # PERFORMANCE FIX: Cache body ID on first call (eliminates 200 string lookups/sec)
        if self._robot_body_id is None:
            self._robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'base_link')
            if self._robot_body_id < 0:
                raise ValueError("Odometry: 'base_link' body not found in model")

        # MOP FIX: For freejoint robots, position comes from qpos[0:2], NOT xpos!
        # Freejoint robots (like Stretch) use qpos[0:3] = [x, y, z] as authoritative position.
        # xpos is derived from qpos during mj_forward(), but qpos is the source of truth.
        # Reading qpos directly ensures correct position for keyframe-spawned robots!
        self.x = float(data.qpos[0])
        self.y = float(data.qpos[1])

        # Get orientation (quaternion -> euler -> yaw)
        quat = data.xquat[self._robot_body_id]
        # Convert quaternion to euler angles
        # For yaw (z-axis rotation): theta = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        instant_theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))  # Wrapped to [-π, π]

        # MOP: Track cumulative rotation (handles >360° rotations!)
        # Unwrap angle discontinuities to get unbounded cumulative theta
        if self._prev_theta is not None:
            # Calculate angular change (shortest path)
            delta = instant_theta - self._prev_theta

            # Unwrap: if we jumped across ±π boundary, adjust
            if delta > np.pi:
                delta -= 2 * np.pi  # We wrapped backwards (e.g., 179° → -179°)
            elif delta < -np.pi:
                delta += 2 * np.pi  # We wrapped forwards (e.g., -179° → 179°)

            # Accumulate the change
            self._cumulative_theta += delta
        else:
            # First call: initialize cumulative theta to current instant theta
            self._cumulative_theta = instant_theta

        # Store instant theta for next unwrap calculation
        self._prev_theta = instant_theta

        # MOP: theta is now CUMULATIVE (unbounded), not wrapped!
        # Can track 360°, 720°, 1080°, etc.
        self.theta = self._cumulative_theta

        # Get velocities (linear and angular)
        # NOTE: data.cvel[body_id] = [angular_vel (3), linear_vel (3)]
        cvel = data.cvel[self._robot_body_id]
        self.vx = cvel[3]  # Linear velocity X
        self.vy = cvel[4]  # Linear velocity Y
        self.vtheta = cvel[2]  # Angular velocity Z (yaw rate)

        # Update timestamp
        self.timestamp = data.time
        self.frame_id = int(data.time / model.opt.timestep)

        # Store trajectory point (deque gives us O(1) operations with automatic maxlen truncation)
        self._trajectory.append([self.x, self.y])

    def render_visualization(self):
        """Render odometry visualization - OFFENSIVE!

        Robot trajectory plot OR CRASH!
        """
        import matplotlib.pyplot as plt
        from ..visualization_protocol import fig_to_numpy

        if len(self._trajectory) == 0:
            raise RuntimeError(
                "❌ Odometry.render_visualization() called but no trajectory data!\n"
                "   FIX: Call sync_from_mujoco() first."
            )

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot trajectory
        traj_arr = np.array(self._trajectory)
        ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, label='Trajectory')

        # Mark current position
        ax.plot(self.x, self.y, 'ro', markersize=12, label='Current Position')

        # Draw heading direction arrow
        arrow_len = 0.3
        dx = arrow_len * np.cos(self.theta)
        dy = arrow_len * np.sin(self.theta)
        ax.arrow(self.x, self.y, dx, dy, head_width=0.1, head_length=0.1, fc='red', ec='red')

        # Mark start position
        if len(traj_arr) > 0:
            ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=10, label='Start')

        ax.set_aspect('equal')
        ax.set_title('Robot Trajectory (Odometry)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig_to_numpy(fig)

    def get_data(self):
        """Get odometry data"""
        return {
            "x": self.x,
            "y": self.y,
            "theta": self.theta,
            "vx": self.vx,
            "vy": self.vy,
            "vtheta": self.vtheta
        }

    def get_rl(self):
        """Get normalized vector for RL"""
        return np.array([self.x, self.y, self.theta, self.vx, self.vy, self.vtheta])



# Union of all sensor outputs
SensorOutput = Union[
    NavCamera,
    D405Camera,
    Lidar2D,
    IMU,
    GripperForceSensor,
    ReSpeakerArray,
    Odometry
]



# Sensor registry - maps sensors to their properties and which actions control them
# This helps understand sensor-action relationships (e.g., head actions control nav camera)
Robot_Input = {
    'nav': {
        'type': 'camera',
        'data_keys': ['cam_nav_rgb'],
        'video_dims': (800, 600),
        'description': 'Navigation camera mounted on head for rooms overview',
        'sensor_class': [NavCamera],  # Sensor class as list
        'action_class': ['HeadPanMoveTo', 'HeadTiltMoveTo']  # Actions that control this sensor
    },
    'd405': {
        'type': 'camera',
        'data_keys': ['cam_d405_rgb', 'cam_d405_depth'],
        'video_dims': (480, 270),
        'description': 'Wrist-mounted depth camera for precise manipulation',
        'sensor_class': [D405Camera],
        'action_class': ['WristYawMoveTo', 'WristPitchMoveTo', 'WristRollMoveTo']
    },
    'lidar': {
        'type': 'lidar',
        'data_keys': ['lidar_ranges'],
        'video_dims': (480, 480),
        'description': '2D lidar scanner for obstacle detection and mapping',
        'sensor_class': [Lidar2D],
        'action_class': None  # No actions control lidar
    },
    'gripper_force': {
        'type': 'force',
        'data_keys': ['gripper_force', 'gripper_contact'],
        'video_dims': (320, 240),  # For visualization
        'description': 'Force and contact sensors in gripper fingers',
        'sensor_class': [GripperForceSensor],
        'action_class': ['GripperMoveTo']  # Gripper action affects force
    },
    'imu': {
        'type': 'imu',
        'data_keys': ['imu_accel', 'imu_gyro', 'imu_quat'],
        'video_dims': None,
        'description': 'Inertial measurement unit for robot body dynamics',
        'sensor_class': [IMU],
        'action_class': None  # No direct control
    },
    'respeaker': {
        'type': 'audio',
        'data_keys': ['audio_channels', 'voice_activity', 'direction'],
        'video_dims': None,
        'description': '4-mic array for audio capture and localization',
        'sensor_class': [ReSpeakerArray],
        'action_class': None  # ReSpeaker just listens, no action controls it
    },
    'odometry': {
        'type': 'odometry',
        'data_keys': ['x', 'y', 'theta', 'vx', 'vy', 'vtheta'],
        'video_dims': None,
        'description': 'Position tracking from wheel encoders',
        'sensor_class': [Odometry],
        'action_class': None  # Odometry is passive sensing
    }
}








# ============================================
# SENSOR REGISTRY FUNCTION
# ============================================

def get_all_sensors() -> Dict[str, Any]:
    """Get all stretch sensors - ready to use"""
    return {
        "nav_camera": NavCamera(rgb_image=None, depth_image=None, timestamp=0.0, frame_id=0),
        "d405_camera": D405Camera(rgb_image=None, depth_image=None, timestamp=0.0, frame_id=0),
        "lidar": Lidar2D(ranges=[], angles=[], timestamp=0.0, frame_id=0),
        "imu": IMU(linear_acceleration=[0,0,0], angular_velocity=[0,0,0], orientation=[0,0,0,1], timestamp=0.0, frame_id=0),
        "gripper_force": GripperForceSensor(force_left=0.0, force_right=0.0, contact_left=False, contact_right=False, timestamp=0.0, frame_id=0),
        "respeaker": ReSpeakerArray(audio_channels=[], sample_rate=16000, direction_of_arrival=0.0,
                                   sound_intensity=0.0, voice_activity=False, detected_speech=None,
                                   timestamp=0.0, frame_id=0),
        "odometry": Odometry(x=0.0, y=0.0, theta=0.0, vx=0.0, vy=0.0, vtheta=0.0, timestamp=0.0, frame_id=0)
    }


def get_sensor_presets() -> Dict[str, List[str]]:
    """Stretch-specific sensor presets for common tasks - AUTO-DISCOVERED

    Each robot type defines its own presets based on its sensors.
    Ops discovers this dynamically - no hardcoding in general code!
    """
    return {
        "manipulation": ["d405_camera", "gripper_force", "odometry"],
        "navigation": ["nav_camera", "lidar", "odometry", "imu"],
        "proprioceptive": ["gripper_force", "odometry", "imu"],
        "vision": ["nav_camera", "d405_camera", "lidar"],
        "minimal": ["odometry"],
        "all": list(get_all_sensors().keys()),  # All sensors
    }

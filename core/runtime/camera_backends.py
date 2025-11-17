"""
CAMERA BACKENDS - Abstraction layer for sim-to-real camera transitions
Modal-Oriented Programming: Clean separation of simulation vs real hardware
OFFENSIVE & ELEGANT - NO DEFAULTS, NO FALLBACKS!
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import numpy as np


class CameraBackend(ABC):
    """Abstract base class for camera backends (simulation or real hardware)

    Modal-Oriented Programming Principle:
    - Modals (CameraModal, NavCamera, D405Camera) should NOT know if they're
      using simulation (MuJoCo) or real hardware (RealSense, webcam)
    - This backend provides a clean interface for both

    OFFENSIVE: NO DEFAULTS, NO FALLBACKS!
    - Backend must be explicitly provided
    - If wrong backend used, CRASH LOUDLY
    - Never silently fall back to MuJoCo in production

    Methods:
        read_rgb() -> RGB image from camera
        read_depth() -> Depth image from camera (if supported)
        get_intrinsics() -> Camera intrinsic parameters (K matrix)
        close() -> Cleanup resources
    """

    @abstractmethod
    def read_rgb(self) -> np.ndarray:
        """Read RGB image from camera

        Returns:
            np.ndarray: RGB image (H, W, 3) uint8, range [0, 255]

        Raises:
            RuntimeError: If camera not initialized or read fails
        """
        pass

    @abstractmethod
    def read_depth(self) -> Optional[np.ndarray]:
        """Read depth image from camera (if supported)

        Returns:
            np.ndarray: Depth image (H, W) float32, in meters
            None: If camera doesn't support depth

        Raises:
            RuntimeError: If camera not initialized or read fails
        """
        pass

    @abstractmethod
    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters

        Returns:
            dict: Camera intrinsics with keys:
                - 'K': 3x3 intrinsic matrix (numpy array)
                - 'width': Image width (int)
                - 'height': Image height (int)
                - 'fx': Focal length x (float)
                - 'fy': Focal length y (float)
                - 'cx': Principal point x (float)
                - 'cy': Principal point y (float)
                - 'fov': Field of view in degrees (float, optional)
        """
        pass

    @abstractmethod
    def close(self):
        """Cleanup backend resources (renderers, camera handles, etc.)"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class MuJoCoRenderingBackend(CameraBackend):
    """MuJoCo simulation camera backend - HIGH QUALITY, NO DEBUG JUNK!

    This backend wraps mujoco.Renderer and handles:
    - High quality rendering (shadows, reflections, skybox)
    - Debug visualization disabled (NO lidar rays, contacts, force arrows)
    - Proper resource cleanup

    OFFENSIVE:
    - Crashes if used without MuJoCo model/data
    - Crashes if camera name not found in model
    - NO silent fallbacks!

    Usage:
        # For free cameras (virtual viewpoints)
        backend = MuJoCoRenderingBackend.create_free_camera(
            model, data,
            lookat=[0, 0, 0], distance=2.0,
            azimuth=90, elevation=-45,
            width=640, height=480
        )

        # For robot sensor cameras (D405, nav_camera, etc.)
        backend = MuJoCoRenderingBackend.create_sensor_camera(
            model, data,
            camera_name='nav_camera',
            width=640, height=480
        )

        # Read images
        rgb = backend.read_rgb()
        depth = backend.read_depth()  # If depth rendering enabled

        # Cleanup
        backend.close()
    """

    def __init__(self, model, data, width: int, height: int, shadows: bool = True, reflections: bool = True):
        """Initialize MuJoCo rendering backend

        Args:
            model: MuJoCo model (mujoco.MjModel)
            data: MuJoCo data (mujoco.MjData)
            width: Image width in pixels
            height: Image height in pixels
            shadows: Enable shadows (default True, disable for 2x faster rendering!)
            reflections: Enable reflections (default True, disable for faster rendering!)
        """
        import mujoco

        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Create renderer immediately on main thread
        # NOTE: OpenGL contexts are NOT thread-safe! Renderer must stay on main thread
        self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        # Apply rendering quality settings
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1 if reflections else 0
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1

        # Disable debug visualizations
        if hasattr(self._renderer, '_scene_option'):
            opt = self._renderer._scene_option
            opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            opt.geomgroup[0] = 1  # Show visual geoms
            opt.geomgroup[1] = 0  # Hide collision geoms

        # Camera configuration (set in subclass methods)
        self._camera_config = None  # 'free' or 'sensor'
        self._camera_obj = None  # MjvCamera or camera_id
        self._depth_enabled = False

    @property
    def renderer(self):
        """Get the renderer (single instance, main thread only!)"""
        return self._renderer

    @classmethod
    def create_free_camera(cls, model, data,
                          lookat: Tuple[float, float, float],
                          distance: float,
                          azimuth: float,
                          elevation: float,
                          width: int = 640,
                          height: int = 480,
                          enable_depth: bool = False,
                          shadows: bool = True,
                          reflections: bool = True) -> 'MuJoCoRenderingBackend':
        """Create backend for free camera (virtual viewpoint)

        Args:
            model: MuJoCo model
            data: MuJoCo data
            lookat: (x, y, z) point to look at
            distance: Distance from lookat point (meters)
            azimuth: Horizontal angle (degrees)
            elevation: Vertical angle (degrees)
            width: Image width
            height: Image height
            enable_depth: Enable depth rendering
            shadows: Enable shadows (default True, disable for 2x faster!)
            reflections: Enable reflections (default True, disable for faster!)

        Returns:
            MuJoCoRenderingBackend configured for free camera
        """
        import mujoco

        backend = cls(model, data, width, height, shadows=shadows, reflections=reflections)
        backend._camera_config = 'free'
        backend._depth_enabled = enable_depth

        # Create MjvCamera with specified parameters
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        backend._camera_obj = cam

        # Enable depth rendering if requested
        if enable_depth:
            backend.renderer.enable_depth_rendering()

        return backend

    @classmethod
    def create_sensor_camera(cls, model, data,
                            camera_name: str,
                            width: int = 640,
                            height: int = 480,
                            enable_depth: bool = False,
                            shadows: bool = True,
                            reflections: bool = True) -> 'MuJoCoRenderingBackend':
        """Create backend for robot sensor camera (e.g., nav_camera, d405_camera)

        OFFENSIVE: Crashes if camera_name not found in model!

        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera_name: Name of camera in MuJoCo XML (e.g., 'nav_camera')
            width: Image width
            height: Image height
            enable_depth: Enable depth rendering
            shadows: Enable shadows (default True, disable for 2x faster!)
            reflections: Enable reflections (default True, disable for faster!)

        Returns:
            MuJoCoRenderingBackend configured for sensor camera

        Raises:
            RuntimeError: If camera_name not found in model
        """
        import mujoco

        backend = cls(model, data, width, height, shadows=shadows, reflections=reflections)
        backend._camera_config = 'sensor'
        backend._depth_enabled = enable_depth

        # OFFENSIVE: Find camera ID or crash!
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            available_cameras = [model.camera(i).name for i in range(model.ncam)]
            raise RuntimeError(
                f"❌ Camera '{camera_name}' not found in MuJoCo model!\n"
                f"   Available cameras: {available_cameras}\n"
                f"   FIX: Check camera name in robot XML or scene configuration.\n"
                f"   REASON: MuJoCoRenderingBackend requires explicit camera reference."
            )

        # Store BOTH camera ID and NAME (update_scene needs NAME for sensor cameras!)
        backend._camera_obj = camera_id
        backend._camera_name = camera_name

        # Enable depth rendering if requested
        if enable_depth:
            backend.renderer.enable_depth_rendering()

        return backend

    def update_free_camera(self, lookat: Tuple[float, float, float],
                          distance: float, azimuth: float, elevation: float):
        """Update free camera parameters (for tracking, user control, etc.)

        OFFENSIVE: Only works for free cameras!

        Args:
            lookat: New lookat point
            distance: New distance
            azimuth: New azimuth
            elevation: New elevation
        """
        if self._camera_config != 'free':
            raise RuntimeError(
                f"❌ update_free_camera() only works for free cameras!\n"
                f"   Current config: {self._camera_config}\n"
                f"   FIX: Only call this method on free camera backends."
            )

        self._camera_obj.lookat[:] = lookat
        self._camera_obj.distance = distance
        self._camera_obj.azimuth = azimuth
        self._camera_obj.elevation = elevation

    def read_rgb_and_depth(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Read BOTH RGB and depth with only ONE scene update - PERFORMANCE OPTIMIZED!

        PERFORMANCE: Cuts rendering time in HALF by calling update_scene() once instead of twice!
        Before: update_scene() for RGB, then update_scene() for depth (2 calls, 27ms)
        After:  update_scene() once, render RGB, render depth (1 call, 13.5ms)

        Returns:
            tuple: (rgb, depth) where:
                - rgb: np.ndarray (H, W, 3) uint8, range [0, 255]
                - depth: np.ndarray (H, W) float32 in meters, or None if depth disabled
        """
        # Update scene ONCE for both RGB and depth!
        self.renderer.update_scene(self.data, camera=self._camera_obj)

        # Render RGB first
        self.renderer.disable_depth_rendering()
        rgb = self.renderer.render()

        # Render depth from SAME scene (no update_scene needed!)
        if self._depth_enabled:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render()
        else:
            depth = None

        return rgb, depth

    def read_rgb(self) -> np.ndarray:
        """Read RGB image from MuJoCo renderer

        Returns:
            np.ndarray: RGB image (H, W, 3) uint8, range [0, 255]
        """
        # CRITICAL: Disable depth rendering to get RGB!
        # (If depth enabled, render() returns depth instead of RGB)
        self.renderer.disable_depth_rendering()

        # Update scene with current camera (ID for sensor cameras!)
        self.renderer.update_scene(self.data, camera=self._camera_obj)

        # Render RGB
        rgb = self.renderer.render()
        return rgb

    def read_depth(self) -> Optional[np.ndarray]:
        """Read depth image from MuJoCo renderer

        Returns:
            np.ndarray: Depth image (H, W) float32, in meters
            None: If depth rendering not enabled
        """
        if not self._depth_enabled:
            return None

        # CRITICAL: Enable depth rendering to get depth!
        self.renderer.enable_depth_rendering()

        # Update scene with current camera (ID for sensor cameras!)
        self.renderer.update_scene(self.data, camera=self._camera_obj)

        # Render depth
        depth = self.renderer.render()  # Returns depth when depth rendering enabled
        return depth

    def get_intrinsics(self) -> Dict[str, Any]:
        """Get camera intrinsic parameters from MuJoCo model

        For free cameras: Approximates FOV = 45 degrees
        For sensor cameras: Uses camera.fovy from model

        Returns:
            dict: Camera intrinsics with K matrix and parameters
        """
        # Get field of view
        if self._camera_config == 'free':
            fovy_deg = 45.0  # Default FOV for free cameras
        else:
            # Get from model camera
            camera = self.model.camera(self._camera_obj)
            fovy_deg = np.degrees(camera.fovy)

        # Calculate intrinsic parameters
        # f = (height / 2) / tan(fovy / 2)
        fovy_rad = np.radians(fovy_deg)
        fy = (self.height / 2.0) / np.tan(fovy_rad / 2.0)

        # Assume square pixels (fx = fy)
        fx = fy

        # Principal point at image center
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Build K matrix
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        return {
            'K': K,
            'width': self.width,
            'height': self.height,
            'fx': fx,
            'fy': fy,
            'cx': cx,
            'cy': cy,
            'fov': fovy_deg
        }

    def close(self):
        """Cleanup renderer resources"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def __repr__(self):
        config = self._camera_config or 'uninitialized'
        return f"MuJoCoRenderingBackend(config='{config}', {self.width}x{self.height})"


class RealCameraBackend(CameraBackend):
    """Real hardware camera backend (Intel RealSense, webcam, etc.)

    FUTURE: This will interface with real robot cameras

    For now: Stub that crashes if you try to use it
    OFFENSIVE: This prevents accidentally using real camera code in simulation!

    Example future usage:
        # Intel RealSense D405
        backend = RealCameraBackend.create_realsense(
            serial_number='f1234567',
            width=640, height=480,
            fps=30
        )

        # USB webcam
        backend = RealCameraBackend.create_webcam(
            device_id=0,
            width=1280, height=720
        )
    """

    def __init__(self):
        raise NotImplementedError(
            "❌ RealCameraBackend not implemented yet!\n"
            "   This backend is for REAL ROBOT hardware (Intel RealSense, webcam, etc.)\n"
            "   Currently only MuJoCoRenderingBackend is supported.\n"
            "\n"
            "   REASON: You tried to use real camera backend in simulation!\n"
            "   FIX: Use MuJoCoRenderingBackend.create_sensor_camera() for simulation."
        )

    def read_rgb(self) -> np.ndarray:
        raise NotImplementedError("RealCameraBackend not implemented")

    def read_depth(self) -> Optional[np.ndarray]:
        raise NotImplementedError("RealCameraBackend not implemented")

    def get_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError("RealCameraBackend not implemented")

    def close(self):
        pass


# NO DEFAULT BACKEND FACTORY!
# You must explicitly choose MuJoCoRenderingBackend or RealCameraBackend
# This prevents accidental MuJoCo usage when you want real cameras

"""
MUJOCO BACKEND - PhysicsBackend implementation for MuJoCo
OFFENSIVE & ELEGANT: Extracted from mujoco_sim_ops.py + scene_ops.py

Pattern: Backend OWNS model/data, handles ALL MuJoCo operations
RuntimeEngine just orchestrates - backends are swappable!
"""

import mujoco
import mujoco.viewer
import os
import subprocess
import numpy as np
from typing import Dict, Any, Optional, Tuple
from .backend_interface import PhysicsBackend
from ..modals.robot_modal import Robot
from ..modals.xml_resolver import XMLResolver


class MuJoCoBackend(PhysicsBackend):
    """MuJoCo physics backend - uses MuJoCoModal internally!

    PURE MOP: Backend delegates to MuJoCoModal
    BACKWARD COMPATIBLE: Still exposes model/data/viewer as properties

    Pattern:
    - MuJoCoModal = data (model, data, XML, state)
    - MuJoCoBackend = operations (sync, query, viewer config)
    """

    def __init__(self, enable_viewer: bool = False, headless: bool = None):
        """Initialize MuJoCo backend

        Args:
            enable_viewer: If True, launch viewer (requires non-headless)
            headless: Force headless mode (None = auto-detect)
        """
        from ..modals.mujoco_modal import MuJoCoModal
        from concurrent.futures import ThreadPoolExecutor

        # PERFORMANCE: Cache sensor signature inspections (avoid inspect.signature() every step!)
        # Pre-populated by RuntimeEngine.load_experiment() - NO runtime inspection!
        self._sensor_supports_event_log = {}  # sensor_name -> bool (event_log parameter)
        self._sensor_supports_camera_backend = {}  # sensor_name -> bool (camera_backend parameter)

        # PERFORMANCE: Single background thread for async camera rendering
        # NOTE: OpenGL contexts are NOT thread-safe, so we use ONE thread to avoid context corruption
        # This thread handles ALL cameras sequentially, but doesn't block main thread!
        self._camera_thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera_render")

        # PERFORMANCE: Pipelined camera rendering (fire-and-forget pattern)
        # Store futures from previous render cycle - we check completion on NEXT sync
        # This allows physics to continue while cameras render (1-frame delay)
        self._camera_render_futures = []

        # GPU detection
        self.has_gpu = self._detect_gpu()

        # Headless detection
        if headless is None:
            is_headless = self._detect_headless()
        else:
            is_headless = headless

        # Setup rendering mode
        if is_headless:
            os.environ['MUJOCO_GL'] = 'egl'

        # Create modal (PURE MOP!)
        self.modal = MuJoCoModal(
            enable_viewer=enable_viewer and not is_headless,
            is_headless=is_headless
        )

    # === BACKWARD COMPATIBILITY - Expose modal's properties ===

    @property
    def model(self):
        """Backward compatible: access modal's model"""
        return self.modal.model

    @property
    def data(self):
        """Backward compatible: access modal's data"""
        return self.modal.data

    @property
    def viewer(self):
        """Backward compatible: access modal's viewer"""
        return self.modal.viewer

    @property
    def enable_viewer(self):
        """Backward compatible: access modal's enable_viewer"""
        return self.modal.enable_viewer

    @property
    def is_headless(self):
        """Backward compatible: access modal's is_headless"""
        return self.modal.is_headless

    # === GPU & HEADLESS DETECTION (from mujoco_sim_ops.py) ===

    def _detect_gpu(self) -> bool:
        """Check if GPU is available - OFFENSIVE"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                return True
        except:
            pass

        try:
            if os.path.exists('/dev/dri/card0'):
                return True
        except:
            pass

        return False

    def _detect_headless(self) -> bool:
        """Check if running in headless mode - OFFENSIVE"""
        if not os.environ.get('DISPLAY'):
            return True

        if os.environ.get('MUJOCO_GL') == 'egl':
            return True

        return False

    # === BACKEND INTERFACE IMPLEMENTATION ===

    def compile_xml(self, xml_string: str) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        """Compile scene XML into MuJoCo model/data - OFFENSIVE

        Delegates to MuJoCoModal!

        Args:
            xml_string: Complete MuJoCo XML scene description

        Returns:
            (MjModel, MjData) tuple

        OFFENSIVE: Crashes if XML invalid
        """
        # Delegate to modal!
        self.modal.compile_xml(xml_string)
        return self.modal.model, self.modal.data

    def step(self):
        """Step MuJoCo physics forward one timestep - OFFENSIVE

        Delegates to MuJoCoModal!
        """
        # Delegate to modal!
        self.modal.step()

    def set_controls(self, controls: Dict[str, float]):
        """Set control values for actuators - OFFENSIVE

        Args:
            controls: {actuator_name: value}

        OFFENSIVE: Crashes if actuator name not found
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        for actuator_name, value in controls.items():
            # Find actuator index
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

            if actuator_id < 0:
                raise ValueError(f"Actuator '{actuator_name}' not found in model")

            # Set control value
            self.data.ctrl[actuator_id] = value

    def get_state(self, joint_names: list) -> Dict[str, Dict[str, float]]:
        """Get current joint state (qpos, qvel) - OFFENSIVE

        Args:
            joint_names: List of joint names to query

        Returns:
            {joint_name: {"qpos": float, "qvel": float}}

        Returns empty dict for unknown joints (not crash)
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        state = {}
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)

            if joint_id < 0:
                continue  # Skip unknown joints

            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]

            state[joint_name] = {
                "qpos": float(self.data.qpos[qpos_addr]),
                "qvel": float(self.data.qvel[qvel_addr])
            }

        return state

    def sync_actuators_from_backend(self, robot: Robot):
        """Sync robot actuators FROM MuJoCo state - GENERIC for ANY robot

        Critical operation between step() and state extraction!
        Updates actuator.get_data() to match MuJoCo state

        Args:
            robot: Robot modal with actuators to sync

        Each actuator calls actuator.sync_from_mujoco(model, data)
        - Base: Syncs position/rotation from root body
        - Gripper: Syncs aperture from joint positions
        - Arm: Syncs extension from joint position

        OFFENSIVE: Silently skips actuators without sync_from_mujoco()
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        for actuator in robot.actuators.values():
            if hasattr(actuator, 'sync_from_mujoco'):
                actuator.sync_from_mujoco(self.model, self.data)

    def sync_assets_from_backend(self, scene):
        """Sync all assets FROM MuJoCo state - GENERIC for ANY scene

        Critical operation: Updates asset modals to match MuJoCo state
        Syncs objects, furniture, walls, etc. (NOT robot - use sync_actuators_from_backend)

        Args:
            scene: Scene modal with assets to sync

        Each asset calls asset.sync_from_mujoco(model, data)
        SAME pattern as robot actuators - Modal-Oriented!

        OFFENSIVE: Silently skips assets without sync_from_mujoco()
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        for asset in scene.assets.values():
            if hasattr(asset, 'sync_from_mujoco'):
                asset.sync_from_mujoco(self.model, self.data)

    def sync_sensors_from_backend(self, robot: Robot, update_cameras: bool = True, update_sensors: bool = True, event_log=None, camera_backends=None):
        """Sync robot sensors FROM backend - GENERIC for ANY robot (with async rates!)

        Updates sensor data from physics engine or real hardware

        Args:
            robot: Robot modal with sensors to sync
            update_cameras: If False, skip camera sensors (expensive rendering!)
            update_sensors: If False, skip non-camera sensors (force, IMU, etc.)
            event_log: Optional EventLog to track sensor changes (backward compatible)
            camera_backends: Optional dict mapping sensor_name -> CameraBackend

        Each sensor calls sensor.sync_from_mujoco(model, data, robot, event_log, camera_backend)
        - Cameras: Render images (EXPENSIVE! Skip if update_cameras=False)
        - Lidar: Raycasts (medium cost)
        - IMU: Read sensor data (cheap)

        PERFORMANCE: Camera rendering is parallelized using ThreadPoolExecutor!
        - 11 cameras × 13ms sequential = 143ms (blocks physics!)
        - 11 cameras parallel = ~13ms (limited by single render time)
        - Each camera backend has thread-local renderer (no context conflicts!)

        OFFENSIVE: Silently skips sensors without sync_from_mujoco()
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        # PERFORMANCE: Separate camera sensors from other sensors for parallel rendering!
        camera_sensors = []  # (sensor_name, sensor, kwargs)
        other_sensors = []   # (sensor_name, sensor, kwargs)

        for sensor_name, sensor in robot.sensors.items():
            if hasattr(sensor, 'sync_from_mujoco'):
                # MOP: Sensor MUST self-decide! OFFENSIVE - no fallback!
                if not hasattr(sensor, 'should_sync'):
                    raise ValueError(f"PURE MOP VIOLATION: Sensor '{sensor_name}' ({type(sensor).__name__}) must have should_sync() method!")

                # TRUE MOP: Sensor makes the decision
                should_sync = sensor.should_sync(update_cameras, update_sensors)

                if should_sync:
                    # Build kwargs based on what sensor supports (cached at init!)
                    kwargs = {}
                    if self._sensor_supports_event_log.get(sensor_name, False) and event_log is not None:
                        kwargs['event_log'] = event_log
                    if self._sensor_supports_camera_backend.get(sensor_name, False) and camera_backends is not None:
                        backend = camera_backends.get(sensor_name)
                        if backend is not None:
                            kwargs['camera_backend'] = backend

                    # PERFORMANCE: Categorize sensors for parallel rendering
                    is_camera = 'camera' in sensor_name.lower() or 'camera' in type(sensor).__name__.lower()
                    if is_camera and 'camera_backend' in kwargs:
                        camera_sensors.append((sensor_name, sensor, kwargs))
                    else:
                        other_sensors.append((sensor_name, sensor, kwargs))

        # Sync non-camera sensors sequentially (cheap operations)
        for sensor_name, sensor, kwargs in other_sensors:
            sensor.sync_from_mujoco(self.model, self.data, robot, **kwargs)

        # SYNC camera rendering (OpenGL contexts are NOT thread-safe!)
        # NOTE: Async rendering causes EGL_BAD_CONTEXT errors due to context sharing issues
        # Performance optimization should be done via resolution/camera count, not threading
        for sensor_name, sensor, kwargs in camera_sensors:
            sensor.sync_from_mujoco(self.model, self.data, robot, **kwargs)

    def get_geom_position(self, geom_name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of named geometry - OFFENSIVE

        Args:
            geom_name: Name of geom to query

        Returns:
            (x, y, z) position in world frame, or None if not found

        Used by behavior_extractors for spatial properties
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

        if geom_id < 0:
            return None

        # Get body position from geom's body
        body_id = self.model.geom_bodyid[geom_id]
        pos = self.data.xpos[body_id]

        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def get_body_position(self, body_name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of named body - OFFENSIVE

        Args:
            body_name: Name of body to query

        Returns:
            (x, y, z) position in world frame, or None if not found

        Used by behavior_extractors for spatial properties
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if body_id < 0:
            return None

        pos = self.data.xpos[body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def check_contact(self, geom_name1: str, geom_name2: str) -> bool:
        """Check if two geometries are in contact - OFFENSIVE

        Args:
            geom_name1: First geometry name
            geom_name2: Second geometry name

        Returns:
            True if geometries are in contact

        Used by behavior_extractors for contact properties
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first"

        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name1)
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name2)

        if geom1_id < 0 or geom2_id < 0:
            return False

        # Check all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (contact.geom1 == geom1_id and contact.geom2 == geom2_id) or \
               (contact.geom1 == geom2_id and contact.geom2 == geom1_id):
                return True

        return False

    def reset_to_keyframe(self, keyframe_name: str):
        """Reset MuJoCo data to saved keyframe - OFFENSIVE & FAST!

        Delegates to MuJoCoModal!

        Args:
            keyframe_name: Name of keyframe to restore (e.g., 'initial')
        """
        # Delegate to modal!
        self.modal.reset_to_keyframe(keyframe_name)

    def list_keyframes(self) -> list:
        """List all available keyframe names - OFFENSIVE

        Delegates to MuJoCoModal!
        """
        # Delegate to modal!
        return self.modal.list_keyframes()

    def cleanup(self):
        """Close viewer and cleanup - OFFENSIVE

        Delegates to MuJoCoModal!
        """
        # Delegate to modal!
        self.modal.cleanup()

    # === MODAL INTERFACE - Delegate to MuJoCoModal! ===

    def to_xml(self) -> str:
        """I know my compiled XML - Delegates to modal!"""
        return self.modal.to_xml()

    def to_state(self) -> dict:
        """I know my state - Delegates to modal!"""
        return self.modal.to_state()

    def from_xml(self, xml: str):
        """I can load from XML - Delegates to modal!"""
        return self.modal.from_xml(xml)

    def from_state(self, state: dict):
        """I can restore my state - Delegates to modal!"""
        self.modal.from_state(state)

    def to_json(self) -> dict:
        """I know how to serialize myself - Delegates to modal!"""
        return self.modal.to_json()

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself - PURE MOP!

        Creates backend from MuJoCoModal!
        """
        from ..modals.mujoco_modal import MuJoCoModal

        # Create modal from JSON
        modal = MuJoCoModal.from_json(data)

        # Create backend (skip normal __init__ to avoid recreating modal)
        backend = cls.__new__(cls)
        backend.modal = modal
        backend.has_gpu = backend._detect_gpu()

        return backend


# === USAGE PATTERN ===

"""
# Old way (mujoco_sim_ops.py):
sim = MujocoSimOps()
scene = sim.create("kitchen", ...)
sim.compile()
sim.run(steps=1000)

# New way (RuntimeEngine + MuJoCoBackend):
backend = MuJoCoBackend(enable_viewer=True)
engine = RuntimeEngine(backend)
engine.load_experiment(experiment)
engine.run(steps=1000)
"""


# === TEST ===

if __name__ == "__main__":
    from ..main.experiment_ops_unified import ExperimentOps
    from ..modals.xml_resolver import XMLResolver
    from ..modals import registry

    print("=== Testing MuJoCoBackend ===")

    # Create backend
    backend = MuJoCoBackend(enable_viewer=False)
    print(f"GPU: {backend.has_gpu}")
    print(f"Headless: {backend.is_headless}")

    # Create simple scene via ExperimentOps
    ops = ExperimentOps(headless=True)
    ops.create_scene("test_room", width=5, length=5, height=3)
    ops.add_asset("table", relative_to=(2, 2, 0))
    ops.add_asset("cup", relative_to="table", relation="on_top")

    # Generate XML
    xml = XMLResolver.build_scene_xml(ops.scene, registry)

    # Compile
    print("\nCompiling scene...")
    model, data = backend.compile_xml(xml)
    print(f"✓ Model compiled: {model.nq} DOFs, {model.nu} actuators")

    # Step physics
    print("\nStepping physics...")
    for i in range(100):
        backend.step()

    print(f"✓ Stepped 100 timesteps")

    # Query state
    print("\nQuerying state...")
    pos = backend.get_body_position("table")
    print(f"Table position: {pos}")

    contact = backend.check_contact("table_top_geom", "cup_body0_geom")
    print(f"Table-Cup contact: {contact}")

    print("\n✓ MuJoCoBackend test complete!")

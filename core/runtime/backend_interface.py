"""
BACKEND INTERFACE - Abstract physics backend for RuntimeEngine
OFFENSIVE & ELEGANT: Enables swapping MuJoCo/Real/Isaac Gym without code changes

Pattern: RuntimeEngine owns backend, delegates ALL physics operations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from ..modals.scene_modal import Scene
from ..modals.robot_modal import Robot


class PhysicsBackend(ABC):
    """Abstract backend interface - ALL physics engines must implement this

    Design Philosophy:
    - Backend OWNS model/data (MuJoCo model, Isaac Gym handle, real hardware client)
    - Backend handles ALL physics/state extraction
    - RuntimeEngine just orchestrates (compile → step → sync → extract)
    - Backends are SWAPPABLE - same RuntimeEngine code works for all

    Lifecycle:
    1. RuntimeEngine creates backend: MuJoCoBackend(), RealBackend(), IsaacGymBackend()
    2. RuntimeEngine calls compile_xml(scene_xml) → backend creates model
    3. RuntimeEngine loop: set_controls() → step() → sync_actuators() → sync_sensors()
    4. RuntimeEngine calls get_state() for state extraction
    """

    @abstractmethod
    def compile_xml(self, xml_string: str) -> Tuple[Any, Any]:
        """Compile scene XML into physics model

        Args:
            xml_string: Complete MuJoCo XML scene description

        Returns:
            (model, data) - Backend-specific model/data objects
            - MuJoCo: (MjModel, MjData)
            - Isaac Gym: (gym handle, sim handle)
            - Real: (hardware client, state dict)

        OFFENSIVE: Crashes if XML invalid or compilation fails
        """
        pass

    @abstractmethod
    def step(self):
        """Step physics forward one timestep

        - MuJoCo: mj_step(model, data)
        - Isaac Gym: gym.simulate(sim)
        - Real: Send controls, wait for next sensor frame

        OFFENSIVE: Must be called AFTER set_controls()
        """
        pass

    @abstractmethod
    def set_controls(self, controls: Dict[str, float]):
        """Set control values for actuators

        Args:
            controls: {actuator_name: value} from ActionExecutor

        - MuJoCo: Set data.ctrl[actuator_index] = value
        - Isaac Gym: gym.set_dof_position_target()
        - Real: Send control commands to hardware

        OFFENSIVE: Crashes if actuator name not found
        """
        pass

    @abstractmethod
    def get_state(self, joint_names: list) -> Dict[str, Dict[str, float]]:
        """Get current joint state (qpos, qvel)

        Args:
            joint_names: List of joint names to query

        Returns:
            {joint_name: {"qpos": float, "qvel": float}}

        Used by StateExtractor for behavior property extraction

        OFFENSIVE: Returns empty dict for unknown joints (not crash)
        """
        pass

    @abstractmethod
    def sync_actuators_from_backend(self, robot: Robot):
        """Sync robot actuators FROM backend state

        Critical operation between step() and state extraction!
        Updates actuator.get_data() to match backend state

        Args:
            robot: Robot modal with actuators to sync

        Each actuator calls actuator.sync_from_mujoco(model, data)
        - Base: Syncs position/rotation from root body
        - Gripper: Syncs aperture from joint positions
        - Arm: Syncs extension from joint position

        OFFENSIVE: Silently skips actuators without sync_from_mujoco()
        """
        pass

    @abstractmethod
    def sync_sensors_from_backend(self, robot: Robot):
        """Sync robot sensors FROM backend data

        Updates sensor data from physics engine

        Args:
            robot: Robot modal with sensors to sync

        Each sensor calls sensor.sync_from_mujoco(model, data, robot)
        - Cameras: Render images
        - Lidar: Raycasts
        - IMU: Read sensor data

        OFFENSIVE: Silently skips sensors without sync_from_mujoco()
        """
        pass

    @abstractmethod
    def get_geom_position(self, geom_name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of named geometry

        Args:
            geom_name: Name of geom to query

        Returns:
            (x, y, z) position in world frame, or None if not found

        Used by behavior_extractors for spatial properties
        """
        pass

    @abstractmethod
    def get_body_position(self, body_name: str) -> Optional[Tuple[float, float, float]]:
        """Get 3D position of named body

        Args:
            body_name: Name of body to query

        Returns:
            (x, y, z) position in world frame, or None if not found

        Used by behavior_extractors for spatial properties
        """
        pass

    @abstractmethod
    def check_contact(self, geom_name1: str, geom_name2: str) -> bool:
        """Check if two geometries are in contact

        Args:
            geom_name1: First geometry name
            geom_name2: Second geometry name

        Returns:
            True if geometries are in contact

        Used by behavior_extractors for contact properties
        """
        pass

    def cleanup(self):
        """Optional cleanup (viewer close, hardware disconnect)

        Not abstract - backends can override if needed
        """
        pass


# === USAGE PATTERN ===

"""
# Create backend
backend = MuJoCoBackend()  # or RealBackend(), IsaacGymBackend()

# RuntimeEngine uses it
engine = RuntimeEngine(backend)
engine.load_experiment(experiment)

# Engine's step() method:
def step():
    # 1. Get control commands from ActionExecutor
    commands = self.action_executor.tick(self.robot)

    # 2. Send to backend
    self.backend.set_controls(commands)

    # 3. Step physics
    self.backend.step()

    # 4. CRITICAL: Sync robot state from backend
    self.backend.sync_actuators_from_backend(self.robot)
    self.backend.sync_sensors_from_backend(self.robot)

    # 5. Extract state (now actuators/sensors have latest data!)
    state = self.state_extractor.extract(self.scene, self.robot)

    # 6. Compute reward
    reward = self.reward_computer.compute(state, self.scene.reward_modal)
"""

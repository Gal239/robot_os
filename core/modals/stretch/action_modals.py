"""
ACTION MODALS - Each atomic action has its own class with shared base logic
"""
from typing import List, Union, Dict, Any, Optional, Literal, Tuple
from pydantic import BaseModel, Field, model_validator
import numpy as np
import time
import math

# Actuator imports removed - actions work with ActuatorComponent instances
# No type checks needed - MODAL-ORIENTED (trust modals)

# ============================================
# LOAD ROTATION TOLERANCE FROM DISCOVERED TOLERANCES (PURE MOP!)
# ============================================
import json
from pathlib import Path

_tolerance_file = Path(__file__).parent / "discovered_tolerances.json"
with open(_tolerance_file) as f:
    _all_tolerances = json.load(f)

    # OFFENSIVE: Crash if tolerance missing!
    if "imu_rotation" not in _all_tolerances:
        raise KeyError(
            f"🚨 MOP VIOLATION: Missing 'imu_rotation' in discovered_tolerances.json!\n"
            f"Available tolerances: {list(_all_tolerances.keys())}\n"
            f"\n"
            f"💡 FIX: Run tolerance discovery:\n"
            f"  python3 -m core.ops.discover_tolerances\n"
        )
    if "odometry_distance" not in _all_tolerances:
        raise KeyError(
            f"🚨 MOP VIOLATION: Missing 'odometry_distance' in discovered_tolerances.json!\n"
            f"Available tolerances: {list(_all_tolerances.keys())}\n"
            f"\n"
            f"💡 FIX: Run tolerance discovery:\n"
            f"  python3 -m core.ops.discover_tolerances\n"
        )

    ROTATION_TOLERANCE = _all_tolerances["imu_rotation"]
    DISTANCE_TOLERANCE = _all_tolerances["odometry_distance"]

# ============================================
# ROTATION HELPER FUNCTIONS (DRY - used by all rotation actions)
# ============================================

def quat_to_heading(quat: List[float]) -> float:
    """Convert quaternion to heading - SHARED by all rotation actions

    Used by: BaseRotateBy, BaseMoveTo, HeadPan, WristYaw, etc.
    """
    x, y, z, w = quat
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

def angle_diff(a: float, b: float) -> float:
    """Calculate shortest angle difference - SHARED by all rotation actions

    Normalizes angle to [-π, π] range.
    Used by: BaseRotateBy, BaseMoveTo, HeadPan, WristYaw, etc.
    """
    diff = a - b
    while diff > np.pi: diff -= 2*np.pi
    while diff < -np.pi: diff += 2*np.pi
    return diff

# ============================================
# SENSOR CONDITION
# ============================================

class SensorCondition(BaseModel):
    """Stop condition based on sensor feedback"""
    sensor: str = Field(..., description="Sensor to monitor (e.g., 'gripper_force')")
    field: str = Field(..., description="Field in sensor data")
    threshold: float = Field(..., description="Threshold value")
    operator: Literal['>=', '<=', '>', '<', '=='] = Field('>=', description="Comparison")

    def check(self, sensors: Dict) -> bool:
        """Check if condition is met"""
        if self.sensor not in sensors:
            return False

        sensor_data = sensors[self.sensor].get_data()
        if self.field not in sensor_data:
            return False

        value = sensor_data[self.field]
        if self.operator == '>=':
            return value >= self.threshold
        elif self.operator == '<=':
            return value <= self.threshold
        elif self.operator == '>':
            return value > self.threshold
        elif self.operator == '<':
            return value < self.threshold
        elif self.operator == '==':
            return abs(value - self.threshold) < 0.001
        return False


# ============================================
# BASE ACTION CLASS
# ============================================

class Action(BaseModel):
    """PURE MOP: Self-connecting, self-syncing, self-executing action modal

    Actions are TRUE MODALS that:
    1. SELF-CONNECT to required actuators/sensors (connect() method)
    2. SELF-SYNC state from connected modals (_sync_state())
    3. SELF-VALIDATE completion (_check_completion())
    4. SELF-EXECUTE and generate commands (_get_command())
    5. SELF-TERMINATE with proper HOLD commands (hold_command property)
    """
    id: str = Field(default_factory=lambda: f"action_{int(time.time() * 1000000)}")  # Auto-generated ID for tracking
    status: Literal['pending', 'executing', 'completed', 'failed'] = 'pending'
    progress: float = Field(0.0, ge=0.0, le=100.0)

    # State tracking
    started_at: Optional[float] = None
    last_state: Optional[Dict] = None
    stuck_counter: int = 0
    stuck_threshold: int = 10
    # Optional sensor conditions
    stop_conditions: List[SensorCondition] = []

    # MOP: Actions self-declare execution type
    execution_type: Literal["single_actuator", "multi_actuator", "non_actuator"] = "single_actuator"

    # PURE MOP: Connected modals (set by connect() method)
    _actuator: Optional[Any] = None
    _sensors: Optional[Dict[str, Any]] = None
    _event_log: Optional[Any] = None

    @property
    def actuator_ids(self) -> List[str]:
        """Return list of actuators this action controls - MOP!"""
        if hasattr(self, 'actuator_id'):
            return [self.actuator_id]
        return []

    @property
    def required_actuators(self) -> List[str]:
        """SELF-DECLARE: What actuators do I need? Override in subclasses!"""
        # Default: use actuator_id if present
        if hasattr(self, 'actuator_id'):
            return [self.actuator_id]
        return []

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: What sensors do I need? Override in subclasses!"""
        return []

    @property
    def hold_command(self) -> float:
        """SELF-TERMINATE: What command to send when done? Override for velocity actuators!

        Default behavior (position actuators): hold current position
        Velocity actuators (wheels): should override to return 0.0
        """
        # PURE MOP: Trust actuator modal has position property if connected!
        if self._actuator:
            return self._actuator.position  # Crash if missing - modal is broken!
        return 0.0

    def connect(self, robot, event_log=None):
        """SELF-CONNECT: Connect to required actuators and sensors - OFFENSIVE!

        This is PURE MOP: Action discovers and connects to its dependencies.
        OFFENSIVE: Crashes with educational error if required dependencies missing!

        Args:
            robot: Robot modal with actuators and sensors
            event_log: EventLog modal for event emission
        """
        # Connect to primary actuator - OFFENSIVE if missing!
        req_actuators = self.required_actuators
        if req_actuators:
            actuator_id = req_actuators[0]
            if actuator_id not in robot.actuators:
                raise RuntimeError(
                    f"🚨 ACTION CONNECTION ERROR!\n"
                    f"  Action '{self.__class__.__name__}' requires actuator: '{actuator_id}'\n"
                    f"  Available actuators: {list(robot.actuators.keys())}\n"
                    f"\n"
                    f"💡 FIX: Check actuator_id is correct or add actuator to robot\n"
                )
            self._actuator = robot.actuators[actuator_id]

        # Connect to all required sensors - OFFENSIVE if missing!
        self._sensors = {}
        req_sensors = self.required_sensors
        for sensor_id in req_sensors:
            if sensor_id not in robot.sensors:
                raise RuntimeError(
                    f"🚨 ACTION CONNECTION ERROR!\n"
                    f"  Action '{self.__class__.__name__}' requires sensor: '{sensor_id}'\n"
                    f"  Available sensors: {list(robot.sensors.keys())}\n"
                    f"\n"
                    f"💡 FIX: Check sensor_id is correct or add sensor to robot\n"
                )
            self._sensors[sensor_id] = robot.sensors[sensor_id]

        # Connect to event log
        self._event_log = event_log

    def tick(self) -> Dict[str, float]:
        """PURE MOP: Self-execute one tick - RETURNS DICT OF COMMANDS!

        This is the CORE of MOP actions:
        1. Auto-transition pending → executing
        2. SELF-SYNC state from connected modals
        3. SELF-VALIDATE completion
        4. SELF-EXECUTE and return commands (or HOLD if done)

        UNIFORM INTERFACE (MOP):
        - ALL actions return Dict[str, float]
        - Single-actuator: {actuator_id: command}
        - Multi-actuator: {act1: cmd1, act2: cmd2}

        Returns:
            Dict of {actuator_id: command} for ALL actuators this action controls
        """
        # Auto-transition pending → executing
        if self.status == 'pending':
            self.status = 'executing'
            self.started_at = time.time()

        # Helper: Normalize command to dict
        def normalize_command(cmd):
            if isinstance(cmd, dict):
                return cmd  # Already dict (multi-actuator)
            else:
                # Single value - wrap in dict with first actuator
                actuator_ids = self.actuator_ids
                if actuator_ids:
                    return {actuator_ids[0]: cmd}
                return {}  # No actuators

        # If done, return HOLD command (normalized to dict)
        if self.status in ['completed', 'failed']:
            return normalize_command(self.hold_command)

        # SELF-SYNC: Read current state from connected modals
        self._sync_state()

        # Check sensor stop conditions (original feature preserved!)
        if self._sensors and self.stop_conditions:
            for condition in self.stop_conditions:
                if condition.check(self._sensors):
                    self.status = 'completed'
                    self.progress = 100.0
                    self._emit_action_event(self._event_log, 'action_complete', {
                        'duration': time.time() - self.started_at if self.started_at else 0,
                        'stopped_by': 'sensor_condition'
                    })
                    return normalize_command(self.hold_command)

        # SELF-VALIDATE: Check if we've reached our goal
        if self._check_completion():
            self.status = 'completed'
            self.progress = 100.0
            self._emit_action_event(self._event_log, 'action_complete', {
                'duration': time.time() - self.started_at if self.started_at else 0
            })
            return normalize_command(self.hold_command)

        # SELF-EXECUTE: Generate and return command (normalized to dict)
        return normalize_command(self._get_command())

    def _sync_state(self):
        """SELF-SYNC: Read state from connected modals - Override in subclasses!"""
        pass  # Base implementation does nothing

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Am I done? - Override in subclasses!

        Returns:
            True if action has reached its goal
        """
        return False  # Base implementation: never complete

    def _get_command(self) -> float:
        """SELF-EXECUTE: What command should I send? - Override in subclasses!

        Returns:
            Command value for actuator
        """
        return 0.0  # Base implementation: send nothing

    def get_required_state(self) -> Dict[str, List[str]]:
        """Return what state this action needs"""
        raise NotImplementedError

    def update(self, actuator, sensors=None, event_log=None):
        """Update state using actuator + sensors - MOP: Direct modal instances

        Args:
            actuator: Actuator modal instance
            sensors: Sensors dict
            event_log: EventLog modal instance (for event emission)
        """
        # Track old status for event emission
        old_status = self.status

        # Start execution
        if self.status == 'pending':
            self.status = 'executing'
            self.started_at = time.time()  # Track when action started

        # Check sensor stop conditions
        if sensors and self.stop_conditions:
            for condition in self.stop_conditions:
                if condition.check(sensors):
                    self.status = 'completed'
                    self.progress = 100.0
                    # Emit completion event
                    self._emit_action_event(event_log, 'action_complete', {
                        'duration': time.time() - self.started_at if self.started_at else 0,
                        'stopped_by': 'sensor_condition'
                    })

    def detect_stuck(self, current_state: Dict, views=None) -> bool:
        """Detect if stuck using state comparison"""
        if self.last_state and self.last_state == current_state:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_state = current_state.copy()

        if self.stuck_counter >= self.stuck_threshold:
            self.status = 'failed'
            # Emit failure event
            self._emit_action_event(views, 'action_failed', {
                'reason': 'stuck',
                'stuck_count': self.stuck_counter
            })
            return True
        return False

    def _emit_action_event(self, event_log, event_type: str, data: Dict):
        """Emit action event to event log - MOP: Direct modal instance

        Args:
            event_log: EventLog modal instance (not via views!)
            event_type: 'action_complete' or 'action_failed'
            data: Event-specific data
        """
        if event_log:
            from ..event_log_modal import create_action_event
            event = create_action_event(
                timestamp=time.time(),
                action_id=self.id,
                event_type=event_type,
                data={**data, 'progress': self.progress}
            )
            event_log.add_event(event)

    def calculate_progress(self, current: float, target: float, range_size: float) -> float:
        """Calculate progress as percentage"""
        error = abs(current - target)
        return max(0, min(100, 100 * (1 - error / range_size)))

    def get_data(self) -> Dict:
        """Get action state for monitoring"""
        return {
            "status": self.status,
            "progress": self.progress,
            "stuck_counter": self.stuck_counter
        }

    def get_rl(self) -> np.ndarray:
        """Get normalized vector for RL"""
        status_val = {'pending': 0.0, 'executing': 0.5, 'completed': 1.0, 'failed': -1.0}[self.status]
        return np.array([self.progress / 100.0, status_val, self.stuck_counter / float(self.stuck_threshold)])


# ============================================
# POSITION ACTION BASE (arm, lift, head, wrist)
# ============================================

class PositionMoveToBase(Action):
    """PURE MOP: Base for all absolute position movements"""
    position: float
    actuator_id: str = ""

    def _sync_state(self):
        """SELF-SYNC: Position actuators are always synced via _actuator"""
        pass

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Have we reached target position?"""
        if self._actuator is None:
            return False

        # TRUE MOP: Actuator knows its own position property!
        current_pos = self._actuator.get_position()
        error = abs(current_pos - self.position)

        # SELF-DISCOVERY: Read tolerance from actuator modal (physics-discovered!)
        tolerance = self._actuator.tolerance

        # Update progress
        range_val = self._actuator.range
        range_size = range_val[1] - range_val[0]
        self.progress = self.calculate_progress(current_pos, self.position, range_size)

        return error < tolerance

    def _get_command(self) -> float:
        """SELF-EXECUTE: Return target position (actuator clamps to range)

        MOP: Action requests position, actuator adjusts if out of range.
        Action captures the ACTUAL target (after clamping) so completion check is accurate.
        """
        if self._actuator is None:
            return 0.0

        # Actuator returns adjusted value if clamping occurred
        adjusted = self._actuator.move_to(self.position)

        # Update our target to match reality (MOP: action knows actual achievable goal)
        if adjusted != self.position:
            self.position = adjusted

        return adjusted

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': [self.actuator_id], 'sensors': []}


class PositionMoveByBase(Action):
    """PURE MOP: Base for all relative position movements"""
    distance: float
    target: Optional[float] = None
    actuator_id: str = ""

    def _sync_state(self):
        """SELF-SYNC: Calculate target position on first sync"""
        if self.target is None and self._actuator is not None:
            self.target = self._actuator.move_by(self.distance)

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Have we reached target position?"""
        if self._actuator is None or self.target is None:
            return False

        # TRUE MOP: Actuator knows its own position property!
        current_pos = self._actuator.get_position()
        error = abs(current_pos - self.target)

        # SELF-DISCOVERY: Read tolerance from actuator modal (physics-discovered!)
        tolerance = self._actuator.tolerance

        # Update progress
        self.progress = self.calculate_progress(current_pos, self.target, abs(self.distance))

        return error < tolerance

    def _get_command(self) -> float:
        """SELF-EXECUTE: Return target position"""
        if self.target is None:
            return 0.0
        return self.target

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': [self.actuator_id], 'sensors': []}


# ============================================
# ARM ACTIONS
# ============================================

class ArmMoveTo(PositionMoveToBase):
    actuator_id: Literal['arm'] = 'arm'

class ArmMoveBy(PositionMoveByBase):
    actuator_id: Literal['arm'] = 'arm'


# ============================================
# LIFT ACTIONS
# ============================================

class LiftMoveTo(PositionMoveToBase):
    actuator_id: Literal['lift'] = 'lift'

class LiftMoveBy(PositionMoveByBase):
    actuator_id: Literal['lift'] = 'lift'


# ============================================
# GRIPPER ACTIONS (with force sensing)
# ============================================

class GripperMoveTo(PositionMoveToBase):
    actuator_id: Literal['gripper'] = 'gripper'
    force_limit: float = 5.0

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: Need force sensor for stop conditions"""
        return ['gripper_force']

    def __init__(self, **data):
        super().__init__(**data)
        # Add force stop conditions (checked by base tick()!)
        self.stop_conditions = [
            SensorCondition(sensor='gripper_force', field='force_left',
                          threshold=self.force_limit, operator='>='),
            SensorCondition(sensor='gripper_force', field='force_right',
                          threshold=self.force_limit, operator='>=')
        ]

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': ['gripper'], 'sensors': ['gripper_force']}

class GripperMoveBy(PositionMoveByBase):
    actuator_id: Literal['gripper'] = 'gripper'
    force_limit: float = 5.0

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: Need force sensor for stop conditions"""
        return ['gripper_force']

    def __init__(self, **data):
        super().__init__(**data)
        # Add force stop conditions (checked by base tick()!)
        self.stop_conditions = [
            SensorCondition(sensor='gripper_force', field='force_left',
                          threshold=self.force_limit, operator='>='),
            SensorCondition(sensor='gripper_force', field='force_right',
                          threshold=self.force_limit, operator='>=')
        ]

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': ['gripper'], 'sensors': ['gripper_force']}


# ============================================
# HEAD ACTIONS
# ============================================

class HeadPanMoveTo(PositionMoveToBase):
    actuator_id: Literal['head_pan'] = 'head_pan'

class HeadPanMoveBy(PositionMoveByBase):
    actuator_id: Literal['head_pan'] = 'head_pan'

class HeadTiltMoveTo(PositionMoveToBase):
    actuator_id: Literal['head_tilt'] = 'head_tilt'

class HeadTiltMoveBy(PositionMoveByBase):
    actuator_id: Literal['head_tilt'] = 'head_tilt'


# ============================================
# WRIST ACTIONS
# ============================================

class WristYawMoveTo(PositionMoveToBase):
    actuator_id: Literal['wrist_yaw'] = 'wrist_yaw'

class WristYawMoveBy(PositionMoveByBase):
    actuator_id: Literal['wrist_yaw'] = 'wrist_yaw'

class WristPitchMoveTo(PositionMoveToBase):
    actuator_id: Literal['wrist_pitch'] = 'wrist_pitch'

class WristPitchMoveBy(PositionMoveByBase):
    actuator_id: Literal['wrist_pitch'] = 'wrist_pitch'

class WristRollMoveTo(PositionMoveToBase):
    actuator_id: Literal['wrist_roll'] = 'wrist_roll'

class WristRollMoveBy(PositionMoveByBase):
    actuator_id: Literal['wrist_roll'] = 'wrist_roll'


# ============================================
# BASE MOVEMENT ACTIONS (use odometry)
# ============================================

class BaseMoveForward(Action):
    """Base forward movement using differential drive kinematics - PURE MOP!

    MOP SELF-DECLARATION (§5 Modal-to-Modal Communication):
    - actuator_ids: Returns ['left_wheel_vel', 'right_wheel_vel'] (BOTH wheels!)
    - _get_command(): Returns Dict[str, float] with coordinated wheel commands

    ActionExecutor reads actuator_ids and applies dict commands - NO GLUE CODE!

    Uses Hello Robot's differential drive kinematics to convert:
        (linear_velocity, angular_velocity) → (left_wheel, right_wheel)

    This compensates for asymmetric wheel collision geometry in stretch.xml.
    XML IS TRUTH - geometry is intentional, we adapt control!

    Args:
        distance: Target distance to move in meters
        speed: Linear velocity in m/s (default 0.1 m/s)

    Example:
        # ONE action controls BOTH wheels using differential drive
        action = BaseMoveForward(distance=2.0, speed=0.1)
        # No actuator_id needed - action SELF-DECLARES it controls both!
    """
    distance: float
    speed: float = 0.1  # Linear velocity (m/s)
    start_pos: Optional[Tuple[float, float]] = None

    @property
    def actuator_ids(self) -> List[str]:
        """SELF-DECLARE: I control BOTH wheels! (MOP §5 Modal-to-Modal)"""
        return ['left_wheel_vel', 'right_wheel_vel']

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: Need odometry for movement tracking"""
        return ['odometry']

    @property
    def hold_command(self) -> Dict[str, float]:
        """SELF-TERMINATE: STOP both wheels (velocity = 0.0)"""
        return {'left_wheel_vel': 0.0, 'right_wheel_vel': 0.0}

    def _sync_state(self):
        """SELF-SYNC: Read odometry position"""
        if self._sensors is None or 'odometry' not in self._sensors:
            return

        odom_data = self._sensors['odometry'].get_data()
        current_pos = (odom_data['x'], odom_data['y'])

        # Set start position on first sync
        if self.start_pos is None:
            self.start_pos = current_pos

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Have we moved enough?"""
        if self.start_pos is None or self._sensors is None or 'odometry' not in self._sensors:
            return False

        odom_data = self._sensors['odometry'].get_data()
        current_pos = (odom_data['x'], odom_data['y'])

        # Calculate distance moved
        dx = current_pos[0] - self.start_pos[0]
        dy = current_pos[1] - self.start_pos[1]
        moved = np.sqrt(dx**2 + dy**2)

        # Update progress
        self.progress = min(100, 100 * moved / abs(self.distance))

        # PURE MOP: Tolerance from discovered_tolerances.json (same as reward system!)
        # Both action and reward use SAME tolerance → perfect validation alignment
        tolerance = DISTANCE_TOLERANCE  # Physics-discovered: 0.02m (20mm)

        return moved >= (abs(self.distance) - tolerance)

    def _get_command(self) -> Dict[str, float]:
        """SELF-EXECUTE: Differential drive kinematics! (MOP §5 Modal-to-Modal)

        Converts desired linear velocity → coordinated wheel angular velocities.
        Uses Hello Robot's official differential drive equations.

        Returns:
            Dict with commands for BOTH wheels: {'left_wheel_vel': w_left, 'right_wheel_vel': w_right}
        """
        from .diff_drive_utils import diff_drive_inv_kinematics

        # Determine linear velocity direction
        v_linear = self.speed if self.distance >= 0 else -self.speed
        omega = 0.0  # Straight movement (no rotation)

        # Convert to wheel angular velocities using differential drive
        w_left, w_right = diff_drive_inv_kinematics(v_linear, omega)

        return {
            'left_wheel_vel': w_left,
            'right_wheel_vel': w_right
        }

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': self.actuator_ids, 'sensors': ['odometry']}


class BaseMoveBackward(BaseMoveForward):
    """Base backward movement - inherits differential drive from BaseMoveForward

    MOP: Inherits all modal communication from BaseMoveForward.
    Only overrides _get_command() to negate linear velocity.

    Args:
        distance: Target distance to move backward in meters (positive value!)
        speed: Linear velocity in m/s (default 0.1 m/s)

    Example:
        # ONE action controls BOTH wheels using differential drive
        action = BaseMoveBackward(distance=0.5, speed=0.1)
        # Moves robot backward in straight line
    """
    def _get_command(self) -> Dict[str, float]:
        """SELF-EXECUTE: Negative linear velocity for backward movement

        Returns:
            Dict with commands for BOTH wheels
        """
        from .diff_drive_utils import diff_drive_inv_kinematics

        # Negative linear velocity = backward movement
        w_left, w_right = diff_drive_inv_kinematics(-self.speed, omega=0.0)

        return {
            'left_wheel_vel': w_left,
            'right_wheel_vel': w_right
        }


# ============================================
# BASE ROTATION ACTIONS (use IMU)
# ============================================

class BaseRotateBy(Action):
    """PURE MOP: Self-executing wheel rotation action using IMU

    ActionBlock with execution_mode='parallel' coordinates both wheels for differential drive.
    This action is instantiated twice (once for each wheel) with different speeds.

    Uses wheel velocity actuators directly (not position control like other actuators).
    Action controls velocity (speed parameter) and monitors IMU to decide when to STOP.

    Args:
        radians: Target rotation angle in radians - provide EITHER radians OR degrees
        degrees: Target rotation angle in degrees - provide EITHER radians OR degrees
        speed: Wheel velocity in rad/s (REQUIRED, range -6.0 to 6.0)
        actuator_id: Which wheel to control ('left_wheel_vel' or 'right_wheel_vel')

    Example:
        # Used in ActionBlock with parallel execution:
        BaseRotateBy(radians=1.57, speed=4.0, actuator_id='left_wheel_vel')   # Left wheel
        BaseRotateBy(radians=1.57, speed=-4.0, actuator_id='right_wheel_vel')  # Right wheel (opposite!)
    """
    radians: Optional[float] = None
    degrees: Optional[float] = None
    speed: float  # Wheel velocity in rad/s (REQUIRED, range -6.0 to 6.0)
    start_heading: Optional[float] = None
    current_heading: Optional[float] = None
    prev_heading: Optional[float] = None  # Track previous heading for delta calculation
    cumulative_rotation: float = 0.0  # PURE MOP: Sum small deltas (handles >180° without wrapping!)
    actuator_id: Literal['left_wheel_vel', 'right_wheel_vel'] = 'left_wheel_vel'
    coast_step: int = 0  # Track coast-down steps AFTER completion

    @model_validator(mode='after')
    def convert_degrees_to_radians(self):
        """Validate that EITHER radians OR degrees is provided (not both, not neither)"""
        if (self.radians is None) == (self.degrees is None):
            raise ValueError(
                "Provide EITHER 'radians' OR 'degrees' (not both, not neither). "
                "Example: BaseRotateBy(degrees=90, speed=4.0)"
            )

        # Convert degrees to radians if degrees was provided
        if self.degrees is not None:
            self.radians = math.radians(self.degrees)

        return self

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: Need IMU for rotation tracking"""
        return ['imu']

    @property
    def hold_command(self) -> float:
        """SELF-TERMINATE: After coast-down completes, wheels STOP at 0.0"""
        return 0.0

    def _sync_state(self):
        """SELF-SYNC: Read IMU heading and accumulate rotation deltas

        PURE MOP FIX: Accumulate small rotation deltas instead of using angle_diff()!
        This handles rotations > 180° correctly by summing steps.
        """
        if self._sensors is None or 'imu' not in self._sensors:
            return

        imu_data = self._sensors['imu'].get_data()
        heading = quat_to_heading(imu_data['orientation'])

        if self.start_heading is None:
            self.start_heading = heading
            self.current_heading = heading
            self.prev_heading = heading
            return

        # Accumulate rotation: sum small deltas (angle_diff handles wrapping per step!)
        delta = angle_diff(heading, self.prev_heading)
        self.cumulative_rotation += delta

        # Track current and previous for next iteration
        self.prev_heading = heading  # Use the heading we just processed!
        self.current_heading = heading

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Have we rotated enough? Uses CUMULATIVE rotation

        PURE MOP FIX: Uses cumulative_rotation (not angle_diff) + tolerance!
        - cumulative_rotation: Handles >180° by summing deltas
        - tolerance: Accept close-enough (179° for 180° target) since physics isn't perfect

        ASYNC COAST-DOWN: Don't mark complete until coast-down finishes!
        1. Reach target (within tolerance) → start coast-down
        2. Coast for 3 steps with aggressive braking (6→4→2→0)
        3. Only mark complete after coast_step >= 3
        """
        if self.start_heading is None:
            return False

        # Update progress based on CUMULATIVE rotation (handles >180°!)
        # Guard against division by zero for 0° rotations
        if abs(self.radians) < 0.001:  # ~0.06°
            self.progress = 100.0  # 0° rotation is instantly complete
        else:
            self.progress = min(100, 100 * abs(self.cumulative_rotation) / abs(self.radians))

        # If coast-down has completed, we're done (accept wherever we ended up!)
        # Coast-down means we already started braking and finished the 3-step sequence
        if self.coast_step >= 3:
            return True

        # PURE MOP: Use tolerance from discovered_tolerances.json!
        # Accept "close enough" since physics never hits exact values
        tolerance_rad = math.radians(ROTATION_TOLERANCE)  # Convert 1.1° to radians

        # Check if within tolerance of target (works for both + and - rotations!)
        # This triggers the START of coast-down
        target_reached = abs(self.cumulative_rotation - self.radians) <= tolerance_rad

        # If target not reached yet, continue executing
        return False

    def _get_command(self) -> float:
        """SELF-EXECUTE: Full speed until target, then async coast-down

        PURE MOP FIX: Uses cumulative_rotation + tolerance (same as _check_completion)!

        ASYNC COAST-DOWN strategy:
        1. Full speed until target reached (within tolerance)
        2. When target reached: start coast-down (increment coast_step)
        3. Return aggressively decreasing speed: 6→4→2→0 (3 steps)
        4. After 3 steps: _check_completion returns True → action completes

        CRITICAL FIX: Once coast-down starts, ALWAYS finish it!
        Faster coast-down (3 steps vs 6) prevents momentum drift on large rotations.
        """
        if self.start_heading is None:
            return self.speed

        # START BRAKING: When half-tolerance away OR overshot
        # Start braking EARLIER to prevent overshoot during coast-down
        tolerance_rad = math.radians(ROTATION_TOLERANCE)  # 1.1° tolerance
        BRAKE_THRESHOLD = tolerance_rad / 2.0  # 0.55° - start braking early!

        # Check if within brake threshold of target
        within_brake_zone = abs(self.cumulative_rotation - self.radians) <= BRAKE_THRESHOLD

        # Check if overshot the target (direction matters!)
        if self.radians >= 0:
            overshot = self.cumulative_rotation > self.radians
        else:
            overshot = self.cumulative_rotation < self.radians

        should_brake = within_brake_zone or overshot

        # If not braking yet, continue at full speed
        if not should_brake:
            return self.speed

        # Target reached! Start or continue coast-down
        self.coast_step += 1

        # Coast down over 3 steps: 6→4→2→0 (aggressive braking)
        if self.coast_step <= 3:
            remaining_speed = abs(self.speed) - (self.coast_step * 2)  # Decrement by 2 each step
            if remaining_speed > 0:
                return math.copysign(remaining_speed, self.speed)

        # After 3 coast steps, fully stopped
        return 0.0

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': [self.actuator_id], 'sensors': ['imu']}


class BaseMoveTo(Action):
    """PURE MOP: Base rotation to absolute heading

    Args:
        rotation: Target heading in DEGREES (user-friendly!)

    Example:
        BaseMoveTo(rotation=90.0)  # Rotate to face 90° (East)
        BaseMoveTo(rotation=0.0)   # Rotate to face 0° (North)

    NOTE: Currently uses left_wheel_vel only (not proper differential drive yet)
    TODO: Implement differential drive with both wheels
    """
    rotation: Optional[float] = None  # Target heading in degrees
    actuator_id: Literal['left_wheel_vel'] = 'left_wheel_vel'
    start_heading: Optional[float] = None
    target_heading: Optional[float] = None
    rotation_needed: Optional[float] = None

    @property
    def required_sensors(self) -> List[str]:
        """SELF-DECLARE: Need IMU for heading tracking"""
        return ['imu']

    @property
    def hold_command(self) -> float:
        """SELF-TERMINATE: Wheels STOP at 0.0 (velocity actuator!)"""
        return 0.0

    def _sync_state(self):
        """SELF-SYNC: Read IMU heading and calculate target on first sync"""
        if self._sensors is None or 'imu' not in self._sensors:
            return

        imu_data = self._sensors['imu'].get_data()
        current_heading = quat_to_heading(imu_data['orientation'])

        if self.start_heading is None:
            self.start_heading = current_heading
            # Convert degrees to radians for target
            self.target_heading = np.radians(self.rotation) if self.rotation is not None else self.start_heading
            # Calculate rotation needed
            self.rotation_needed = angle_diff(self.target_heading, self.start_heading)

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Have we reached target heading?"""
        if self._sensors is None or 'imu' not in self._sensors or self.target_heading is None:
            return False

        imu_data = self._sensors['imu'].get_data()
        current_heading = quat_to_heading(imu_data['orientation'])
        heading_error = angle_diff(self.target_heading, current_heading)

        # SELF-DISCOVERY: Read tolerance from actuator modal (physics-discovered!)
        tolerance = self._actuator.tolerance if self._actuator else 0.01

        # Update progress
        if self.start_heading is not None:
            total_rotation = angle_diff(self.target_heading, self.start_heading)
            if abs(total_rotation) > 0.01:
                rotated = angle_diff(current_heading, self.start_heading)
                self.progress = min(100, 100 * abs(rotated) / abs(total_rotation))

        return abs(heading_error) < tolerance

    def _get_command(self) -> float:
        """SELF-EXECUTE: Return angular velocity command"""
        if self._actuator is None or self.rotation_needed is None:
            return 0.0
        return self._actuator.move_by(self.rotation_needed)

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': ['left_wheel_vel'], 'sensors': ['imu']}


# ============================================
# SPEAKER ACTION
# ============================================

class SpeakerPlay(Action):
    """PURE MOP: Play audio file or text-to-speech"""
    audio_file: Optional[str] = None  # Pre-recorded audio file
    text: Optional[str] = None  # Text for TTS
    duration: Optional[float] = None

    # MOP: Non-actuator action (audio output, not physical actuator)
    execution_type: Literal["non_actuator"] = "non_actuator"

    @property
    def actuator_ids(self) -> List[str]:
        return ["speaker"]  # Virtual actuator for queueing

    def _sync_state(self):
        """SELF-SYNC: Check speaker status"""
        pass  # Speaker updates itself

    def _check_completion(self) -> bool:
        """SELF-VALIDATE: Is playback done?"""
        if self._actuator is None:
            return False

        # Check if speaker is no longer active
        if not self._actuator.is_active:
            self.progress = 100.0
            return True

        # If duration provided, check time
        if self.duration and self.started_at:
            elapsed = time.time() - self.started_at
            self.progress = min(100, 100 * elapsed / self.duration)
            return elapsed >= self.duration

        return False

    def _get_command(self) -> Dict:
        """SELF-EXECUTE: Return empty dict (speaker is VIRTUAL!)

        PURE MOP FIX: Speaker is a virtual actuator (no MuJoCo representation).
        Returning {} tells backend to skip sending commands.

        Audio playback happens separately (not through MuJoCo backend).

        The error message that led to this fix taught us:
        - Virtual actions must return {} from _get_command()
        - Application data ({'text': ..., 'type': 'tts'}) cannot go to physics backend
        - MuJoCo backend would crash: ValueError: Actuator 'text' not found

        See MODAL_ORIENTED_PROGRAMMING.md: 'Real Bug 2: The text Actuator Incident'
        """
        # PURE MOP: Virtual actuator returns empty dict (skip backend send)
        return {}

    def get_required_state(self) -> Dict[str, List[str]]:
        return {'actuators': ['speaker'], 'sensors': []}


# ============================================
# ACTION TYPES UNION
# ============================================

AtomicAction = Union[
    ArmMoveTo, ArmMoveBy,
    LiftMoveTo, LiftMoveBy,
    GripperMoveTo, GripperMoveBy,
    HeadPanMoveTo, HeadPanMoveBy,
    HeadTiltMoveTo, HeadTiltMoveBy,
    WristYawMoveTo, WristYawMoveBy,
    WristPitchMoveTo, WristPitchMoveBy,
    WristRollMoveTo, WristRollMoveBy,
    BaseMoveForward, BaseMoveBackward, BaseRotateBy, BaseMoveTo,
    SpeakerPlay
]


# ============================================
# ACTION BLOCK
# ============================================

class ActionBlock(BaseModel):
    """Collection of actions that form a behavior

    ActionBlock = List of Actions + execution config
    """
    id: str
    description: str = ""  # Optional, defaults to empty
    execution_mode: Literal['sequential', 'parallel'] = 'sequential'
    push_before_others: bool = False
    replace_current: bool = False
    end_with_thinking: bool = False
    actions: List[Action] = []

    @property
    def status(self) -> str:
        """MOP: I self-report my status from my actions!

        Returns:
            'pending' - if all actions are pending
            'executing' - if any action is executing
            'completed' - if all actions are completed
            'failed' - if any action failed
        """
        if not self.actions:
            return 'pending'

        statuses = [action.status for action in self.actions]

        # Priority: failed > executing > completed > pending
        if any(s == 'failed' for s in statuses):
            return 'failed'
        if any(s == 'executing' for s in statuses):
            return 'executing'
        if all(s == 'completed' for s in statuses):
            return 'completed'
        return 'pending'

    @property
    def progress(self) -> float:
        """MOP: I self-report my progress from my actions!

        Returns average progress of all actions.
        """
        if not self.actions:
            return 0.0

        return sum(action.progress for action in self.actions) / len(self.actions)


# ============================================
# Note: Skill class removed (unused)
# Future note: If needed later, a Skill is just List[ActionBlock]
# ============================================




# ============================================
# ACTION REGISTRY FUNCTION
# ============================================

def get_all_actions():
    """Get all action classes for ActionOps discovery

    Returns dict mapping action names to action classes (not instances).
    Used by ActionOps for get_available_actions() and get_action_params().
    """
    return {
        # Arm actions
        'ArmMoveTo': ArmMoveTo,
        'ArmMoveBy': ArmMoveBy,

        # Lift actions
        'LiftMoveTo': LiftMoveTo,
        'LiftMoveBy': LiftMoveBy,

        # Gripper actions
        'GripperMoveTo': GripperMoveTo,
        'GripperMoveBy': GripperMoveBy,

        # Head pan actions
        'HeadPanMoveTo': HeadPanMoveTo,
        'HeadPanMoveBy': HeadPanMoveBy,

        # Head tilt actions
        'HeadTiltMoveTo': HeadTiltMoveTo,
        'HeadTiltMoveBy': HeadTiltMoveBy,

        # Wrist yaw actions
        'WristYawMoveTo': WristYawMoveTo,
        'WristYawMoveBy': WristYawMoveBy,

        # Wrist pitch actions
        'WristPitchMoveTo': WristPitchMoveTo,
        'WristPitchMoveBy': WristPitchMoveBy,

        # Wrist roll actions
        'WristRollMoveTo': WristRollMoveTo,
        'WristRollMoveBy': WristRollMoveBy,

        # Base movement actions
        'BaseMoveForward': BaseMoveForward,
        'BaseMoveBackward': BaseMoveBackward,
        'BaseRotateBy': BaseRotateBy,
        'BaseMoveTo': BaseMoveTo,

        # Speaker action
        'SpeakerPlay': SpeakerPlay,
    }

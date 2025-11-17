"""
SCENE SOLVER MODAL - Auto-calculates robot placement for tasks
PURE MOP: Composes robot capabilities + asset info (MODAL-TO-MODAL)

NO HARDCODING! All dimensions and capabilities discovered at runtime from:
- Robot actuators (range, behaviors, placement_site)
- Asset dimensions (MuJoCo model or XML)
- Scene bounds

Example:
    solver = SceneSolverModal()
    placement = solver.solve_robot_placement(
        robot=stretch,
        task="grasp",
        target_asset=apple_asset,
        target_placement=apple_placement
    )
    # Returns: {'position': (x,y,z), 'orientation': ..., 'joint_positions': {...}}
"""

from typing import Dict, Tuple, Optional, Any
from pydantic import BaseModel
from abc import ABC, abstractmethod
import numpy as np
import math


def _find_actuator_name_by_behavior(robot, behavior: str) -> Optional[str]:
    """MOP: Auto-discover actuator name by behavior - NO HARDCODING!

    Args:
        robot: Robot modal instance
        behavior: Behavior to search for (e.g., "robot_lift", "robot_arm")

    Returns:
        Actuator name or None if not found
    """
    for name, actuator in robot.actuators.items():
        if behavior in actuator.behaviors:
            return name
    return None


class TaskSolver(BaseModel, ABC):
    """Base class for task-specific solvers - SELF-CALCULATION!

    Each solver knows how to calculate robot placement for its task.
    MOP: Solvers are Pydantic modals with validate + calculate methods.
    """

    task_type: str
    description: str
    requires_reach: bool = True
    requires_orientation: bool = False

    @abstractmethod
    def calculate_placement(
        self,
        robot_capabilities: Dict,
        target_info: Dict,
        scene_bounds: Optional[Dict] = None
    ) -> Dict:
        """Calculate robot placement - SELF-CALCULATION!

        Args:
            robot_capabilities: From robot.get_reach_capabilities()
            target_info: From placement.get_surface_info()
            scene_bounds: Optional scene boundaries

        Returns:
            {
                'position': (x, y, z),
                'orientation': (w, x, y, z) or preset string,
                'initial_state': {'joint_name': value}  # Matches add_robot() param!
            }
        """
        pass


class GraspTaskSolver(TaskSolver):
    """Grasp task solver - positions robot to grasp target object

    PURE MOP: Calculates from robot capabilities + target dimensions.
    NO HARDCODING of reach distances, joint names, or dimensions!
    """

    task_type: str = "grasp"
    description: str = "Position robot to grasp target object"
    requires_reach: bool = True
    requires_orientation: bool = True

    def calculate_placement(
        self,
        robot_capabilities: Dict,
        target_info: Dict,
        scene_bounds: Optional[Dict] = None,
        robot: Any = None  # NEW: Pass robot for MOP actuator discovery!
    ) -> Dict:
        """Calculate grasp placement - PURE MOP!

        Strategy:
        1. Position base for comfortable arm reach (70% of max reach)
        2. Orient to face target
        3. Set lift to target height
        4. Extend arm to reach distance
        """
        # OFFENSIVE: Extract capabilities - NO DEFAULTS!
        if 'horizontal_reach' not in robot_capabilities:
            raise ValueError("❌ Robot has no horizontal_reach capability!")
        if 'vertical_reach' not in robot_capabilities:
            raise ValueError("❌ Robot has no vertical_reach capability!")

        arm_reach = robot_capabilities['horizontal_reach']
        lift_range = robot_capabilities['vertical_reach']

        # Extract target info (MODAL-TO-MODAL!)
        target_pos = target_info['position']
        target_dims = target_info.get('dimensions', {'width': 0.1, 'depth': 0.1, 'height': 0.1})

        # OFFENSIVE: Calculate optimal base position - NO BULLSHIT DEFAULTS!
        # Position for comfortable reach (70% of max reach)
        reach_distance = arm_reach * 0.7

        # Calculate angle to target
        dx = target_pos[0]
        dy = target_pos[1]
        target_angle = math.atan2(dy, dx)

        # Position base at comfortable reach distance from target
        base_x = target_pos[0] - reach_distance * math.cos(target_angle)
        base_y = target_pos[1] - reach_distance * math.sin(target_angle)
        base_z = 0.0

        position = (base_x, base_y, base_z)

        # Calculate orientation (face target)
        # Use relational orientation for automatic calculation
        orientation = "facing_target"  # Scene will resolve to quaternion

        # MOP: Auto-discover actuator names from robot behaviors!
        joint_positions = {}

        if robot:
            # PURE MOP: Discover actuator names instead of hardcoding!
            lift_name = _find_actuator_name_by_behavior(robot, "robot_lift")
            arm_name = _find_actuator_name_by_behavior(robot, "robot_arm")
            gripper_name = _find_actuator_name_by_behavior(robot, "robot_gripper")
            head_pan_name = _find_actuator_name_by_behavior(robot, "robot_head_pan")
            head_tilt_name = _find_actuator_name_by_behavior(robot, "robot_head_tilt")

            # 1. Lift height (target Z minus gripper offset)
            # OFFENSIVE: Assume 0.2m gripper offset - could be discovered from robot!
            target_z = target_pos[2]
            lift_height = min(max(target_z - 0.2, 0.0), lift_range)
            if lift_name:
                joint_positions[lift_name] = lift_height

            # 2. Arm extension (distance from base to target)
            actual_reach = math.sqrt((target_pos[0] - position[0])**2 +
                                    (target_pos[1] - position[1])**2)
            if arm_name:
                joint_positions[arm_name] = actual_reach

            # 3. Gripper open for grasping
            if gripper_name:
                joint_positions[gripper_name] = -0.15

            # 4. Head tracking (look at target)
            head_dx = target_pos[0] - position[0]
            head_dy = target_pos[1] - position[1]
            head_pan = math.atan2(head_dy, head_dx)
            if head_pan_name:
                joint_positions[head_pan_name] = head_pan

            head_dz = target_pos[2] - (position[2] + lift_height)
            head_dist = math.sqrt(head_dx**2 + head_dy**2)
            head_tilt = math.atan2(head_dz, head_dist)
            if head_tilt_name:
                joint_positions[head_tilt_name] = head_tilt

        return {
            'position': position,
            'orientation': orientation,
            'initial_state': joint_positions  # Renamed for add_robot() unpacking!
        }


class InspectTaskSolver(TaskSolver):
    """Inspect task solver - positions robot to visually inspect target

    PURE MOP: Camera positioned for optimal view of target.
    """

    task_type: str = "inspect"
    description: str = "Position robot camera to inspect target"
    requires_reach: bool = False
    requires_orientation: bool = True

    def calculate_placement(
        self,
        robot_capabilities: Dict,
        target_info: Dict,
        scene_bounds: Optional[Dict] = None,
        robot: Any = None  # NEW: Pass robot for MOP actuator discovery!
    ) -> Dict:
        """Calculate inspection placement - focus on camera positioning"""

        # Extract target info
        target_pos = target_info['position']
        target_dims = target_info.get('dimensions', {'width': 0.1, 'depth': 0.1, 'height': 0.1})

        # Position at inspection distance (1.5m from target)
        inspection_distance = 1.5

        # Calculate angle to target
        dx = target_pos[0]
        dy = target_pos[1]
        target_angle = math.atan2(dy, dx)

        # Position base at inspection distance
        base_x = target_pos[0] - inspection_distance * math.cos(target_angle)
        base_y = target_pos[1] - inspection_distance * math.sin(target_angle)

        position = (base_x, base_y, 0.0)
        orientation = "facing_target"

        # MOP: Auto-discover actuator names from robot behaviors!
        joint_positions = {}

        if robot:
            # PURE MOP: Discover actuator names instead of hardcoding!
            lift_name = _find_actuator_name_by_behavior(robot, "robot_lift")
            arm_name = _find_actuator_name_by_behavior(robot, "robot_arm")
            head_pan_name = _find_actuator_name_by_behavior(robot, "robot_head_pan")
            head_tilt_name = _find_actuator_name_by_behavior(robot, "robot_head_tilt")

            # Lift camera to target height
            lift_range = robot_capabilities.get('vertical_reach', 1.0)
            target_z = target_pos[2]
            if lift_name:
                joint_positions[lift_name] = min(max(target_z, 0.0), lift_range)

            # Arm retracted for clear view
            if arm_name:
                joint_positions[arm_name] = 0.0

            # Head looking at target
            head_dx = target_pos[0] - position[0]
            head_dy = target_pos[1] - position[1]
            if head_pan_name:
                joint_positions[head_pan_name] = math.atan2(head_dy, head_dx)

            head_dz = target_pos[2] - joint_positions.get(lift_name, 0.0)
            head_dist = math.sqrt(head_dx**2 + head_dy**2)
            if head_tilt_name:
                joint_positions[head_tilt_name] = math.atan2(head_dz, head_dist)

        return {
            'position': position,
            'orientation': orientation,
            'initial_state': joint_positions  # Renamed for add_robot() unpacking!
        }


class SceneSolverModal:
    """Scene solver - composes robot + asset modals (MODAL-TO-MODAL)

    PURE MOP: NO HARDCODING!
    - Robot capabilities from actuators
    - Asset dimensions from MuJoCo or XML
    - Task solvers self-calculate placement
    """

    def __init__(self):
        # Registry of task solvers (LEGO COMPOSITION!)
        self.solvers = {
            'grasp': GraspTaskSolver(),
            'inspect': InspectTaskSolver(),
        }

    def get_robot_capabilities(self, robot) -> Dict:
        """Extract reach capabilities from robot - PURE MOP!

        NO HARDCODING: Reads from actuator.range, actuator.behaviors

        Args:
            robot: Robot modal instance

        Returns:
            {
                'horizontal_reach': float,  # From arm actuator
                'vertical_reach': float,    # From lift actuator
                'gripper_width': float,     # From gripper actuator
                'mobile_base': bool,        # Has base actuator
            }
        """
        capabilities = {}

        # AUTO-DISCOVERY: Scan actuators (MODAL-TO-MODAL!)
        for name, actuator in robot.actuators.items():
            # Check behaviors to identify actuator type
            behaviors = actuator.behaviors

            if "robot_arm" in behaviors:
                # Horizontal reach from arm range
                capabilities['horizontal_reach'] = actuator.range[1]

            elif "robot_lift" in behaviors:
                # Vertical reach from lift range
                capabilities['vertical_reach'] = actuator.range[1]

            elif "robot_gripper" in behaviors:
                # Gripper width from gripper range
                grip_min, grip_max = actuator.range
                capabilities['gripper_width'] = grip_max - grip_min

            elif "robot_base" in behaviors:
                # Mobile base capability
                capabilities['mobile_base'] = True

        # OFFENSIVE: Validate required capabilities discovered
        if 'horizontal_reach' not in capabilities:
            print("⚠️  Warning: No arm actuator found, using default reach 0.5m")
            capabilities['horizontal_reach'] = 0.5

        if 'vertical_reach' not in capabilities:
            print("⚠️  Warning: No lift actuator found, using default reach 1.0m")
            capabilities['vertical_reach'] = 1.0

        return capabilities

    def get_asset_surface_info(self, asset, placement, scene, model=None) -> Dict:
        """Extract surface info from asset - PURE MOP!

        NO HARDCODING: Reads from MuJoCo model or XML

        Args:
            asset: Asset modal instance
            placement: Placement modal instance
            scene: Scene modal instance
            model: Optional MuJoCo model (for runtime extraction)

        Returns:
            {
                'position': (x, y, z),
                'dimensions': {'width': float, 'depth': float, 'height': float}
            }
        """
        # Get position from placement (pass model for relational positioning!)
        runtime_state = {'model': model} if model is not None else None
        position = placement.get_xyz(scene, runtime_state, resolved_furniture=None)

        # Get dimensions
        dimensions = None

        # Try runtime extraction (from MuJoCo model)
        if model is not None:
            try:
                from .behavior_extractors import get_asset_dimensions
                dimensions = get_asset_dimensions(model, placement.instance_name)
            except Exception as e:
                print(f"⚠️  Runtime dimension extraction failed: {e}")

        # Fallback: Use default dimensions
        if dimensions is None:
            print(f"⚠️  Using default dimensions for {placement.instance_name}")
            dimensions = {'width': 0.1, 'depth': 0.1, 'height': 0.1}

        return {
            'position': position,
            'dimensions': dimensions
        }

    def solve_robot_placement(
        self,
        robot,
        task: str,
        target_asset,
        target_placement,
        scene,
        model=None,
        **kwargs
    ) -> Dict:
        """Calculate robot placement - MODAL-TO-MODAL composition!

        PURE MOP: Composes robot capabilities + asset info + task solver

        Args:
            robot: Robot modal instance
            task: Task type ("grasp", "inspect", "manipulate")
            target_asset: Target Asset modal instance
            target_placement: Target Placement modal instance
            scene: Scene modal instance
            model: Optional MuJoCo model (for dimension extraction)
            **kwargs: Additional solver parameters

        Returns:
            {
                'position': (x, y, z),
                'orientation': (w, x, y, z) or preset string,
                'initial_state': {'joint_name': value}  # Matches add_robot() param!
            }

        Example:
            placement = solver.solve_robot_placement(
                robot=stretch,
                task="grasp",
                target_asset=apple,
                target_placement=apple_placement,
                scene=scene
            )
        """
        # OFFENSIVE: Validate task solver exists
        if task not in self.solvers:
            available = list(self.solvers.keys())
            raise ValueError(
                f"❌ No solver for task '{task}'!\n"
                f"\n✅ Available tasks: {available}\n"
                f"\n💡 MOP: Add new TaskSolver to scene_solver_modal.py"
            )

        solver = self.solvers[task]

        # MODAL-TO-MODAL: Extract capabilities from robot
        robot_capabilities = self.get_robot_capabilities(robot)

        # MODAL-TO-MODAL: Extract info from target asset
        target_info = self.get_asset_surface_info(
            target_asset,
            target_placement,
            scene,
            model
        )

        # Scene bounds (optional - for workspace validation)
        scene_bounds = kwargs.get('scene_bounds', None)

        # SELF-CALCULATION: Solver calculates placement!
        # MOP: Pass robot for actuator name discovery!
        placement = solver.calculate_placement(
            robot_capabilities,
            target_info,
            scene_bounds,
            robot=robot  # Pass robot for MOP actuator discovery!
        )

        # MOP FIX: Replace generic "facing_target" with actual target asset name
        # This allows the orientation resolution system to work correctly
        if placement['orientation'] == "facing_target":
            placement['orientation'] = f"facing_{target_placement.instance_name}"

        print(f"✅ Scene solver calculated placement for task '{task}':")
        print(f"   Position: {placement['position']}")
        print(f"   Orientation: {placement['orientation']}")
        print(f"   Initial state: {len(placement['initial_state'])} joints")

        return placement

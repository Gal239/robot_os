"""
REACHABILITY MODAL - Decoupled IK for Stretch Robot
MODAL-ORIENTED: Geometry-based inverse kinematics

Pattern: Stretch has decoupled kinematics (each DOF independent)
- Base (x,y): 2D positioning
- Lift (z): Vertical height
- Arm: Horizontal extension
- Wrist: Pure rotation
- Gripper: Width adjustment

This is SIMPLE geometry, not complex polynomial solving!
"""

import numpy as np
from typing import Dict, Tuple


class ReachabilityModal:
    """Decoupled IK solver for Stretch robot (MOP!)

    Stretch robot design makes IK trivial:
    - Mobile base provides gross positioning
    - Prismatic lift provides vertical reach
    - Prismatic arm provides horizontal reach
    - Revolute wrists provide orientation

    No Jacobians, no iterative solvers - just geometry!
    """

    def __init__(self):
        # Stretch robot constants (from design)
        self.BASE_HEIGHT = 0.2  # Height of base above ground
        self.GRIPPER_OFFSET_FROM_ARM = 0.15  # Gripper extends beyond arm tip
        self.MIN_BASE_DISTANCE = 0.3  # Don't get too close (collision)
        self.MAX_ARM_REACH = 0.5  # Maximum arm extension

    def solve_for_grasp(self, target_pos: np.ndarray, object_width: float,
                       robot_config: Dict) -> Dict[str, float]:
        """Calculate joint values to position gripper at target

        Args:
            target_pos: [x, y, z] world position of grasp point
            object_width: Width of object (for gripper opening)
            robot_config: Robot configuration dict (for constraints)

        Returns:
            Dict of joint values for initial_state

        MOP: Returns data-driven joint values (no hardcoding!)
        """
        target_x, target_y, target_z = target_pos

        # ================================================================
        # DECOUPLED IK - Each DOF independent!
        # ================================================================

        # 1. BASE POSITION (gross positioning)
        #    Position base so arm can reach target
        #    Keep safe distance to avoid collision
        base_x, base_y = self._solve_base_position(target_x, target_y)

        # 2. LIFT HEIGHT (vertical)
        #    Match target Z height (account for base height)
        lift_height = self._solve_lift_height(target_z)

        # 3. ARM EXTENSION (horizontal)
        #    Extend arm to reach target distance
        arm_extension = self._solve_arm_extension(base_x, base_y, target_x, target_y)

        # 4. GRIPPER OPENING (object-dependent)
        #    Open gripper to fit object + clearance
        gripper_opening = self._solve_gripper_opening(object_width)

        # 5. WRIST ORIENTATION (future enhancement)
        #    For now: neutral orientation (0, 0, 0)
        wrist_yaw, wrist_pitch, wrist_roll = 0.0, 0.0, 0.0

        # ================================================================
        # BUILD JOINT VALUES DICT
        # ================================================================

        joint_values = {
            # Base position (mobile base is special - not a joint!)
            "_base_position": [base_x, base_y, 0.0],  # Special key for base

            # Lift joint
            "joint_lift": lift_height,

            # Arm joints (prismatic tendon split across 4 joints)
            "joint_arm_l0": arm_extension / 4.0,
            "joint_arm_l1": arm_extension / 4.0,
            "joint_arm_l2": arm_extension / 4.0,
            "joint_arm_l3": arm_extension / 4.0,

            # Wrist joints (orientation)
            "joint_wrist_yaw": wrist_yaw,
            "joint_wrist_pitch": wrist_pitch,
            "joint_wrist_roll": wrist_roll,

            # Gripper joint (parallel jaw)
            "joint_gripper_finger_left": gripper_opening / 2.0,
            "joint_gripper_finger_right": -gripper_opening / 2.0,
        }

        return joint_values

    def _solve_base_position(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """Calculate base (x,y) position for gross positioning

        Strategy: Position base slightly behind target so arm can reach
        """
        # Calculate direction to target
        distance_to_target = np.sqrt(target_x**2 + target_y**2)

        if distance_to_target < 0.01:  # Target at origin
            return (0.0, 0.0)

        # Position base MIN_BASE_DISTANCE away from target
        # This leaves room for arm extension
        offset_distance = max(self.MIN_BASE_DISTANCE, distance_to_target - self.MAX_ARM_REACH)

        # Calculate base position (move back along line to target)
        scale = offset_distance / distance_to_target
        base_x = target_x * (1.0 - scale)
        base_y = target_y * (1.0 - scale)

        return (base_x, base_y)

    def _solve_lift_height(self, target_z: float) -> float:
        """Calculate lift joint value for vertical positioning

        Simple: target_z minus base height
        """
        # Lift joint 0 = gripper at base height
        # Lift joint N = gripper at base height + N
        lift_height = target_z - self.BASE_HEIGHT - self.GRIPPER_OFFSET_FROM_ARM

        # Clamp to physical limits (0.0 to 1.1m for Stretch)
        lift_height = max(0.0, min(1.1, lift_height))

        return lift_height

    def _solve_arm_extension(self, base_x: float, base_y: float,
                            target_x: float, target_y: float) -> float:
        """Calculate arm extension to reach target

        Horizontal distance from base to target
        """
        # Calculate distance from base to target (in XY plane)
        dx = target_x - base_x
        dy = target_y - base_y
        distance = np.sqrt(dx**2 + dy**2)

        # Arm extension is simply this distance
        # Clamp to physical limits (0.0 to 0.52m for Stretch)
        arm_extension = max(0.0, min(0.52, distance))

        return arm_extension

    def _solve_gripper_opening(self, object_width: float) -> float:
        """Calculate gripper opening for object

        Args:
            object_width: Width of object to grasp

        Returns:
            Gripper opening (distance between fingers)
        """
        # Add clearance for contact (2cm = 0.02m)
        clearance = 0.02
        gripper_opening = object_width + clearance

        # Clamp to physical limits (0.0 to 0.1m for Stretch gripper)
        gripper_opening = max(0.0, min(0.1, gripper_opening))

        return gripper_opening

    def is_reachable(self, target_pos: np.ndarray, robot_config: Dict) -> bool:
        """Check if target position is within robot workspace

        Args:
            target_pos: [x, y, z] world position
            robot_config: Robot configuration

        Returns:
            True if reachable, False otherwise

        Future: Add workspace constraints checking
        """
        # For now: assume everything is reachable
        # Future: Check against workspace bounds
        target_x, target_y, target_z = target_pos

        # Simple bounds check
        horizontal_distance = np.sqrt(target_x**2 + target_y**2)

        # Maximum reach = MAX_ARM_REACH from base
        if horizontal_distance > self.MAX_ARM_REACH + 1.0:
            return False

        # Vertical bounds (0.0 to 1.5m above ground)
        if target_z < 0.0 or target_z > 1.5:
            return False

        return True


# === USAGE EXAMPLE ===

if __name__ == "__main__":
    """Test decoupled IK solver"""

    # Create modal
    ik_modal = ReachabilityModal()

    # Test grasp calculation
    target_pos = np.array([0.5, 0.0, 0.8])  # 50cm forward, 80cm high
    object_width = 0.05  # 5cm wide object
    robot_config = {}  # Empty for now

    # Solve IK
    joint_values = ik_modal.solve_for_grasp(target_pos, object_width, robot_config)

    print("IK Solution:")
    print("=" * 50)
    for joint_name, value in joint_values.items():
        if isinstance(value, list):
            print(f"  {joint_name}: {value}")
        else:
            print(f"  {joint_name}: {value:.4f}")

    # Check reachability
    reachable = ik_modal.is_reachable(target_pos, robot_config)
    print(f"\nReachable: {reachable}")

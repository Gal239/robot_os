#!/usr/bin/env python3
"""
CENTRAL KEYFRAME BUILDER - The ONLY place that generates keyframes for scene initialization

MOP Principle: Single source of truth for keyframe generation, order-independent.

Problem this solves:
- Before: Keyframe generation scattered across 3 files (170 lines)
- Before: Each robot generates keyframe independently (duplicate 'initial' keyframes)
- Before: No clear ownership of keyframe generation

After: ONE central KeyframeBuilder class, clean architecture.

Note: Keyframes only include robot base freejoint (7 DOF).
Freejoint objects (apples, blocks, etc.) initialize from <body pos=...> tags naturally.
This is correct MuJoCo behavior!
"""

from typing import List, Tuple, Optional


class KeyframeBuilder:
    """Central keyframe generation - PURE MOP!

    Replaces:
    - robot_modal.py:render_initial_state_xml() (87 lines)
    - xml_resolver.py:merge_with_robot() keyframe logic (40 lines)

    This is the SINGLE SOURCE OF TRUTH for initial state keyframes.
    """

    @staticmethod
    def build_initial_keyframe(
        robot,
        robot_position: Tuple[float, float, float],
        robot_orientation: Tuple[float, float, float, float] = None,
        freejoint_objects: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    ) -> str:
        """Generate initial keyframe for robot - OFFENSIVE!

        This is the ONLY function that should generate 'initial' keyframes.
        Generates keyframe with robot base freejoint ONLY (7 DOF).

        Freejoint objects (apples, blocks, etc.) are NOT included in keyframe!
        They initialize from their <body pos="..."> tags naturally (MuJoCo behavior).

        Args:
            robot: Robot modal with actuators (RobotModal instance)
            robot_position: Robot (x,y,z) from Placement SSOT
            robot_orientation: Robot quaternion (w,x,y,z) from Placement SSOT (optional, defaults to identity)
            freejoint_objects: List of freejoint objects (for tracking only, not used in qpos)

        Returns:
            Keyframe XML with robot base freejoint (7 DOF) + ctrl (10 actuators)

        Raises:
            AssertionError: If robot or robot_position is None (OFFENSIVE - no defaults!)

        Example:
            Robot at (0, 0.8, 0) facing north:
            qpos = "0 0.8 0 1 0 0 0"  (7 DOFs - robot base only!)
                   ^^^^^^^^^^^^^^^^
                   Robot freejoint (position + quaternion)
        """
        # OFFENSIVE: No None defaults - crash if missing required data!
        assert robot is not None, "Robot cannot be None!"
        assert robot_position is not None, "Robot position cannot be None!"

        # SIMPLIFIED: No qpos needed! MuJoCo auto-initializes ALL bodies from <body pos="..."> tags
        # Robot position and orientation are already set in the worldbody XML
        # We only need ctrl values for actuators (arm, lift, gripper, etc.)

        # Build ctrl from robot actuators
        actuator_order = [
            "left_wheel", "right_wheel", "lift", "arm", "wrist_yaw",
            "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt"
        ]

        ctrl_values = []
        for act_name in actuator_order:
            if act_name in robot.actuators:
                ctrl_values.append(str(robot.actuators[act_name].position))
            else:
                ctrl_values.append("0")

        ctrl_str = " ".join(ctrl_values)

        # Generate keyframe XML (NO qpos - MuJoCo uses body positions from XML!)
        keyframe_xml = f'<keyframe><key name="initial" ctrl="{ctrl_str}"/></keyframe>'
        return keyframe_xml

    @staticmethod
    def build_multi_robot_keyframe(
        robots: List[Tuple[object, Tuple[float, float, float]]],
        freejoint_objects: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    ) -> str:
        """Generate initial keyframe for MULTIPLE robots + freejoint objects

        This extends build_initial_keyframe() to handle multiple robots.
        Each robot contributes 7 DOFs for its freejoint.

        Args:
            robots: List of (robot_modal, position) tuples
            freejoint_objects: List of (body_name, position, quaternion) for other freejoint objects

        Returns:
            Complete <keyframe> XML with all robots + all freejoint objects

        Example:
            Robot1 at (0, 0, 0) + Robot2 at (2, 0, 0) + Apple at (1, 0, 1):
            qpos = "0 0 0 1 0 0 0  2 0 0 1 0 0 0  1 0 1 1 0 0 0"
                   ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^
                   Robot1 (7)      Robot2 (7)      Apple (7)
                   Total: 21 DOFs
        """
        assert robots, "robots list cannot be empty!"
        assert all(r is not None and p is not None for r, p in robots), "All robots and positions must be non-None!"

        qpos_parts = []
        ctrl_parts = []

        # Process each robot
        for robot, robot_position in robots:
            # Add robot freejoint qpos (7 DOFs)
            x, y, z = robot_position
            qw, qx, qy, qz = 1, 0, 0, 0  # Identity quaternion

            robot_qpos = [str(x), str(y), str(z), str(qw), str(qx), str(qy), str(qz)]
            qpos_parts.extend(robot_qpos)

            print(f"  [KeyframeBuilder] Robot '{robot.name}' qpos (7 DOFs): {' '.join(robot_qpos)}")

            # Add robot ctrl
            actuator_order = [
                "left_wheel", "right_wheel", "lift", "arm", "wrist_yaw",
                "wrist_pitch", "wrist_roll", "gripper", "head_pan", "head_tilt"
            ]

            for act_name in actuator_order:
                if act_name in robot.actuators:
                    ctrl_parts.append(str(robot.actuators[act_name].position))
                else:
                    ctrl_parts.append("0")

        # Add all freejoint objects
        if freejoint_objects:
            print(f"  [KeyframeBuilder] Adding {len(freejoint_objects)} freejoint objects:")
            for obj_name, obj_pos, obj_quat in freejoint_objects:
                obj_x, obj_y, obj_z = obj_pos
                obj_qw, obj_qx, obj_qy, obj_qz = obj_quat

                obj_qpos = [
                    str(obj_x), str(obj_y), str(obj_z),
                    str(obj_qw), str(obj_qx), str(obj_qy), str(obj_qz)
                ]
                qpos_parts.extend(obj_qpos)

                print(f"    - {obj_name}: pos=({obj_x}, {obj_y}, {obj_z})")

        # Build final strings
        qpos_str = " ".join(qpos_parts)
        ctrl_str = " ".join(ctrl_parts)
        total_dofs = len(qpos_parts)

        print(f"  [KeyframeBuilder] Total qpos size: {total_dofs} DOFs ({len(robots)} robots + {len(freejoint_objects) if freejoint_objects else 0} objects)")

        keyframe_xml = f'<keyframe><key name="initial" qpos="{qpos_str}" ctrl="{ctrl_str}"/></keyframe>'

        print(f"  [KeyframeBuilder] ✅ Generated multi-robot 'initial' keyframe with {total_dofs} DOFs")

        return keyframe_xml

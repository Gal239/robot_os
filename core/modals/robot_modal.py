"""
Robot Modal - Self-building robot modal that dynamically loads components
Modal-Oriented Programming: Modal IS the code, builds itself from robot type
Offensive Programming: No classes, just dataclasses and utility functions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import importlib


@dataclass
class Robot:
    """Robot modal - dynamically builds itself from robot type"""
    name: str
    robot_type: str  # "stretch", "fetch", "pr2", etc.
    sensors: Dict[str, Any] = field(default_factory=dict)  # Actual sensor modal instances
    actuators: Dict[str, Any] = field(default_factory=dict)  # Actual actuator modal instances
    actions: Dict[str, Any] = field(default_factory=dict)  # Actual action modal instances
    views: Dict[str, Any] = field(default_factory=dict)  # View instances
    _view_factory: Optional[callable] = None  # Factory function that creates views on-demand
    initial_position: Tuple[float, float, float] = (0, 0, 0)
    initial_orientation: Tuple[float, float, float] = (0, 0, 0)
    xml_path: str = ""  # Path to robot XML file, set by robot_ops
    _built: bool = field(default=False, init=False)  # Track if robot has been built
    _view_deps: Dict[str, List[str]] = field(default_factory=dict, init=False)  # View dependencies
    _is_tracked: bool = field(default=True, init=False)  # Robots ALWAYS need full physics collisions!

    def __post_init__(self):
        """Initialize dependency tracking"""
        # View dependencies will be set when views are loaded
        # The robot_ops loads views which contain their own dependencies
        self._view_deps = {}

    # ============================================
    # FLUENT API - Start full, remove what you don't need
    # ============================================

    def remove_sensors(self, sensor_names: List[str]) -> "Robot":
        """Remove sensors and their dependent views"""
        for sensor in sensor_names:
            # Remove sensor
            self.sensors.pop(sensor, None)

            # NOTE: Views are managed by ViewAggregator, not Robot
            # The sensor removal is enough - views won't be created without sensors

        return self  # For chaining

    def remove_actuators(self, actuator_names: List[str]) -> "Robot":
        """Remove actuators and their dependent views"""
        for actuator in actuator_names:
            # Remove actuator
            self.actuators.pop(actuator, None)

            # NOTE: Views are managed by ViewAggregator, not Robot
            # The actuator removal is enough - views won't be created without actuators

        return self  # For chaining

    def remove_actions(self, action_names: List[str]) -> "Robot":
        """Remove actions and their views"""
        for action in action_names:
            self.actions.pop(action, None)

            # NOTE: Views are managed by ViewAggregator, not Robot
            # The action removal is enough - views won't be created without actions

        return self

    def basic_views_only(self) -> "Robot":
        """Keep only direct sensor/actuator/action views, remove analysis and composite views"""
        # Keep only views that end with _view (sensor/actuator/action views)
        # Remove everything else (analysis and composite views)
        basic_view_names = [k for k in self.views.keys() if k.endswith('_view')]
        self.views = {k: v for k, v in self.views.items() if k in basic_view_names}
        return self

    def keep_views(self, view_names: List[str]) -> "Robot":
        """Keep only specified views"""
        self.views = {k: v for k, v in self.views.items() if k in view_names}
        return self

    def sensors_only(self, sensor_names: List[str]) -> "Robot":
        """Keep only specified sensors (removes others)"""
        all_sensors = list(self.sensors.keys())
        sensors_to_remove = [s for s in all_sensors if s not in sensor_names]
        return self.remove_sensors(sensors_to_remove)

    def actuators_only(self, actuator_names: List[str]) -> "Robot":
        """Keep only specified actuators (removes others)"""
        all_actuators = list(self.actuators.keys())
        actuators_to_remove = [a for a in all_actuators if a not in actuator_names]
        return self.remove_actuators(actuators_to_remove)

    def no_views(self) -> "Robot":
        """Remove all views"""
        self.views = {}
        return self

    def render_data(self) -> Dict:
        """MOP: Simple data representation - just names"""
        return {
            "name": self.name,
            "type": self.robot_type,
            "sensors": list(self.sensors.keys()),
            "actuators": list(self.actuators.keys()),
            "actions": list(self.actions.keys()),
            "views": list(self.views.keys()),
            "initial_position": list(self.initial_position),
            "initial_orientation": list(self.initial_orientation)
        }

    def render_full_data(self) -> Dict:
        """MOP: Full data with nested renders from each modal"""
        data = {
            "name": self.name,
            "type": self.robot_type,
            "initial_position": list(self.initial_position),
            "initial_orientation": list(self.initial_orientation)
        }

        # Render each modal's data
        # OFFENSIVE: Trust all modals implement render_data()
        if self.sensors:
            data["sensors"] = {k: v.render_data() for k, v in self.sensors.items()}

        if self.actuators:
            data["actuators"] = {k: v.render_data() for k, v in self.actuators.items()}

        if self.actions:
            data["actions"] = {k: v.render_data() for k, v in self.actions.items()}

        if self.views:
            data["views"] = {k: v.render_data() for k, v in self.views.items()}

        return data

    def get_specs(self) -> Dict[str, Any]:
        """MOP: Get robot specifications for dynamic positioning calculations

        Returns actuator ranges + geometry constants for calculating robot positions.
        Used by ops.get_robot_info() to enable dynamic positioning (no hardcoding!)

        Returns:
            dict with:
                - actuators: {name: {min_position, max_position, unit, ...}}
                - geometry: {gripper_length, base_to_arm_offset, base_height}
                - margins: {reach_safety, placement_safety, grasp_threshold}
                - comfortable_pct: {arm_reach, lift_height}
        """
        # Import constants from robot-specific actuator module
        if self.robot_type == "stretch":
            try:
                from .stretch import actuator_modals as stretch_actuators

                # DYNAMIC: Extract geometry from XML (MOP!)
                geometry = stretch_actuators._extract_robot_geometry()

                # Combine with physics-derived constants
                constants = {
                    # From XML (dynamic extraction)
                    "gripper_length": geometry["gripper_length"],
                    "base_height": geometry["base_height"],
                    "base_to_arm_offset": geometry["base_to_arm_offset"],
                    # From physics testing (constants)
                    "gripper_grasp_threshold": stretch_actuators.GRIPPER_GRASP_THRESHOLD,
                    "arm_comfortable_reach_pct": stretch_actuators.ARM_COMFORTABLE_REACH_PCT,
                    "lift_comfortable_height_pct": stretch_actuators.LIFT_COMFORTABLE_HEIGHT_PCT,
                    "reach_safety_margin": stretch_actuators.REACH_SAFETY_MARGIN,
                    "placement_safety_margin": stretch_actuators.PLACEMENT_SAFETY_MARGIN,
                }
            except ImportError:
                # Fallback if can't import (shouldn't happen)
                constants = {}
        else:
            constants = {}

        # Extract actuator specs (ranges, etc.)
        # Only include position actuators (have range) - skip velocity actuators
        actuator_specs = {}
        for name, actuator in self.actuators.items():
            # Skip actuators without render_data() (shouldn't happen but be safe)
            if not hasattr(actuator, 'render_data'):
                continue

            data = actuator.render_data()

            # Only include actuators with range (position actuators)
            # Velocity actuators don't have range, skip them
            if "range" not in data:
                continue

            # OFFENSIVE: Position actuators MUST have these fields!
            try:
                actuator_specs[name] = {
                    "min_position": data["range"][0],
                    "max_position": data["range"][1],
                    "unit": data["unit"],
                    "type": data["type"],
                }
            except (KeyError, IndexError) as e:
                raise KeyError(
                    f"Position actuator '{name}' missing required field!\n"
                    f"Expected: range, unit, type\n"
                    f"Got: {list(data.keys())}\n"
                    f"Error: {e}"
                )

        # OFFENSIVE: Expect constants to be populated (no defaults!)
        if not constants:
            raise ValueError(
                f"Robot constants not loaded for type '{self.robot_type}'!\n"
                f"Expected geometry/margins constants from actuator_modals.\n"
                f"Check if robot type '{self.robot_type}' is supported."
            )

        return {
            "robot_type": self.robot_type,
            "actuators": actuator_specs,
            "geometry": {
                "gripper_length": constants["gripper_length"],  # OFFENSIVE!
                "base_to_arm_offset": constants["base_to_arm_offset"],  # OFFENSIVE!
                "base_height": constants["base_height"],  # OFFENSIVE!
            },
            "margins": {
                "reach_safety": constants["reach_safety_margin"],  # OFFENSIVE!
                "placement_safety": constants["placement_safety_margin"],  # OFFENSIVE!
                "grasp_threshold": constants["gripper_grasp_threshold"],  # OFFENSIVE!
            },
            "comfortable_pct": {
                "arm_reach": constants["arm_comfortable_reach_pct"],  # OFFENSIVE!
                "lift_height": constants["lift_comfortable_height_pct"],  # OFFENSIVE!
            }
        }

    def apply_initial_state(self, model, data, initial_state: Dict[str, float]):
        """Apply initial actuator states to MuJoCo - MOP! Robot knows itself!

        Robot modal knows how to apply its own initial state to the backend.
        Handles both joint and tendon actuators correctly.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            initial_state: Dict of actuator_name -> target_value

        Example:
            robot.apply_initial_state(model, data, {"arm": 0.0, "lift": 0.76, "gripper": 0.04})
        """
        import mujoco

        print("  [Applying initial_state directly to joint qpos (INSTANT!)...]")

        for actuator_name, target_value in initial_state.items():
            # Get actuator modal (MOP: actuator knows how to move!)
            actuator = self.actuators[actuator_name]

            # Get actuator ID in MuJoCo
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if act_id < 0:
                continue  # Actuator not found in compiled model

            # Get transmission type and handle accordingly
            trntype = model.actuator_trntype[act_id]
            clamped = actuator.move_to(target_value)  # Actuator clamps to range

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:  # Direct joint control
                jnt_id = model.actuator_trnid[act_id][0]
                qpos_adr = model.jnt_qposadr[jnt_id]
                data.qpos[qpos_adr] = clamped
                print(f"    {actuator_name} (joint): qpos[{qpos_adr}] = {clamped:.3f}")

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:  # Tendon control (e.g., arm telescope)
                # Get tendon ID and all wraps (joints) in the tendon
                tendon_id = model.actuator_trnid[act_id][0]
                tendon_adr = model.tendon_adr[tendon_id]
                tendon_num = model.tendon_num[tendon_id]

                # Set ALL joints in the tendon (synchronized movement!)
                for i in range(tendon_num):
                    # Get joint ID from wrap (MuJoCo stores tendon wraps)
                    jnt_id = model.wrap_objid[tendon_adr + i]
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    # For telescoping arm with coef=1, all joints move equally
                    data.qpos[qpos_adr] = clamped
                    print(f"    {actuator_name} (tendon joint {i}): qpos[{qpos_adr}] = {clamped:.3f}")

            else:
                # OFFENSIVE: Crash if unsupported transmission type
                raise ValueError(
                    f"Unsupported actuator transmission type for '{actuator_name}'!\n"
                    f"  trntype = {trntype} ({mujoco.mjtTrn(trntype).name})\n"
                    f"  Supported: mjTRN_JOINT (0), mjTRN_TENDON (3)\n"
                    f"  This is a bug - add support for this transmission type!"
                )

            # Also set ctrl for consistency
            data.ctrl[act_id] = clamped

        # Forward kinematics to update positions (NO physics steps needed!)
        mujoco.mj_forward(model, data)

        # Zero velocities (robot is stationary)
        data.qvel[:] = 0

    def get_qpos_range(self, model):
        """MOP: Robot knows its own qpos range in the compiled model!

        Returns the (start, end) indices for this robot's qpos.
        This includes the base freejoint + all actuator joints.

        Args:
            model: MuJoCo model

        Returns:
            tuple: (qpos_start, qpos_end) indices

        Example:
            start, end = robot.get_qpos_range(model)
            robot_qpos = data.qpos[start:end]  # Robot's configuration
        """
        import mujoco

        # Find robot's base body (base_link for Stretch)
        robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        if robot_body_id < 0:
            raise ValueError(f"Robot base body 'base_link' not found in model!")

        # Robot qpos starts at base_link's freejoint
        robot_jnt_id = model.body_jntadr[robot_body_id]
        qpos_start = model.jnt_qposadr[robot_jnt_id]

        # Robot qpos ends at end of model (robot is last in qpos)
        # This assumes robot is added AFTER all scene objects (floor, table, apple, etc.)
        qpos_end = model.nq

        return (qpos_start, qpos_end)

    def freeze_during_settling(self, model, data):
        """MOP: Robot knows how to freeze itself during settling!

        Settling is for OBJECTS (apples, blocks), not for robot!
        Robot is explicitly placed and configured - it should stay FROZEN.

        This method returns a snapshot of the robot's current qpos that can
        be restored after each physics step during settling.

        Args:
            model: MuJoCo model
            data: MuJoCo data

        Returns:
            numpy array: Saved robot qpos (base + all joints)

        Example:
            saved_qpos = robot.freeze_during_settling(model, data)
            start, end = robot.get_qpos_range(model)
            # During settling:
            for _ in range(100):
                mujoco.mj_step(model, data)
                data.qpos[start:end] = saved_qpos  # Restore robot!
        """
        qpos_start, qpos_end = self.get_qpos_range(model)

        # Save robot's current configuration (base + ALL joints)
        saved_robot_qpos = data.qpos[qpos_start:qpos_end].copy()

        # Log what we're preserving
        robot_base = data.qpos[qpos_start:qpos_start+7]
        print(f"  [Robot] FREEZING during settling qpos[{qpos_start}:{qpos_end}]: "
              f"pos=({robot_base[0]:.3f}, {robot_base[1]:.3f}, {robot_base[2]:.3f}), "
              f"quat=({robot_base[3]:.3f}, {robot_base[4]:.3f}, {robot_base[5]:.3f}, {robot_base[6]:.3f})")

        return saved_robot_qpos

    def render_json(self) -> str:
        """MOP: Robot as JSON"""
        return json.dumps(self.render_data(), indent=2)

    def render_xml(self) -> str:
        """MOP: Robot as MuJoCo XML body element"""
        x, y, z = self.initial_position
        roll, pitch, yaw = self.initial_orientation

        # Use the xml_path set by robot_ops
        assert self.xml_path, "XML path must be set by robot_ops"

        return f"""<body name="{self.name}" pos="{x} {y} {z}" euler="{roll} {pitch} {yaw}">
      <include file="{self.xml_path}"/>
    </body>"""

    # NOTE: render_initial_state_xml() method REMOVED!
    # Keyframe generation is now centralized in keyframe_builder.py (KeyframeBuilder class)
    # This fixes:
    # - Order-dependent compilation bug (robot first vs apple first)
    # - Multi-robot scenarios (was generating duplicate 'initial' keyframes)
    # - Missing freejoint object initialization (freejoint_objects param was ignored!)
    #
    # See: core/modals/keyframe_builder.py and core/modals/xml_resolver.py:merge_with_robot()

    def render_summary(self) -> str:
        """MOP: Robot as text summary"""
        lines = [
            f"Robot: {self.name} ({self.robot_type})",
            f"Sensors: {len(self.sensors)} - {', '.join(list(self.sensors.keys())[:5])}{'...' if len(self.sensors) > 5 else ''}",
            f"Actuators: {len(self.actuators)} - {', '.join(list(self.actuators.keys())[:5])}{'...' if len(self.actuators) > 5 else ''}",
            f"Actions: {len(self.actions)} - {', '.join(list(self.actions.keys())[:5])}{'...' if len(self.actions) > 5 else ''}",
            f"Position: ({self.initial_position[0]:.2f}, {self.initial_position[1]:.2f}, {self.initial_position[2]:.2f})"
        ]
        return "\n".join(lines)

    def get_data(self) -> Dict[str, Any]:
        """I know my complete state - OFFENSIVE & MODAL-ORIENTED

        Aggregates data from ALL actuators and sensors.
        Matches pattern established in other modals (room, asset, scene, reward).
        OFFENSIVE: Crashes if actuators/sensors lack get_data() - good, reveals bugs.

        🎓 EDUCATIONAL: Returns dict with helpful suggestions when wrong keys accessed.
        """
        data = {
            "name": self.name,
            "robot_type": self.robot_type,
            "initial_position": list(self.initial_position),
            "initial_orientation": list(self.initial_orientation)  # MOP FIX: Was copy-paste bug!
        }

        # Aggregate actuator data - OFFENSIVE, no checks
        data["actuators"] = {
            name: actuator.get_data()
            for name, actuator in self.actuators.items()
        }

        # Aggregate sensor data - OFFENSIVE, no checks
        data["sensors"] = {
            name: sensor.get_data()
            for name, sensor in self.sensors.items()
        }

        # 🎓 Wrap in educational dict that suggests fixes
        return _RobotDataDict(data)

    def get_reach_capabilities(self) -> Dict[str, Any]:
        """Extract reach capabilities from actuators - PURE MOP!

        NO HARDCODING: Discovers capabilities from actuator.range and actuator.behaviors.
        AUTO-DISCOVERY: Scans ALL actuators and extracts physical limits.

        Returns:
            {
                'horizontal_reach': float,  # From arm actuator range (meters)
                'vertical_reach': float,    # From lift actuator range (meters)
                'gripper_width': float,     # From gripper actuator range (radians)
                'mobile_base': bool,        # Has base actuator
                'workspace_bounds': Dict    # Calculated reachable workspace
            }

        Example:
            >>> robot = create_robot("stretch", "stretch_1")
            >>> caps = robot.get_reach_capabilities()
            >>> caps['horizontal_reach']
            0.52  # Discovered from arm actuator!
        """
        capabilities = {}

        # AUTO-DISCOVERY: Scan actuators (MODAL-TO-MODAL!)
        for name, actuator in self.actuators.items():
            # Check behaviors to identify actuator type
            behaviors = actuator.behaviors

            if "robot_arm" in behaviors:
                # Horizontal reach from arm range
                capabilities['horizontal_reach'] = actuator.range[1]
                capabilities['arm_unit'] = actuator.unit

            elif "robot_lift" in behaviors:
                # Vertical reach from lift range
                capabilities['vertical_reach'] = actuator.range[1]
                capabilities['lift_unit'] = actuator.unit

            elif "robot_gripper" in behaviors:
                # Gripper width from gripper range
                grip_min, grip_max = actuator.range
                capabilities['gripper_width'] = grip_max - grip_min
                capabilities['gripper_range'] = actuator.range
                capabilities['gripper_unit'] = actuator.unit

            elif "robot_base" in behaviors:
                # Mobile base capability
                capabilities['mobile_base'] = True

        # Calculate workspace bounds (if arm + lift available)
        if 'horizontal_reach' in capabilities and 'vertical_reach' in capabilities:
            h_reach = capabilities['horizontal_reach']
            v_reach = capabilities['vertical_reach']

            capabilities['workspace_bounds'] = {
                'x_min': -h_reach,
                'x_max': h_reach,
                'y_min': -h_reach,
                'y_max': h_reach,
                'z_min': 0.0,
                'z_max': v_reach
            }

        return capabilities

    def get_viewable_components(self) -> Dict[str, Tuple[Any, str, str]]:
        """I know what views need - MODAL-ORIENTED

        Returns dict of {component_id: (modal, view_type, modal_category)} for view creation.
        ViewAggregator calls this instead of digging into robot.sensors/robot.actuators.

        MOP-CORRECT: Robot controls what's exposed AND provides type metadata.
        NO external system needs to check robot internals!

        Returns:
            Dict mapping component IDs to (modal, view_type, modal_category) tuples
            - modal: Sensor/Actuator modal instance
            - view_type: "video", "video_and_data", or "data" (3-type system!)
            - modal_category: "sensor" or "actuator" (for directory organization)
        """
        from .stretch.sensors_modals import NavCamera, D405Camera
        from .camera_modal import CameraModal

        components = {}

        # Sensors with type metadata (3-TYPE SYSTEM!)
        for sensor_id, sensor_modal in self.sensors.items():
            # TYPE 1: Cameras are video-only (pixels ARE the data)
            if isinstance(sensor_modal, (NavCamera, D405Camera, CameraModal)):
                components[sensor_id] = (sensor_modal, "video", "sensor")
            # TYPE 2: All other sensors get data + video
            else:
                components[sensor_id] = (sensor_modal, "video_and_data", "sensor")

        # Actuators with type metadata (TYPE 2: data + video)
        for actuator_id, actuator_modal in self.actuators.items():
            components[actuator_id] = (actuator_modal, "video_and_data", "actuator")

        return components

    def _generate_behavior_from_actuator(self, actuator_name: str, actuator, behavior: str) -> Dict:
        """Generate behavior spec from actuator characteristics

        MODAL-ORIENTED: Infers properties from actuator's range, unit, etc.
        """
        # Map actuator names to human-readable descriptions
        descriptions = {
            "robot_base": "Mobile base with differential drive",
            "robot_gripper": "Robot gripper/end effector",
            "robot_arm": "Telescoping arm extension",
            "robot_lift": "Vertical lift mechanism",
            "robot_head_pan": "Head pan rotation",
            "robot_head_tilt": "Head tilt rotation",
            "robot_wrist_yaw": "Wrist yaw rotation",
            "robot_wrist_pitch": "Wrist pitch rotation",
            "robot_wrist_roll": "Wrist roll rotation",
            "robot_speaker": "Audio output speaker"
        }

        # Map actuator names to primary property names
        property_names = {
            "robot_base": "position",
            "robot_gripper": "aperture",
            "robot_arm": "extension",
            "robot_lift": "height",
            "robot_head_pan": "angle_rad",
            "robot_head_tilt": "angle_rad",
            "robot_wrist_yaw": "angle_rad",
            "robot_wrist_pitch": "angle_rad",
            "robot_wrist_roll": "angle_rad",
            "robot_speaker": "volume"
        }

        description = descriptions.get(behavior, f"{actuator_name.replace('_', ' ').title()} control")
        primary_prop = property_names.get(behavior, "position")

        # Build properties from actuator characteristics
        properties = {}

        # Primary property (position, angle, aperture, etc.)
        # SINGLE SOURCE OF TRUTH: actuator.range from XML joint limits!
        min_val, max_val = actuator.range if hasattr(actuator, 'range') else (0.0, 1.0)
        unit = actuator.unit if hasattr(actuator, 'unit') else ""
        natural_range = max_val - min_val  # Physical range from XML joint limits

        properties[primary_prop] = {
            "description": f"{primary_prop.replace('_', ' ').title()} value",
            "unit": unit,
            "natural_range": natural_range,  # NEW! For target-based rewards
            "states": {},  # Could infer from range
            "default": None
        }

        # Add boolean derived properties for certain behaviors
        if behavior in ["robot_gripper"]:
            properties["closed"] = {
                "description": "True when gripper closed",
                "unit": "boolean",
                "states": {},
                "default": None
            }
            properties["holding"] = {
                "description": "True when gripping object",
                "unit": "boolean",
                "states": {},
                "default": None
            }
        elif behavior in ["robot_arm", "robot_lift"]:
            properties["extended" if "arm" in behavior else "raised"] = {
                "description": f"True when {'extended' if 'arm' in behavior else 'raised'} past threshold",
                "unit": "percentage",
                "states": {},
                "default": 0.3 if "arm" in behavior else 0.5
            }
        elif behavior == "robot_base":
            properties["rotation"] = {
                "description": "Rotation angle around Z axis",
                "unit": "radians",  # Note: extracted as DEGREES in behavior_extractors.py!
                "natural_range": 360.0,  # NEW! Full rotation in DEGREES (matches extraction)
                "states": {},
                "default": None
            }
            properties["at_location"] = {
                "description": "True when base near target XY",
                "unit": "meters",
                # NO natural_range - scene-dependent! Same as distance_to
                "states": {"0.5": "Close", "1.0": "Near"},
                "default": 0.5
            }

        return {
            "description": description,
            "properties": properties
        }

    def _generate_spatial_behavior(self, behavior: str) -> Dict:
        """Generate spatial behavior spec (position tracking)

        MODAL-ORIENTED: Spatial behaviors are generic across all components.
        """
        return {
            "description": f"Spatial position tracking for {behavior.replace('_spatial', '')}",
            "properties": {
                "position": {
                    "description": "XYZ position in world frame",
                    "unit": "meters",
                    # NO natural_range - absolute position has no target!
                    "states": {},
                    "default": None
                },
                "distance_to": {
                    "description": "Distance to target position",
                    "unit": "meters",
                    # NO natural_range - scene-dependent! Discover at runtime from scene bounds
                    "states": {},
                    "default": None
                }
            }
        }

    def _generate_behavior_from_sensor(self, sensor_name: str, sensor, behavior: str) -> Dict:
        """Generate behavior spec from sensor characteristics

        MODAL-ORIENTED: Infers properties from sensor type and capabilities.
        """
        descriptions = {
            "vision": "Visual sensing and object detection",
            "thermal_vision": "Thermal imaging and heat detection",
            "distance_sensing": "Distance measurement and obstacle detection",
            "motion_sensing": "Motion and acceleration sensing",
            "tactile": "Touch and force sensing",
            "audio_sensing": "Sound capture and localization"
        }

        # Generate properties based on sensor behavior type
        properties = {}

        if behavior == "vision":
            properties["target_visible"] = {
                "description": "True when target in view",
                "unit": "boolean",
                "states": {},
                "default": None
            }
            properties["target_distance"] = {
                "description": "Distance to visible target",
                "unit": "meters",
                "states": {},
                "default": None
            }
        elif behavior == "distance_sensing":
            properties["min_distance"] = {
                "description": "Minimum detected distance",
                "unit": "meters",
                "states": {},
                "default": None
            }
            properties["obstacle_detected"] = {
                "description": "True when obstacle within threshold",
                "unit": "boolean",
                "states": {},
                "default": None
            }
        elif behavior == "motion_sensing":
            properties["acceleration"] = {
                "description": "Linear acceleration magnitude",
                "unit": "m/s^2",
                "states": {},
                "default": None
            }
            properties["angular_velocity"] = {
                "description": "Angular velocity magnitude",
                "unit": "rad/s",
                "states": {},
                "default": None
            }
        elif behavior == "tactile":
            properties["force"] = {
                "description": "Applied force",
                "unit": "Newtons",
                "states": {},
                "default": None
            }
            properties["contact"] = {
                "description": "True when contact detected",
                "unit": "boolean",
                "states": {},
                "default": None
            }
        elif behavior == "audio_sensing":
            properties["sound_detected"] = {
                "description": "True when sound above threshold",
                "unit": "boolean",
                "states": {},
                "default": None
            }
            properties["sound_direction"] = {
                "description": "Direction of sound source",
                "unit": "radians",
                "states": {},
                "default": None
            }
        elif behavior == "robot_base":
            # Special case: robot_base from odometry sensor (composite behavior)
            properties["position"] = {
                "description": "Base XYZ position",
                "unit": "meters",
                # NO natural_range - scene-dependent!
                "states": {},
                "default": None
            }
            properties["rotation"] = {
                "description": "Rotation angle around Z axis",
                "unit": "radians",  # Note: extracted as DEGREES in behavior_extractors.py!
                "natural_range": 360.0,  # Full rotation in DEGREES (matches extraction)
                "states": {},
                "default": None
            }
            properties["at_location"] = {
                "description": "True when base near target XY",
                "unit": "meters",
                # NO natural_range - scene-dependent!
                "states": {"0.5": "Close", "1.0": "Near"},
                "default": 0.5
            }

        return {
            "description": descriptions.get(behavior, f"{sensor_name.replace('_', ' ').title()} sensing"),
            "properties": properties
        }

    def create_robot_asset_package(self) -> Dict:
        """
        AUTO-GENERATE robot asset definition from actuators/sensors

        MODAL-ORIENTED: Scans ALL actuators and their declared behaviors.
        No hardcoding! Reads from ActuatorComponent.behaviors list.

        Returns:
            {
                "behaviors": {...},      # ALL robot behaviors from actuators
                "components": {...},     # Component definitions
                "state_provider": func,  # Callable that returns state dict
                "xml_path": str         # Path to robot XML
            }
        """

        # 1. AUTO-GENERATE BEHAVIORS - Scan ALL actuators AND sensors!
        behaviors = {}

        # Scan actuators
        for actuator_name, actuator in self.actuators.items():
            # Each actuator declares its behaviors
            for behavior in actuator.behaviors:
                # Skip spatial behaviors - they're generic position tracking
                if behavior.endswith('_spatial'):
                    if behavior not in behaviors:
                        behaviors[behavior] = self._generate_spatial_behavior(behavior)
                    continue

                # Generate behavior spec from actuator characteristics
                if behavior not in behaviors:
                    behaviors[behavior] = self._generate_behavior_from_actuator(
                        actuator_name, actuator, behavior
                    )

        # Scan sensors (NEW!)
        for sensor_name, sensor in self.sensors.items():
            # Only process sensors that have behaviors (trackable sensors)
            if hasattr(sensor, 'behaviors') and sensor.behaviors:
                for behavior in sensor.behaviors:
                    # Skip spatial behaviors - already handled above
                    if behavior.endswith('_spatial'):
                        if behavior not in behaviors:
                            behaviors[behavior] = self._generate_spatial_behavior(behavior)
                        continue

                    # Generate sensor behavior spec
                    if behavior not in behaviors:
                        behaviors[behavior] = self._generate_behavior_from_sensor(
                            sensor_name, sensor, behavior
                        )

        # OFFENSIVE validation: ALL declared behaviors must be generated!
        declared_behaviors = set()
        for actuator in self.actuators.values():
            declared_behaviors.update(actuator.behaviors)
        for sensor in self.sensors.values():
            if hasattr(sensor, 'behaviors'):
                declared_behaviors.update(sensor.behaviors)

        generated_behaviors = set(behaviors.keys())
        missing = declared_behaviors - generated_behaviors

        if missing:
            raise RuntimeError(
                f"GENERATION INCOMPLETE!\n"
                f"  Robot: {self.robot_type}\n"
                f"  Actuators declare: {sorted(declared_behaviors)}\n"
                f"  Generated: {sorted(generated_behaviors)}\n"
                f"  MISSING: {sorted(missing)}\n"
                f"\n"
                f"  FIX: Update Robot._generate_behavior_from_actuator() to handle all behavior types!"
            )

        # 2. MAP ACTUATORS → COMPONENTS
        # MOP: Read everything from actuator modals (no external config needed!)
        components = {}

        # AUTO-DISCOVER ALL ACTUATORS (MOP!) - No hardcoded list!
        for actuator_name, actuator in self.actuators.items():
            components[actuator_name] = {
                "name": actuator_name,
                "behaviors": actuator.behaviors,  # MOP: Direct from modal!
                "geom_names": actuator.geom_names,  # MOP: Direct from modal!
                "joint_names": actuator.joint_names,
                "actuator_ref": actuator,
                "sync_mode": actuator.sync_mode,
                "placement_site": actuator.placement_site  # MOP: Direct from modal!
            }

        # 3. CREATE STATE PROVIDER (uses view.render_scene() for translation)
        robot_name = self.name
        robot_views = self.views  # Reference to robot's views

        def state_provider() -> Dict:
            """Convert actuator data → behavior properties via view.render_scene()

            NOTE: Actuators must be synced BEFORE calling this!
            - In simulation: ops.sync_actuators_from_mujoco()
            - In real: ops.sync_actuators_from_hardware()
            """
            state = {}

            for comp_name, comp_def in components.items():
                # Get the view for this component - MUST exist!
                view_name = f"{comp_name}_view"
                assert view_name in robot_views, f"View '{view_name}' not found for component '{comp_name}'"

                view = robot_views[view_name]
                # View uses actuator.get_data() - actuators already synced!
                comp_state = view.render_scene()

                state[f"{robot_name}.{comp_name}"] = comp_state

            return state

        # 4. RETURN COMPLETE PACKAGE
        return {
            "behaviors": behaviors,
            "components": components,
            "state_provider": state_provider,
            "xml_path": self.xml_path
        }


class _RobotDataDict(dict):
    """🎓 Educational dictionary for Robot data

    When you access a wrong key, it SUGGESTS what you might want instead.
    Doesn't auto-fix - you decide what to use!
    """

    # Map of common wrong keys → suggested correct keys
    SUGGESTIONS = {
        "type": ["robot_type", "name"],
        "position": ["initial_position"],
        "pos": ["initial_position"],
        "orientation": ["initial_orientation"],
        "orient": ["initial_orientation"],
    }

    def __getitem__(self, key):
        """Get item with educational error messages"""
        try:
            return super().__getitem__(key)
        except KeyError:
            # Key not found - suggest alternatives
            suggestions = self.SUGGESTIONS.get(key, [])
            available = list(self.keys())

            msg = f"\n❌ Key '{key}' not found in Robot data!\n\n"

            if suggestions:
                msg += "🎓 SUGGESTION:\n"
                msg += f"   Based on your input '{key}', you might want:\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    # ✅ PURE: Just show key names, NO data access!
                    msg += f"   Option {i}: robot_data['{suggestion}']\n"
                msg += "\n"

            msg += f"Available keys: {available}\n"

            if suggestions:
                msg += f"\n💡 TIP: Use robot_data['{suggestions[0]}'] instead\n"

            raise KeyError(msg)


# ============================================================================
# UTILITY: Percentage-Based initial_state Conversion - MOP!
# ============================================================================
def convert_initial_state_percentages(initial_state: Dict, robot: Robot) -> Dict:
    """Convert percentage strings to numeric values - PURE MOP UTILITY!

    Supports:
        {"arm": "100%"} → {"arm": 0.52} (max)
        {"arm": "50%"}  → {"arm": 0.26} (middle)
        {"arm": "0%"}   → {"arm": 0.0}  (min)
        {"arm": 0.3}    → {"arm": 0.3}  (unchanged)

    Args:
        initial_state: Dict with actuator names and values (numeric or percentage strings)
        robot: Robot modal instance with actuators

    Returns:
        Dict with all values converted to numeric (floats)

    Raises:
        ValueError: If percentage format is invalid
    """
    converted = {}

    for actuator_name, value in initial_state.items():
        if isinstance(value, str) and value.endswith('%'):
            # Parse percentage: "50%" → 50.0
            try:
                percent = float(value.rstrip('%'))
            except ValueError:
                raise ValueError(f"Invalid percentage format for '{actuator_name}': {value}")

            # Clamp to valid range [0, 100]
            if percent < 0 or percent > 100:
                print(f"⚠️  {actuator_name}: Percentage {percent}% out of range [0, 100], clamping")
                percent = max(0.0, min(100.0, percent))

            # Get actuator range and convert
            actuator = robot.actuators[actuator_name]
            min_val, max_val = actuator.range
            numeric_value = min_val + (percent / 100.0) * (max_val - min_val)
            converted[actuator_name] = numeric_value
        else:
            # Already numeric (int or float)
            converted[actuator_name] = float(value)

    return converted

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

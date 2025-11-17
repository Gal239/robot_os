"""
Robot Operations - Dynamic robot creation using registry functions
Offensive Programming: Clean, minimal, fully dynamic
"""

import sys
from pathlib import Path
import importlib

# MOP: Dual import support (works both internally + externally)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from simulation_center.core.modals.robot_modal import Robot
except ModuleNotFoundError:
    from core.modals.robot_modal import Robot

# Helper for dynamic imports (used by importlib below)
def _try_import(module_path):
    """Try external path first, fallback to internal"""
    try:
        return importlib.import_module(f"simulation_center.{module_path}")
    except ModuleNotFoundError:
        return importlib.import_module(module_path)


def create_robot(robot_type: str, name: str = None, sensors=None, task_hint=None) -> Robot:
    """Create robot with optional sensor/task configuration

    Args:
        robot_type: Robot type (e.g., "stretch")
        name: Robot name (defaults to robot_type)
        sensors: Sensor configuration:
            - None: All sensors (default)
            - List[str]: Only these sensors (e.g., ["nav_camera", "gripper_force"])
            - Dict[str, bool]: Enable/disable specific sensors
        task_hint: Auto-configure for task:
            - "manipulation": d405_camera, gripper_force, arm sensors
            - "navigation": nav_camera, lidar, odometry
            - "proprioceptive": No cameras, only joint/force sensors
            - "vision": All cameras

    Examples:
        robot = create_robot("stretch")  # All sensors
        robot = create_robot("stretch", sensors=["nav_camera", "lidar"])  # Only these
        robot = create_robot("stretch", task_hint="manipulation")  # Auto-config
    """

    # Default name is robot type
    if name is None:
        name = robot_type

    # Create empty robot
    robot = Robot(name=name, robot_type=robot_type)

    module_base = f"core.modals.{robot_type.replace('-', '_')}"

    # Dynamically load registry functions
    sensors_mod = _try_import(f"{module_base}.sensors_modals")
    all_sensors = sensors_mod.get_all_sensors()

    # Apply task hints (presets)
    if task_hint:
        sensors = _get_sensors_for_task(robot_type, task_hint)

    # Filter sensors based on configuration
    if sensors is not None:
        if isinstance(sensors, list):
            # List of sensor names to keep
            robot.sensors = {k: v for k, v in all_sensors.items() if k in sensors}
        elif isinstance(sensors, dict):
            # Dict of sensor_name -> enabled
            robot.sensors = {k: v for k, v in all_sensors.items() if sensors.get(k, True)}
        else:
            robot.sensors = all_sensors
    else:
        # No filter - all sensors
        robot.sensors = all_sensors

    actuators_mod = _try_import(f"{module_base}.actuator_modals")
    robot.actuators = actuators_mod.create_all_actuators()

    actions_mod = _try_import(f"{module_base}.action_modals")
    robot.actions = actions_mod.get_all_actions()

    # MOP-CORRECT: No view factory! ViewAggregator creates views on-demand from modals.
    # Views are EXTERNAL-ONLY (for UI, logging) - NOT for internal data flow!

    robot.xml_path = f"core/modals/{robot_type}/mujoco_assets/robot/{robot_type}.xml"

    return robot


def _get_sensors_for_task(robot_type: str, task: str) -> list:
    """Get sensor list for task - DISCOVERED from robot modals

    Each robot defines its own sensor presets in its modals.
    Ops discovers them dynamically - no hardcoding!
    """
    module_base = f"core.modals.{robot_type.replace('-', '_')}"
    sensors_mod = importlib.import_module(f"{module_base}.sensors_modals")

    # Check if robot has presets defined
    if hasattr(sensors_mod, 'get_sensor_presets'):
        presets = sensors_mod.get_sensor_presets()
        return presets.get(task, None)

    return None  # No presets for this robot


# ============================================
# USAGE EXAMPLES
# ============================================

if __name__ == "__main__":
    # 1. Create full robot (everything included)
    robot = create_robot("stretch")
    print("Full robot:")
    print(robot.render_summary())
    print(f"Total components: {len(robot.sensors)} sensors, {len(robot.actuators)} actuators, " +
          f"{len(robot.actions)} actions, {len(robot.views)} views")
    print()

    # 2. Navigation robot (remove unnecessary components)
    nav_robot = (create_robot("stretch", "nav_robot")
                  .remove_sensors(["gripper_force", "respeaker"])  # Don't need these
                  .remove_actuators(["wrist_roll", "wrist_pitch", "gripper"])  # Skip complex wrist/gripper
                  .basic_views_only())  # No complex analysis views
    print("Navigation robot:")
    print(nav_robot.render_summary())
    print(f"After removal: {len(nav_robot.sensors)} sensors, {len(nav_robot.views)} views")
    print()

    # 3. Manipulation robot (focused on grasping)
    manip_robot = (create_robot("stretch", "manip_robot")
                    .remove_sensors(["respeaker", "imu"])  # Don't need audio or IMU
                    .keep_views(["d405_camera_view", "gripper_force_view", "manipulation_grid", "d405_surface"]))
    print("Manipulation robot:")
    print(manip_robot.render_summary())
    print()

    # 4. Minimal robot (just base and camera)
    minimal_robot = (create_robot("stretch", "minimal_robot")
                     .sensors_only(["nav_camera"])
                     .actuators_only(["base", "base_rotation"])
                     .no_views())
    print("Minimal robot:")
    print(minimal_robot.render_summary())
    print()

    # 5. Custom robot (specific configuration)
    custom_robot = (create_robot("stretch", "custom_robot")
                    .remove_sensors(["respeaker"])
                    .remove_actuators(["speaker"])
                    .remove_actions(["scan_area"])
                    .basic_views_only())
    print("Custom robot:")
    print(f"Sensors: {list(custom_robot.sensors.keys())}")
    print(f"Views: {list(custom_robot.views.keys())[:5]}...")

    # Show how dependencies work
    print("\n--- Dependency Example ---")
    test_robot = create_robot("stretch", "test")
    print(f"Before removing nav_camera: nav_edges in views? {'nav_edges' in test_robot.views}")

    test_robot.remove_sensors(["nav_camera"])
    print(f"After removing nav_camera: nav_edges in views? {'nav_edges' in test_robot.views}")
    print(f"(nav_edges was automatically removed because it depends on nav_camera)")
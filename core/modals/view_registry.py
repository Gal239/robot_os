"""
VIEW REGISTRY - Self-documenting view discovery system
OFFENSIVE & SELF-CREATING: Naming conventions + runtime discovery

Pattern: Like asset registry, but for views
- Static: Naming conventions (documentation in code)
- Dynamic: Runtime discovery (what's actually available)
"""

from typing import Dict, List, Optional


# === NAMING CONVENTIONS (STATIC DOCUMENTATION) ===

VIEW_NAMING_CONVENTIONS = {
    "sensors": "{sensor_id}_sensor_view",
    "actuators": "{actuator_id}_actuator_view",
    "scene_assets": "asset_{asset_name}_view",
    "scene_room": "room_view",
    "rewards": "rewards_view",
    "runtime_status": "runtime_status",
    "action_queue": "action_queue"
}

VIEW_EXAMPLES = {
    "sensors": [
        "nav_camera_sensor_view",
        "d405_camera_sensor_view",
        "lidar_sensor_view",
        "imu_sensor_view",
        "gripper_force_sensor_view",
        "respeaker_sensor_view",
        "odometry_sensor_view"
    ],
    "actuators": [
        "gripper_actuator_view",
        "arm_actuator_view",
        "lift_actuator_view",
        "base_actuator_view",
        "head_pan_actuator_view",
        "head_tilt_actuator_view",
        "wrist_yaw_actuator_view",
        "wrist_pitch_actuator_view",
        "wrist_roll_actuator_view"
    ],
    "scene": [
        "room_view",
        "asset_table_view",
        "asset_door_view",
        "asset_cup_view"
    ],
    "system": [
        "rewards_view",
        "runtime_status",
        "action_queue"
    ]
}


# === RUNTIME DISCOVERY (DYNAMIC) ===

def list_available(views_dict: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Discover all available views grouped by category - DYNAMIC

    Args:
        views_dict: Dict of view_name -> view_data (from ops.engine.last_views)

    Returns:
        Dict categorizing views by type
    """
    categorized = {
        "sensors": [],
        "actuators": [],
        "scene": [],
        "system": [],
        "other": []
    }

    for view_name in views_dict.keys():
        if view_name.endswith('_sensor_view'):
            categorized["sensors"].append(view_name)
        elif view_name.endswith('_actuator_view'):
            categorized["actuators"].append(view_name)
        elif view_name.startswith('asset_') or view_name == 'room_view':
            categorized["scene"].append(view_name)
        elif view_name in ['rewards_view', 'runtime_status', 'action_queue']:
            categorized["system"].append(view_name)
        else:
            categorized["other"].append(view_name)

    return categorized


def list_sensors(views_dict: Dict[str, Dict]) -> List[str]:
    """List all sensor views - CONVENIENCE"""
    return [v for v in views_dict.keys() if v.endswith('_sensor_view')]


def list_actuators(views_dict: Dict[str, Dict]) -> List[str]:
    """List all actuator views - CONVENIENCE"""
    return [v for v in views_dict.keys() if v.endswith('_actuator_view')]


def list_scene(views_dict: Dict[str, Dict]) -> List[str]:
    """List all scene views - CONVENIENCE"""
    return [v for v in views_dict.keys()
            if v.startswith('asset_') or v == 'room_view']


def list_system(views_dict: Dict[str, Dict]) -> List[str]:
    """List all system views - CONVENIENCE"""
    return [v for v in views_dict.keys()
            if v in ['rewards_view', 'runtime_status', 'action_queue']]


def get_view_info(view_name: str) -> Optional[Dict]:
    """Get info about a view from naming convention - INFORMATIVE

    Args:
        view_name: View name (e.g., 'gripper_actuator_view')

    Returns:
        Dict with view type, component name, etc.
    """
    if view_name.endswith('_sensor_view'):
        component = view_name.replace('_sensor_view', '')
        return {
            'type': 'sensor',
            'component': component,
            'full_name': view_name,
            'convention': VIEW_NAMING_CONVENTIONS['sensors']
        }

    if view_name.endswith('_actuator_view'):
        component = view_name.replace('_actuator_view', '')
        return {
            'type': 'actuator',
            'component': component,
            'full_name': view_name,
            'convention': VIEW_NAMING_CONVENTIONS['actuators']
        }

    if view_name.startswith('asset_') and view_name.endswith('_view'):
        component = view_name.replace('asset_', '').replace('_view', '')
        return {
            'type': 'scene_asset',
            'component': component,
            'full_name': view_name,
            'convention': VIEW_NAMING_CONVENTIONS['scene_assets']
        }

    if view_name == 'room_view':
        return {
            'type': 'scene_room',
            'component': 'room',
            'full_name': view_name,
            'convention': VIEW_NAMING_CONVENTIONS['scene_room']
        }

    if view_name in ['rewards_view', 'runtime_status', 'action_queue']:
        return {
            'type': 'system',
            'component': view_name,
            'full_name': view_name,
            'convention': f"Fixed name: {view_name}"
        }

    return None


def print_conventions():
    """Print naming conventions - DOCUMENTATION"""
    print("=" * 80)
    print("VIEW NAMING CONVENTIONS")
    print("=" * 80)

    for category, convention in VIEW_NAMING_CONVENTIONS.items():
        print(f"\n{category.upper()}:")
        print(f"  Pattern: {convention}")
        if category in VIEW_EXAMPLES:
            print(f"  Examples:")
            for example in VIEW_EXAMPLES[category][:3]:
                print(f"    • {example}")


def print_available(views_dict: Dict[str, Dict]):
    """Print all available views - DISCOVERY"""
    categorized = list_available(views_dict)

    print("=" * 80)
    print("AVAILABLE VIEWS (Runtime Discovery)")
    print("=" * 80)

    for category, views in categorized.items():
        if views:
            print(f"\n{category.upper()} ({len(views)}):")
            for view in sorted(views):
                print(f"  • {view}")
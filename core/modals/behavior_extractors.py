"""
BEHAVIOR EXTRACTORS - Pure physics extraction from MuJoCo
Maps MuJoCo data → semantic properties from BEHAVIORS.json
OFFENSIVE & ELEGANT: One function per behavior, COMPLETE implementations
"""

import mujoco
import numpy as np
import json
import os
from typing import Dict, Any

# Load BEHAVIORS.json - SINGLE SOURCE OF TRUTH for defaults
_BEHAVIORS_PATH = os.path.join(os.path.dirname(__file__), "../behaviors/BEHAVIORS.json")
with open(_BEHAVIORS_PATH) as f:
    BEHAVIORS = json.load(f)

# Load ROTATION TOLERANCE from discovered_tolerances.json (PURE MOP!)
# Same tolerance used by actions and rewards for perfect alignment
_TOLERANCE_PATH = os.path.join(os.path.dirname(__file__), "stretch/discovered_tolerances.json")
with open(_TOLERANCE_PATH) as f:
    _tolerances = json.load(f)
    ROTATION_TOLERANCE = _tolerances["imu_rotation"]  # OFFENSIVE - crash if missing!


# ============================================================================
# SNAPSHOT WRAPPERS - Duck-typing for thread-safe MuJoCo data access
# ============================================================================

class ContactStruct:
    """Wrapper for individual contact - provides .geom1, .geom2 attributes"""
    def __init__(self, contact_dict: Dict[str, int]):
        self.geom1 = contact_dict['geom1']
        self.geom2 = contact_dict['geom2']


class ContactWrapper:
    """Wrapper for contact array - allows data.contact[i] indexing"""
    def __init__(self, contact_list: list):
        self._contacts = contact_list

    def __getitem__(self, i: int) -> ContactStruct:
        """Return ContactStruct for data.contact[i] access"""
        return ContactStruct(self._contacts[i])

    def __len__(self) -> int:
        return len(self._contacts)


class SnapshotData:
    """Duck-types as MjData - provides thread-safe snapshot access

    Background threads use this instead of live mjdata to avoid race conditions.
    Provides the same interface as MjData for behavior extractors.

    MOP: Uses duck-typing marker _is_snapshot instead of isinstance() checks!
    """
    def __init__(self, snapshot: Dict[str, Any], model: mujoco.MjModel):
        # MOP: Duck-typing marker (avoids isinstance checks!)
        self._is_snapshot = True

        # Handle incomplete snapshots (e.g., when robot is None during init)
        # Use .get() with None defaults so SnapshotData can be created
        # but will fail gracefully if behavior extractors try to access missing data

        # Arrays (copied from live data)
        self.qpos = snapshot.get('qpos')
        self.qvel = snapshot.get('qvel')
        self.xpos = snapshot.get('xpos')
        self.xmat = snapshot.get('xmat')
        self.xquat = snapshot.get('xquat')  # Quaternions for IMU-based rotation
        self.cvel = snapshot.get('cvel')
        self.site_xpos = snapshot.get('site_xpos')
        self.geom_xpos = snapshot.get('geom_xpos')  # Geom positions (for surface height calculations)

        # Contact data (thread-safe wrapper)
        self.ncon = snapshot.get('ncon', 0)
        self.contact = ContactWrapper(snapshot.get('contact', []))

        # Metadata
        self.time = snapshot.get('time', 0.0)

        # Keep model reference for mj_contactForce calls
        self._model = model
        self._snapshot = snapshot


# ============================================================================
# HELPER FUNCTIONS - Pure physics utilities & unit conversion
# ============================================================================

def _convert_to_human_unit(value: float, behavior: str, prop_name: str,
                          min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Convert raw physics value to human-friendly unit from BEHAVIORS.json

    Args:
        value: Raw physics value (radians, Newtons, meters, etc.)
        behavior: Behavior name (e.g., "hinged", "graspable")
        prop_name: Property name (e.g., "open", "held")
        min_val: Minimum physics value (for percentage conversion)
        max_val: Maximum physics value (for percentage conversion)

    Returns:
        Converted value in human-friendly units
    """
    # Get property metadata from BEHAVIORS.json
    prop_meta = BEHAVIORS[behavior]["properties"][prop_name]  # OFFENSIVE - crash if missing!
    unit = prop_meta["unit"]  # OFFENSIVE - crash if missing!

    # Convert to percentage if unit is "percentage"
    if unit == "percentage":
        if max_val == min_val:
            return 0.0 if value < max_val else 100.0
        percentage = ((value - min_val) / (max_val - min_val)) * 100.0
        return max(0.0, min(100.0, percentage))  # Clamp to 0-100

    # Otherwise keep as-is (Newtons, meters, rad/s, etc.)
    return value


def _get_body_position(model: mujoco.MjModel, data: mujoco.MjData, geom_name: str,
                       extraction_cache: dict = None) -> np.ndarray:
    """Get body position from geom name - OFFENSIVE

    PURE MOP: For OBJECTS, geoms are named (use geom lookup).
              For FURNITURE, bodies are named with asset name (use body lookup).

    Args:
        extraction_cache: REQUIRED cache from build_extraction_cache().
                         If None: CRASHES with clear error message.

    MOP: NO SLOW PATH! Cache is required. If missing, tell user to build it!
    """
    if extraction_cache is None:
        raise ValueError(
            "❌ PERFORMANCE ERROR: _get_body_position() called without extraction_cache!\n"
            "This would cause 745K redundant mj_name2id() calls (EXTREMELY SLOW).\n\n"
            "FIX: Build extraction cache in runtime_engine.py and pass it to extractors."
        )

    geom_name_to_id = extraction_cache['geom_name_to_id']
    body_name_to_id = extraction_cache['body_name_to_id']

    # Try geom first (works for OBJECTS which have named geoms)
    if geom_name in geom_name_to_id:
        geom_id = geom_name_to_id[geom_name]
        body_id = model.geom_bodyid[geom_id]
        return data.xpos[body_id]

    # FURNITURE fallback: Try as body name with common suffix stripping
    body_name_candidates = [
        geom_name,
        geom_name.replace("_top", "").replace("_surface", "").replace("_geom", "")
    ]

    for body_name in body_name_candidates:
        if body_name in body_name_to_id:
            body_id = body_name_to_id[body_name]
            return data.xpos[body_id]

    return np.array([0, 0, 0])


def _get_body_position_by_name(model: mujoco.MjModel, data: mujoco.MjData, body_name: str,
                                extraction_cache: dict = None) -> np.ndarray:
    """Get body position from body name directly - OFFENSIVE

    Args:
        extraction_cache: REQUIRED cache from build_extraction_cache().

    MOP: NO SLOW PATH! Cache is required.
    """
    if extraction_cache is None:
        raise ValueError(
            "❌ PERFORMANCE ERROR: _get_body_position_by_name() called without extraction_cache!\n"
            "FIX: Build extraction cache and pass it to extractors."
        )

    body_name_to_id = extraction_cache['body_name_to_id']
    if body_name in body_name_to_id:
        body_id = body_name_to_id[body_name]
        return data.xpos[body_id]
    return np.array([0, 0, 0])


def _get_contact_force(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int,
                       extraction_cache: dict = None) -> float:
    """Get total contact force for a geom - MOP: handles snapshots!

    NEW: Uses cached geom forces to eliminate 31K redundant contact scans!

    Snapshots don't have force data (requires real MjData), so return 0.0.
    This centralizes snapshot handling in ONE place instead of every extractor!

    MOP: Uses duck-typing (hasattr) instead of isinstance!

    Args:
        extraction_cache: Optional cache from build_extraction_cache().
                         If provided, uses cached force (FAST!)
                         If None, scans all contacts (SLOW - for backward compat)
    """
    if hasattr(data, '_is_snapshot'):
        return 0.0  # Snapshots don't compute forces (not critical for state)

    # MOP: Use cached force if available (eliminate 31K redundant scans!)
    if extraction_cache is not None:
        geom_forces = extraction_cache.get('geom_forces', {})
        return geom_forces.get(geom_id, 0.0)

    # SLOW PATH: Scan all contacts (backward compatibility)
    # This path should NEVER be used in production after cache is enabled!
    total_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        if contact.geom1 == geom_id or contact.geom2 == geom_id:
            c_array = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_array)
            total_force += float(np.linalg.norm(c_array[:3]))
    return total_force


def build_extraction_cache(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Build unified extraction cache - AUTO-DISCOVER ONCE at 30Hz (MOP principle!)

    Eliminates 6M+ redundant mj_name2id() calls by caching ALL name→ID mappings.
    NEW: Caches pairwise distances to eliminate 22x redundancy in distance calculations!

    Returns:
        extraction_cache: Dict with:
            # Contact data (O(ncon) scan instead of O(n²))
            'contact_pairs': Dict[(geom1_id, geom2_id), force] - contact forces
            'geom_contacts': Dict[geom_id, Set[contacted_geom_ids]] - adjacency list

            # Name→ID mappings (eliminate 6.15M mj_name2id calls!)
            'geom_name_to_id': Dict[str, int] - geom name → geom ID
            'body_name_to_id': Dict[str, int] - body name → body ID
            'joint_name_to_id': Dict[str, int] - joint name → joint ID
            'site_name_to_id': Dict[str, int] - site name → site ID
            'actuator_name_to_id': Dict[str, int] - actuator name → actuator ID

            # Reverse mappings (eliminate 132K list comprehensions!)
            'body_to_geoms': Dict[int, List[int]] - body ID → list of geom IDs

            # Geometry data (eliminate 850K size lookups!)
            'geom_sizes': Dict[int, np.ndarray] - geom ID → size array
            'geom_radii': Dict[int, float] - geom ID → effective radius

            # Pairwise distance cache (eliminate 22x redundancy!)
            'pairwise_distances': Dict[(geom_name1, geom_name2), distance] - lazy cache

    Built at state extraction frequency (30Hz), reused for all queries in that cycle.
    Physics runs at 200Hz but we only rebuild cache 30x/sec → 6x fewer cache builds!
    """
    cache = {}

    # Add pairwise distance cache (populated lazily by _get_surface_distance)
    cache['pairwise_distances'] = {}

    # ================================================================
    # 1. CONTACT DATA - O(ncon) scan (already optimized)
    # ================================================================
    contact_pairs = {}
    geom_contacts = {}
    geom_forces = {}  # NEW: Total contact force per geom (eliminate 31K redundant scans!)
    is_snapshot = hasattr(data, '_is_snapshot')

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = contact.geom1, contact.geom2

        force = 0.0
        if not is_snapshot:
            c_array = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, c_array)
            force = float(np.linalg.norm(c_array[:3]))

        pair_key = (min(geom1, geom2), max(geom1, geom2))
        if pair_key in contact_pairs:
            contact_pairs[pair_key] += force
        else:
            contact_pairs[pair_key] = force

        if geom1 not in geom_contacts:
            geom_contacts[geom1] = set()
        if geom2 not in geom_contacts:
            geom_contacts[geom2] = set()
        geom_contacts[geom1].add(geom2)
        geom_contacts[geom2].add(geom1)

        # NEW: Accumulate total force per geom (same loop, zero overhead!)
        geom_forces[geom1] = geom_forces.get(geom1, 0.0) + force
        geom_forces[geom2] = geom_forces.get(geom2, 0.0) + force

    cache['contact_pairs'] = contact_pairs
    cache['geom_contacts'] = geom_contacts
    cache['geom_forces'] = geom_forces  # NEW: Per-geom contact forces!

    # ================================================================
    # 2. NAME→ID MAPPINGS - Eliminate 6.15M mj_name2id() calls!
    # ================================================================
    # Build ALL name→ID mappings ONCE instead of millions of times

    # Geom names
    geom_name_to_id = {}
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name:  # Skip unnamed geoms
            geom_name_to_id[name] = i

    # Body names
    body_name_to_id = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_name_to_id[name] = i

    # Joint names
    joint_name_to_id = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_name_to_id[name] = i

    # Site names
    site_name_to_id = {}
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if name:
            site_name_to_id[name] = i

    # Actuator names (PERFORMANCE: eliminate 47.5x mj_name2id per step!)
    actuator_name_to_id = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_name_to_id[name] = i

    cache['geom_name_to_id'] = geom_name_to_id
    cache['body_name_to_id'] = body_name_to_id
    cache['joint_name_to_id'] = joint_name_to_id
    cache['site_name_to_id'] = site_name_to_id
    cache['actuator_name_to_id'] = actuator_name_to_id

    # ================================================================
    # 3. REVERSE MAPPINGS - Eliminate 132K list comprehensions!
    # ================================================================
    # Build body→geoms mapping ONCE instead of scanning all geoms repeatedly
    body_to_geoms = {}
    for body_id in range(model.nbody):
        body_to_geoms[body_id] = []

    for geom_id in range(model.ngeom):
        body_id = model.geom_bodyid[geom_id]
        body_to_geoms[body_id].append(geom_id)

    cache['body_to_geoms'] = body_to_geoms

    # ================================================================
    # 4. GEOMETRY DATA - Eliminate 850K size lookups!
    # ================================================================
    # Cache geom sizes and effective radii ONCE
    geom_sizes = {}
    geom_radii = {}

    for geom_id in range(model.ngeom):
        size = model.geom_size[geom_id]
        geom_sizes[geom_id] = size
        geom_radii[geom_id] = float(np.max(size))  # Effective radius

    cache['geom_sizes'] = geom_sizes
    cache['geom_radii'] = geom_radii

    return cache


# Backward compatibility alias
def build_contact_cache(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """DEPRECATED: Use build_extraction_cache() instead.

    This is a compatibility shim for old code that uses build_contact_cache().
    Returns the full extraction cache (includes contact data + name mappings).
    """
    return build_extraction_cache(model, data)


def _check_contact(model: mujoco.MjModel, data: mujoco.MjData, geom1_name: str, geom2_name: str,
                   contact_cache: dict = None) -> tuple:
    """Check if two geoms (or bodies) are in contact - returns (has_contact, force)

    Handles both geom names AND body names - if geom lookup fails, tries as body name
    and checks ALL geoms attached to that body. This allows config.json to use stable
    body names instead of unnamed geoms.

    Args:
        contact_cache: REQUIRED extraction cache from build_extraction_cache().
                      If None: CRASHES with clear error message.

    MOP: NO SLOW PATH! Cache is required. If missing, tell user to build it!
    """
    if contact_cache is None:
        raise ValueError(
            "❌ PERFORMANCE ERROR: _check_contact() called without extraction_cache!\n"
            "This would cause 6.15M redundant mj_name2id() calls (EXTREMELY SLOW).\n\n"
            "FIX: Build extraction cache in runtime_engine.py:\n"
            "  from simulation_center.core.modals.behavior_extractors import build_extraction_cache\n"
            "  extraction_cache = build_extraction_cache(model, data)\n\n"
            "Then pass it to all extraction functions."
        )

    geom_name_to_id = contact_cache['geom_name_to_id']
    body_name_to_id = contact_cache['body_name_to_id']
    body_to_geoms = contact_cache['body_to_geoms']

    # Try geom lookup, fallback to body if not found
    geom1_ids = []
    if geom1_name in geom_name_to_id:
        geom1_ids = [geom_name_to_id[geom1_name]]
    elif geom1_name in body_name_to_id:
        body1_id = body_name_to_id[geom1_name]
        geom1_ids = body_to_geoms.get(body1_id, [])

    geom2_ids = []
    if geom2_name in geom_name_to_id:
        geom2_ids = [geom_name_to_id[geom2_name]]
    elif geom2_name in body_name_to_id:
        body2_id = body_name_to_id[geom2_name]
        geom2_ids = body_to_geoms.get(body2_id, [])

    if not geom1_ids or not geom2_ids:
        return False, 0.0

    # Use cached contact pairs for O(1) lookup
    contact_pairs = contact_cache['contact_pairs']
    total_force = 0.0
    has_contact = False

    for g1 in geom1_ids:
        for g2 in geom2_ids:
            pair_key = (min(g1, g2), max(g1, g2))
            if pair_key in contact_pairs:
                has_contact = True
                total_force += contact_pairs[pair_key]

    return has_contact, float(total_force)


def _point_in_box(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> bool:
    """Check if point is inside axis-aligned box - OFFENSIVE"""
    half_size = size / 2.0
    return all(
        center[i] - half_size[i] <= point[i] <= center[i] + half_size[i]
        for i in range(3)
    )


def _iter_all_components(all_assets):
    """Iterate through all components - SINGLE SOURCE OF TRUTH for asset access!

    MOP: Unified interface - all assets have .components (Component instances)
    If interface changes, fix HERE only! No more changing 3+ places!

    Yields:
        (asset_name, component) tuples
    """
    for asset_name, asset in all_assets.items():
        for component in asset.components.values():
            yield asset_name, component


# ============================================================================
# BEHAVIOR EXTRACTORS - One per behavior in BEHAVIORS.json
# ============================================================================

def extract_hinged(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                   asset_config: Dict = None) -> Dict[str, Any]:
    """
    Extract hinged behavior: angle, velocity, open, closed, opening

    From: Joint position and velocity
    """
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos = float(data.qpos[model.jnt_qposadr[joint_id]])
    qvel = float(data.qvel[model.jnt_dofadr[joint_id]])

    # Get joint range from config for unit conversion
    min_angle, max_angle = 0.0, 1.57  # Default range
    if asset_config and "joint_ranges" in asset_config:
        joint_ranges = asset_config["joint_ranges"]
        if joint_name in joint_ranges:
            min_angle, max_angle = joint_ranges[joint_name]

    # Convert to human-friendly units (percentage for "open")
    open_percentage = _convert_to_human_unit(qpos, "hinged", "open", min_angle, max_angle)

    return {
        "angle": qpos,  # Keep raw radians for angle
        "velocity": qvel,  # Keep raw rad/s for velocity
        "open": open_percentage,  # PERCENTAGE (0-100)!
        "closed": qpos < 0.1,  # Near zero
        "opening": abs(qvel) > 0.01  # Moving
    }


def extract_graspable(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                      all_assets: Dict = None, contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract graspable behavior: geometry + contact_force + held + height + velocity

    Returns:
        object_width: meters (X dimension)
        object_depth: meters (Y dimension)
        object_height: meters (Z dimension)
        object_radius: approximate radius (max of half-sizes)
        contact_force: total contact force
        height: Z position in world
        velocity: linear velocity magnitude

    From: Body position and contact forces with GRIPPER GEOMS ONLY

    IMPORTANT: "held" now checks contact specifically with robot gripper geoms,
    not total contact (which includes floor, table, walls). This prevents false
    positives when object falls on floor.

    Args:
        contact_cache: Pre-built contact cache for O(1) lookups
    """
    if not component.geom_names:
        return {}

    geom_name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return {}

    body_id = model.geom_bodyid[geom_id]
    height = float(data.xpos[body_id][2])

    # Calculate total contact force (all contacts - for debugging)
    # MOP: Uses cached force (eliminates redundant contact scans!)
    total_contact_force = _get_contact_force(model, data, geom_id, contact_cache)

    # Extract object dimensions (PURE MOP!)
    half_sizes = model.geom_size[geom_id]
    object_width = float(half_sizes[0] * 2)
    object_depth = float(half_sizes[1] * 2)
    object_height_dim = float(half_sizes[2] * 2)
    object_radius = float(np.max(half_sizes))

    # Calculate held_by_{gripper} for each gripper (relational property)
    # Creates dynamic properties like: held_by_stretch.gripper, held_by_robot2.gripper
    result = {
        "object_width": object_width,
        "object_depth": object_depth,
        "object_height": object_height_dim,
        "object_radius": object_radius,
        "contact_force": float(total_contact_force),  # ALL contacts (floor, table, gripper, etc.)
        "height": height,
        "velocity": float(np.linalg.norm(data.cvel[body_id][:3]))  # Linear velocity
    }

    if all_assets:
        for asset_name, asset in all_assets.items():
            # ALL assets have .components - no checks needed!
            for comp_name, comp in asset.components.items():
                # Check if this component is a robot gripper
                if "robot_gripper" in comp.behaviors:
                    # Calculate contact force with THIS specific gripper
                    gripper_contact_force = 0.0
                    for gripper_geom_name in comp.geom_names:
                        has_contact, force = _check_contact(model, data, geom_name, gripper_geom_name, contact_cache)
                        if has_contact:
                            gripper_contact_force += force

                    # Create dynamic property: held_by_stretch.gripper
                    gripper_asset_name = f"{asset_name}.{comp_name}" if '.' not in asset_name else asset_name
                    result[f"held_by_{gripper_asset_name}"] = float(gripper_contact_force)

    return result


def extract_sliding(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract sliding behavior: position, velocity, extended, sliding

    From: Linear joint position and velocity
    """
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos = float(data.qpos[model.jnt_qposadr[joint_id]])
    qvel = float(data.qvel[model.jnt_dofadr[joint_id]])

    return {
        "position": qpos,
        "velocity": qvel,
        "extended": qpos > 0.1,  # Pulled out beyond threshold
        "sliding": abs(qvel) > 0.01  # Moving
    }


def extract_container(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                      all_assets: Dict[str, Any], contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract container behavior: contains_X, empty, full, num_contained

    From: Bounds checking - which objects are inside this container
    """
    if not component.geom_names or not all_assets:
        return {}

    # Get container position and size (approximate from geom)
    geom_name = component.geom_names[0]
    container_pos = _get_body_position(model, data, geom_name, contact_cache)

    # Approximate container size (hardcoded for now, should be from config)
    container_size = np.array([0.3, 0.3, 0.3])  # 30cm cube default

    result = {}
    num_contained = 0

    # Check each other asset
    for asset_name, asset in all_assets.items():
        if asset_name == component.name:
            continue

        # Check if object is inside bounds
        for other_comp in asset.components.values():
            if not other_comp.geom_names:
                continue

            obj_pos = _get_body_position(model, data, other_comp.geom_names[0], contact_cache)
            is_inside = _point_in_box(obj_pos, container_pos, container_size)

            if is_inside:
                result[f"contains_{asset_name}"] = True
                num_contained += 1
            else:
                result[f"contains_{asset_name}"] = False

    result["num_contained"] = num_contained
    result["empty"] = num_contained == 0
    result["full"] = num_contained >= 3  # Default capacity

    return result


def extract_surface(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                    all_assets: Dict[str, Any], contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract surface behavior: geometry info + contacts

    Returns:
        surface_position: [x, y, z] center of surface geom
        surface_width: meters (X dimension)
        surface_depth: meters (Y dimension)
        surface_thickness: meters (Z dimension)
        surface_top_z: absolute height where objects sit
        contact_X: True/False for each asset
        contact_force: total contact force

    From: MuJoCo geom properties + contact detection
    """
    if not component.geom_names or not all_assets:
        return {}

    surface_geom = component.geom_names[0]
    result = {}

    # Extract surface geometry (DIMENSIONS!)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, surface_geom)
    if geom_id >= 0:
        # Surface position in world coordinates
        surface_pos = data.geom_xpos[geom_id]
        result["surface_position"] = [float(surface_pos[0]), float(surface_pos[1]), float(surface_pos[2])]

        # Surface dimensions (half-sizes in MuJoCo)
        half_sizes = model.geom_size[geom_id]
        result["surface_width"] = float(half_sizes[0] * 2)  # Full width (X)
        result["surface_depth"] = float(half_sizes[1] * 2)  # Full depth (Y)
        result["surface_thickness"] = float(half_sizes[2] * 2)  # Full thickness (Z)

        # Top of surface (where objects sit)
        result["surface_top_z"] = float(surface_pos[2] + half_sizes[2])

    # Check contact with each other asset - MOP: Use helper for uniform access!
    total_force = 0.0
    for asset_name, other_comp in _iter_all_components(all_assets):
        if not other_comp.geom_names:
            continue

        for obj_geom in other_comp.geom_names:
            has_contact, force = _check_contact(model, data, surface_geom, obj_geom, contact_cache)

            if has_contact:
                result[f"contact_{asset_name}"] = True
                total_force += force
            else:
                result[f"contact_{asset_name}"] = False

    result["contact_force"] = total_force

    return result


def extract_room_boundary(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                          all_assets: Dict[str, Any], contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract room_boundary behavior: contact_X, contact_force

    From: MuJoCo contact detection (walls/floor/ceiling detecting objects)

    Note: Uses 'contact' prefix (same as surface behavior) for consistency

    Args:
        contact_cache: Pre-built contact cache for O(1) lookups
    """
    if not component.geom_names or not all_assets:
        return {}

    boundary_geom = component.geom_names[0]
    result = {}
    total_force = 0.0

    # Check contact with each other asset - MOP: Use helper for uniform access!
    for asset_name, other_comp in _iter_all_components(all_assets):
        if not other_comp.geom_names:
            continue

        for obj_geom in other_comp.geom_names:
            has_contact, force = _check_contact(model, data, boundary_geom, obj_geom, contact_cache)

            if has_contact:
                result[f"contact_{asset_name}"] = True
                total_force += force
            else:
                result[f"contact_{asset_name}"] = False

    result["contact_force"] = total_force

    return result


def extract_rollable(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract rollable behavior: angular_velocity, rolling, stationary

    From: Body angular velocity
    """
    if not component.geom_names:
        return {}

    geom_name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return {}

    body_id = model.geom_bodyid[geom_id]
    ang_vel = data.cvel[body_id][3:]  # Angular velocity (last 3 components)
    ang_speed = float(np.linalg.norm(ang_vel))

    return {
        "angular_velocity": ang_speed,
        "rolling": ang_speed > 0.1,  # Threshold for rolling
        "stationary": ang_speed < 0.01  # Nearly stopped
    }


def extract_stackable(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                      all_assets: Dict[str, Any], contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract stackable behavior: stacked_on_X, supporting_X, stack_height, stable

    From: Contact detection + height comparison

    OPTIMIZED: Uses contact cache to SKIP non-contacting objects!
    Instead of checking ALL components (O(N)), only check contacted geoms (O(k) where k~2-5)
    Reduces from 100 checks/extraction to ~3 checks/extraction!

    Args:
        contact_cache: Pre-built contact cache for O(1) lookups
    """
    if not component.geom_names or not all_assets:
        return {}

    my_geom = component.geom_names[0]

    # Get my geom ID for cache lookup
    if contact_cache is None:
        # Fallback to slow path if no cache
        my_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, my_geom)
    else:
        geom_name_to_id = contact_cache.get('geom_name_to_id', {})
        my_geom_id = geom_name_to_id.get(my_geom, -1)

    if my_geom_id < 0:
        return {}

    my_pos = _get_body_position(model, data, my_geom, contact_cache)
    my_height = float(my_pos[2])

    result = {}
    stacked_on = []
    supporting = []

    # OPTIMIZATION: Only check geoms that are ACTUALLY in contact!
    # Cache already knows which geoms touch → skip all others!
    contacted_geoms = set()
    if contact_cache is not None:
        geom_contacts = contact_cache.get('geom_contacts', {})
        contacted_geoms = geom_contacts.get(my_geom_id, set())

    # Build reverse lookup: geom_id → asset_name
    geom_to_asset = {}
    if contact_cache is not None:
        geom_name_to_id = contact_cache.get('geom_name_to_id', {})
        for asset_name, other_comp in _iter_all_components(all_assets):
            if other_comp.geom_names:
                for other_geom_name in other_comp.geom_names:
                    other_geom_id = geom_name_to_id.get(other_geom_name)
                    if other_geom_id is not None:
                        geom_to_asset[other_geom_id] = (asset_name, other_geom_name)

    # Only process contacted geoms (typically 2-5 instead of 100!)
    for contacted_geom_id in contacted_geoms:
        if contacted_geom_id not in geom_to_asset:
            continue

        asset_name, other_geom = geom_to_asset[contacted_geom_id]

        other_pos = _get_body_position(model, data, other_geom, contact_cache)
        other_height = float(other_pos[2])

        # Am I above? Then stacked on them
        # PURE MOP FIX: 1cm threshold (not 5cm!) - objects sitting on surfaces have ~same height!
        # Floor/table should use SAME threshold - if banana on table works, apple on floor must work!
        if my_height > other_height - 0.01:  # Allow 1cm tolerance (including penetration!)
            result[f"stacked_on_{asset_name}"] = True
            stacked_on.append(asset_name)
        # Are they above? Then I'm supporting them
        # PURE MOP: Same threshold for both directions! (Symmetric relationship!)
        elif other_height > my_height - 0.01:
            result[f"supporting_{asset_name}"] = True
            supporting.append(asset_name)

    result["stack_height"] = len(stacked_on) + 1  # 1 = on floor, 2+ = stacked
    result["stable"] = True  # TODO: Check angular velocity for wobbling

    return result


def extract_pressable(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                      extraction_cache: dict = None) -> Dict[str, Any]:
    """
    Extract pressable behavior: pressed, released, force, activation_count

    From: Contact force detection (buttons)

    Args:
        extraction_cache: Optional cache from build_extraction_cache() for cached forces
    """
    if not component.geom_names:
        return {}

    geom_name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return {}

    # Calculate total force on button
    # MOP: Uses cached force (eliminates redundant contact scans!)
    total_force = _get_contact_force(model, data, geom_id, extraction_cache)
    pressed = total_force > 1.0  # Threshold for button press

    return {
        "pressed": pressed,
        "released": not pressed,
        "force": float(total_force),
        "activation_count": 0  # TODO: Track state changes for counting
    }


def extract_pourable(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract pourable behavior: volume, flow_rate, pouring, empty

    From: Body orientation (tilt angle) as proxy for pouring
    """
    if not component.geom_names:
        return {}

    geom_name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return {}

    body_id = model.geom_bodyid[geom_id]

    # Get body rotation matrix
    rotation = data.xmat[body_id].reshape(3, 3)
    # Check tilt - z-axis of body (up direction)
    up_vector = rotation[:, 2]
    tilt_angle = float(np.arccos(np.clip(up_vector[2], -1, 1)))  # Angle from vertical

    is_pouring = tilt_angle > np.pi / 4  # Tilted more than 45 degrees

    return {
        "volume": 1.0,  # Placeholder - would need fluid simulation
        "flow_rate": 0.5 if is_pouring else 0.0,  # Simple proxy
        "pouring": is_pouring,
        "empty": False  # Placeholder
    }


def extract_fragile(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                    extraction_cache: dict = None) -> Dict[str, Any]:
    """
    Extract fragile behavior: intact, broken, impact_force

    From: Maximum contact force detection

    Args:
        extraction_cache: Optional cache from build_extraction_cache() for cached forces
    """
    if not component.geom_names:
        return {}

    geom_name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return {}

    # Calculate max impact force
    # MOP: Uses cached force (eliminates redundant contact scans!)
    # Note: total force, not max per-contact (good enough for fragile detection)
    max_force = _get_contact_force(model, data, geom_id, extraction_cache)
    broken = max_force > 50.0  # Threshold for breaking

    return {
        "intact": not broken,
        "broken": broken,
        "impact_force": float(max_force)
    }


# ============================================================================
# ROBOT BEHAVIOR EXTRACTORS - Robot actuator physics → semantics
# ============================================================================

def extract_robot_arm(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract robot_arm behavior: extension, extended

    From: Sum of telescoping joint positions (sync_mode="sum")
    """
    if not component.joint_names:
        return {}

    # Sum all joint positions (telescoping arm)
    total_extension = 0.0
    for joint_name in component.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        total_extension += float(data.qpos[qpos_addr])

    # Extended = past threshold (default 0.3m)
    threshold = 0.3  # TODO: Get from BEHAVIORS.json

    return {
        "extension": total_extension,
        "extended": total_extension >= threshold
    }


def extract_robot_lift(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract robot_lift behavior: height, raised

    From: Single lift joint position
    """
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    height = float(data.qpos[qpos_addr])

    # Raised = past threshold (default 0.5m)
    threshold = 0.5

    return {
        "height": height,
        "raised": height >= threshold
    }


def extract_robot_gripper(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                          all_assets: Dict[str, Any] = None, robot=None,
                          extraction_cache: dict = None) -> Dict[str, Any]:
    """
    Extract robot_gripper behavior: aperture, closed, holding

    SELF-VALIDATING (MOP Principle #6): Uses dual signals in simulation!
    - Sensor: GripperForceSensor (mimics real robot)
    - Physics: MuJoCo contact forces (ground truth)
    - Validates: sensor vs physics match

    From: Gripper joint position + force (sensor OR physics)

    Args:
        extraction_cache: Optional cache from build_extraction_cache() for cached forces
    """
    if not component.joint_names:
        return {}

    # Aperture from joint (average of left/right fingers)
    total_aperture = 0.0
    joint_count = 0
    for joint_name in component.joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        total_aperture += float(data.qpos[qpos_addr])
        joint_count += 1

    aperture = total_aperture / joint_count if joint_count > 0 else 0.0

    # ================================================================
    # SELF-VALIDATION: Dual-Signal Pattern (MOP Principle #6)
    # ================================================================

    # 1. Get SENSOR signal (if robot available - real or simulated!)
    sensor_force = 0.0
    if robot and hasattr(robot, 'sensors') and 'gripper_force' in robot.sensors:
        try:
            sensor_data = robot.sensors['gripper_force'].get_data()
            sensor_force = max(sensor_data.get('force_left', 0.0),
                             sensor_data.get('force_right', 0.0))
        except (AttributeError, KeyError):
            pass  # Sensor not available, will use physics

    # 2. Get PHYSICS signal (ground truth from MuJoCo contacts)
    # MOP: Uses cached force (eliminates redundant contact scans!)
    physics_force = 0.0
    if component.geom_names:
        for geom_name in component.geom_names:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id < 0:
                continue
            physics_force += _get_contact_force(model, data, geom_id, extraction_cache)

    # 3. SELF-VALIDATE: Compare sensor vs physics (simulation only!)
    validated = True
    if sensor_force > 0.1 and physics_force > 0.1:
        validated = abs(sensor_force - physics_force) < 1.0  # Within 1N tolerance

    # 4. Use sensor if available (real robot!), otherwise physics (simulation)
    holding_force = sensor_force if sensor_force > 0.1 else physics_force

    # 5. Behavior based on FORCE ONLY! (No position - force is the signal!)
    # Real gripping physics: you grip when you FEEL force, not when reaching arbitrary position!
    # Object size is unknown - force tells you if grip succeeded!
    closed = (holding_force > 2.0)  # 2.0N = physics threshold

    return {
        "aperture": aperture,
        "closed": closed,
        "holding": holding_force > 1.0,  # Threshold for holding

        # SELF-VALIDATION METADATA (MOP! - debuggable!)
        "_sensor_force": sensor_force,
        "_physics_force": physics_force,
        "_validated": validated
    }


def extract_robot_head_pan(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract robot_head_pan behavior: angle_rad

    From: Single pan joint position
    """
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    angle = float(data.qpos[qpos_addr])

    return {"angle_rad": angle}


def extract_robot_head_tilt(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract robot_head_tilt behavior: angle_rad

    From: Single tilt joint position
    """
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    angle = float(data.qpos[qpos_addr])

    return {"angle_rad": angle}


def extract_robot_wrist_yaw(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """Extract robot_wrist_yaw: angle_rad"""
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    return {"angle_rad": float(data.qpos[qpos_addr])}


def extract_robot_wrist_pitch(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """Extract robot_wrist_pitch: angle_rad"""
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    return {"angle_rad": float(data.qpos[qpos_addr])}


def extract_robot_wrist_roll(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """Extract robot_wrist_roll: angle_rad"""
    if not component.joint_names:
        return {}

    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {}

    qpos_addr = model.jnt_qposadr[joint_id]
    return {"angle_rad": float(data.qpos[qpos_addr])}


def extract_robot_speaker(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """Extract robot_speaker: volume (stub - no MuJoCo representation)"""
    return {"volume": 50.0}  # Default volume


def extract_robot_wheel(model: mujoco.MjModel, data: mujoco.MjData, component: Any) -> Dict[str, Any]:
    """
    Extract robot_wheel behavior: position (velocity for wheels)

    From: Joint qpos (for velocity actuators, qpos is the velocity)
    """
    if not component.joint_names:
        return {}

    # Get velocity from first joint (wheels have single joint)
    joint_name = component.joint_names[0]
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return {"position": 0.0}

    qpos_addr = model.jnt_qposadr[joint_id]
    velocity = float(data.qpos[qpos_addr])

    return {"position": velocity}


def extract_robot_base(model: mujoco.MjModel, data: mujoco.MjData, component: Any, robot=None) -> Dict[str, Any]:
    """
    Extract robot_base behavior: position, rotation, at_location

    From: Base body position (from base_link) and IMU orientation (from base_imu)

    CRITICAL FIX: Use IMU quaternion for rotation (same source as actions!)
    Actions use IMU to decide when to stop, so state MUST use IMU too.
    Otherwise actions complete but state shows different rotation.

    CUMULATIVE ROTATION: If robot has odometry sensor, use cumulative theta (handles >360°)
    Otherwise use instant yaw angle (wraps at ±180°)

    ROBUST: Handles both geom names and body names (like _check_contact)
    """
    if not component.geom_names:
        return {}

    # Try geom name first, fallback to body name (ROBUST!)
    name = component.geom_names[0]
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)

    if geom_id >= 0:
        # Found as geom - get body from geom
        body_id = model.geom_bodyid[geom_id]
    else:
        # Not a geom - try as body name (common for robot bases)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            return {}  # Not found as either geom or body

    # MOP FIX: For freejoint robots, read position DIRECTLY from qpos (source of truth!)
    # Freejoint robots have position in qpos[0:3] - this is authoritative!
    # xpos is derived from qpos, so reading qpos directly ensures correct placement.
    if name == "base_link":  # Freejoint robot (like Stretch)
        base_pos = np.array([data.qpos[0], data.qpos[1], data.qpos[2]])
        # MOP FIX: Also extract quaternion from qpos[3:7] for freejoint robots!
        base_quat = [float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6])]
    else:
        # Non-freejoint robots - use xpos
        base_pos = data.xpos[body_id]
        # Non-freejoint robots - extract quaternion from xquat
        base_quat = data.xquat[body_id].tolist()

    # Get rotation - use odometry for cumulative rotation if available!
    if robot and hasattr(robot, 'sensors') and 'odometry' in robot.sensors:
        # Use cumulative theta from odometry (handles >360° rotations!)
        odom_data = robot.sensors['odometry'].get_data()
        yaw_deg = float(np.degrees(odom_data['theta']))  # Cumulative rotation in degrees
    else:
        # Fallback: Extract instant yaw angle from rotation matrix (wraps at ±180°)
        rotation_matrix = data.xmat[body_id].reshape(3, 3)
        yaw_rad = float(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        yaw_deg = float(np.degrees(yaw_rad))

    return {
        "position": base_pos.tolist(),  # [x, y, z]
        "quaternion": base_quat,  # [w, x, y, z] - MOP: Auto-extracted orientation!
        "rotation": yaw_deg,  # Degrees around Z (cumulative if odometry available!)
        "at_location": False,  # TODO: Compare to target location
        "_tolerance": np.degrees(ROTATION_TOLERANCE)  # Convert radians to degrees (same unit as rotation!)
    }


def _get_surface_distance(model: mujoco.MjModel, pos1: np.ndarray, geom1_name: str,
                          pos2: np.ndarray, geom2_name: str,
                          extraction_cache: dict = None) -> float:
    """Calculate surface-to-surface distance between two geoms

    MOP FIX: distance_to should report SURFACE distance, not center distance!
    When robot touches table → distance = 0.0m (correct physics)

    NEW: Eliminates 22x redundancy by caching pairwise distances!
    Same distance pair queried 22x per extraction → now calculated ONCE!

    Args:
        model: MuJoCo model
        pos1: Center position of first geom [x, y, z]
        geom1_name: Name of first geom
        pos2: Center position of second geom [x, y, z]
        geom2_name: Name of second geom
        extraction_cache: REQUIRED cache from build_extraction_cache().

    Returns:
        Surface-to-surface distance in meters (0.0 if touching/colliding)

    MOP: NO SLOW PATH! Cache is required.
    """
    if extraction_cache is None:
        raise ValueError(
            "❌ PERFORMANCE ERROR: _get_surface_distance() called without extraction_cache!\n"
            "This would cause 850K redundant mj_name2id() calls (EXTREMELY SLOW).\n\n"
            "FIX: Build extraction cache and pass it to extractors."
        )

    # MOP: Check pairwise distance cache FIRST (eliminate 22x redundancy!)
    pairwise_distances = extraction_cache.get('pairwise_distances', {})
    pair_key = tuple(sorted([geom1_name, geom2_name]))

    if pair_key in pairwise_distances:
        # Cache hit! Reuse cached distance (saves numpy.linalg.norm call + lookups)
        return pairwise_distances[pair_key]

    # Cache miss - calculate distance and cache it
    # Calculate center-to-center distance
    center_dist = float(np.linalg.norm(pos1 - pos2))

    geom_name_to_id = extraction_cache['geom_name_to_id']
    geom_radii = extraction_cache['geom_radii']

    # Get radius for geom1
    radius1 = 0.0
    if geom1_name in geom_name_to_id:
        geom1_id = geom_name_to_id[geom1_name]
        radius1 = geom_radii.get(geom1_id, 0.0)

    # Get radius for geom2
    radius2 = 0.0
    if geom2_name in geom_name_to_id:
        geom2_id = geom_name_to_id[geom2_name]
        radius2 = geom_radii.get(geom2_id, 0.0)

    # Surface distance = center distance - both radii
    surface_dist = max(0.0, center_dist - radius1 - radius2)

    # Cache the result for reuse in this extraction cycle
    pairwise_distances[pair_key] = surface_dist

    return surface_dist


def extract_spatial(model: mujoco.MjModel, data: mujoco.MjData, component: Any,
                    all_assets: Dict[str, Any], contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract spatial behavior: position, distance_to_X, distance_to

    From: Body position in world coordinates (universal - works for ANY component)

    Supports PURE MOP: Uses geoms OR sites (fallback for gripper which has no geoms)

    Returns:
        - position: [x, y, z] in meters
        - distance_to_X: Distance to specific asset X (relational)
        - distance_to: Minimum distance to ANY other asset (aggregate)
    """
    # Get component position - PURE MOP: sites FIRST (accurate endpoint), then geoms!
    my_pos = None
    use_site = False

    if component.site_names:
        # PREFERRED: Use site position (accurate endpoint for arm, gripper, etc.)
        site_name = component.site_names[0]
        # Use cached site lookup if available
        if contact_cache is not None:
            site_name_to_id = contact_cache.get('site_name_to_id', {})
            site_id = site_name_to_id.get(site_name, -1)
            if site_id < 0:
                # Fallback if site not in cache
                site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        else:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            my_pos = data.site_xpos[site_id]
            use_site = True
    elif component.geom_names:
        # Fallback: Use geom position (for objects without sites)
        my_pos = _get_body_position(model, data, component.geom_names[0], contact_cache)

    if my_pos is None:
        # Spatial extraction failed - component has no position
        return {}

    result = {
        "position": my_pos.tolist()  # [x, y, z] in meters
    }

    # NEW: Extract orientation and direction (cached - O(1)!)
    # Orientation: quaternion [w, x, y, z]
    # Direction: forward vector [dx, dy, dz] (normalized)
    body_id = -1
    if use_site and component.site_names:
        # Get body from site (sites attached to bodies)
        site_name = component.site_names[0]
        if contact_cache is not None:
            site_name_to_id = contact_cache.get('site_name_to_id', {})
            site_id = site_name_to_id.get(site_name, -1)
        else:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            body_id = model.site_bodyid[site_id]
    elif component.geom_names:
        # Get body from geom
        geom_name = component.geom_names[0]
        if contact_cache is not None:
            geom_name_to_id = contact_cache.get('geom_name_to_id', {})
            geom_id = geom_name_to_id.get(geom_name, -1)
        else:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            body_id = model.geom_bodyid[geom_id]

    if body_id >= 0:
        # Orientation (quaternion)
        result["orientation"] = data.xquat[body_id].tolist()  # [w, x, y, z]

        # Direction (forward vector from rotation matrix)
        # xmat is 3x3 rotation matrix flattened to 9 elements
        xmat = data.xmat[body_id].reshape(3, 3)
        # Forward is Z-axis (3rd column) - this is where component is "pointing"
        result["direction"] = xmat[:, 2].tolist()  # [dx, dy, dz] - normalized!

    # Calculate distance to each other asset (relational property!)
    min_distance = float('inf')
    if all_assets:
        for asset_name, asset in all_assets.items():
            # Skip self - OFFENSIVE: Trust all components have .name
            if component.name == asset_name:
                continue

            # UNIFORM: All assets have .components!
            for other_comp in asset.components.values():
                if not other_comp.geom_names:
                    continue

                other_pos = _get_body_position(model, data, other_comp.geom_names[0], contact_cache)
                # MOP FIX: Use SURFACE distance, not center distance!
                distance = _get_surface_distance(model, my_pos, component.geom_names[0],
                                                  other_pos, other_comp.geom_names[0], contact_cache)

                # Relational property (specific)
                result[f"distance_to_{asset_name}"] = distance

                # Track minimum for aggregate property
                min_distance = min(min_distance, distance)

                break  # Only need one component per asset for distance

    # Aggregate property: distance to closest object
    result["distance_to"] = min_distance if min_distance != float('inf') else 0.0

    return result


# ============================================================================
# DIMENSION EXTRACTION - PURE MOP (Runtime Only!)
# ============================================================================

def get_geom_dimensions(model: mujoco.MjModel, geom_name: str) -> Dict[str, float]:
    """Extract dimensions from MuJoCo model - SINGLE SOURCE OF TRUTH!

    PURE MOP: Dimensions extracted from compiled model (RUNTIME ONLY!)
    MuJoCo stores HALF-SIZES - we return FULL dimensions.

    Args:
        model: Compiled MuJoCo model
        geom_name: Name of geom to measure

    Returns:
        Dict with 'width', 'depth', 'height' in meters (full sizes, not half)

    Raises:
        ValueError: If geom not found (OFFENSIVE!)
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)

    if geom_id < 0:
        # OFFENSIVE: Geom not found!
        raise ValueError(
            f"❌ Geom '{geom_name}' not found in compiled model!\n"
            f"\n💡 MOP: Dimensions extracted from MuJoCo model (RUNTIME ONLY!)\n"
            f"Ensure scene is compiled before querying dimensions."
        )

    # Get HALF-sizes from model (MuJoCo convention!)
    half_size = model.geom_size[geom_id]

    return {
        'width': float(half_size[0] * 2),
        'depth': float(half_size[1] * 2),
        'height': float(half_size[2] * 2)
    }


def get_asset_dimensions(model: mujoco.MjModel, asset_name: str) -> Dict[str, float]:
    """Get asset dimensions - FULL BOUNDING BOX across all geoms!

    PURE MOP: Auto-discovers dimensions from compiled model.
    Calculates bounding box across ALL geoms in the asset body.

    Args:
        model: Compiled MuJoCo model
        asset_name: Asset name (e.g., "table", "apple")

    Returns:
        Dict with 'width', 'depth', 'height' in meters (FULL bounding box!)

    Raises:
        ValueError: If no geom found (OFFENSIVE!)
    """
    import numpy as np

    # Find the asset's body by trying common naming patterns
    body_patterns = [
        asset_name,                       # Direct body name
        f"{asset_name}_body",             # Common pattern
        f"{asset_name}_0"                 # Numbered pattern
    ]

    body_id = -1
    for body_name in body_patterns:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            break

    if body_id < 0:
        # Fallback: Try single geom approach (for simple objects)
        geom_patterns = [
            f"{asset_name}_top_geom",
            f"{asset_name}_geom",
            f"{asset_name}_{asset_name}_mesh_geom",
        ]
        for geom_name in geom_patterns:
            try:
                return get_geom_dimensions(model, geom_name)
            except ValueError:
                continue

        raise ValueError(
            f"❌ Cannot find body or geoms for '{asset_name}'!\n"
            f"\nTried body names: {body_patterns}\n"
            f"\n💡 MOP: Asset should have a body with geoms!"
        )

    # Find ALL geoms attached to this body OR its children (MOP: recursive!)
    # Many assets have empty parent body with child body containing geoms
    # Example: "table" parent → "simpleWoodTable" child with table_top geom
    geom_ids = []

    def find_geoms_recursive(search_body_id):
        """Recursively find geoms in this body and all child bodies"""
        # Add geoms from this body
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] == search_body_id:
                geom_ids.append(geom_id)

        # Recursively search child bodies
        for child_body_id in range(model.nbody):
            if model.body_parentid[child_body_id] == search_body_id:
                find_geoms_recursive(child_body_id)

    find_geoms_recursive(body_id)

    if not geom_ids:
        raise ValueError(f"❌ Body '{asset_name}' and children have no geoms!")

    # Calculate bounding box across ALL geoms
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for geom_id in geom_ids:
        # Get geom position and size
        geom_pos = model.geom_pos[geom_id]
        geom_size = model.geom_size[geom_id]

        # Calculate geom bounding box (assuming box type - most common)
        # MuJoCo stores half-sizes
        min_x = min(min_x, geom_pos[0] - geom_size[0])
        max_x = max(max_x, geom_pos[0] + geom_size[0])
        min_y = min(min_y, geom_pos[1] - geom_size[1])
        max_y = max(max_y, geom_pos[1] + geom_size[1])
        min_z = min(min_z, geom_pos[2] - geom_size[2])
        max_z = max(max_z, geom_pos[2] + geom_size[2])

    # Return FULL bounding box dimensions
    return {
        'width': float(max_x - min_x),
        'depth': float(max_y - min_y),
        'height': float(max_z - min_z)  # FULL height including legs!
    }


# ============================================================================
# REGISTRY & MAIN EXTRACTION
# ============================================================================

EXTRACTORS = {
    # Furniture behaviors
    "hinged": extract_hinged,
    "graspable": extract_graspable,
    "sliding": extract_sliding,
    "container": extract_container,
    "surface": extract_surface,
    "room_boundary": extract_room_boundary,
    "rollable": extract_rollable,
    "stackable": extract_stackable,
    "pressable": extract_pressable,
    "pourable": extract_pourable,
    "fragile": extract_fragile,
    "spatial": extract_spatial,

    # Robot actuator behaviors
    "robot_arm": extract_robot_arm,
    "robot_lift": extract_robot_lift,
    "robot_gripper": extract_robot_gripper,
    "robot_gripper_spatial": extract_spatial,  # Gripper position from sites
    "robot_head_pan": extract_robot_head_pan,
    "robot_head_tilt": extract_robot_head_tilt,
    "robot_wrist_yaw": extract_robot_wrist_yaw,
    "robot_wrist_pitch": extract_robot_wrist_pitch,
    "robot_wrist_roll": extract_robot_wrist_roll,
    "robot_base": extract_robot_base,
    "robot_speaker": extract_robot_speaker,
    "robot_wheel": extract_robot_wheel,

    # Robot spatial behaviors (use generic spatial extractor)
    "robot_spatial": extract_spatial,
    "robot_arm_spatial": extract_spatial,
    "robot_gripper_spatial": extract_spatial,
    "robot_head_spatial": extract_spatial,
    "robot_base_spatial": extract_spatial,  # Base spatial tracking (distance_to)
}


def extract_component_state(model: mujoco.MjModel, data: mujoco.MjData,
                            component: Any, all_assets: Dict = None,
                            asset_config: Dict = None, event_log=None,
                            asset_id: str = None, robot=None,
                            contact_cache: dict = None) -> Dict[str, Any]:
    """
    Extract all properties for a component - OFFENSIVE + SELF-VALIDATING

    Calls appropriate extractor for each behavior the component has.
    Returns component-prefixed keys: {"component_name.property": value}

    Args:
        model: MuJoCo model
        data: MuJoCo data
        component: Component to extract from
        all_assets: All assets in scene (for relational properties)
        asset_config: Asset config with joint_ranges for unit conversion
        event_log: EventLog to track behavior changes (ALWAYS-ON!)
        asset_id: Asset ID for event tracking (e.g., "door", "stretch.arm")
        robot: Robot object for sensor access (enables self-validation in simulation!)
        contact_cache: Pre-built contact cache from build_contact_cache() for O(1) lookups

    Returns:
        Dict with component-prefixed keys: {"body.position": [1,0,0], "body.height": 0.5}
    """
    # Store old state for event tracking (ALWAYS-ON!)
    old_state = getattr(component, '_prev_state', None)

    state = {}

    for behavior in component.behaviors:
        if behavior not in EXTRACTORS:
            continue

        extractor = EXTRACTORS[behavior]

        # Call extractor with appropriate parameters
        # Most extractors need extraction_cache (contacts + name→ID + sizes + forces)
        if behavior in ["surface", "room_boundary", "stackable", "graspable",
                        "container", "spatial", "robot_spatial", "robot_arm_spatial",
                        "robot_gripper_spatial", "robot_head_spatial", "robot_base_spatial"]:
            props = extractor(model, data, component, all_assets or {}, contact_cache)
        # robot_gripper needs robot for sensor access + extraction_cache for forces
        elif behavior == "robot_gripper":
            props = extractor(model, data, component, all_assets or {}, robot, contact_cache)
        # robot_base needs robot for sensor access (odometry)
        elif behavior == "robot_base":
            props = extractor(model, data, component, robot)
        # hinged needs asset_config for unit conversion
        elif behavior == "hinged":
            props = extractor(model, data, component, asset_config)
        # pressable, fragile need extraction_cache for cached forces
        elif behavior in ["pressable", "fragile"]:
            props = extractor(model, data, component, contact_cache)
        else:
            props = extractor(model, data, component)

        # Add properties directly - NO PREFIX!
        # Asset name already contains component info ("stretch.arm")
        # Don't double-prefix!
        state.update(props)

    # ================================================================
    # MOP: Component SELF-DESCRIBES! Add ALL modal metadata
    # ================================================================

    # Behaviors - what this component can do
    state["behaviors"] = component.behaviors

    # Source of truth
    state["source"] = "mujoco" if hasattr(component, '_mujoco_data') else "real_robot"

    # Geom/joint/site names (spatial identity)
    if component.geom_names:
        state["geom_names"] = component.geom_names
    if component.joint_names:
        state["joint_names"] = component.joint_names
    if component.site_names:
        state["site_names"] = component.site_names

    # For ActuatorComponent - operational state!
    if hasattr(component, 'range'):
        state["range"] = list(component.range)
        state["target"] = component.position  # Commanded target

    if hasattr(component, 'tolerance'):
        state["tolerance"] = component.tolerance

        # Compute at_target: Is current position within tolerance of target?
        current_pos = None
        if "extension" in state:  # Arm
            current_pos = state["extension"]
        elif "height" in state:  # Lift
            current_pos = state["height"]
        elif "aperture" in state:  # Gripper
            current_pos = state["aperture"]
        elif "angle_rad" in state:  # Wrist/head
            current_pos = state["angle_rad"]
        elif "position" in state and isinstance(state["position"], (int, float)):  # Wheels
            current_pos = state["position"]

        if current_pos is not None and hasattr(component, 'position'):
            state["current"] = current_pos
            state["at_target"] = abs(current_pos - component.position) < component.tolerance
            state["is_busy"] = not state["at_target"]

            # ================================================================
            # PERCENTAGE & RANGE INFO - for cleaner control
            # ================================================================
            # Add percentage info for actuators (makes control easier)
            min_val, max_val = component.range
            if max_val != min_val:
                position_percent = ((current_pos - min_val) / (max_val - min_val)) * 100.0
            else:
                position_percent = 0.0

            state["position_percent"] = position_percent
            state["min_position"] = min_val
            state["max_position"] = max_val

    # Emit change events (ALWAYS-ON!)
    if old_state and asset_id and event_log:
        from .event_log_modal import create_behavior_change_event
        for prop, new_val in state.items():
            old_val = old_state.get(prop)
            # Only emit if value changed significantly
            if _should_emit_event(prop, old_val, new_val):
                # Find which behavior this property belongs to
                for behavior in component.behaviors:
                    event = create_behavior_change_event(
                        timestamp=data.time,
                        asset_id=asset_id,
                        behavior=behavior,
                        field=prop,
                        old_value=old_val,
                        new_value=new_val,
                        step=int(data.time / model.opt.timestep)
                    )
                    event_log.add_event(event)
                    break  # Only emit once per property

    # Update previous state for next comparison
    component._prev_state = state.copy()

    return state


def _should_emit_event(prop: str, old_val: Any, new_val: Any) -> bool:
    """Check if event should be emitted for property change - ELEGANT

    Avoids noise from minor sensor fluctuations
    """
    if old_val is None or new_val is None:
        return False  # First reading, no change yet

    if old_val == new_val:
        return False  # No change

    # For numeric values, check if change is significant
    if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
        # Percentage changes: only emit if change > 1%
        if prop in ['open', 'closed', 'filled']:
            return abs(new_val - old_val) > 1.0
        # Force: only emit if change > 0.1N
        elif 'force' in prop:
            return abs(new_val - old_val) > 0.1
        # Position/distance: only emit if change > 0.01m (1cm)
        elif prop in ['height', 'position', 'distance_to']:
            return abs(new_val - old_val) > 0.01
        # Velocity: only emit if change > 0.05 m/s
        elif 'velocity' in prop:
            return abs(new_val - old_val) > 0.05
        # Default: any change
        else:
            return abs(new_val - old_val) > 0.001

    # For boolean/other types, emit on any change
    return True
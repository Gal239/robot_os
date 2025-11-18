"""
ACTUATOR MODALS - Hardware actuator definitions

UNIFIED: All actuators now use ActuatorComponent from asset_modals.py
Old actuator classes (ArmActuator, LiftActuator, etc.) removed - no longer needed!
Components are self-building, self-syncing, multi-format rendering.
"""
from typing import Dict

# MOP: Dual import support
try:
    from simulation_center.core.modals.asset_modals import ActuatorComponent
except ModuleNotFoundError:
    from core.modals.asset_modals import ActuatorComponent


# ============================================
# OLD ACTUATOR CLASSES REMOVED
# ============================================
# All actuators now use ActuatorComponent from asset_modals.py
# - Single unified class with behaviors from ROBOT_BEHAVIORS.json
# - Self-syncing from MuJoCo
# - Multi-format rendering (get_data, get_rl)
# - Spatial relations support (holding, looks_at, distance_to)
#
# Deleted ~780 lines of duplicate actuator classes:
# - Actuator (base class)
# - ArmActuator, LiftActuator, GripperActuator
# - HeadPanActuator, HeadTiltActuator
# - WristYawActuator, WristPitchActuator, WristRollActuator
# - BaseActuator, BaseRotationActuator
# - SpeakerActuator
#
# See create_all_actuators() below for new implementation
# ============================================


# Removed old __all__ export list - no longer needed


# ============================================
# PHYSICS-DERIVED CONSTANTS - MOP!
# ============================================
# These are from PHYSICS TESTING, not geometry (can't extract from XML!)
# Geometry (gripper length, base height, etc.) is extracted dynamically from XML

# Comfortable reach percentages (physics-tested for stability)
ARM_COMFORTABLE_REACH_PCT = 0.7  # 70% of max reach (stable, not max extension)
LIFT_COMFORTABLE_HEIGHT_PCT = 0.7  # 70% of max lift (stable positioning)

# Safety margins (physics-tested minimum clearances)
REACH_SAFETY_MARGIN = 0.05  # meters - clearance from objects
PLACEMENT_SAFETY_MARGIN = 0.1  # meters - clearance when positioning robot
GRIPPER_GRASP_THRESHOLD = 0.15  # meters - max distance for successful grasp

# ============================================
# AUTO-DISCOVERY FROM KEYFRAME (MOP!)
# ============================================
def _discover_robot_specs() -> Dict[str, Dict[str, any]]:
    """Auto-discover ALL actuator specs from stretch.xml - PURE MOP!

    MOP Principle #1: COMPLETE AUTO-DISCOVERY
    - Scans stretch.xml for ALL actuators (position + velocity)
    - Discovers: names, types, joints, geoms, ranges, positions, units
    - Single source of truth: NO HARDCODED VALUES!
    - Adding actuator to XML → automatically discovered!

    OFFENSIVE: Crashes if XML malformed or missing required data

    Returns:
        Dict mapping actuator names to complete specs:
        {
            "lift": {
                "type": "position",
                "joints": ["joint_lift"],
                "geoms": ["link_lift_0", "link_lift_2", ...],
                "range": (0.0, 1.1),
                "position": 0.6,
                "unit": "meters"
            },
            "left_wheel_vel": {
                "type": "velocity",
                "joints": ["joint_left_wheel"],
                "geoms": ["link_left_wheel_0", "link_left_wheel_1", ...],
                "range": (-6.0, 6.0),
                "position": 0.0,
                "unit": "rad/s"
            },
            ...
        }
    """
    from pathlib import Path
    import xml.etree.ElementTree as ET

    # Find stretch.xml
    robot_xml = Path(__file__).parent.parent / "mujoco_assets" / "robots" / "stretch" / "robot" / "stretch.xml"

    if not robot_xml.exists():
        raise FileNotFoundError(
            f"stretch.xml not found at {robot_xml}\n"
            f"Cannot auto-discover actuator specs!"
        )

    # Parse XML
    tree = ET.parse(robot_xml)
    root = tree.getroot()

    # 1. Discover keyframe ctrl values (for position actuators)
    keyframe = root.find(".//keyframe/key[@name='home']")
    ctrl_values = {}
    if keyframe is not None:
        ctrl_str = keyframe.get("ctrl")
        if ctrl_str:
            ctrl_list = [float(x) for x in ctrl_str.split()]
            # Map by actuator index (0:left_wheel_vel, 1:right_wheel_vel, 2:lift, 3:arm, ...)
            ctrl_values = {i: val for i, val in enumerate(ctrl_list)}

    # 2. Helper: Discover ctrlrange from class defaults
    def get_ctrlrange(act_elem):
        # Try direct attribute
        ctrlrange_str = act_elem.get("ctrlrange")
        if ctrlrange_str:
            vals = [float(x) for x in ctrlrange_str.split()]
            return tuple(vals) if len(vals) == 2 else None

        # Try class defaults
        class_name = act_elem.get("class")
        if class_name:
            act_type = act_elem.tag  # "position", "velocity", etc.
            # Try <act_type ctrlrange="..."/>
            default_elem = root.find(f".//default[@class='{class_name}']/{act_type}[@ctrlrange]")
            if default_elem is not None:
                ctrlrange_str = default_elem.get("ctrlrange")
            else:
                # Try <general ctrlrange="..."/>
                default_elem = root.find(f".//default[@class='{class_name}']/general[@ctrlrange]")
                if default_elem is not None:
                    ctrlrange_str = default_elem.get("ctrlrange")

            if ctrlrange_str:
                vals = [float(x) for x in ctrlrange_str.split()]
                return tuple(vals) if len(vals) == 2 else None

        return None

    # 3. Helper: Discover geoms from joints
    def get_geoms_from_joints(joint_names):
        geoms = []
        for joint_name in joint_names:
            # Find body containing this joint (iterate through all bodies)
            for body in root.findall(".//body"):
                # Check if this body contains the joint
                joint_elem = body.find(f"joint[@name='{joint_name}']")
                if joint_elem is not None:
                    # Get ALL geoms in this body (visual + collision)
                    for geom in body.findall("geom"):
                        # Geom name from: explicit name OR mesh attribute (MuJoCo auto-names from mesh)
                        geom_name = geom.get("name") or geom.get("mesh")
                        if geom_name:
                            geoms.append(geom_name)
                    break
        return geoms

    # 4. Helper: Infer unit from joint type
    def infer_unit(joint_names):
        if not joint_names:
            return ""
        # Find first joint (iterate to find it)
        for joint in root.findall(".//joint"):
            if joint.get("name") == joint_names[0]:
                # Check direct type attribute
                joint_type = joint.get("type")
                if joint_type:
                    return "meters" if joint_type == "slide" else "radians"

                # Check class defaults
                class_name = joint.get("class")
                if class_name:
                    # Look up class default for joint type
                    class_joint = root.find(f".//default[@class='{class_name}']/joint[@type]")
                    if class_joint is not None:
                        joint_type = class_joint.get("type", "hinge")
                        return "meters" if joint_type == "slide" else "radians"

                # Default to hinge (rotation)
                return "radians"
        return "radians"  # Default

    # 5. Helper: Load MuJoCo model and calculate tolerances (PURE MOP!)
    def calculate_all_tolerances():
        """Calculate ALL actuator tolerances from MuJoCo compiled model - PURE MOP!

        Instead of manually parsing XML, load MuJoCo model and let MuJoCo
        compute all physics parameters (resolves class defaults automatically!)

        Returns dict mapping actuator names to tolerances
        """
        import mujoco

        # Load MuJoCo model (MuJoCo resolves all XML inheritance!)
        model = mujoco.MjModel.from_xml_path(str(robot_xml))

        tolerances = {}

        # Iterate through all actuators in the model
        for act_id in range(model.nu):
            act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_id)
            if not act_name:
                continue

            # Get actuator transmission type
            trntype = model.actuator_trntype[act_id]

            # Skip velocity actuators (no tolerance needed)
            if trntype == mujoco.mjtTrn.mjTRN_JOINT and model.actuator_dyntype[act_id] == mujoco.mjtDyn.mjDYN_INTEGRATOR:
                tolerances[act_name] = 0.0
                continue

            # Get transmission ID
            trnid = model.actuator_trnid[act_id]

            # Get actuator gain (kp) from gainprm[0]
            kp = model.actuator_gainprm[act_id, 0]

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                # Direct joint actuator
                joint_id = trnid[0]
                dof_adr = model.jnt_dofadr[joint_id]

                # Read physics parameters from MuJoCo (already resolved from XML!)
                stiffness = model.jnt_stiffness[joint_id]
                damping = model.dof_damping[dof_adr]
                frictionloss = model.dof_frictionloss[dof_adr]

                # Calculate friction force
                friction = frictionloss if frictionloss > 0 else damping

                if friction == 0:
                    # No friction - very precise (numerical precision)
                    tolerances[act_name] = 1e-6
                    continue

                # Calculate tolerance based on control type
                if stiffness > 0:
                    tolerance = friction / stiffness
                elif kp > 0:
                    tolerance = friction / kp
                else:
                    tolerance = friction * 0.01

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                # Tendon actuator (e.g., arm telescope)
                tendon_id = trnid[0]

                # Use tendon stiffness from MuJoCo model
                tendon_stiffness = model.tendon_stiffness[tendon_id]

                if tendon_stiffness > 0:
                    # Tendon has spring - use spring stiffness
                    # Estimate friction as 1% of stiffness (conservative)
                    tolerance = 0.01 / tendon_stiffness
                elif kp > 0:
                    # No spring, use actuator gain
                    # Estimate friction as 1% of gain (conservative)
                    tolerance = 0.01 / kp
                else:
                    # OFFENSIVE: Crash if we can't calculate tolerance!
                    raise RuntimeError(
                        f"TOLERANCE CALCULATION FAILED!\\n"
                        f"  Actuator: {act_name}\\n"
                        f"  Type: TENDON\\n"
                        f"  Tendon stiffness: {tendon_stiffness}\\n"
                        f"  Actuator gain (kp): {kp}\\n"
                        f"\\n"
                        f"  Cannot calculate tolerance without stiffness or gain!\\n"
                        f"  FIX: Add stiffness to tendon or gain to actuator in stretch.xml"
                    )

            else:
                # OFFENSIVE: Crash on unknown transmission type!
                raise RuntimeError(
                    f"UNKNOWN ACTUATOR TRANSMISSION TYPE!\\n"
                    f"  Actuator: {act_name}\\n"
                    f"  Transmission type: {trntype}\\n"
                    f"\\n"
                    f"  Expected: mjTRN_JOINT or mjTRN_TENDON\\n"
                    f"  FIX: Add support for this transmission type in calculate_all_tolerances()"
                )

            # Add safety margin - REALISTIC for actual physics!
            # Tendon actuators (arm) need MUCH larger margin due to:
            # - Cable compliance/stretch
            # - Gravity sag
            # - Friction in pulleys
            # - Model inaccuracies
            # Joint actuators (lift, wrist) converge faster
            if trntype == mujoco.mjtTrn.mjTRN_TENDON:
                # EMPIRICAL: Arm settles ~3mm from target in 500 steps
                # Theoretical tolerance ~0.13mm, actual ~3mm = 23x difference
                # Use 50x safety margin to account for real settling behavior
                tolerances[act_name] = tolerance * 50.0
            else:
                # Joint actuators - use smaller margin
                tolerances[act_name] = tolerance * 2.0

        return tolerances

    # 5. Load EMPIRICALLY DISCOVERED tolerances (PURE MOP!)
    # These were measured from actual physics simulation (no magic numbers!)
    import json
    tolerance_file = Path(__file__).parent / "discovered_tolerances.json"

    if not tolerance_file.exists():
        # OFFENSIVE: Crash if tolerances not discovered yet!
        raise FileNotFoundError(
            f"TOLERANCE FILE NOT FOUND!\n"
            f"  Expected: {tolerance_file}\n"
            f"\n"
            f"  Tolerances must be empirically discovered from physics simulation!\n"
            f"  This is the MOP way - NO MAGIC NUMBERS, NO GUESSING!\n"
            f"\n"
            f"  FIX: Run tolerance discovery script:\n"
            f"    cd {Path(__file__).parent.parent}\n"
            f"    python3 discover_tolerances.py\n"
            f"\n"
            f"  This will:\n"
            f"  - Run 1000-step settling tests for each actuator\n"
            f"  - Measure ACTUAL physics settling errors\n"
            f"  - Save to discovered_tolerances.json\n"
            f"  - Takes ~2 minutes (one-time per robot)\n"
        )

    with open(tolerance_file) as f:
        tolerance_data = json.load(f)
        # Extract just the tolerances dict (file has _meta + tolerances structure)
        all_tolerances = tolerance_data.get("tolerances", tolerance_data)

    # 6. AUTO-DISCOVER ALL ACTUATORS!
    all_specs = {}
    actuator_index = 0

    for act_elem in root.findall(".//actuator/*"):  # ALL actuator types
        name = act_elem.get("name")
        if not name:
            continue

        act_type = act_elem.tag  # "position", "velocity", etc.
        joint = act_elem.get("joint")
        tendon = act_elem.get("tendon")

        # Discover joints
        if joint:
            joints = [joint]
        elif tendon:
            # Discover joints from tendon definition
            tendon_elem = root.find(f".//tendon/fixed[@name='{tendon}']")
            if tendon_elem is not None:
                joints = [j.get("joint") for j in tendon_elem.findall("joint") if j.get("joint")]
            else:
                raise ValueError(
                    f"Actuator '{name}' references tendon '{tendon}' but tendon not found in stretch.xml!"
                )
        else:
            joints = []

        # Discover range
        ctrlrange = get_ctrlrange(act_elem)
        if not ctrlrange:
            raise ValueError(
                f"PURE AUTO-DISCOVERY FAILED!\n"
                f"Actuator '{name}' has no ctrlrange!\n"
                f"Expected: ctrlrange attribute or class default"
            )

        # Discover geoms
        geoms = get_geoms_from_joints(joints)

        # Discover position from keyframe
        position = ctrl_values.get(actuator_index, 0.0)

        # Infer unit
        unit = "rad/s" if act_type == "velocity" else infer_unit(joints)

        # Get tolerance from MuJoCo (PURE MOP!) - MUST exist (no defaults!)
        if name not in all_tolerances:
            raise KeyError(
                f"Tolerance for '{name}' not found in discovered tolerances!\n"
                f"Available: {list(all_tolerances.keys())}\n"
                f"\n"
                f"RUN LEVEL 0 BOOTSTRAP FIRST to discover tolerances from physics!\n"
                f"Command: PYTHONPATH=$PWD python3 core/tests/levels/level_0_bootstrap.py"
            )
        tolerance = all_tolerances[name]

        all_specs[name] = {
            "type": act_type,
            "joints": joints,
            "geoms": geoms,
            "range": ctrlrange,
            "position": position,
            "unit": unit,
            "tolerance": tolerance  # FROM MUJOCO MODEL!
        }

        actuator_index += 1

    return all_specs


def _extract_robot_geometry() -> Dict[str, float]:
    """Extract robot geometry from XML - MOP!

    Extracts:
    - gripper_length: Length of gripper fingers (from finger aruco positions)
    - base_height: Height of robot base from floor
    - base_to_arm_offset: Distance from base center to arm mount

    Returns geometry dict with measurements in meters
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path

    # Load stretch XML
    xml_path = Path(__file__).parent.parent / "mujoco_assets" / "robots" / "stretch" / "stretch.xml"

    if not xml_path.exists():
        # Return defaults if XML not found
        return {
            "gripper_length": 0.13,
            "base_height": 0.3,
            "base_to_arm_offset": 0.1,
        }

    tree = ET.parse(xml_path)
    root = tree.getroot()

    geometry = {}

    # Extract gripper length from finger aruco positions
    # Left finger aruco at pos="-0.14425 0.0014877 -0.005189"
    # Right finger aruco at pos="0.14426 0 0.005189"
    left_finger = root.find(".//body[@name='link_SG3_gripper_left_finger_aruco']")
    right_finger = root.find(".//body[@name='link_SG3_gripper_right_finger_aruco']")

    if left_finger is not None and right_finger is not None:
        left_pos = left_finger.get("pos", "0 0 0").split()
        right_pos = right_finger.get("pos", "0 0 0").split()
        # Gripper length is the X distance from center to finger tip
        left_x = abs(float(left_pos[0]))
        right_x = abs(float(right_pos[0]))
        geometry["gripper_length"] = max(left_x, right_x)  # Use max of both fingers
    else:
        geometry["gripper_length"] = 0.13  # Fallback

    # Extract base height (from link_mast body position)
    mast = root.find(".//body[@name='link_mast']")
    if mast is not None:
        mast_pos = mast.get("pos", "0 0 0").split()
        geometry["base_height"] = float(mast_pos[2])  # Z position
    else:
        geometry["base_height"] = 0.3  # Fallback

    # Extract base to arm offset (from arm mount position)
    # This is approximate - could measure from base to arm joint
    geometry["base_to_arm_offset"] = 0.1  # This one is harder to extract, keep as constant

    return geometry


# ============================================
# ACTUATOR REGISTRY FUNCTION
# ============================================
def create_all_actuators(tolerances: Dict[str, float] = None) -> Dict[str, "ActuatorComponent"]:
    """Create all stretch actuators - COMPLETE AUTO-DISCOVERY!

    MOP Principle #1 (COMPLETE AUTO-DISCOVERY):
    - Scans stretch.xml for ALL actuators (no hardcoded lists!)
    - Discovers: names, types, joints, geoms, ranges, positions, TOLERANCES
    - Tolerances calculated from XML physics (damping, stiffness, frictionloss)
    - Adding actuator to XML → automatically discovered!
    - Single source of truth: stretch.xml

    UNIFIED: Uses ActuatorComponent from asset_modals.py
    Components are self-building, self-syncing, multi-format rendering

    Args:
        tolerances: REMOVED - tolerances now auto-discovered from XML!
    """
    # AUTO-DISCOVER ALL actuators from stretch.xml!
    # Includes tolerance calculation from physics parameters!
    specs = _discover_robot_specs()

    # Helper: Infer behaviors from actuator name and type
    def infer_behaviors(name, spec):
        # Velocity actuators (wheels)
        if spec["type"] == "velocity":
            return ["robot_wheel"]

        # Speaker (no joints)
        if not spec["joints"]:
            return ["robot_speaker"]

        # Position actuators - infer from name (base behavior only!)
        behaviors_map = {
            "arm": ["robot_arm"],
            "lift": ["robot_lift"],
            "gripper": ["robot_gripper"],
            "base": ["robot_base"],
            "head_pan": ["robot_head_pan"],
            "head_tilt": ["robot_head_tilt"],
            "wrist_yaw": ["robot_wrist_yaw"],
            "wrist_pitch": ["robot_wrist_pitch"],
            "wrist_roll": ["robot_wrist_roll"],
        }
        behaviors = list(behaviors_map.get(name, ["robot_generic"]))

        # PURE MOP: Auto-add spatial variant if actuator has position data (sites OR geoms)
        # Sites are preferred (base_imu, reach_point), but geoms work too (base_link body)
        base_behavior = behaviors[0]  # e.g., "robot_base"
        if spec.get("sites") or spec.get("geoms"):
            spatial_behavior = f"{base_behavior}_spatial"
            if spatial_behavior not in behaviors:
                behaviors.append(spatial_behavior)

        return behaviors

    # Helper: Infer sync mode
    def infer_sync_mode(joints):
        if len(joints) > 1:
            return "sum"  # Multiple joints (e.g., arm telescope)
        return "single"

    # Helper: Discover sites for spatial behaviors
    def get_sites(name):
        sites_map = {
            "arm": ["reach_point"],
            "gripper": ["grasp_left", "grasp_right", "grasp_center"],
            "base": ["base_imu"],  # Base IMU site for distance_to tracking
            "head_tilt": ["gaze_forward"],
            "wrist_pitch": ["manipulation_point"],
        }
        return sites_map.get(name, [])

    # Helper: Get placement_site (MOP - actuator knows its placement!)
    def get_placement_site(name):
        """Return placement_site for actuators that need spatial anchoring.

        MOP: Self-declaration - actuator modal knows where it's placed!
        Only 3 actuators have placement sites (base, gripper, arm).
        All others are None (attached to robot structure).
        """
        placement_map = {
            "base": "base_center",
            "gripper": "gripper_center",
            "arm": "link_grasp_center",
        }
        return placement_map.get(name, None)

    # AUTO-CREATE ALL ACTUATORS!
    actuators = {}

    for name, spec in specs.items():
        # Components can have MULTIPLE behaviors - no splitting needed!
        actuators[name] = ActuatorComponent(
            name=name,
            behaviors=infer_behaviors(name, spec),
            geom_names=spec["geoms"],
            joint_names=spec["joints"],
            site_names=get_sites(name),
            position=spec["position"],
            unit=spec["unit"],
            range=spec["range"],
            sync_mode=infer_sync_mode(spec["joints"]),
            tolerance=spec["tolerance"],  # FROM XML PHYSICS!
            placement_site=get_placement_site(name)  # MOP: Self-declaration!
        )

    # Add speaker (not a MuJoCo actuator, but a logical component)
    actuators["speaker"] = ActuatorComponent(
        name="speaker",
        behaviors=["robot_speaker"],
        geom_names=[],
        joint_names=[],
        site_names=[],
        position=0.0,
        unit="",
        range=(0.0, 1.0),
        sync_mode="single",
        tolerance=0.0,
        placement_site=None  # No placement (logical component)
    )

    return actuators

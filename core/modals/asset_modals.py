"""
ASSET MODALS - Self-contained component-based asset system
Auto-discovers from XML, manages state, connects to live MuJoCo
OFFENSIVE & ELEGANT & MODAL-ORIENTED
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from .visualization_protocol import fig_to_numpy


 # Load BEHAVIORS.json and ROBOT_BEHAVIORS.json - Single source of truth
BEHAVIORS_FILE = Path(__file__).parent.parent / "behaviors" / "BEHAVIORS.json"
ROBOT_BEHAVIORS_FILE = Path(__file__).parent.parent / "behaviors" / "ROBOT_BEHAVIORS.json"

# Merge both behavior files into unified BEHAVIORS dict
BEHAVIORS = {}
if BEHAVIORS_FILE.exists():
    BEHAVIORS.update(json.loads(BEHAVIORS_FILE.read_text()))
if ROBOT_BEHAVIORS_FILE.exists():
    BEHAVIORS.update(json.loads(ROBOT_BEHAVIORS_FILE.read_text()))


def is_smooth_capable(unit: str) -> bool:
    """OFFENSIVE - determine if unit type supports smooth mode"""
    unit_types = BEHAVIORS["_unit_types"]  # OFFENSIVE - crash if missing!

    # Check if unit is in our defined types
    for unit_name, unit_meta in unit_types.items():
        if unit == unit_meta["physical_units"]:  # OFFENSIVE - crash if missing!
            return unit_meta["smooth"]  # OFFENSIVE - crash if missing!

    # Fallback - if has states, can be smooth
    return False


# ============================================================================
# BASE CLASS (for RoomModal inheritance)
# ============================================================================

@dataclass
class AssetModal:
    """Minimal base for modal pattern - just serialization interface"""
    name: str

    def to_json(self) -> dict:
        """Subclasses implement their own serialization"""
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_json()")

    @classmethod
    def from_json(cls, data: dict):
        """Subclasses implement their own deserialization"""
        raise NotImplementedError(f"{cls.__name__} must implement from_json()")


# ============================================================================
# COMPONENT & ASSET CLASSES
# ============================================================================

@dataclass
class Component:
    """Base component - self-building, self-syncing, multi-format rendering

    Modal-Oriented: Component knows how to build itself from XML,
    sync itself from MuJoCo, and render itself in multiple formats.

    MOP SELF-DECLARATION: Components declare which behaviors create trackable assets!
    No string parsing, no ifs - I KNOW which of my behaviors are trackable.
    """
    name: str
    behaviors: List[str]  # From BEHAVIORS.json (single source of truth)
    trackable_behaviors: List[str] = None  # MOP: Which behaviors create assets (defaults to all)
    geom_names: List[str] = field(default_factory=list)
    joint_names: List[str] = field(default_factory=list)
    site_names: List[str] = None  # NEW: Sites for semantic markers

    def __post_init__(self):
        """Initialize defaults - MOP: Default trackable = all behaviors"""
        if self.site_names is None:
            self.site_names = []
        if self.trackable_behaviors is None:
            self.trackable_behaviors = self.behaviors  # Default: all behaviors are trackable

    # Live simulation connection (set by Asset.connect())
    _mujoco_model: Any = None
    _mujoco_data: Any = None
    _instance_prefix: str = ""  # For multi-instance scenes
    _scene_assets: Dict[str, Any] = None  # Reference to scene.assets for relational properties
    _extraction_cache: Dict = None  # MOP: Piggybacked cache from runtime_engine (auto-discover ONCE!)

    def get_state(self, prop_name: str) -> Any:
        """Read live physics data from MuJoCo - OFFENSIVE

        Returns actual values from running simulation:
        - held → bool (contact_force > threshold)
        - angle → float (joint qpos)
        - rolling → bool (angular_velocity > threshold)
        - contact_force → float (sum of contact forces)

        ARCHITECTURE: Uses behavior_extractors.py - SINGLE SOURCE OF TRUTH!
        NO hardcoded logic, NO fallbacks, NO backwards compatibility!
        """
        if not self._mujoco_model or not self._mujoco_data:
            raise RuntimeError(
                f"Component '{self.name}' not connected to MuJoCo. "
                f"Call Asset.connect(model, data) first."
            )

        import mujoco

        # Get property metadata from BEHAVIORS - OFFENSIVE!
        prop_data = None
        behavior_name = None
        for behavior in self.behaviors:
            if behavior in BEHAVIORS:
                if prop_name in BEHAVIORS[behavior]['properties']:  # OFFENSIVE - crash if no properties!
                    prop_data = BEHAVIORS[behavior]['properties'][prop_name]
                    behavior_name = behavior
                    break

        if not prop_data:
            raise AttributeError(
                f"Component '{self.name}' has no property '{prop_name}' "
                f"in behaviors {self.behaviors}"
            )

        # ==================================================================
        # ARCHITECTURE: Use behavior extractors - OFFENSIVE & ELEGANT!
        # ==================================================================
        # behavior_extractors.py = SINGLE SOURCE OF TRUTH for all extraction
        # NO hardcoded logic, NO fallbacks, NO backwards compatibility
        # CRASH if fails → teaches you to fix BEHAVIORS.json!
        from .behavior_extractors import extract_component_state, build_extraction_cache, SnapshotData

        model = self._mujoco_model

        # ASYNC MODE: Use snapshot if piggybacked (thread-safe!)
        if hasattr(self, '_mujoco_snapshot') and self._mujoco_snapshot is not None:
            data = SnapshotData(self._mujoco_snapshot, model)
        else:
            data = self._mujoco_data

        # MOP: Use piggybacked cache if available (from runtime_engine), otherwise build
        # This avoids rebuilding cache 19,373x when called during extraction!
        if self._extraction_cache is not None:
            extraction_cache = self._extraction_cache
        else:
            # Fallback: Build cache (only happens outside of extraction loop)
            extraction_cache = build_extraction_cache(model, data)

        # Extract ALL properties using extractors (NO PREFIX - direct property names!)
        # Self-aware: Component knows its neighbors via _scene_assets
        extracted_state = extract_component_state(model, data, self, all_assets=self._scene_assets,
                                                   contact_cache=extraction_cache)

        # Get our property (direct, no prefix!)
        if prop_name not in extracted_state:
            raise RuntimeError(
                f"MODAL ERROR: Property '{prop_name}' not extracted for component '{self.name}'!\n"
                f"  Behaviors: {self.behaviors}\n"
                f"  Extracted properties: {list(extracted_state.keys())}\n"
                f"\n"
                f"FIX: Add extractor for this behavior in behavior_extractors.py\n"
                f"     or check BEHAVIORS.json has this property defined!"
            )

        return extracted_state[prop_name]

    def get_data(self, extracted_state: Dict = None) -> Dict:
        """Multi-format renderer: Express self as dictionary

        Args:
            extracted_state: Pre-extracted state to avoid duplicate extraction (PERFORMANCE!)

        Modal-Oriented: Component knows its own data representation.
        Aggregates all behavior properties from BEHAVIORS.json.

        OPTIMIZATION: Extract ALL properties in ONE call to avoid redundant MuJoCo queries!
        """
        data = {
            "name": self.name,
            "behaviors": self.behaviors,
            "geoms": self.geom_names,
            "joints": self.joint_names,
            "sites": self.site_names
        }

        # PERFORMANCE: Use pre-extracted state if provided (avoids duplicate work!)
        # This eliminates 2000+ redundant extractions per experiment when timeline is enabled
        if extracted_state is not None:
            data.update(extracted_state)
            return data

        # OPTIMIZATION: Extract state ONCE instead of per-property!
        # Old code called extract_component_state() separately for each property (10+ times!)
        # New code: Extract ALL properties in one call, then merge
        if self._mujoco_model and self._mujoco_data:
            from .behavior_extractors import extract_component_state, build_extraction_cache, SnapshotData
            try:
                # ASYNC MODE: Use snapshot if piggybacked (thread-safe!)
                if hasattr(self, '_mujoco_snapshot') and self._mujoco_snapshot is not None:
                    data_source = SnapshotData(self._mujoco_snapshot, self._mujoco_model)
                else:
                    data_source = self._mujoco_data

                # MOP: Use piggybacked cache if available (from runtime_engine), otherwise build
                if self._extraction_cache is not None:
                    extraction_cache = self._extraction_cache
                else:
                    # Fallback: Build cache (only happens outside of extraction loop)
                    extraction_cache = build_extraction_cache(self._mujoco_model, data_source)

                extracted_state = extract_component_state(
                    self._mujoco_model, data_source, self, all_assets=self._scene_assets,
                    contact_cache=extraction_cache
                )
                data.update(extracted_state)  # Add all extracted properties at once!
            except Exception as e:
                raise RuntimeError(
                    f"MODAL ERROR: Failed to extract state for Component '{self.name}'!\n"
                    f"  Behaviors: {self.behaviors}\n"
                    f"  Component geoms: {self.geom_names}\n"
                    f"  Component joints: {self.joint_names}\n"
                    f"  Component sites: {self.site_names}\n"
                    f"  Original error: {type(e).__name__}: {e}\n"
                    f"\n"
                    f"  This means the modal's structure doesn't match BEHAVIORS.json definition.\n"
                    f"  FIX: Either update modal structure OR regenerate BEHAVIORS.json from modals!"
                ) from e

        return data

    def get_rl(self) -> 'np.ndarray':
        """Multi-format renderer: Express self as normalized vector for RL

        Modal-Oriented: Component knows its own RL representation.
        Uses BEHAVIORS.json for normalization ranges.
        """
        import numpy as np
        values = []

        for behavior in self.behaviors:
            if behavior in BEHAVIORS:
                for prop_name, prop_meta in BEHAVIORS[behavior]["properties"].items():
                    try:
                        val = self.get_state(prop_name)

                        # Boolean → 0 or 1
                        if isinstance(val, bool):
                            values.append(1.0 if val else 0.0)

                        # Numeric → normalize using BEHAVIORS.json
                        elif isinstance(val, (int, float)):
                            # OFFENSIVE: CRASH if min/max missing! Fix the modal!
                            min_val = prop_meta["min"]  # No default - must exist!
                            max_val = prop_meta["max"]  # No default - must exist!
                            if max_val != min_val:
                                normalized = (val - min_val) / (max_val - min_val)
                                values.append(np.clip(normalized, 0.0, 1.0))
                            else:
                                values.append(0.0)
                    except Exception as e:
                        # OFFENSIVE: Re-raise with educational context
                        raise RuntimeError(
                            f"MODAL ERROR: Failed to get RL value for property '{prop_name}' from Component '{self.name}'!\n"
                            f"  Behavior: {behavior}\n"
                            f"  Component geoms: {self.geom_names}\n"
                            f"  Component joints: {self.joint_names}\n"
                            f"  Component sites: {self.site_names}\n"
                            f"  Original error: {type(e).__name__}: {e}\n"
                            f"\n"
                            f"  This means the modal's structure doesn't match BEHAVIORS.json definition.\n"
                            f"  FIX: Either update modal structure OR regenerate BEHAVIORS.json from modals!"
                        ) from e

        return np.array(values) if values else np.array([0.0])

    def sync_from_mujoco(self, model, data):
        """Sync protocol: Update self from MuJoCo state

        Modal-Oriented: Component syncs itself (not synced by manager).
        """
        self._mujoco_model = model
        self._mujoco_data = data

    # Component.__getattr__ removed - using explicit syntax now
    # Properties accessed via scene.add_reward(tracked_asset="...", behavior="...", threshold=...)


# PropertyProxy class removed - using explicit syntax now
# See scene_modal.add_reward() for new API


# ============================================================================
# ACTUATOR & SENSOR COMPONENTS - Extend base Component
# ============================================================================

@dataclass
class ActuatorComponent(Component):
    """Component with hardware control capabilities

    Modal-Oriented: Extends Component with actuator-specific methods.
    Inherits get_data(), get_rl(), sync protocol from Component.
    Adds hardware control (move_to, move_by).
    """

    # Hardware state (synced from MuJoCo)
    position: float = field(default=0.0)
    is_active: bool = field(default=False)
    unit: str = field(default="meters")
    range: Tuple[float, float] = field(default=(0.0, 1.0))
    sync_mode: str = field(default="single")  # "single", "sum", "average" for multi-joint
    tolerance: float = field(default=0.0)  # Physics-based position error (auto-discovered from simulation)
    placement_site: Optional[str] = field(default=None)  # MOP: Actuator knows its own placement site!

    # Visualization tracking (for render_visualization())
    target_position: float = field(default=0.0)  # Commanded position from data.ctrl
    _position_history: List[float] = field(default_factory=list)  # Actual position over time
    _target_history: List[float] = field(default_factory=list)  # Target position over time
    _max_history_len: int = field(default=100)  # Rolling window size

    def __post_init__(self):
        """MODAL SELF-GENERATION: Auto-update ROBOT_BEHAVIORS.json with min/max from range!

        This is OFFENSIVE & ELEGANT - modals CREATE their own JSON definitions!
        """
        super().__post_init__()

        # Auto-generate behavior definitions with min/max from actuator range - OFFENSIVE!
        for behavior in self.behaviors:
            if behavior in BEHAVIORS:
                # Update existing behavior properties with min/max
                for prop_name, prop_meta in BEHAVIORS[behavior]["properties"].items():
                    # Skip boolean properties and lists (position, etc.)
                    # KEEP .get() here - checking if property has these optional fields
                    if prop_meta.get("unit") == "boolean" or isinstance(prop_meta.get("default"), list):
                        continue

                    # Auto-add min/max from actuator range if missing
                    if "min" not in prop_meta and "max" not in prop_meta:
                        prop_meta["min"] = self.range[0]
                        prop_meta["max"] = self.range[1]

    def move_to(self, target: float) -> float:
        """Hardware control: Return clipped target position

        MOP: Actuator knows its range and adjusts command if needed.
        Prints warning if clamping occurs.

        Args:
            target: Requested position

        Returns:
            Adjusted position (clamped to range if needed)
        """
        clipped = np.clip(target, *self.range)

        # Warn if clamping occurred
        if clipped != target:
            if target < self.range[0]:
                print(f"⚠️  {self.name}: Requested {target:.3f} below min range {self.range[0]:.3f}, clamping to {clipped:.3f}")
            else:
                print(f"⚠️  {self.name}: Requested {target:.3f} exceeds max range {self.range[1]:.3f}, clamping to {clipped:.3f}")

        return clipped

    def move_by(self, delta: float) -> float:
        """Hardware control: Relative movement"""
        return np.clip(self.position + delta, *self.range)

    def get_position(self) -> float:
        """MOP: Actuator knows its own position property name - NO IFS in actions!

        TRUE MODAL-ORIENTED: Component self-discovers which property to read based on behaviors.
        Actions call actuator.get_position() instead of needing if/elif chains!
        """
        # SELF-DISCOVERED: Read the right property based on behaviors
        for behavior in self.behaviors:
            if behavior == 'robot_arm':
                return self.get_state('extension')
            elif behavior == 'robot_lift':
                return self.get_state('height')
            elif behavior == 'robot_gripper':
                return self.get_state('aperture')
            elif behavior in ['robot_wrist_yaw', 'robot_wrist_pitch', 'robot_wrist_roll',
                             'robot_head_pan', 'robot_head_tilt']:
                return self.get_state('angle_rad')

        # Fallback to synced position field
        return self.position

    def sync_from_mujoco(self, model, data, cache=None):
        """Sync protocol: Update position and is_active from MuJoCo

        Modal-Oriented: Actuator syncs itself from physics.
        Also tracks ctrl (target) and position history for visualization.

        Args:
            cache: Optional extraction cache with name→ID mappings (PERFORMANCE!)
        """
        super().sync_from_mujoco(model, data)

        import mujoco

        # PERFORMANCE: Use cached name→ID mappings if available
        joint_name_to_id = cache.get('joint_name_to_id', {}) if cache else {}
        actuator_name_to_id = cache.get('actuator_name_to_id', {}) if cache else {}

        # Read joint positions
        joint_values = []
        for joint_name in self.joint_names:
            prefixed = f"{self._instance_prefix}{joint_name}" if self._instance_prefix else joint_name
            try:
                # PERFORMANCE: Use cache instead of mj_name2id (eliminates 47.5x calls per step!)
                if cache and prefixed in joint_name_to_id:
                    joint_id = joint_name_to_id[prefixed]
                else:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
                if joint_id >= 0:
                    qpos_addr = model.jnt_qposadr[joint_id]
                    joint_values.append(data.qpos[qpos_addr])
            except:
                pass

        if joint_values:
            # Apply sync mode (for multi-joint actuators like telescoping arm)
            if self.sync_mode == "sum":
                self.position = sum(joint_values)
            elif self.sync_mode == "average":
                self.position = sum(joint_values) / len(joint_values)
            else:  # single
                self.position = joint_values[0]

        # Read ctrl values (target positions) - OFFENSIVE
        # Map joint_names → actuator_names (typically "actuator_<joint_name>")
        ctrl_values = []
        for joint_name in self.joint_names:
            # Common MuJoCo naming: actuator names are "actuator_<joint_name>"
            actuator_name = f"actuator_{joint_name}"
            prefixed = f"{self._instance_prefix}{actuator_name}" if self._instance_prefix else actuator_name

            try:
                # PERFORMANCE: Use cache instead of mj_name2id
                if cache and prefixed in actuator_name_to_id:
                    actuator_id = actuator_name_to_id[prefixed]
                else:
                    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed)
                if actuator_id >= 0:
                    ctrl_values.append(data.ctrl[actuator_id])
            except:
                pass

        if ctrl_values:
            # Apply same sync mode to ctrl as position
            if self.sync_mode == "sum":
                self.target_position = sum(ctrl_values)
            elif self.sync_mode == "average":
                self.target_position = sum(ctrl_values) / len(ctrl_values)
            else:  # single
                self.target_position = ctrl_values[0]
        else:
            # No ctrl found - use position as target (for passive joints)
            self.target_position = self.position

        # Read velocity to determine is_active
        velocities = []
        for joint_name in self.joint_names:
            prefixed = f"{self._instance_prefix}{joint_name}" if self._instance_prefix else joint_name
            try:
                # PERFORMANCE: Use cache instead of mj_name2id
                if cache and prefixed in joint_name_to_id:
                    joint_id = joint_name_to_id[prefixed]
                else:
                    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefixed)
                if joint_id >= 0:
                    qvel_addr = model.jnt_dofadr[joint_id]
                    velocities.append(abs(data.qvel[qvel_addr]))
            except:
                pass

        self.is_active = max(velocities) > 0.001 if velocities else False

        # Track history for visualization (rolling window)
        self._position_history.append(self.position)
        self._target_history.append(self.target_position)

        # Maintain rolling window
        if len(self._position_history) > self._max_history_len:
            self._position_history.pop(0)
        if len(self._target_history) > self._max_history_len:
            self._target_history.pop(0)

    def get_data(self) -> Dict:
        """Multi-format renderer: Actuator data includes hardware state

        SEMANTIC PROPERTY MAPPING: Provides human-readable property names
        - position_m: For arm (extension) actuators
        - height_m: For lift actuators
        - angle_rad: Already provided by Component.get_data()
        - angle_deg: Already provided by Component.get_data()
        """
        data = super().get_data()

        data.update({
            "position": self.position,
            "is_active": self.is_active,
            "unit": self.unit,
            "range": self.range
        })

        # ================================================================
        # PERCENTAGE & RANGE INFO - for cleaner control
        # ================================================================
        min_val, max_val = self.range
        if max_val != min_val:
            position_percent = ((self.position - min_val) / (max_val - min_val)) * 100.0
        else:
            position_percent = 0.0

        data.update({
            "position_percent": position_percent,
            "min_position": min_val,
            "max_position": max_val
        })

        # ================================================================
        # SEMANTIC PROPERTY MAPPING - human-readable names
        # ================================================================
        # Map canonical properties to semantic names

        # Arm extension → position_m
        if "robot_arm" in self.behaviors and "extension" in data:
            data["position_m"] = data["extension"]

        # Lift height → height_m
        if "robot_lift" in self.behaviors and "height" in data:
            data["height_m"] = data["height"]

        # All rotational actuators already have angle_rad from ROBOT_BEHAVIORS
        # angle_deg is computed by Component.get_state()

        return data

    def get_rl(self) -> np.ndarray:
        """Multi-format renderer: Actuator RL includes position + active state"""
        behavior_vec = super().get_rl()
        normalized_pos = (self.position - self.range[0]) / (self.range[1] - self.range[0])
        actuator_vec = np.array([normalized_pos, float(self.is_active)])
        return np.concatenate([actuator_vec, behavior_vec])

    def render_visualization(self) -> Optional[np.ndarray]:
        """Render actuator as position vs target graph - OFFENSIVE!

        Returns:
            RGB image (H, W, 3) uint8 showing position tracking over time

        Raises:
            RuntimeError if no history data available (OFFENSIVE!)

        Graph shows:
        - Blue solid line: Actual position from MuJoCo (data.qpos)
        - Red dashed line: Target position from controller (data.ctrl)
        - Reveals tracking performance, lag, overshoot
        """
        if len(self._position_history) == 0:
            raise RuntimeError(
                f"❌ ActuatorComponent '{self.name}' has no position history!\n"
                f"   This means sync_from_mujoco() was never called or actuator not active.\n"
                f"   FIX: Ensure actuator is connected to robot and simulation is running.\n"
                f"   Component joint_names: {self.joint_names}"
            )

        # Create figure with position tracking
        fig, ax = plt.subplots(figsize=(10, 5))

        steps = np.arange(len(self._position_history))
        position_arr = np.array(self._position_history)
        target_arr = np.array(self._target_history)

        # Plot actual vs target
        ax.plot(steps, position_arr, 'b-', linewidth=2, label='Actual Position', alpha=0.9)
        ax.plot(steps, target_arr, 'r--', linewidth=2, label='Target (Commanded)', alpha=0.8)

        # Add range limits as horizontal lines
        ax.axhline(y=self.range[0], color='gray', linestyle=':', alpha=0.5, label='Range Min')
        ax.axhline(y=self.range[1], color='gray', linestyle=':', alpha=0.5, label='Range Max')

        # Styling
        ax.set_title(f'{self.name.title()} Actuator - Position Tracking', fontsize=14, fontweight='bold')
        ax.set_xlabel('Steps', fontsize=11)
        ax.set_ylabel(f'Position ({self.unit})', fontsize=11)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add current position annotation
        current_pos = position_arr[-1]
        current_target = target_arr[-1]
        tracking_error = abs(current_pos - current_target)
        ax.text(
            0.02, 0.98,
            f'Current: {current_pos:.4f} {self.unit}\nTarget: {current_target:.4f} {self.unit}\nError: {tracking_error:.4f} {self.unit}',
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        )

        plt.tight_layout()
        return fig_to_numpy(fig)


@dataclass
class SensorComponent(Component):
    """Component with sensor data capabilities

    Modal-Oriented: Extends Component with sensor-specific methods.
    Inherits get_data(), get_rl(), sync protocol from Component.
    """

    # Sensor readings (populated by sync_from_mujoco)
    sensor_data: Dict = field(default_factory=dict)
    timestamp: float = field(default=0.0)
    frame_id: int = field(default=0)

    def get_data(self) -> Dict:
        """Multi-format renderer: Sensor data includes readings"""
        data = super().get_data()
        data.update({
            "sensor_data": self.sensor_data,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id
        })
        return data


class Asset:
    """Asset with auto-discovery and live tracking - OFFENSIVE & ELEGANT"""

    def __init__(self, name: str, config: Dict[str, Any], instance_name: str = None):
        self.name = name  # Asset type (e.g., "wood_block")
        self.instance_name = instance_name  # Unique instance ID (e.g., "block_red") - NEW!
        self.config = config
        self.components: Dict[str, Component] = {}

        # MOP: Asset knows if it's tracked by rewards (collision optimization!)
        # Default False = visual only (no physics collisions)
        # Set to True when reward is added for this asset
        self._is_tracked = False

        # MOP: Normalize config to have Component instances (not dicts!)
        # This ensures _build_from_config() has ONE path only
        if 'components' in config and config['components']:
            self._normalize_components()
            self.components = self._build_from_config()
        else:
            # Auto-discover from XML
            self.components = self._discover_from_xml()

        # PERFORMANCE: Cache component signature inspections (eliminates 4.175x inspect.signature per step!)
        # Signatures don't change during runtime - inspect ONCE at init
        self._component_sync_param_counts = {}
        import inspect
        for comp_name, component in self.components.items():
            if hasattr(component, 'sync_from_mujoco'):
                sig = inspect.signature(component.sync_from_mujoco)
                self._component_sync_param_counts[comp_name] = len(sig.parameters)

    def _normalize_components(self):
        """Convert dict specs to Component instances - ONE place only!

        MOP: Modal-to-modal = instances only! Dicts are converted HERE,
        not scattered across multiple methods.
        """
        normalized = {}
        for comp_name, comp_data in self.config['components'].items():
            # PURE MODAL TRUST: Duck typing - if it has behaviors, it's a Component!
            if hasattr(comp_data, 'behaviors'):
                # Modal instance (Component/ActuatorComponent/SensorComponent) - trust it!
                normalized[comp_name] = comp_data
            else:
                # Dict from JSON config - convert to Component instance
                behaviors = comp_data.get('behaviors', [])
                # ALWAYS add spatial if not already there
                if 'spatial' not in behaviors:
                    behaviors = ['spatial'] + behaviors

                normalized[comp_name] = Component(
                    name=comp_name,
                    behaviors=behaviors,
                    geom_names=comp_data.get('geom_names', []),
                    joint_names=comp_data.get('joint_names', []),
                    site_names=comp_data.get('site_names')
                )
        self.config['components'] = normalized

    def _build_from_config(self) -> Dict[str, Component]:
        """Build components from config - OFFENSIVE + PURE MOP

        MOP: Modal-to-modal = Component instances ONLY! (pure LEGO)
        ONE path only - no if/else, no dual type handling, no bugs!
        """
        components = {}
        for comp_name, comp_data in self.config['components'].items():
            # OFFENSIVE: Modal-to-modal = instances only! (duck typing - trust behaviors)
            assert hasattr(comp_data, 'behaviors'), (
                f"OFFENSIVE MOP violation! Asset._build_from_config() expects Component instances only.\n"
                f"Component '{comp_name}' is {type(comp_data).__name__}, missing 'behaviors' attribute!\n"
                f"Dicts should be converted in _normalize_components() before calling this method."
            )
            # Pure MOP - use Component instance directly (preserves ALL fields!)
            components[comp_name] = comp_data
        return components

    def _discover_from_xml(self) -> Dict[str, Component]:
        """Self-building: Auto-discover components from XML with site inference

        Modal-Oriented: Asset builds itself from minimal input (name).
        """
        from .xml_resolver import XMLResolver

        category = self.config.get('category', 'furniture')

        try:
            # Get resolved XML with all names
            xml_str = XMLResolver.get_full_xml(self.name, category)
            xml_components = XMLResolver.extract_components(xml_str)

            components = {}

            # Each body becomes a component
            for body_name in xml_components['bodies']:
                # Find geoms, joints, and sites belonging to this body
                geom_names = [g for g in xml_components['geoms'] if body_name in g]
                joint_names = [j for j in xml_components['joints'] if body_name in j]
                site_names = [s for s in xml_components['sites'] if body_name in s]  # NEW!

                # Infer base behaviors from structure
                behaviors = self._infer_behaviors(joint_names, geom_names, body_name)

                # Enhance with site-based inference (self-discovery!)
                behaviors = XMLResolver.infer_behaviors_from_sites(site_names, behaviors)

                components[body_name] = Component(
                    name=body_name,
                    geom_names=geom_names,
                    joint_names=joint_names,
                    site_names=site_names,  # NEW: Track sites!
                    behaviors=behaviors
                )

            # If no bodies found, create single component from asset name
            if not components:
                behaviors = self._infer_behaviors(xml_components['joints'], xml_components['geoms'], self.name)
                behaviors = XMLResolver.infer_behaviors_from_sites(xml_components['sites'], behaviors)

                components[self.name] = Component(
                    name=self.name,
                    geom_names=xml_components['geoms'],
                    joint_names=xml_components['joints'],
                    site_names=xml_components['sites'],  # NEW!
                    behaviors=behaviors
                )

            return components

        except Exception as e:
            # OFFENSIVE: Crash with educational error showing available assets
            from . import registry
            available_furniture = sorted(registry.FURNITURE)
            available_objects = sorted(registry.OBJECTS)

            raise RuntimeError(
                f"MODAL ERROR: Failed to discover Asset '{self.name}' from XML!\n"
                f"  Category: {category}\n"
                f"  Original error: {type(e).__name__}: {e}\n"
                f"\n"
                f"  AVAILABLE FURNITURE ({len(available_furniture)}):\n"
                f"    {available_furniture}\n"
                f"\n"
                f"  AVAILABLE OBJECTS ({len(available_objects)}):\n"
                f"    {available_objects}\n"
                f"\n"
                f"  FIX OPTIONS:\n"
                f"  1. Check asset name spelling - is '{self.name}' in the lists above?\n"
                f"  2. Ensure XML file exists at: modals/mujoco_assets/{category}/{self.name}/{self.name}.xml\n"
                f"  3. Run registry.refresh() to pick up new assets\n"
            ) from e

    def _infer_behaviors(self, joints: List[str], geoms: List[str], name: str) -> List[str]:
        """Infer behaviors from XML structure - OFFENSIVE

        ALL assets get 'spatial' behavior for position tracking!
        """
        behaviors = []

        # ALWAYS add spatial - ALL assets need position tracking
        behaviors.append('spatial')

        # Check joints for hinged/sliding
        for joint_name in joints:
            joint_lower = joint_name.lower()
            if any(keyword in joint_lower for keyword in ['hinge', 'revolute', 'door']):
                if 'hinged' not in behaviors:
                    behaviors.append('hinged')
            elif any(keyword in joint_lower for keyword in ['slide', 'drawer', 'prismatic']):
                if 'sliding' not in behaviors:
                    behaviors.append('sliding')

        # Check name for surface types
        name_lower = name.lower()
        if any(keyword in name_lower for keyword in ['floor', 'wall', 'ceiling', 'table', 'desk', 'counter']):
            if 'surface' not in behaviors:
                behaviors.append('surface')

        # Has geoms? → graspable (unless it's a surface/room part)
        if geoms and 'surface' not in behaviors:
            if not any(keyword in name_lower for keyword in ['floor', 'wall', 'ceiling', 'room']):
                if 'graspable' not in behaviors:
                    behaviors.append('graspable')

        # Check for rollable shapes
        for geom_name in geoms:
            geom_lower = geom_name.lower()
            if any(keyword in geom_lower for keyword in ['sphere', 'ball', 'cylinder', 'wheel']):
                if 'rollable' not in behaviors:
                    behaviors.append('rollable')

        return behaviors

    def set_preset(self, preset_name: str) -> Dict[str, float]:
        """Get joint values for preset state - OFFENSIVE

        Returns dict of {joint_name: value} for XML generation

        Example:
            door.set_preset("open") → {"door_hinge": 1.57}
            door.set_preset("closed") → {"door_hinge": 0.0}
        """
        if 'state_presets' not in self.config:
            raise ValueError(f"Asset '{self.name}' has no state presets")

        if preset_name not in self.config['state_presets']:
            raise ValueError(
                f"Preset '{preset_name}' not found for '{self.name}'. "
                f"Available: {list(self.config['state_presets'].keys())}"
            )

        if 'joint_ranges' not in self.config:
            raise ValueError(f"Asset '{self.name}' has no joint ranges")

        # Get preset value (0.0-1.0)
        preset_value = self.config['state_presets'][preset_name]

        # Convert to joint values
        joint_values = {}
        for joint_name, joint_range in self.config['joint_ranges'].items():
            min_val, max_val = joint_range
            joint_values[joint_name] = min_val + preset_value * (max_val - min_val)

        return joint_values

    def connect(self, model, data, instance_prefix: str = "", scene_assets: Dict = None):
        """Connect to live MuJoCo simulation

        Modal-Oriented: Links all components to simulation for live tracking

        Args:
            model: MuJoCo model
            data: MuJoCo data
            instance_prefix: Prefix for multi-instance scenes (e.g., "door1_")
            scene_assets: Reference to scene.assets for relational properties (self-awareness)
        """
        for component in self.components.values():
            component._mujoco_model = model
            component._mujoco_data = data
            component._instance_prefix = instance_prefix
            component._scene_assets = scene_assets  # Self-aware: component knows its neighbors

    def sync_from_mujoco(self, model, data):
        """Sync protocol: Update all components from MuJoCo

        Modal-Oriented: Asset syncs itself by syncing all its components.
        SAME PATTERN as Robot.sync_all() - uniform interface!

        Note: Sensors need robot arg, skip them here (synced via RobotModal)
        """
        for comp_name, component in self.components.items():
            if hasattr(component, 'sync_from_mujoco'):
                # PERFORMANCE: Use cached param count instead of inspect.signature (eliminates 4.175x calls per step!)
                param_count = self._component_sync_param_counts.get(comp_name, 3)
                # Only sync if method accepts (model, data), not (model, data, robot)
                if param_count == 2:
                    component.sync_from_mujoco(model, data)

    def analyze(self) -> Dict[str, Dict[str, Any]]:
        """Get all trackable properties from live simulation - OFFENSIVE

        Returns:
            {component_name: {property_name: value}}
        """
        result = {}
        for comp_name, component in self.components.items():
            comp_data = {}

            # Get all properties from component's behaviors
            for behavior in component.behaviors:
                if behavior in BEHAVIORS:
                    for prop_name in BEHAVIORS[behavior].get('properties', {}).keys():
                        try:
                            comp_data[prop_name] = component.get_state(prop_name)
                        except:
                            comp_data[prop_name] = None

            result[comp_name] = comp_data

        return result

    def mark_as_tracked(self):
        """Mark this asset as tracked by rewards - COLLISION OPTIMIZATION

        MOP: Reward tells asset "I'm tracking you!"
        Asset responds: "I need full physics then" (_is_tracked = True)

        Tracked assets get full collision detection.
        Non-tracked assets are visual only (contype=0, conaffinity=0).

        Called by Scene.add_reward() when reward is added for this asset.
        """
        self._is_tracked = True

    def get_data(self, extracted_state: Dict = None) -> Dict[str, Any]:
        """Flatten analyze() output - VIEW INTERFACE

        Args:
            extracted_state: Pre-extracted component state to avoid duplicate extraction (PERFORMANCE!)

        Matches robot modal pattern (actuators/sensors have get_data()).
        Returns flat dict for AtomicView to wrap.

        Returns:
            Flat dict: {component.property: value}
        """
        # PERFORMANCE: Use pre-extracted state if provided (avoids redundant component.get_state() calls!)
        if extracted_state is not None:
            # extracted_state is already flat: {component.property: value}
            return extracted_state

        # Fallback: Call analyze() which extracts state from components
        nested = self.analyze()
        flat = {}
        for comp_name, props in nested.items():
            for prop_name, value in props.items():
                flat[f"{comp_name}.{prop_name}"] = value
        return flat

    def get_rl(self) -> 'np.ndarray':
        """Normalize properties for RL - OFFENSIVE & MODAL-ORIENTED

        Uses BEHAVIORS.json as single source of truth for normalization ranges.
        Boolean → 0/1, Numeric → normalized by min/max from BEHAVIORS.

        Returns:
            np.ndarray: Normalized values in 0-1 range
        """
        import numpy as np

        data = self.get_data()
        normalized = []

        for key, value in data.items():
            # None → 0.0
            if value is None:
                normalized.append(0.0)
            # Boolean → 0 or 1
            elif isinstance(value, bool):
                normalized.append(1.0 if value else 0.0)
            # Numeric → normalize by BEHAVIORS.json range
            else:
                comp_name, prop_name = key.split('.')
                component = self.components[comp_name]  # OFFENSIVE - crash if missing

                # Find behavior with this property
                prop_meta = None
                for behavior in component.behaviors:
                    prop_meta = BEHAVIORS[behavior]['properties'].get(prop_name)
                    if prop_meta:
                        break

                # Normalize using metadata - OFFENSIVE: CRASH if min/max missing!
                min_val = prop_meta["min"]  # No default - must exist!
                max_val = prop_meta["max"]  # No default - must exist!

                if max_val != min_val:
                    norm_val = (value - min_val) / (max_val - min_val)
                    normalized.append(np.clip(norm_val, 0.0, 1.0))
                else:
                    normalized.append(0.0)

        return np.array(normalized)

    # Asset.__getattr__ removed - using explicit syntax now
    # Use scene.add_reward(tracked_asset=asset.name, behavior="...", threshold=...)

    def reward_hints(self):
        """OFFENSIVE - show all reward syntax examples with smooth capability"""
        print(f"\n{self.name.upper()} - Reward Syntax Examples")
        print("=" * 60)

        unit_types = BEHAVIORS.get("_unit_types", {})

        for comp_name, comp in self.components.items():
            for behavior in comp.behaviors:
                if behavior not in BEHAVIORS:
                    continue

                print(f"\n  [{behavior}]")
                for prop_name, meta in BEHAVIORS[behavior]["properties"].items():
                    default = meta.get("default")
                    states = meta.get("states", {})
                    desc = meta.get("description", "")
                    unit = meta.get("unit", "")

                    # Get unit info from _unit_types
                    unit_info = None
                    for ut_name, ut_meta in unit_types.items():
                        if ut_meta.get("physical_units") == unit:
                            unit_info = ut_meta
                            break

                    # Threshold syntax
                    if default:
                        smooth_hint = " [smooth]" if unit_info and unit_info.get("smooth") else " [discrete]"
                        example = unit_info.get("input_example", default) if unit_info else default
                        what = unit_info.get("what", desc) if unit_info else desc
                        print(f"    {comp_name}.{prop_name}({example}){smooth_hint}  # {what}")

                    # State presets
                    for val, label in states.items():
                        semantic = label.lower().replace(" ", "_")
                        print(f"    {comp_name}.{prop_name}.{semantic}  # {label}")

                    # Boolean
                    if not default and not states:
                        print(f"    {comp_name}.{prop_name} [discrete]  # {desc}")

    def list_components(self) -> List[str]:
        """List all components"""
        return list(self.components.keys())

    def list_properties(self) -> Dict[str, List[str]]:
        """List all properties by component"""
        result = {}
        for comp_name, component in self.components.items():
            props = []
            for behavior in component.behaviors:
                if behavior in BEHAVIORS:
                    behavior_props = BEHAVIORS[behavior].get('properties', {})
                    props.extend(behavior_props.keys())
            result[comp_name] = props
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Export asset structure as dict for ASSETS.json"""
        asset_dict = {
            "type": self.config.get('category', 'unknown'),
            "source": self.config.get('xml_file', ''),
            "components": {}
        }

        # Add each component with its properties from BEHAVIORS.json
        for comp_name, component in self.components.items():
            comp_dict = {}

            # Get all properties from component's behaviors
            for behavior in component.behaviors:
                if behavior in BEHAVIORS:
                    behavior_data = BEHAVIORS[behavior]
                    for prop_name, prop_data in behavior_data.get('properties', {}).items():
                        comp_dict[prop_name] = {
                            "behavior": behavior,
                            "description": prop_data.get('description', ''),
                            "unit": prop_data.get('unit', ''),
                            "states": prop_data.get('states', {}),
                            "default": prop_data.get('default'),
                            "reward_examples": []
                        }

            if comp_dict:  # Only add component if it has properties
                asset_dict["components"][comp_name] = comp_dict

        return asset_dict

    def __repr__(self):
        comp_count = len(self.components)
        return f"Asset('{self.name}', {comp_count} components)"


def load_asset(name: str, config: Dict[str, Any]) -> Asset:
    """Load asset from config - OFFENSIVE"""
    return Asset(name, config)

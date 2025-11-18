"""
SCENE MODAL - Scene composition
Pure data structures - OFFENSIVE & ELEGANT
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional, Union, Dict, Tuple
from . import registry
from .xml_resolver import XMLResolver
from .asset_modals import Asset


def _resolve_state_preset(config: dict, state: str) -> dict:
    """
    Resolve state preset to joint values - SINGLE SOURCE OF TRUTH

    Args:
        config: Asset config dict with state_presets and joint_ranges
        state: State preset name (e.g., "open", "closed")

    Returns:
        Dict of {joint_name: joint_value}
    """
    state_presets = config["state_presets"]  # OFFENSIVE - crash if missing!
    if state not in state_presets:
        raise ValueError(
            f"State preset '{state}' not found. "
            f"Available: {list(state_presets.keys())}"
        )

    preset_value = state_presets[state]
    joint_ranges = config["joint_ranges"]  # OFFENSIVE - crash if missing!
    initial_state = {}

    for joint_name, joint_range in joint_ranges.items():
        min_val, max_val = joint_range
        initial_state[joint_name] = min_val + preset_value * (max_val - min_val)

    return initial_state


@dataclass
class Placement:
    """Where an asset is - pure data"""
    asset: str  # Asset type from registry (e.g., "door", "table")
    position: Any  # (x,y,z) tuple or dict with relative info
    initial_state: Optional[dict] = None  # {joint_name: value} for initial state
    instance_name: Optional[str] = None  # Unique instance name (e.g., "north_door")
    # NEW: Surface sub-positioning (HYBRID API - semantic names OR numeric offsets)
    surface_position: Optional[str] = None  # "top_left", "top_right", "center", "bottom_left", "bottom_right"
    offset: Optional[Tuple[float, float]] = None  # (dx, dy) manual offset in meters
    # NEW: Orientation support for stable stacking
    orientation: Optional[Union[Tuple[float, float, float, float], str]] = None  # Quaternion (w,x,y,z) or preset like "upright"

    def get_xyz(self, scene=None, runtime_state=None, resolved_furniture=None) -> tuple:
        """Calculate absolute position - OFFENSIVE + PURE MOP!

        PURE MOP: Dimensions extracted from MuJoCo model (runtime) OR XML (compile-time)
        OFFENSIVE: Crashes if dimensions needed but unavailable

        Args:
            scene: Scene with placements
            runtime_state: MuJoCo runtime state (optional, for post-compile)
            resolved_furniture: List of parsed XML roots (optional, for compile-time)
        """
        if isinstance(self.position, tuple):
            return self.position

        # FIX: Handle list positions (solver returns lists, not tuples!)
        if isinstance(self.position, list) and len(self.position) == 3:
            return tuple(self.position)

        if isinstance(self.position, dict):
            # Relative placement
            if scene is None:
                return (0, 0, 0)

            rel = self.position
            base_name = rel["relative_to"]  # OFFENSIVE - crash if missing!
            relation = rel["relation"]  # OFFENSIVE - crash if missing!
            distance = rel.get("distance", None)  # May be None!

            # Find base placement - could be regular asset OR robot component!
            base = scene.find(base_name)
            if base is None:
                # Check if it's a robot component (e.g., "stretch.gripper")
                if base_name not in scene.assets:
                    return (0, 0, 0)

                # Robot component exists as asset
                # At compile time: return default (0,0,0), will be repositioned post-compile
                return (0, 0, 0)

            base_pos = base.get_xyz(scene, runtime_state, resolved_furniture)

            # Check if we need dimensions (dimension-aware relations)
            # MOP: Ask relation modal if it needs dimensions!
            from .relation_modal import get_relation
            relation_modal = get_relation(relation)
            needs_dimensions = relation_modal.dimension_aware
            force_dimensions = relation_modal.requires_dimensions  # MOP: stack_on MUST use real dimensions!

            # USER OVERRIDE: If user provides explicit distance, respect it and skip dimension extraction
            # EXCEPT for stack_on which REQUIRES accurate dimensions for proper stacking
            if distance is not None and not force_dimensions:
                # User provided explicit distance - use simple calculation!
                # This allows manual placement when user knows exact height
                return self._calculate_simple(relation, base_pos, distance)

            # ALWAYS try dimension extraction for dimension-aware relations (if no distance override)
            # This ensures semantic positioning works correctly with surface_position
            if needs_dimensions:
                # MOP: Modals communicate! Extract dimensions from MuJoCo model OR XML
                # Try 1: MuJoCo model (runtime - highest priority)
                model = runtime_state.get('model') if runtime_state else None

                if model is not None:
                    # RUNTIME: Model available - extract dimensions and calculate!
                    from .behavior_extractors import get_asset_dimensions
                    try:
                        base_dims = get_asset_dimensions(model, base_name)
                        placed_dims = get_asset_dimensions(model, self.asset)
                        return self._calculate_with_dimensions(
                            relation, base_pos, base_dims, placed_dims
                        )
                    except ValueError as e:
                        # OFFENSIVE: Dimension extraction failed!
                        raise ValueError(
                            f"❌ Cannot extract dimensions for '{self.asset}' or '{base_name}'!\n"
                            f"\nOriginal error: {e}\n"
                            f"\n💡 SOLUTION: Provide explicit distance= parameter:\n"
                            f"   ops.add_asset('{self.asset}', relative_to='{base_name}',\n"
                            f"                 relation='{relation}', distance=0.75)"
                        ) from e

                # Try 2: Parsed XML (compile-time - NEW!)
                if resolved_furniture is not None and scene is not None:
                    from .xml_resolver import extract_dimensions_from_xml
                    # Find index of base asset in placements - MOP: Search by instance_name!
                    base_idx = None
                    placed_idx = None
                    for i, p in enumerate(scene.placements):
                        if p.instance_name == base_name:  # MOP: Search by asset_id!
                            base_idx = i
                        if p.instance_name == self.instance_name:  # MOP: Search by asset_id!
                            placed_idx = i

                    if base_idx is not None and placed_idx is not None:
                        if base_idx < len(resolved_furniture) and placed_idx < len(resolved_furniture):
                            # Extract dimensions from XML - OFFENSIVE: Will crash if extraction fails!
                            # MOP: Use instance_name for body lookup (XML bodies renamed with asset_id)
                            base_dims = extract_dimensions_from_xml(resolved_furniture[base_idx], base_name)
                            placed_dims = extract_dimensions_from_xml(resolved_furniture[placed_idx], self.instance_name)
                            return self._calculate_with_dimensions(
                                relation, base_pos, base_dims, placed_dims
                            )

                # Try 3: Manual distance fallback (if not force_dimensions)
                if not force_dimensions:
                    if distance is not None:
                        # ✨ User provided distance - use simple calculation!
                        # This allows manual placement when dimensions unavailable
                        return self._calculate_simple(relation, base_pos, distance)

                    raise ValueError(
                        f"❌ Cannot place '{self.asset}' {relation} '{base_name}' - dimensions unknown!\n"
                        f"\n💡 MOP SOLUTIONS:\n"
                        f"\n1. Provide explicit distance= parameter:\n"
                        f"   ops.add_asset('{self.asset}', relative_to='{base_name}',\n"
                        f"                 relation='{relation}', distance=0.75)\n"
                        f"\n2. OR compile scene first (dimensions auto-extracted):\n"
                        f"   ops.compile()  # Then dimensions available!\n"
                        f"\n🧵 MOP: Dimensions extracted from MuJoCo model OR XML!"
                    )

                # OFFENSIVE: stack_on REQUIRES dimensions but none available!
                raise ValueError(
                    f"❌ Cannot calculate '{relation}' for '{self.asset}' on '{base_name}'!\n"
                    f"\nDimensions required but unavailable (no MuJoCo model, no XML, no distance).\n"
                    f"\n💡 This should not happen - check that resolved_furniture is passed correctly!"
                )

            # Distance provided OR relation doesn't need dimensions - use simple calculation
            return self._calculate_simple(relation, base_pos, distance or 0)

        # Default
        return (0, 0, 0)

    def _calculate_with_dimensions(self, relation: str, base_pos: tuple,
                                    base_dims: Dict, placed_dims: Dict) -> tuple:
        """MODAL-TO-MODAL: Ask relation modal to calculate!

        MOP PRINCIPLE: Relations are Pydantic modals that know how to calculate.
        NO if/elif chains - relations self-calculate!

        Args:
            relation: Relation name ("on_top", "stack_on", etc.)
            base_pos: Base object position (x, y, z)
            base_dims: Base object dimensions {width, depth, height}
            placed_dims: Placed object dimensions {width, depth, height}

        Returns:
            Calculated position (x, y, z)
        """
        from .relation_modal import get_relation

        # MODAL-TO-MODAL: Get relation modal
        relation_modal = get_relation(relation)

        # SELF-VALIDATION: Relation validates if it can be applied
        relation_modal.validate(has_dimensions=True)

        # Prepare kwargs (surface positioning for on_top)
        kwargs = {}
        if relation == "on_top" and self.surface_position:
            # Apply surface positioning
            x_offset, y_offset = self._apply_surface_positioning(base_dims)
            kwargs['offset'] = (x_offset, y_offset)

        # SELF-CALCULATION: Relation calculates position!
        return relation_modal.calculate(
            base_pos=base_pos,
            base_dims=base_dims,
            placed_dims=placed_dims,
            **kwargs
        )

    # OLD CODE DELETED (76 lines of if/elif chains!)
    # NOW: 3 lines of modal-to-modal communication!
    # NO FALLBACKS! If relation unknown, get_relation() will CRASH with helpful error!

    def _calculate_simple(self, relation: str, base_pos: tuple, distance: float) -> tuple:
        """Simple calculation with user-provided distance

        Used when distance= parameter provided OR relation doesn't need dimensions.
        """
        if relation == "on_top":
            # User-provided height offset + surface positioning
            x_offset, y_offset = self._apply_surface_positioning({'width': 1.0, 'depth': 1.0})
            return (base_pos[0] + x_offset, base_pos[1] + y_offset, base_pos[2] + distance)

        elif relation == "stack_on":
            # Centered stacking with manual distance (fallback)
            return (base_pos[0], base_pos[1], base_pos[2] + distance)

        elif relation == "front":
            return (base_pos[0], base_pos[1] + distance, base_pos[2])

        elif relation == "back":
            return (base_pos[0], base_pos[1] - distance, base_pos[2])

        elif relation == "left":
            return (base_pos[0] - distance, base_pos[1], base_pos[2])

        elif relation == "right":
            return (base_pos[0] + distance, base_pos[1], base_pos[2])

        elif relation == "next_to":
            return (base_pos[0] + distance, base_pos[1], base_pos[2])

        elif relation == "inside":
            # Place at same position (container center)
            return base_pos

        elif relation == "in_gripper":
            # Use default estimate: gripper at 0.95m height
            return (base_pos[0], base_pos[1], base_pos[2] + 0.95)

        else:
            # Unknown relation - return base position
            return base_pos

    def _apply_surface_positioning(self, base_dims: Dict) -> Tuple[float, float]:
        """Apply surface sub-positioning - HYBRID API (semantic OR numeric)

        Returns (x_offset, y_offset) in meters.
        """
        # Check for semantic surface position
        if self.surface_position:
            # Use proportional factors (0-1 range) to work with any surface size
            # 0.35 = 35% from center, leaves 15% margin on each edge
            SURFACE_POSITIONS = {
                "top_left": (-0.35, 0.35),
                "top_right": (0.35, 0.35),
                "center": (0.0, 0.0),
                "bottom_left": (-0.35, -0.35),
                "bottom_right": (0.35, -0.35),
            }

            if self.surface_position not in SURFACE_POSITIONS:
                # OFFENSIVE: Unknown surface position
                raise ValueError(
                    f"❌ Unknown surface_position '{self.surface_position}'!\n"
                    f"\n✅ Valid positions: {list(SURFACE_POSITIONS.keys())}\n"
                    f"\n💡 MOP: Use semantic names for clarity!"
                )

            factors = SURFACE_POSITIONS[self.surface_position]
            return (
                factors[0] * base_dims['width'],
                factors[1] * base_dims['depth']
            )

        # Check for manual offset
        if self.offset:
            return self.offset

        # Default: center
        return (0.0, 0.0)

    def get_quat(self, scene=None, runtime_state=None) -> Tuple[float, float, float, float]:
        """Get orientation quaternion - OFFENSIVE + PURE MOP!

        Handles:
        - Direct quaternions: (w, x, y, z)
        - Preset strings: "north", "south", "east", "west", "upright", "sideways", "inverted"
        - Relational: "facing_table", "facing_apple", "facing_origin"
        - None: Identity quaternion (1, 0, 0, 0)

        Args:
            scene: Scene with assets (for relational orientation)
            runtime_state: Optional runtime state with compiled model (for dimension extraction)

        Returns:
            Quaternion tuple (w, x, y, z) in MuJoCo format

        Example:
            >>> placement.get_quat()  # No orientation
            (1, 0, 0, 0)

            >>> placement.orientation = "north"
            >>> placement.get_quat()
            (1, 0, 0, 0)  # Face +Y

            >>> placement.orientation = "facing_table"
            >>> placement.get_quat(scene)
            # Calculated quaternion to face table position
        """
        # No orientation specified - identity quaternion
        if self.orientation is None:
            return (1.0, 0.0, 0.0, 0.0)

        # Direct quaternion tuple - pass through
        if isinstance(self.orientation, tuple):
            if len(self.orientation) != 4:
                raise ValueError(
                    f"❌ Orientation quaternion must be (w, x, y, z), got {self.orientation}!\n"
                    f"\n💡 MOP: Use 4-tuple for quaternions or string preset"
                )
            return self.orientation

        # String orientation - check for relational pattern first
        if isinstance(self.orientation, str):
            # Check for "facing_X" relational pattern
            if self.orientation.startswith("facing_"):
                # Extract target asset name
                target_name = self.orientation[7:]  # Remove "facing_" prefix

                if target_name == "origin":
                    # Special case: facing origin (0, 0, 0)
                    my_pos = self.get_xyz(scene, runtime_state, resolved_furniture=None)
                    target_pos = (0.0, 0.0, 0.0)
                else:
                    # Regular relational orientation - face another asset
                    if scene is None:
                        raise ValueError(
                            f"❌ Relational orientation '{self.orientation}' requires scene!\n"
                            f"\n💡 MOP: Pass scene to get_quat() for relational orientation"
                        )

                    # Find target asset in scene
                    target_placement = scene.find(target_name)
                    if target_placement is None:
                        available_assets = [p.instance_name for p in scene.placements]
                        raise ValueError(
                            f"❌ Target asset '{target_name}' not found in scene!\n"
                            f"\n✅ Available assets: {available_assets}\n"
                            f"\n💡 MOP: Use exact asset_id from add_asset(asset_id='...')"
                        )

                    # Get positions (pass runtime_state for dimension extraction!)
                    my_pos = self.get_xyz(scene, runtime_state, resolved_furniture=None)
                    target_pos = target_placement.get_xyz(scene, runtime_state, resolved_furniture=None)

                # Calculate quaternion to face target
                from ..utils.quaternion_utils import calculate_facing_quaternion
                return calculate_facing_quaternion(my_pos, target_pos)

            # Not relational - use preset resolver from xml_resolver
            from .xml_resolver import XMLResolver
            return XMLResolver._resolve_orientation(self.orientation)

        # Should never reach here
        raise ValueError(
            f"❌ Invalid orientation type: {type(self.orientation)}!\n"
            f"\n💡 Must be: tuple (w,x,y,z), preset string, or relational 'facing_X'"
        )

    def get_surface_info(self, scene=None, runtime_state=None) -> Dict:
        """Get surface info for this placement - PURE MOP!

        Extracts position + dimensions for scene solver calculations.
        NO HARDCODING: Uses existing dimension extraction patterns.

        Args:
            scene: Scene modal instance (for position calculation)
            runtime_state: Optional dict with 'model' key (for runtime extraction)

        Returns:
            {
                'position': (x, y, z),
                'dimensions': {'width': float, 'depth': float, 'height': float},
                'surface_z': float  # Absolute surface height (for furniture)
            }

        Example:
            >>> placement = scene.find("table")
            >>> info = placement.get_surface_info(scene, {'model': model})
            >>> info['position']
            (2.0, 0.0, 0.0)
            >>> info['dimensions']
            {'width': 1.2, 'depth': 0.8, 'height': 0.74}
        """
        # MOP FIX: If runtime_state available, extract position from compiled model!
        # Don't recalculate relational positions - use actual MuJoCo body position
        if runtime_state and 'model' in runtime_state:
            model = runtime_state['model']
            data = runtime_state.get('data')  # Optional

            try:
                # Extract position from MuJoCo body (RUNTIME!)
                import mujoco
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.instance_name)

                if body_id >= 0 and data is not None:
                    # Use actual body position from physics
                    position = tuple(data.xpos[body_id])
                elif body_id >= 0:
                    # No data, use model's default position
                    position = tuple(model.body_pos[body_id])
                else:
                    # Fallback: calculate from placement (may fail for relational)
                    position = self.get_xyz(scene, runtime_state, resolved_furniture=None)
            except:
                # Fallback if runtime extraction fails
                position = self.get_xyz(scene, runtime_state, resolved_furniture=None)
        else:
            # No runtime state - calculate from placement
            position = self.get_xyz(scene, runtime_state, resolved_furniture=None)

        # Get dimensions
        dimensions = None
        surface_z = None

        # Try runtime extraction (from MuJoCo model)
        if runtime_state and 'model' in runtime_state:
            model = runtime_state['model']
            try:
                from .behavior_extractors import get_asset_dimensions
                dimensions = get_asset_dimensions(model, self.instance_name)

                # Calculate surface_z (top of object)
                surface_z = position[2] + dimensions.get('height', 0) / 2

            except Exception as e:
                # Runtime extraction failed - will use fallback
                pass

        # Fallback: Use default dimensions
        if dimensions is None:
            dimensions = {'width': 0.1, 'depth': 0.1, 'height': 0.1}
            surface_z = position[2]

        return {
            'position': position,
            'dimensions': dimensions,
            'surface_z': surface_z if surface_z is not None else position[2]
        }

    def to_json(self) -> dict:
        """I know how to serialize myself"""
        return {"asset": self.asset, "position": self.position, "initial_state": self.initial_state}

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself"""
        return cls(**data)


@dataclass
class Scene:
    """Scene composition - just a room and placements"""
    room: Any  # RoomModal
    placements: List[Placement] = field(default_factory=list)
    assets: dict = field(default_factory=dict)  # name -> AssetModal
    robots: dict = field(default_factory=dict)  # name -> robot info (asset, state_provider, robot_modal)
    cameras: dict = field(default_factory=dict)  # name -> CameraModal (NEW!)
    conditions: dict = field(default_factory=dict)  # id -> Condition (named conditions)
    reward_modal: Any = field(default=None)  # RewardModal
    agents: dict = field(default_factory=dict)  # name -> AgentModal (PURE MOP!)

    def __post_init__(self):
        """Initialize reward modal and auto-create doors from room openings - OFFENSIVE"""
        if self.reward_modal is None:
            from .reward_modal import RewardModal
            self.reward_modal = RewardModal()

        # Auto-create door/window assets from room openings using scene.add() with state
        if hasattr(self.room, 'openings'):
            for opening in self.room.openings:
                # Calculate door position based on wall
                wall = opening["wall"]  # OFFENSIVE - crash if missing!
                if wall == "north":
                    pos = (0, self.room.length/2, 0)
                elif wall == "south":
                    pos = (0, -self.room.length/2, 0)
                elif wall == "east":
                    pos = (self.room.width/2, 0, 0)
                elif wall == "west":
                    pos = (-self.room.width/2, 0, 0)
                else:
                    pos = (0, 0, 0)

                # Get state from opening (normalized in room_modal.__post_init__)
                door_state = opening.get("state", "closed")  # LEGITIMATE - state has default

                # Create unique instance name
                instance_name = f"{wall}_door"

                # Add door with state
                from .asset_modals import Asset
                door_config = registry.load_asset_config("door")
                door = Asset("door", door_config)
                self.assets[instance_name] = door

                # Create placement with state resolved from opening
                config = registry.load_asset_config("door")

                # Resolve state using helper function
                initial_state = _resolve_state_preset(config, door_state) if door_state else None

                # Create placement with asset="door" and instance_name="north_door"
                self.placements.append(Placement("door", pos, initial_state=initial_state, instance_name=instance_name))

        # MODAL-ORIENTED: Room self-declares trackable components!
        # Floor, walls, ceiling are now trackable for rewards
        from .asset_modals import Asset

        if hasattr(self.room, 'components'):
            # MOP FIX: Use .components (Component instances), not get_components() (dicts)!
            # RoomModal now properly initializes .components in __post_init__()
            for comp_name, component in self.room.components.items():
                # Wrap each room component as trackable asset
                asset_config = {
                    "name": comp_name,
                    "category": "room_parts",
                    "xml_file": "",  # Already in room XML
                    "virtual": True,  # Not a separate XML file
                    "components": {
                        comp_name: component  # Component instance, not dict!
                    }
                }
                self.assets[comp_name] = Asset(comp_name, asset_config)

    def add_asset(self, asset: str,
                  asset_id: str = None,  # NEW - unique ID for this asset!
                  relative_to: Optional[Union[str, tuple, Placement]] = None,
                  relation: str = "at",
                  distance: float = None,  # Optional now!
                  surface_position: Optional[str] = None,  # NEW
                  offset: Optional[Tuple[float, float]] = None,  # NEW
                  orientation: Optional[Union[Tuple[float, float, float, float], str]] = None,  # NEW - quat (w,x,y,z) or preset
                  initial_state: Optional[Union[str, Dict[str, float]]] = None,
                  is_tracked: bool = False):  # PERFORMANCE OPTIMIZATION!
        """Add any asset to scene - returns Asset instance - OFFENSIVE + PURE MOP

        Args:
            asset: Asset type from registry (e.g., "wood_block", "table")
            asset_id: Unique ID for this asset (e.g., "block_red", "block_blue") - MUST BE UNIQUE!
                     If not provided, uses asset type as ID (only one instance of that type allowed).
            relative_to: Position - (x,y,z) tuple, asset name, or Placement
            relation: Spatial relation ("at", "front", "back", "left", "right", "on_top", "stack_on", "next_to", "inside")
            distance: Distance for relative placement (optional - auto-extracted from model if None)
            surface_position: Semantic surface position ("top_left", "center", etc.) - NEW
            offset: Manual (x, y) offset in meters - NEW
            orientation: Quaternion (w,x,y,z) or preset like "upright" - NEW for stable stacking
            initial_state: Initial joint state - preset name ("closed", "open") or joint dict {"door_hinge": 0.5}

        PURE MOP: If distance=None, dimensions extracted from MuJoCo model at compile-time.
        """
        from .asset_modals import Asset

        # OFFENSIVE validation - crash if asset doesn't exist
        if (asset not in registry.FURNITURE and
            asset not in registry.OBJECTS and
            asset not in registry.ROBOTS):
            available = sorted(list(registry.FURNITURE) + list(registry.OBJECTS))
            raise ValueError(
                f"Asset '{asset}' not found in registry.\n"
                f"\n✅ Available assets:\n"
                f"   {', '.join(available)}"
            )

        # Determine the name to use for this asset instance
        asset_key = asset_id if asset_id else asset

        # OFFENSIVE validation - crash if name already used (MUST BE UNIQUE!)
        if asset_key in self.assets:
            raise ValueError(
                f"❌ Asset name '{asset_key}' already exists in scene!\n"
                f"\n💡 MUST BE UNIQUE! Use a different asset_id:\n"
                f"   ops.add_asset(asset_name='{asset}', asset_id='unique_name', ...)\n"
                f"\n✅ Existing assets: {list(self.assets.keys())}"
            )

        # Determine position
        if relative_to is None:
            # Default to origin
            pos = (0, 0, 0)
        elif isinstance(relative_to, tuple):
            # Direct coordinates
            pos = relative_to
        elif isinstance(relative_to, str):
            # OFFENSIVE - crash if relative_to asset not in scene
            if relative_to not in self.assets:
                raise ValueError(
                    f"Cannot place '{asset}' relative to '{relative_to}' - "
                    f"'{relative_to}' not in scene. Add it first."
                )
            # Relative to named asset
            pos = {
                "relative_to": relative_to,
                "relation": relation,
            }
            # Only add distance if provided (don't store None)
            if distance is not None:
                pos["distance"] = distance
        elif isinstance(relative_to, Placement):
            # Relative to placement
            pos = {
                "relative_to": relative_to.asset,
                "relation": relation,
            }
            # Only add distance if provided (don't store None)
            if distance is not None:
                pos["distance"] = distance
        else:
            pos = (0, 0, 0)

        # Resolve initial state
        resolved_state = None
        if initial_state is not None:
            # Load asset config to get state presets
            config = registry.load_asset_config(asset)

            # Resolve state using helper function
            resolved_state = _resolve_state_preset(config, initial_state) if isinstance(initial_state, str) else initial_state

        # Create and add placement with resolved state + NEW surface positioning params + orientation + asset_id
        placement = Placement(
            asset, pos,
            initial_state=resolved_state,
            instance_name=asset_key,  # MOP: Use asset_key (defaults to asset type if no asset_id)!
            surface_position=surface_position,  # NEW
            offset=offset,  # NEW
            orientation=orientation  # NEW - for stable stacking
        )
        self.placements.append(placement)

        # Create Asset instance with instance identity (MOP: Asset knows who it is!)
        config = registry.load_asset_config(asset)
        asset_instance = Asset(asset, config, instance_name=asset_key)  # MOP: Use asset_key!

        # PERFORMANCE: Set tracking flag if requested
        if is_tracked:
            asset_instance._is_tracked = True

        # Store in scene using the determined key (asset_id if provided, else asset type)
        self.assets[asset_key] = asset_instance

        return asset_instance

    def add(self, *args, **kwargs):
        """Alias for add_asset() - OFFENSIVE shorthand"""
        return self.add_asset(*args, **kwargs)

    def find(self, asset_name: str) -> Optional[Placement]:
        """Find placement by instance name (asset_id) - PURE MOP + OFFENSIVE!

        Searches ONLY by instance_name (asset_id):
        - relative_to="block_red" → finds placement with instance_name="block_red" ✅
        - relative_to="table" → finds placement with instance_name="table" ✅
          (table has no asset_id, so instance_name defaults to asset type)

        MOP: Aligns lookup with storage - assets stored by asset_id, found by asset_id!
        OFFENSIVE: Crashes explicitly if not found (no silent fallback!)
        """
        # Search by instance_name (asset_id) - matches storage key!
        for p in self.placements:
            if p.instance_name == asset_name:
                return p

        # OFFENSIVE: Not found? Crash with helpful error!
        available_ids = [p.instance_name for p in self.placements]
        raise ValueError(
            f"❌ Asset '{asset_name}' not found in scene!\n"
            f"\n💡 Available asset_ids:\n"
            f"   {available_ids}\n"
            f"\n🧵 MOP: Use exact asset_id from add_asset(asset_id='...')"
        )

    def get_tracked_assets(self) -> Dict[str, Any]:
        """Get only tracked assets for selective state extraction - PERFORMANCE OPTIMIZATION!

        Returns dict of asset_name -> Asset for:
        1. Robot (always tracked - has actuators/sensors)
        2. Assets marked as tracked by add_reward() (_is_tracked=True)
        3. Reward targets (e.g., table if tracking "stacked_on table")

        This allows selective state extraction:
        - 100 blocks in scene, only 3 tracked → extract state for 3 blocks + robot + table = 5 assets
        - 95% reduction in data copying and state extraction!

        Used by:
        - RuntimeEngine._snapshot_mujoco_data() - selective body data copying
        - StateExtractor.extract() - selective state extraction
        """
        tracked = {}

        # 1. Always include robot (has actuators/sensors that must be updated)
        for robot_name in self.robots:
            if robot_name in self.assets:
                tracked[robot_name] = self.assets[robot_name]

        # 2. Include assets marked as tracked by add_reward()
        for asset_name, asset in self.assets.items():
            if hasattr(asset, '_is_tracked') and asset._is_tracked:
                tracked[asset_name] = asset

        # 3. Include reward targets (e.g., table if we have "stacked_on table")
        if self.reward_modal and hasattr(self.reward_modal, 'rewards'):
            for reward in self.reward_modal.rewards.values():
                # Get target asset from reward
                if hasattr(reward, 'target') and reward.target:
                    target_name = reward.target
                    if target_name in self.assets:
                        tracked[target_name] = self.assets[target_name]

        return tracked

    def add_robot(self, robot, relative_to=(0, 0, 0), orientation=None):
        """
        Add robot - ALL components wrapped as Assets!

        UNIFORM: Every entry in scene.assets is an Asset with .components dict.
        Robot actuators and sensors wrapped just like regular assets.

        Args:
            robot: Robot modal instance (from create_robot())
            relative_to: XYZ position in scene
            orientation: Robot orientation (optional):
                - None: Identity quaternion (1, 0, 0, 0) - faces default direction (+Y)
                - Preset string: "north", "south", "east", "west", "upright", "sideways", "inverted"
                - Relational: "facing_table", "facing_apple", "facing_origin" (auto-calculated)
                - Manual quaternion: (w, x, y, z) tuple

        Returns:
            robot name (str) for use in rewards

        Example:
            from core.main.robot_ops import create_robot

            stretch = create_robot("stretch", "stretch_1")

            # Default orientation (facing north/+Y)
            scene.add_robot(stretch, relative_to=(5, 5, 0))

            # Face east
            scene.add_robot(stretch, relative_to=(5, 5, 0), orientation="east")

            # Face toward table (relational)
            scene.add_robot(stretch, relative_to=(5, 5, 0), orientation="facing_table")

            # Manual quaternion (90° rotation around Z)
            scene.add_robot(stretch, relative_to=(5, 5, 0), orientation=(0.707, 0, 0, 0.707))

            # Track robot components in rewards!
            scene.add_reward(
                tracked_asset="stretch_1.gripper",
                behavior="holding",
                threshold=True,
                target="apple",
                reward=50,
                id="gripped"
            )
        """
        # ================================================================
        # MOP: Robot is ONE asset with ALL components inside!
        # state["robot"] = {"arm": {...}, "gripper": {...}, "base": {...}}
        # ================================================================

        # Build components dict - ALL actuators and sensors in ONE asset!
        all_components = {}

        # Add actuators as components
        for actuator_name, actuator_comp in robot.actuators.items():
            all_components[actuator_name] = actuator_comp

        # Add sensors as components (only trackable ones with behaviors)
        for sensor_name, sensor_comp in robot.sensors.items():
            trackable = getattr(sensor_comp, 'trackable_behaviors', sensor_comp.behaviors)
            if trackable:
                # Strip "robot_" prefix for cleaner component names
                # e.g., "robot_base" -> "base"
                for behavior in trackable:
                    comp_name = behavior.replace("robot_", "")
                    # Create component from sensor (has geom_names, joint_names, site_names)
                    all_components[comp_name] = sensor_comp

        # Create ONE asset for the entire robot!
        robot_config = {
            "virtual": True,
            "robot": robot.name,
            "components": all_components
        }

        # Robot as ONE asset (MOP compliant!)
        self.assets[robot.name] = Asset(robot.name, robot_config)

        # Store robot modal reference
        self.robots[robot.name] = {
            "robot_modal": robot,
            "xml_path": robot.xml_path
        }

        # Add robot to placements so it gets included in XML compilation
        # Robot type is the robot's robot_type (e.g., "stretch")
        robot_placement = Placement(
            asset=robot.robot_type,
            position=relative_to,
            orientation=orientation,  # MOP: Pass orientation through to Placement SSOT!
            initial_state=None,  # Robot initial state handled by robot XML
            instance_name=robot.name
        )
        self.placements.append(robot_placement)

        return robot.name

    def solve_robot_placement(
        self,
        robot,
        task: str,
        target_asset_name: str,
        model=None,
        **kwargs
    ) -> Dict:
        """Calculate robot placement for task - PURE MOP delegation!

        Delegates to SceneSolverModal which composes robot + asset modals.
        NO HARDCODING: All dimensions and capabilities auto-discovered.

        Args:
            robot: Robot modal instance (from add_robot)
            task: Task type ("grasp", "inspect", "manipulate")
            target_asset_name: Target asset name in scene
            model: Optional MuJoCo model (for runtime dimension extraction)
            **kwargs: Additional solver parameters

        Returns:
            {
                'position': (x, y, z),
                'orientation': (w, x, y, z) or preset string,
                'joint_positions': {'joint_name': value}
            }

        Example:
            # Add robot and assets first
            ops.add_robot("stretch", position=(0, 0, 0))
            ops.add_asset("apple", relative_to=(2, 0, 0))
            ops.compile()

            # Calculate optimal placement for grasping apple
            placement = ops.scene.solve_robot_placement(
                robot=ops.robot,
                task="grasp",
                target_asset_name="apple",
                model=ops.backend.model
            )

            # Apply placement (would need to reset robot position)
            # This is typically done BEFORE adding robot or compilation
        """
        # OFFENSIVE: Validate target asset exists
        if target_asset_name not in self.assets:
            available = list(self.assets.keys())
            raise ValueError(
                f"❌ Target asset '{target_asset_name}' not found in scene!\n"
                f"\n✅ Available assets: {available}\n"
                f"\n💡 Add it first: ops.add_asset('{target_asset_name}', ...)"
            )

        # Find target placement
        target_placement = self.find(target_asset_name)
        target_asset = self.assets[target_asset_name]

        # Create scene solver and delegate (MODAL-TO-MODAL!)
        from .scene_solver_modal import SceneSolverModal
        solver = SceneSolverModal()

        # Calculate placement
        placement = solver.solve_robot_placement(
            robot=robot,
            task=task,
            target_asset=target_asset,
            target_placement=target_placement,
            scene=self,
            model=model,
            **kwargs
        )

        return placement

    def add_camera(self,
                   camera_id: str = "birds_eye",
                   lookat: Tuple[float, float, float] = (0, 0, 0.5),
                   distance: float = 5.0,
                   azimuth: float = 90.0,
                   elevation: float = -30.0,
                   width: int = 640,
                   height: int = 480,
                   track_target: Optional[str] = None,
                   track_offset: Tuple[float, float, float] = (0, 0, 2.0)):
        """Add virtual camera to scene - SIMULATION ONLY

        Cameras are scene-level viewpoints (not robot sensors).
        Can track assets or robots by setting track_target.

        Args:
            camera_id: Unique camera name (e.g., 'birds_eye', 'side_view')
            lookat: [x, y, z] point to look at
            distance: Distance from lookat point (meters)
            azimuth: Horizontal rotation angle (degrees)
            elevation: Vertical rotation angle (degrees)
            width: Image width (pixels)
            height: Image height (pixels)
            track_target: Optional - asset or robot name to track
            track_offset: Offset from tracked target position

        Returns:
            CameraModal instance

        Example:
            # Static camera
            cam = scene.add_camera('birds_eye', lookat=(5, 5, 0.5), distance=8.0)

            # Tracking camera
            cam = scene.add_camera(
                'follow_robot',
                track_target='stretch',
                track_offset=(0, 0, 2.0)
            )
        """
        from .camera_modal import CameraModal

        # Create camera modal
        camera = CameraModal(
            camera_id=camera_id,
            lookat=list(lookat),
            distance=distance,
            azimuth=azimuth,
            elevation=elevation,
            width=width,
            height=height,
            track_target=track_target,
            track_offset=track_offset
        )

        # Connect to scene for asset tracking
        camera._scene_ref = self

        # Store in scene cameras dict
        self.cameras[camera_id] = camera

        return camera

    def add_reward(self, tracked_asset=None, behavior=None, target=None,
                   condition_id=None, reward=0, spatial_target=None,
                   within=None, after=None, after_event=None, requires=None, id=None,
                   mode=None, speed_bonus=None,
                   tolerance_override=None, natural_range_override=None):
        """
        Add reward - UNIFIED SYNTAX - TRUE MOP!

        Args:
            tracked_asset: Asset name (e.g., "door", "apple", "stretch.gripper") - mutually exclusive with condition_id
            behavior: Behavior property (e.g., "open", "held", "holding") - from BEHAVIORS.json
            target: Target value (e.g., 90 for 90° rotation, True for boolean, "floor" for spatial relations)
                   SMART DETECTION: For spatial properties (stacked_on, supporting, held_by, distance_to),
                   passing a string automatically uses it as spatial_target with val=True
            condition_id: Reference existing condition by ID - mutually exclusive with tracked_asset
            reward: Points to award
            spatial_target: (Optional) Explicit spatial target - usually not needed with smart detection
            within: Seconds from reference - condition must be met WITHIN this time
            after: Seconds from reference - only check AFTER this time
            after_event: Condition ID that starts the timer (if None, uses scene start)
            requires: Condition ID (or list of IDs) that must be met first
            id: REQUIRED - ID to name this condition for later reference
            mode: "discrete" (default), "convergent" (partial credit + penalties), "achievement" (partial credit, forgiving)
            speed_bonus: Extra points for faster completion
            tolerance_override: Override discovered tolerance (optional)
            natural_range_override: Override discovered natural range (optional)

        Usage:
            # Discrete (binary 0/100pts)
            scene.add_reward(tracked_asset="door", behavior="open", target=90,
                           reward=100, id="door_open")

            # Convergent (partial credit + overshooting penalties)
            scene.add_reward(tracked_asset="stretch.base", behavior="rotation", target=90,
                           reward=100, mode="convergent", id="precise_turn")

            # Achievement (partial credit, forgiving)
            scene.add_reward(tracked_asset="stretch.base", behavior="rotation", target=90,
                           reward=100, mode="achievement", id="reach_angle")

            # Spatial relation (CLEAN API - string target auto-detected!)
            scene.add_reward(tracked_asset="apple", behavior="stacked_on", target="table",
                           reward=100, id="apple_on_table")
            scene.add_reward(tracked_asset="stretch.gripper", behavior="holding", target="apple",
                           reward=50, id="holding_apple")

            # Reference existing condition
            scene.add_reward(condition_id="door_open", reward=10, id="bonus")
        """
        from .reward_modal import Condition

        # TRUE MOP! Minimal validation - Condition validates itself!
        # Validate parameters
        if not id:
            raise ValueError("ID is REQUIRED for all reward conditions")

        if tracked_asset and condition_id:
            raise ValueError("Cannot specify both tracked_asset and condition_id")

        if not tracked_asset and not condition_id:
            raise ValueError("Must specify either tracked_asset or condition_id")

        # Create or lookup condition
        if tracked_asset:
            # TRUE MOP! Minimal scene-level validation - Condition validates itself!
            if behavior is None:
                raise ValueError("tracked_asset requires behavior parameter")

            if target is None:
                raise ValueError("tracked_asset requires target parameter")

            # OFFENSIVE: No backward compatibility! spatial_target is DEPRECATED!
            if spatial_target is not None:
                raise ValueError(
                    f"❌ DEPRECATED: spatial_target parameter is no longer supported!\n"
                    f"\n💡 CLEAN API: Just use target=\"{spatial_target}\" instead:\n"
                    f"\n   # OLD (CRASHES):\n"
                    f"   ops.add_reward(\"{tracked_asset}\", \"{behavior}\", target=True, spatial_target=\"{spatial_target}\", ...)\n"
                    f"\n   # NEW (CLEAN!):\n"
                    f"   ops.add_reward(\"{tracked_asset}\", \"{behavior}\", target=\"{spatial_target}\", ...)\n"
                    f"\n🎯 MOP: String targets auto-detected for spatial properties!"
                )

            # Basic scene-level check: asset must exist
            if tracked_asset not in self.assets:
                available = list(self.assets.keys())
                raise ValueError(
                    f"❌ Asset '{tracked_asset}' not found in scene!\n"
                    f"\n✅ Available: {available}\n"
                    f"\n💡 Add it first: ops.add_asset('{tracked_asset}', ...)"
                )

            # TRUE MOP: Discover behavior_name from asset components - NO STRING PARSING!
            asset_modal = self.assets[tracked_asset]

            # MOP: Tell asset it's being tracked (collision optimization!)
            # Asset marks itself as needing full physics collisions
            if hasattr(asset_modal, 'mark_as_tracked'):
                asset_modal.mark_as_tracked()

            behavior_name = None

            # DYNAMIC PROPERTIES: Recognize runtime-generated properties!
            # These are created by extractors, not listed in BEHAVIORS.json
            dynamic_property_patterns = {
                'stacked_on_': 'stackable',     # stacked_on_table, stacked_on_apple, etc.
                'supporting_': 'stackable',     # supporting_apple, supporting_banana, etc.
                'held_by_': 'graspable',        # held_by_stretch.gripper, etc.
                'distance_to_': 'spatial',      # distance_to_apple, distance_to_table, etc.
            }

            # Check if this is a dynamic property
            for pattern, behavior_type in dynamic_property_patterns.items():
                if behavior.startswith(pattern):  # Does property match pattern?
                    # Check if asset has this behavior
                    for comp in asset_modal.components.values():
                        if behavior_type in comp.behaviors:
                            behavior_name = behavior_type
                            break
                    if behavior_name:
                        break

            # If not dynamic, check BEHAVIORS.json
            if behavior_name is None:
                # Load behavior schemas to check which behavior has this property
                from pathlib import Path
                import json
                try:
                    behaviors_path = Path(__file__).parent.parent / "behaviors" / "ROBOT_BEHAVIORS.json"
                    with open(behaviors_path) as f:
                        robot_behaviors = json.load(f)

                    obj_behaviors_path = Path(__file__).parent.parent / "behaviors" / "BEHAVIORS.json"
                    with open(obj_behaviors_path) as f:
                        object_behaviors = json.load(f)

                    all_behaviors = {**robot_behaviors, **object_behaviors}
                except Exception as e:
                    raise ValueError(f"Failed to load behavior schemas: {e}")

                # Find which component has this property
                for comp in asset_modal.components.values():
                    for comp_behavior in comp.behaviors:
                        # Check if this behavior has the requested property
                        if comp_behavior in all_behaviors:
                            behavior_spec = all_behaviors[comp_behavior]
                            if "properties" in behavior_spec and behavior in behavior_spec["properties"]:
                                behavior_name = comp_behavior
                                break
                    if behavior_name:
                        break

                if behavior_name is None:
                    raise ValueError(
                        f"❌ Property '{behavior}' not found in any behavior of asset '{tracked_asset}'!\n"
                        f"\n✅ Available components: {list(asset_modal.components.keys())}\n"
                    )

            # Infer operator based on property semantics
            if "distance" in behavior.lower():
                op = "<="  # Distance: trigger when CLOSE
            else:
                op = ">="  # Most properties: trigger when HIGH

            # SMART TARGET DETECTION: For spatial properties, allow target="floor" instead of target=True, spatial_target="floor"
            # If target is a string, check if this is a BASE spatial property name (without target suffix)
            actual_behavior = behavior  # Property name to use
            actual_target = target  # What goes into val
            actual_spatial_target = spatial_target  # What goes into target (Condition field)

            # Check for BASE spatial property names (without underscore suffix)
            base_spatial_properties = {
                'stacked_on': 'stackable',      # stacked_on + target → stacked_on_table
                'supporting': 'stackable',      # supporting + target → supporting_apple
                'held_by': 'graspable',         # held_by + target → held_by_gripper
                'distance_to': 'spatial',       # distance_to + target → distance_to_apple
            }

            if isinstance(target, str) and not spatial_target and behavior in base_spatial_properties:
                # CLEAN API! Build full property name: "stacked_on" + "_table" = "stacked_on_table"
                actual_behavior = f"{behavior}_{target}"
                actual_spatial_target = None  # Don't set target - it's already embedded in property name!

                # TRUE MOP: Set correct target value based on property type!
                if behavior == 'distance_to':
                    # Distance is NUMERIC (meters) - use tolerance_override as threshold
                    if tolerance_override is not None:
                        # User provides explicit distance threshold
                        actual_target = tolerance_override
                    else:
                        # Default: 10x discovered odometry tolerance (0.02m * 10 = 0.2m)
                        actual_target = 0.2
                else:
                    # Boolean spatial properties (stacked_on, supporting, held_by)
                    actual_target = True

                # TRUE MOP: Pass base property name (not category) for schema lookup!
                # reward_modal will use this to find "distance_to" in schema without parsing "distance_to_table"
                behavior_name = behavior  # "distance_to", "stacked_on", etc.

            # TRUE MOP: Create Condition - it validates itself in __post_init__!
            condition = Condition(
                asset=tracked_asset,
                prop=actual_behavior,  # Use full property name (e.g., "stacked_on_table")
                op=op,
                val=actual_target,  # UNIFIED! target is the value (or True for spatial properties)
                mode=mode if mode else "discrete",  # Default mode
                behavior_name=behavior_name,  # EXPLICIT! No parsing!
                target=actual_spatial_target,  # Spatial target (e.g., "apple")
                tolerance_override=tolerance_override,
                natural_range_override=natural_range_override
            )

            # Store in conditions dict
            self.conditions[id] = condition
        else:
            # Reference existing condition by ID
            if condition_id not in self.conditions:
                raise ValueError(f"Condition ID '{condition_id}' not found in scene.conditions")
            # Pass the ID string, not the condition object, so reward_modal can copy if needed
            condition = condition_id

        # Delegate to reward_modal.add_condition()
        self.reward_modal.add_condition(
            condition,
            reward,
            within=within,
            after=after,
            after_event=after_event,
            requires=requires,
            id=id,
            mode=mode,
            speed_bonus=speed_bonus,
            conditions_registry=self.conditions
        )

        return self

    def add_reward_sequence(self, sequence, reward: float, id: str, within=None, mode="discrete"):
        """
        Add reward for strict sequence of events - delegates to reward_modal
        Args:
            sequence: List of condition IDs that must occur in order
            reward: Points to award when sequence completes
            id: REQUIRED - ID for the sequence (for tracking by ID)
            within: Optional time constraint (seconds from first to last event)
            mode: "discrete" (default) or "smooth"

        Usage:
            # Define named conditions first
            scene.add_reward(door.handle_grabbed, reward=0, id="grabbed")
            scene.add_reward(door.open(1.4), reward=0, id="opened")

            # Reward for correct sequence
            scene.add_reward_sequence(["grabbed", "opened"], reward=200, id="door_sequence")
        """
        # Delegate to reward_modal
        self.reward_modal.add_sequence(
            sequence,
            reward,
            id=id,
            within=within,
            mode=mode,
            conditions_registry=self.conditions
        )

        # Register with ID
        from .reward_modal import SequenceCondition
        seq_condition = SequenceCondition(sequence=sequence, within=within, mode=mode)
        self.conditions[id] = seq_condition

        return self

    def add_reward_composite(self, operator, conditions, reward, id,
                            within=None, after=None, after_event=None, requires=None,
                            mode="discrete", speed_bonus=None):
        """
        Add composite reward (AND/OR/NOT) - EXPLICIT & SELF-DOCUMENTING

        Args:
            operator: "AND", "OR", or "NOT"
            conditions: List of condition IDs (or single ID for NOT)
            reward: Points to award
            id: REQUIRED - ID for this composite
            within: Time constraint (seconds)
            after: Only check after this time
            after_event: Condition ID that starts the timer
            requires: Condition ID (or list) that must be met first
            mode: "discrete" (default), "smooth", or "auto"
            speed_bonus: Extra points for speed

        Usage:
            # AND composite
            scene.add_reward_composite(
                operator="AND",
                conditions=["door_open", "apple_held"],
                reward=200,
                id="both"
            )

            # NOT composite
            scene.add_reward_composite(
                operator="NOT",
                conditions="door_open",
                reward=-50,
                after=60,
                id="penalty"
            )
        """
        from .reward_modal import AND, OR, NOT as NOT_OP

        # Validate
        if not id:
            raise ValueError("ID is REQUIRED for all reward conditions")

        # Create composite based on operator
        if operator == "AND":
            if not isinstance(conditions, list) or len(conditions) < 2:
                raise ValueError("AND requires a list of at least 2 condition IDs")
            composite = AND(*conditions)
        elif operator == "OR":
            if not isinstance(conditions, list) or len(conditions) < 2:
                raise ValueError("OR requires a list of at least 2 condition IDs")
            composite = OR(*conditions)
        elif operator == "NOT":
            if isinstance(conditions, list):
                if len(conditions) != 1:
                    raise ValueError("NOT requires exactly 1 condition ID")
                conditions = conditions[0]
            composite = NOT_OP(conditions)
        else:
            raise ValueError(f"Unknown operator '{operator}'. Use 'AND', 'OR', or 'NOT'")

        # Store composite
        self.conditions[id] = composite

        # Delegate to reward_modal
        self.reward_modal.add_condition(
            composite,
            reward,
            within=within,
            after=after,
            after_event=after_event,
            requires=requires,
            id=id,
            mode=mode,
            speed_bonus=speed_bonus,
            conditions_registry=self.conditions
        )

        return self

    def reward_catalog(self):
        """OFFENSIVE - show reward examples for ALL assets in scene"""
        for asset_name, asset in self.assets.items():
            asset.reward_hints()

    def render_xml(self, registry=None) -> str:
        """Generate complete scene XML - delegates to XMLResolver"""
        return XMLResolver.build_scene_xml(self, registry)

    def to_json(self) -> dict:
        """I know how to serialize myself"""
        return {
            "room": self.room.to_json() if hasattr(self.room, 'to_json') else {"name": self.room.name},
            "placements": [p.to_json() for p in self.placements]
        }

    def get_data(self) -> Dict[str, Any]:
        """I know my complete state - OFFENSIVE & MODAL-ORIENTED

        Scene IS the aggregator. I trust my modals to know themselves.
        No checks, no defensive code. Crashes if modal incomplete? Good.

        Returns:
            Dict with ALL scene data (room, assets, placements, rewards, robots, agents)
        """
        return {
            "room": self.room.get_data(),
            "assets": {name: asset.get_data() for name, asset in self.assets.items()},
            "placements": [{"asset": p.asset, "position": p.get_xyz(self)} for p in self.placements],
            "rewards": self.reward_modal.get_data(),
            "robots": {name: info["robot_modal"].get_data() for name, info in self.robots.items()},
            "agents": {name: agent.get_data() for name, agent in self.agents.items()}
        }

    def get_rl(self) -> 'np.ndarray':
        """I know my RL representation - OFFENSIVE & MODAL-ORIENTED

        Concatenates normalized vectors from room + all assets.

        Returns:
            np.ndarray: Concatenated normalized vectors
        """
        import numpy as np

        vectors = [self.room.get_rl()]
        vectors.extend([asset.get_rl() for asset in self.assets.values()])

        return np.concatenate(vectors)

    def is_facing(self, object1: str, object2: str, runtime_state: dict, threshold: float = 0.7) -> dict:
        """Check if object1 is facing object2 - MOP UTILITY!

        Uses extracted state (position, direction) to calculate spatial facing.
        Works with assets or components: "stretch.arm", "apple", "table.table_surface"

        Args:
            object1: Asset or component name (e.g., "stretch.arm", "apple")
            object2: Asset or component name (e.g., "table", "apple")
            runtime_state: Current state dict from engine (with extracted_state)
            threshold: Dot product threshold for "facing" (default 0.7 = ~45°)

        Returns:
            dict with:
                - facing: bool (True if dot > threshold)
                - dot: float (how much facing: 1.0 = directly facing, 0.0 = perpendicular, -1.0 = opposite)
                - dot_class: str (category: "directly_facing", "facing", "partially_facing", "perpendicular", "partially_away", "facing_away", "directly_opposite")
                - dot_explain: str (human-readable explanation with angle estimate)
                - distance: float (distance between objects in meters)
                - object1_direction: list [dx, dy, dz]
                - object2_position: list [x, y, z]

        Example:
            result = scene.is_facing("stretch.arm", "apple", runtime_state)
            print(result["dot_explain"])  # "stretch.arm is perpendicular to apple (side-by-side, ~90°)"
            if result["facing"]:
                print(f"Arm is facing apple! (dot={result['dot']:.2f})")
        """
        import numpy as np

        # Get extracted state (hierarchical: state[asset][component][property])
        state = runtime_state.get("extracted_state", {})
        if not state:
            raise ValueError("No extracted_state in runtime_state - did you call get_state()?")

        # Parse object names (handle "asset.component" or just "asset")
        def get_component_state(obj_name):
            parts = obj_name.split(".")
            if len(parts) == 2:
                asset_name, comp_name = parts
                asset_state = state.get(asset_name, {})
                return asset_state.get(comp_name, {})
            else:
                # Just asset name - get first component with spatial behavior
                asset_name = obj_name
                asset_state = state.get(asset_name, {})
                # Find first component with position
                for comp_state in asset_state.values():
                    if isinstance(comp_state, dict) and "position" in comp_state:
                        return comp_state
                return {}

        obj1_state = get_component_state(object1)
        obj2_state = get_component_state(object2)

        # Extract required data
        pos1 = obj1_state.get("position")
        pos2 = obj2_state.get("position")
        dir1 = obj1_state.get("direction")

        if not pos1:
            raise ValueError(f"Object '{object1}' has no position in state")
        if not pos2:
            raise ValueError(f"Object '{object2}' has no position in state")
        if not dir1:
            raise ValueError(f"Object '{object1}' has no direction in state (no orientation data)")

        # Calculate facing
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        dir1 = np.array(dir1)

        # Vector from obj1 to obj2
        to_obj2 = pos2 - pos1
        distance = float(np.linalg.norm(to_obj2))

        if distance < 1e-6:
            # Objects at same position - can't determine facing
            return {
                "facing": False,
                "dot": 0.0,
                "distance": 0.0,
                "object1_direction": dir1.tolist(),
                "object2_position": pos2.tolist()
            }

        # Normalize direction to obj2
        to_obj2_norm = to_obj2 / distance

        # Dot product: how aligned is dir1 with direction to obj2?
        dot = float(np.dot(dir1, to_obj2_norm))

        # Classify dot product for human understanding
        if dot > 0.95:
            dot_class = "directly_facing"
            dot_explain = f"{object1} is directly facing {object2} (perfect alignment)"
        elif dot > 0.7:
            dot_class = "facing"
            dot_explain = f"{object1} is facing {object2} (strong alignment)"
        elif dot > 0.3:
            dot_class = "partially_facing"
            dot_explain = f"{object1} is partially facing {object2} (angled ~45-60°)"
        elif dot > -0.3:
            dot_class = "perpendicular"
            dot_explain = f"{object1} is perpendicular to {object2} (side-by-side, ~90°)"
        elif dot > -0.7:
            dot_class = "partially_away"
            dot_explain = f"{object1} is angled away from {object2} (~120-135°)"
        elif dot > -0.95:
            dot_class = "facing_away"
            dot_explain = f"{object1} is facing away from {object2} (strong opposite)"
        else:
            dot_class = "directly_opposite"
            dot_explain = f"{object1} is directly opposite to {object2} (perfect opposite, 180°)"

        return {
            "facing": dot > threshold,
            "dot": dot,
            "dot_class": dot_class,
            "dot_explain": dot_explain,
            "distance": distance,
            "object1_direction": dir1.tolist(),
            "object2_position": pos2.tolist()
        }

    def get_distance(self, object1: str, object2: str, runtime_state: dict) -> dict:
        """Get distance between two objects - MOP UTILITY!"""
        import numpy as np

        state = runtime_state.get("extracted_state", {})

        def get_component_state(obj_name):
            parts = obj_name.split(".")
            if len(parts) == 2:
                asset_name, comp_name = parts
                asset_state = state.get(asset_name, {})
                return asset_state.get(comp_name, {})
            else:
                asset_name = obj_name
                asset_state = state.get(asset_name, {})
                for comp_state in asset_state.values():
                    if isinstance(comp_state, dict) and "position" in comp_state:
                        return comp_state
                return {}

        obj1_state = get_component_state(object1)
        obj2_state = get_component_state(object2)

        # MOP FIX: OFFENSIVE error handling for None positions!
        # If position is missing, tell user what's available
        pos1 = obj1_state.get("position")
        pos2 = obj2_state.get("position")

        if pos1 is None:
            available = list(state.keys())
            raise ValueError(
                f"❌ Object '{object1}' has no position in extracted_state!\n"
                f"✅ Available objects: {available}\n"
                f"💡 Did you call ops.step() before get_distance()?"
            )

        if pos2 is None:
            available = list(state.keys())
            raise ValueError(
                f"❌ Object '{object2}' has no position in extracted_state!\n"
                f"✅ Available objects: {available}\n"
                f"💡 Did you call ops.step() before get_distance()?"
            )

        pos1 = np.array(pos1)
        pos2 = np.array(pos2)

        offset = pos2 - pos1
        distance = float(np.linalg.norm(offset))
        horizontal_distance = float(np.linalg.norm(offset[:2]))
        vertical_distance = float(abs(offset[2]))

        return {
            "distance": distance,
            "horizontal_distance": horizontal_distance,
            "vertical_distance": vertical_distance,
            "offset": {
                "dx": float(offset[0]),
                "dy": float(offset[1]),
                "dz": float(offset[2])
            }
        }

    def get_offset(self, object1: str, object2: str, runtime_state: dict) -> dict:
        """Get offset from object1 to object2 - MOP UTILITY!"""
        dist_result = self.get_distance(object1, object2, runtime_state)
        return dist_result["offset"]

    def get_asset_info(self, asset_name: str, runtime_state: dict) -> dict:
        """Get asset information - MOP! Asset knows itself!

        Returns asset properties like position, height.

        Returns:
            {
                "position": [x, y, z],
                "height": float,  # Height of top surface (z-coordinate)
                "exists": bool
            }
        """
        state = runtime_state.get("extracted_state", {})

        if asset_name not in state:
            return {"exists": False, "error": f"Asset '{asset_name}' not found"}

        asset_state = state[asset_name]

        # Find main body component with position
        position = None
        for comp_name, comp_state in asset_state.items():
            if isinstance(comp_state, dict) and "position" in comp_state:
                position = comp_state["position"]
                break

        if position is None:
            return {"exists": True, "error": f"Asset '{asset_name}' has no position"}

        # Height is z-coordinate of top surface
        height = position[2]

        return {
            "exists": True,
            "position": position,
            "height": height,
            "z": height
        }

    def get_robot_info(self, robot_type: str = "stretch") -> dict:
        """Get robot specifications - MOP! Robot knows itself!

        Returns robot actuator specs + geometry for dynamic positioning calculations.
        Works BEFORE or AFTER adding robot to scene!

        Args:
            robot_type: Type of robot (e.g., "stretch") - NOT the instance name!

        Returns:
            dict with:
                - robot_type: str
                - actuators: {name: {min_position, max_position, unit, type}}
                - geometry: {gripper_length, base_to_arm_offset, base_height}
                - margins: {reach_safety, placement_safety, grasp_threshold}
                - comfortable_pct: {arm_reach, lift_height}

        Raises:
            ValueError: If robot type not supported
        """
        # Check if robot already in scene (use it if exists)
        for robot in self.robots.values():
            if robot.robot_type == robot_type and robot.actuators:
                # Use existing built robot
                return robot.get_specs()

        # Robot not in scene yet - extract specs directly from robot modules!
        # MOP: Robot specs are static per type, don't need full scene

        if robot_type == "stretch":
            # Import robot building utilities
            from .stretch import actuator_modals

            # Get specs by directly calling functions (no temp robot needed!)
            try:
                # Get geometry from XML
                geometry = actuator_modals._extract_robot_geometry()

                # Get actuators from auto-discovery
                all_actuators = actuator_modals.create_all_actuators()

                # Extract actuator specs (only position actuators)
                actuator_specs = {}
                for name, actuator in all_actuators.items():
                    # ActuatorComponent uses get_data(), not render_data()!
                    if not hasattr(actuator, 'range'):
                        continue

                    # Only include actuators with range (position actuators)
                    # Velocity actuators won't have numeric range
                    if actuator.range is None or not isinstance(actuator.range, tuple):
                        continue

                    actuator_specs[name] = {
                        "min_position": actuator.range[0],
                        "max_position": actuator.range[1],
                        "unit": actuator.unit,
                        "type": "position",  # All with range are position actuators
                    }

                # Return specs dict
                return {
                    "robot_type": robot_type,
                    "actuators": actuator_specs,
                    "geometry": {
                        "gripper_length": geometry["gripper_length"],
                        "base_to_arm_offset": geometry["base_to_arm_offset"],
                        "base_height": geometry["base_height"],
                    },
                    "margins": {
                        "reach_safety": actuator_modals.REACH_SAFETY_MARGIN,
                        "placement_safety": actuator_modals.PLACEMENT_SAFETY_MARGIN,
                        "grasp_threshold": actuator_modals.GRIPPER_GRASP_THRESHOLD,
                    },
                    "comfortable_pct": {
                        "arm_reach": actuator_modals.ARM_COMFORTABLE_REACH_PCT,
                        "lift_height": actuator_modals.LIFT_COMFORTABLE_HEIGHT_PCT,
                    }
                }
            except Exception as e:
                raise ValueError(
                    f"Failed to get specs for robot type '{robot_type}'!\n"
                    f"Error: {e}\n"
                    f"Robot type may not be supported or actuator discovery failed"
                )
        else:
            raise ValueError(
                f"Robot type '{robot_type}' not supported!\n"
                f"Currently only 'stretch' is supported.\n"
                f"Add support in scene_modal.get_robot_info()"
            )

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself"""
        from .room_modal import RoomModal
        room = RoomModal.from_json(data["room"]) if "width" in data["room"] else RoomModal(name=data["room"]["name"])
        scene = cls(room)
        for p_data in data["placements"]:  # OFFENSIVE - crash if missing!
            scene.placements.append(Placement.from_json(p_data))
        return scene



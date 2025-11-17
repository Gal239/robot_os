"""
ROOM MODAL - Room composition with openings
OFFENSIVE & ELEGANT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from .asset_modals import AssetModal


@dataclass
class RoomModal(AssetModal):
    """Room with dimensions and openings - OFFENSIVE validation"""
    width: float = 10.0
    length: float = 10.0
    height: float = 3.0
    floor_texture: str = "wood_floor"
    wall_texture: str = "gray_wall"
    ceiling_texture: Optional[str] = None
    openings: Union[List[str], List[Dict]] = field(default_factory=list)

    def __post_init__(self):
        """NORMALIZE and VALIDATE openings - CRASH if invalid - OFFENSIVE"""
        # Normalize simple strings to full dict format
        normalized = []
        for opening in self.openings:
            if isinstance(opening, str):
                # Simple string like "north" -> full dict with state
                normalized.append({
                    "wall": opening,
                    "width_m": 0.9,
                    "height_m": 2.0,
                    "position": "center",
                    "state": "closed"
                })
            else:
                # Already a dict, ensure it has state
                if "state" not in opening:
                    opening["state"] = "closed"
                normalized.append(opening)

        self.openings = normalized

        # OFFENSIVE validation - crash immediately
        for opening in self.openings:
            wall = opening["wall"]  # OFFENSIVE - crash if missing!
            width_m = opening.get("width_m", 0.9)  # LEGITIMATE - has default
            height_m = opening.get("height_m", 2.0)  # LEGITIMATE - has default

            # Determine wall width
            wall_width = self.width if wall in ["north", "south"] else self.length

            # OFFENSIVE validation - crash immediately
            assert width_m < wall_width, \
                f"Opening width {width_m}m too wide for {wall} wall ({wall_width}m)"
            assert height_m < self.height, \
                f"Opening height {height_m}m too tall for room height ({self.height}m)"
            assert width_m > 0 and height_m > 0, \
                f"Opening dimensions must be positive: width_m={width_m}, height_m={height_m}"
            assert wall in ["north", "south", "east", "west"], \
                f"Invalid wall '{wall}'. Must be north/south/east/west"

        # MOP FIX: RoomModal is a dataclass, so AssetModal.__init__() never runs!
        # We must populate .components ourselves so RoomModal is compatible with AssetModal interface
        from .asset_modals import Component
        self.components = {}
        for comp_name, comp_spec in self.get_components().items():
            self.components[comp_name] = Component(
                name=comp_name,
                behaviors=comp_spec["behaviors"],
                geom_names=comp_spec["geom_names"],
                joint_names=comp_spec["joint_names"],
                site_names=comp_spec.get("site_names", [])
            )

    def get_components(self) -> dict:
        """
        MODAL-ORIENTED: Room self-declares trackable components

        Floor, walls, ceiling are trackable with behaviors!
        Example: Track if objects fell on floor, stuck in wall, etc.
        """
        components = {
            "floor": {
                "behaviors": ["surface", "spatial", "stackable"],  # MOP: Floor supports objects like tables do!
                "geom_names": ["floor"],
                "joint_names": [],
                "site_names": ["floor_center"]
            }
        }

        # Add walls
        for wall_name in ["north", "south", "east", "west"]:
            components[f"wall_{wall_name}"] = {
                "behaviors": ["spatial", "room_boundary"],
                "geom_names": [f"wall_{wall_name}"],  # May have _left, _right, _top variants
                "joint_names": [],
                "site_names": []
            }

        # Add ceiling if it exists
        if self.ceiling_texture:
            components["ceiling"] = {
                "behaviors": ["surface", "spatial"],
                "geom_names": ["ceiling"],
                "joint_names": [],
                "site_names": []
            }

        return components

    def render_xml(self, registry=None):
        """Generate room XML from data"""
        lines = []

        # Visual settings - inline
        lines.extend([
            '<visual>',
            '  <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>',
            '  <rgba haze="0.15 0.25 0.35 1"/>',
            '</visual>',
            ''
        ])

        # Assets - textures
        lines.append('<asset>')
        if registry:
            floor_path = registry.resolve("texture", self.floor_texture)
            wall_path = registry.resolve("texture", self.wall_texture)
        else:
            floor_path = f"../textures/{self.floor_texture}.png"
            wall_path = f"../textures/{self.wall_texture}.png"

        lines.append(f'  <texture name="{self.floor_texture}" type="2d" file="{floor_path}"/>')
        lines.append(f'  <material name="floor_mat" texture="{self.floor_texture}" texrepeat="8 8"/>')
        lines.append(f'  <texture name="{self.wall_texture}" type="2d" file="{wall_path}"/>')
        lines.append(f'  <material name="wall_mat" texture="{self.wall_texture}" texrepeat="4 4"/>')
        lines.append('</asset>')
        lines.append('')

        # Worldbody
        lines.append('<worldbody>')

        # Floor (with tracking site) - Use MuJoCo DEFAULT friction (no attribute!)
        # Like original Hello Robot scene.xml - let MuJoCo decide
        lines.append(f'  <geom name="floor" type="plane" size="{self.width/2} {self.length/2} 0.1" '
                    f'pos="0 0 0" material="floor_mat"/>')
        lines.append(f'  <site name="floor_center" pos="0 0 0" size="0.01"/>')

        # Walls with openings
        self._render_walls(lines)

        # Ceiling
        if self.ceiling_texture:
            lines.append(f'  <geom name="ceiling" type="box" size="{self.width/2} {self.length/2} 0.1" '
                        f'pos="0 0 {self.height}" rgba="0.95 0.95 0.95 1"/>')

        # DON'T close worldbody - let scene do it so it can add placements
        # lines.append('</worldbody>')

        return '\n'.join(lines)

    def _render_walls(self, lines: List[str]):
        """Render walls with openings cut out - OFFENSIVE"""
        walls = {
            "north": {
                "pos": [0, self.length/2, self.height/2],
                "size": [self.width/2, 0.15, self.height/2],
                "axis": "x",
                "wall_width": self.width
            },
            "south": {
                "pos": [0, -self.length/2, self.height/2],
                "size": [self.width/2, 0.15, self.height/2],
                "axis": "x",
                "wall_width": self.width
            },
            "east": {
                "pos": [self.width/2, 0, self.height/2],
                "size": [0.15, self.length/2, self.height/2],
                "axis": "y",
                "wall_width": self.length
            },
            "west": {
                "pos": [-self.width/2, 0, self.height/2],
                "size": [0.15, self.length/2, self.height/2],
                "axis": "y",
                "wall_width": self.length
            }
        }

        for wall_name, wall_info in walls.items():
            wall_openings = [o for o in self.openings if o["wall"] == wall_name]  # OFFENSIVE!

            if not wall_openings:
                # Full wall - no openings
                pos = wall_info["pos"]
                size = wall_info["size"]
                lines.append(f'  <geom name="wall_{wall_name}" type="box" '
                           f'size="{size[0]} {size[1]} {size[2]}" '
                           f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                           f'material="wall_mat" rgba="0.9 0.9 0.9 1"/>')  # Explicit light gray
            else:
                # Wall with opening - cut into segments
                opening = wall_openings[0]  # Take first opening
                o_width = opening.get("width_m", 0.9)  # LEGITIMATE - has default
                o_height = opening.get("height_m", 2.0)  # LEGITIMATE - has default
                o_pos = opening.get("position", "center")  # LEGITIMATE - has default

                axis = wall_info["axis"]
                wall_width = wall_info["wall_width"]
                wall_height = self.height

                # Calculate opening center position along wall
                if o_pos == "center":
                    o_center = 0
                elif o_pos == "left":
                    o_center = -wall_width/2 + o_width/2
                elif o_pos == "right":
                    o_center = wall_width/2 - o_width/2
                else:
                    # Numeric offset
                    o_center = float(o_pos) if isinstance(o_pos, (int, float)) else 0

                # Left segment (before opening)
                if o_center - o_width/2 > -wall_width/2 + 0.1:
                    left_width = (o_center - o_width/2) - (-wall_width/2)
                    left_center = -wall_width/2 + left_width/2
                    pos = wall_info["pos"].copy()
                    if axis == "x":
                        pos[0] = left_center
                        size = [left_width/2, 0.15, wall_height/2]
                    else:
                        pos[1] = left_center
                        size = [0.15, left_width/2, wall_height/2]
                    lines.append(f'  <geom name="wall_{wall_name}_left" type="box" '
                               f'size="{size[0]} {size[1]} {size[2]}" '
                               f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                               f'material="wall_mat" rgba="0.9 0.9 0.9 1"/>')

                # Right segment (after opening)
                if o_center + o_width/2 < wall_width/2 - 0.1:
                    right_width = (wall_width/2) - (o_center + o_width/2)
                    right_center = wall_width/2 - right_width/2
                    pos = wall_info["pos"].copy()
                    if axis == "x":
                        pos[0] = right_center
                        size = [right_width/2, 0.15, wall_height/2]
                    else:
                        pos[1] = right_center
                        size = [0.15, right_width/2, wall_height/2]
                    lines.append(f'  <geom name="wall_{wall_name}_right" type="box" '
                               f'size="{size[0]} {size[1]} {size[2]}" '
                               f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                               f'material="wall_mat" rgba="0.9 0.9 0.9 1"/>')

                # Top segment (above opening)
                if o_height < wall_height - 0.1:
                    top_height = wall_height - o_height
                    top_z = o_height + top_height/2
                    pos = wall_info["pos"].copy()
                    pos[2] = top_z
                    if axis == "x":
                        size = [o_width/2, 0.15, top_height/2]
                    else:
                        size = [0.15, o_width/2, top_height/2]
                    lines.append(f'  <geom name="wall_{wall_name}_top" type="box" '
                               f'size="{size[0]} {size[1]} {size[2]}" '
                               f'pos="{pos[0]} {pos[1]} {pos[2]}" '
                               f'material="wall_mat" rgba="0.9 0.9 0.9 1"/>')

    def validate_point_inside(self, x: float, y: float, z: float,
                             entity_name: str, entity_type: str = "entity",
                             extra_context: dict = None) -> None:
        """MOP: Room validates its own boundaries - OFFENSIVE!

        I own the concept of "inside/outside" - entities ask ME to validate positions.
        CRASHES immediately if point is outside my bounds with educational message.

        Args:
            x, y, z: Point coordinates to validate
            entity_name: Name of entity (for error message)
            entity_type: Type of entity (camera, object, robot, etc.)
            extra_context: Optional dict with camera distance, elevation, etc. for better fix suggestions

        Raises:
            RuntimeError: If point is outside room bounds (OFFENSIVE - crashes immediately!)

        Example:
            room.validate_point_inside(camera_x, camera_y, camera_z, "apple_tracker", "camera",
                                      {"distance": 2.0, "elevation": -45, "azimuth": 90})
        """
        # Room bounds (origin-centered)
        x_min, x_max = -self.width / 2, self.width / 2
        y_min, y_max = -self.length / 2, self.length / 2
        z_min, z_max = 0.0, self.height

        # Small numerical margin (0.1m) for edge cases
        margin = 0.1
        violations = []

        if x < x_min - margin or x > x_max + margin:
            violations.append(f"X={x:.2f}m outside [{x_min:.2f}, {x_max:.2f}]m")
        if y < y_min - margin or y > y_max + margin:
            violations.append(f"Y={y:.2f}m outside [{y_min:.2f}, {y_max:.2f}]m")
        if z < z_min - margin or z > z_max + margin:
            violations.append(f"Z={z:.2f}m outside [{z_min:.2f}, {z_max:.2f}]m")

        if violations:
            # Build fix suggestions based on entity type and context
            fix_suggestions = []
            if entity_type == "camera" and extra_context:
                distance = extra_context.get("distance")
                elevation = extra_context.get("elevation")
                azimuth = extra_context.get("azimuth")
                lookat = extra_context.get("lookat")

                if distance and elevation is not None and azimuth is not None and lookat:
                    # Calculate MAXIMUM SAFE distance that keeps camera inside room
                    # This is the OFFENSIVE educational part - teach exact fix!
                    import math

                    elevation_rad = math.radians(elevation)
                    azimuth_rad = math.radians(azimuth)

                    # Camera offset direction (unit vector)
                    z_dir = math.sin(-elevation_rad)
                    horizontal_mag = math.cos(elevation_rad)
                    x_dir = horizontal_mag * math.cos(azimuth_rad)
                    y_dir = horizontal_mag * math.sin(azimuth_rad)

                    # Calculate max distance before hitting each boundary
                    max_distances = []
                    margin = 0.1

                    # X boundaries
                    if abs(x_dir) > 0.001:
                        if x_dir > 0:
                            max_d_x = (self.width/2 - margin - lookat[0]) / x_dir
                        else:
                            max_d_x = (-self.width/2 + margin - lookat[0]) / x_dir
                        if max_d_x > 0:
                            max_distances.append(max_d_x)

                    # Y boundaries
                    if abs(y_dir) > 0.001:
                        if y_dir > 0:
                            max_d_y = (self.length/2 - margin - lookat[1]) / y_dir
                        else:
                            max_d_y = (-self.length/2 + margin - lookat[1]) / y_dir
                        if max_d_y > 0:
                            max_distances.append(max_d_y)

                    # Z boundaries
                    if abs(z_dir) > 0.001:
                        if z_dir > 0:
                            max_d_z = (self.height - margin - lookat[2]) / z_dir
                        else:
                            max_d_z = (0 + margin - lookat[2]) / z_dir
                        if max_d_z > 0:
                            max_distances.append(max_d_z)

                    if max_distances:
                        safe_distance = min(max_distances)
                        fix_suggestions.append(
                            f"1. Reduce camera distance to {safe_distance:.2f}m or less (currently {distance:.2f}m)"
                        )
                    else:
                        fix_suggestions.append(f"1. Reduce camera distance (currently {distance:.2f}m)")
                else:
                    if distance:
                        fix_suggestions.append(f"1. Reduce camera distance (currently {distance:.2f}m)")

                fix_suggestions.append(f"2. Change camera elevation/azimuth")
                fix_suggestions.append(f"3. Move tracked object closer to room center")

            fix_suggestions.append(f"4. Increase room size (currently {self.width}x{self.length}x{self.height}m)")

            reason_map = {
                "camera": "Camera outside room renders GRAY (skybox/nothing visible)",
                "object": "Object outside room may fall into void or behave unexpectedly",
                "robot": "Robot outside room violates physical constraints"
            }
            reason = reason_map.get(entity_type, f"{entity_type} outside room violates spatial constraints")

            raise RuntimeError(
                f"❌ {entity_type.upper()} '{entity_name}' is OUTSIDE room bounds!\n"
                f"\n"
                f"   Position: ({x:.2f}, {y:.2f}, {z:.2f})m\n"
                f"   Room bounds:\n"
                f"     X: [{x_min:.2f}, {x_max:.2f}]m\n"
                f"     Y: [{y_min:.2f}, {y_max:.2f}]m\n"
                f"     Z: [{z_min:.2f}, {z_max:.2f}]m\n"
                f"\n"
                f"   Violations:\n"
                f"     {chr(10).join('     ' + v for v in violations)}\n"
                f"\n"
                f"   FIX OPTIONS:\n"
                f"     {chr(10).join('     ' + s for s in fix_suggestions)}\n"
                f"\n"
                f"   REASON: {reason}.\n"
                f"   MOP: Room validates its own boundaries - you violated my space!"
            )

    def to_json(self) -> dict:
        """I know how to serialize myself"""
        return {
            "name": self.name,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "floor_texture": self.floor_texture,
            "wall_texture": self.wall_texture,
            "ceiling_texture": self.ceiling_texture,
            "openings": self.openings
        }

    def get_data(self) -> dict:
        """Get room state - VIEW INTERFACE

        Matches robot modal pattern (actuators/sensors have get_data()).
        Returns dict for AtomicView to wrap.

        Returns:
            Dict with room dimensions, textures, openings
        """
        return {
            "name": self.name,
            "width": self.width,
            "length": self.length,
            "height": self.height,
            "openings": self.openings,
            "floor_texture": self.floor_texture,
            "wall_texture": self.wall_texture,
            "ceiling_texture": self.ceiling_texture
        }

    def get_rl(self) -> 'np.ndarray':
        """Normalize dimensions - RL INTERFACE

        Matches robot modal pattern (actuators/sensors have get_rl()).
        Returns normalized numpy array for RL training.

        Returns:
            np.ndarray: [width_norm, length_norm, height_norm, opening_count]
        """
        import numpy as np

        return np.array([
            self.width / 20.0,         # Normalize - max 20m
            self.length / 20.0,        # Normalize - max 20m
            self.height / 5.0,         # Normalize - max 5m
            float(len(self.openings))  # Raw count (0-4 range is fine)
        ])

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself"""
        return cls(**data)

    def save(self, directory: Optional[str] = None):
        """
        MODAL SELF-GENERATION: Save room specification to JSON

        Modal creates the JSON, not manual editing!
        """
        import json
        from pathlib import Path

        if directory is None:
            directory = Path(__file__).parent.parent.parent / "assets" / "rooms"
        else:
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / f"{self.name}.json"
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f, indent=2)

        return filepath

    @classmethod
    def load(cls, room_name: str, directory: Optional[str] = None):
        """
        Load room from generated JSON

        Args:
            room_name: Room name (e.g., "kitchen")
            directory: Optional custom directory

        Returns:
            RoomModal instance

        Raises:
            FileNotFoundError: If room JSON doesn't exist
        """
        import json
        from pathlib import Path

        if directory is None:
            directory = Path(__file__).parent.parent.parent / "assets" / "rooms"
        else:
            directory = Path(directory)

        filepath = directory / f"{room_name}.json"

        if not filepath.exists():
            raise FileNotFoundError(
                f"Room '{room_name}' not found at {filepath}\n"
                f"Create it first:\n"
                f"  room = RoomModal('{room_name}', width=5, length=5, height=3)\n"
                f"  room.save()"
            )

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_json(data)
"""
XML RESOLVER - Takes the shitty XMLs and makes them trackable
OFFENSIVE - crashes on bad XML
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union, Dict
from dataclasses import dataclass
from core.modals.keyframe_builder import KeyframeBuilder


def _calculate_mesh_bbox(mesh_file_path: Path) -> Optional[Tuple[float, float, float]]:
    """Calculate bounding box HALF-sizes from OBJ mesh file

    Reads vertex positions from .obj file and calculates tight bounding box.

    Args:
        mesh_file_path: Path to .obj mesh file

    Returns:
        (half_width, half_depth, half_height) or None if file unreadable
    """
    try:
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')

        with open(mesh_file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):  # Vertex line
                    parts = line.split()
                    if len(parts) >= 4:
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        min_x, max_x = min(min_x, x), max(max_x, x)
                        min_y, max_y = min(min_y, y), max(max_y, y)
                        min_z, max_z = min(min_z, z), max(max_z, z)

        # Calculate HALF-sizes (MuJoCo convention)
        half_width = (max_x - min_x) / 2.0
        half_depth = (max_y - min_y) / 2.0
        half_height = (max_z - min_z) / 2.0

        return (half_width, half_depth, half_height)
    except (FileNotFoundError, ValueError, IOError):
        return None


def _parse_geom_size(geom_elem: ET.Element) -> Optional[Tuple[float, float, float]]:
    """Parse geom size attribute into (width, depth, height) HALF-sizes

    MuJoCo convention: size attribute contains HALF-dimensions!

    Args:
        geom_elem: Geom XML element

    Returns:
        (half_width, half_depth, half_height) or None if no size attribute
    """
    size_str = geom_elem.get("size")
    if not size_str:
        return None

    try:
        parts = [float(x) for x in size_str.split()]
        geom_type = geom_elem.get("type", "box")

        if geom_type == "sphere":
            # Sphere: size="radius" → return (r, r, r)
            radius = parts[0]
            return (radius, radius, radius)
        elif geom_type == "box":
            # Box: size="x y z" → return as-is (already half-sizes)
            if len(parts) >= 3:
                return (parts[0], parts[1], parts[2])
        elif geom_type == "mesh":
            # Mesh: size="x y z" (injected by _inject_mesh_sizes!)
            # Same as box - bounding box half-sizes
            if len(parts) >= 3:
                return (parts[0], parts[1], parts[2])
        # Other types: capsule, cylinder, etc. - skip for now

        return None
    except (ValueError, IndexError):
        return None


def extract_dimensions_from_xml(asset_root: ET.Element, asset_name: str) -> Dict[str, float]:
    """Extract dimensions from parsed XML ElementTree (COMPILE-TIME!)

    PURE MOP: Asset config declares its own surface geom!
    - Furniture with "surface" behavior: Uses pos.z + size.z (actual surface height)
    - Objects without "surface": Uses size.z * 2 (bounding box)

    Args:
        asset_root: ET.Element root of asset XML
        asset_name: Name of asset (e.g., "table", "foam_brick")

    Returns:
        Dict with 'width', 'depth', 'height' in meters (FULL sizes, not half)

    Raises:
        ValueError: If no usable geom found (OFFENSIVE!)
    """
    # MOP: Load asset config to check if it has "surface" behavior!
    from . import registry
    try:
        config = registry.load_asset_config(asset_name)

        # Check if this asset has "surface" behavior (furniture with elevated surfaces)
        has_surface = any(
            "surface" in comp.get("behaviors", [])
            for comp in config.get("components", {}).values()
        )

        # Get surface geom name from config
        surface_geom_name = None
        if has_surface:
            for comp in config.get("components", {}).values():
                if "surface" in comp.get("behaviors", []):
                    geom_names = comp.get("geom_names", [])
                    if geom_names:
                        surface_geom_name = geom_names[0]  # First geom is the surface
                        break

        # Extract surface height from declared geom (MOP!)
        if has_surface and surface_geom_name:
            geom = asset_root.find(f".//geom[@name='{surface_geom_name}']")
            if geom is not None:
                half_size = _parse_geom_size(geom)
                if half_size:
                    # Get geom position
                    pos_str = geom.get('pos', '0 0 0')
                    pos = [float(x) for x in pos_str.split()]

                    # Return BOTH bounding box AND surface position!
                    # height = bounding box (for calculations)
                    # surface_z = where objects should sit (absolute position)
                    return {
                        'width': float(half_size[0] * 2),
                        'depth': float(half_size[1] * 2),
                        'height': float(half_size[2] * 2),  # ✅ Bounding box (thickness)
                        'surface_z': pos[2] + half_size[2]   # ✅ Surface position (0.76m for table)
                    }
    except (ValueError, KeyError):
        # Config not found or incomplete - fall back to pattern matching
        pass

    # FALLBACK: Try common geom naming patterns (for objects without config)
    geom_patterns = [
        f"{asset_name}_top_geom",              # Most common for furniture
        f"{asset_name}_top",                    # Explicit names without suffix (e.g., table_top)
        f"{asset_name}_geom",                   # Simple objects
        f"{asset_name}_{asset_name}_mesh_geom", # YCB mesh objects
        f"{asset_name}_body0_geom"              # Auto-generated bodies
    ]

    for geom_name in geom_patterns:
        # Find geom by name
        geom = asset_root.find(f".//geom[@name='{geom_name}']")
        if geom is not None:
            half_size = _parse_geom_size(geom)
            if half_size:
                # Objects (no surface behavior) - use bounding box
                return {
                    'width': float(half_size[0] * 2),
                    'depth': float(half_size[1] * 2),
                    'height': float(half_size[2] * 2)
                }

    # Fallback: Find first non-plane geom with size
    for geom in asset_root.findall(".//geom"):
        geom_type = geom.get("type", "box")
        if geom_type != "plane":  # Skip infinite planes
            half_size = _parse_geom_size(geom)
            if half_size:
                return {
                    'width': float(half_size[0] * 2),
                    'depth': float(half_size[1] * 2),
                    'height': float(half_size[2] * 2)
                }

            # Mesh geom without size attribute - use ESTIMATE!
            # YCB objects are typically 0.05-0.15m in each dimension
            # Use GENEROUS estimate: 0.15m cube (prevents physics collapse during stacking)
            if geom_type == "mesh":
                import warnings
                warnings.warn(
                    f"⚠️  Using estimated dimensions for mesh object '{asset_name}' (0.15m cube).\n"
                    f"    Mesh geoms don't have explicit sizes in XML.\n"
                    f"    For precise placement, use explicit distance= or compile+runtime extraction!"
                )
                return {
                    'width': 0.15,
                    'depth': 0.15,
                    'height': 0.15
                }

    # OFFENSIVE: No usable geom found!
    raise ValueError(
        f"❌ Cannot extract dimensions from XML for '{asset_name}'!\n"
        f"\nTried geom names: {geom_patterns}\n"
        f"\n💡 MOP SOLUTIONS:\n"
        f"1. Ensure asset XML has geom with 'size' attribute\n"
        f"2. Provide explicit distance= parameter:\n"
        f"   ops.add_asset('{asset_name}', relative_to=..., relation=..., distance=0.75)\n"
        f"\n🧵 MOP: Dimensions extracted from XML at COMPILE-TIME!"
    )


@dataclass
class XmlSection:
    """Self-contained XML section merger - OFFENSIVE & ELEGANT

    Eliminates 200+ lines of repetitive _merge_X_sections() methods.
    Each section knows how to merge itself from multiple XML sources.

    Example:
        asset_section = XmlSection("asset")
        asset_section.merge(resolved_xmls, lines)

        keyframe_section = XmlSection("keyframe")
        keyframe_section.merge_with_robot(resolved_xmls, lines, robot_modal)
    """
    tag: str  # Section tag name ("asset", "default", "actuator", etc.)
    indent: str = "    "  # Indent for children (4 spaces)

    def merge(self, resolved_xmls: List[ET.Element], lines: List[str],
              path_fix_fn: Optional[Callable[[ET.Element], None]] = None):
        """Merge all sections with this tag - OFFENSIVE

        Args:
            resolved_xmls: List of parsed XML roots to merge from
            lines: Output lines list (modified in-place)
            path_fix_fn: Optional function to fix paths in children (e.g., make absolute)
        """
        has_content = False
        content = []

        for xml_root in resolved_xmls:
            section = xml_root.find(f".//{self.tag}")
            if section is not None:
                has_content = True
                for child in section:
                    # Apply path fixes if provided
                    if path_fix_fn:
                        path_fix_fn(child)
                    child_str = ET.tostring(child, encoding='unicode')
                    content.append(f'{self.indent}{child_str.strip()}')

        # Only add section if we have content
        if has_content:
            lines.append(f'  <{self.tag}>')
            lines.extend(content)
            lines.append(f'  </{self.tag}>')
            lines.append('')

    def merge_with_robot(self, resolved_xmls: List[ET.Element], lines: List[str], robot=None, robot_position=(0, 0, 0), robot_orientation=None, freejoint_objects=[]):
        """Merge keyframe section using CENTRAL KeyframeBuilder - OFFENSIVE!

        NEW: Uses KeyframeBuilder (single source of truth) instead of scattered generation.
        This fixes:
        - Order-dependent compilation bug
        - Multi-robot scenarios (was broken)
        - Missing freejoint object initialization (was ignored!)

        Args:
            robot_position: Robot's actual position from Placement (SSOT for position)
            robot_orientation: Robot's quaternion (w,x,y,z) from Placement (SSOT for orientation)
            freejoint_objects: List of (body_name, position, quaternion) tuples for objects with freejoint
                              Format: [(name, (x,y,z), (qw,qx,qy,qz)), ...]
        """
        has_content = False
        content = []

        # NEW: Use central KeyframeBuilder instead of robot.render_initial_state_xml()!
        # This generates ONE complete keyframe with robot + ALL freejoint objects
        if robot is not None:
            has_content = True
            # Call central builder - it handles robot only!
            # NOTE: freejoint_objects are IGNORED - MuJoCo initializes them from <body pos="...">
            robot_keyframe_xml = KeyframeBuilder.build_initial_keyframe(
                robot=robot,
                robot_position=robot_position,
                robot_orientation=robot_orientation,
                freejoint_objects=None  # IGNORED - objects init from body pos
            )

            robot_keyframe_root = ET.fromstring(robot_keyframe_xml)
            key_element = robot_keyframe_root.find('.//key')
            if key_element is not None:
                key_str = ET.tostring(key_element, encoding='unicode')
                content.append(f'{self.indent}{key_str.strip()}')

        # Collect OTHER keyframes from assets (e.g., "home" from stretch.xml)
        # Skip any 'initial' keyframes from assets to avoid duplicates
        for xml_root in resolved_xmls:
            section = xml_root.find(f".//{self.tag}")
            if section is not None:
                for child in section:
                    # Skip 'initial' keyframes - we generate that centrally now
                    key_name = child.get('name', '')
                    if key_name == 'initial':
                        print(f"  [XMLResolver] Skipping duplicate 'initial' keyframe from asset (using central builder instead)")
                        continue

                    has_content = True
                    child_str = ET.tostring(child, encoding='unicode')
                    content.append(f'{self.indent}{child_str.strip()}')

        # Only add section if we have content
        if has_content:
            lines.append(f'  <{self.tag}>')
            lines.extend(content)
            lines.append(f'  </{self.tag}>')
            lines.append('')


class XMLResolver:
    """Resolves includes and adds names to EVERYTHING - OFFENSIVE"""

    @staticmethod
    def resolve_and_name(xml_path: Path, asset_name: str) -> str:
        """Take main XML, resolve ALL includes, add names to EVERY geom/joint"""

        # Parse main XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        base_dir = xml_path.parent

        # Resolve all includes recursively
        XMLResolver._resolve_includes(root, base_dir)

        # NEW: Add mesh bounding boxes as size attributes!
        XMLResolver._inject_mesh_sizes(root, base_dir)

        # FIX: Correct furniture geom groups (group 1 → group 0 for visual geoms)
        # XMLResolver._fix_furniture_geom_groups(root)  # TODO: Not implemented, not needed

        # Now add names to EVERYTHING
        XMLResolver._add_names(root, asset_name)

        # Return the full XML with names
        return ET.tostring(root, encoding='unicode')

    @staticmethod
    def _resolve_includes(elem: ET.Element, base_dir: Path):
        """Recursively resolve all <include> tags - OFFENSIVE"""

        # Find all includes
        includes = elem.findall(".//include")

        for include in includes:
            file_ref = include.get("file")
            if not file_ref:
                continue

            # Load the included file
            include_path = base_dir / file_ref
            if not include_path.exists():
                # Try variations
                if "body" in file_ref and not include_path.exists():
                    # Try numbered bodies
                    for i in range(10):
                        alt_path = base_dir / file_ref.replace("body", f"body{i}")
                        if alt_path.exists():
                            include_path = alt_path
                            break

            if not include_path.exists():
                print(f"WARNING: Include not found: {include_path}")
                elem.remove(include)
                continue

            # Parse included file
            inc_tree = ET.parse(include_path)
            inc_root = inc_tree.getroot()

            # Handle mujocoinclude wrapper
            if inc_root.tag == "mujocoinclude":
                # Get contents
                children = list(inc_root)
            else:
                children = [inc_root]

            # Insert contents where include was
            parent = elem
            for parent in elem.iter():
                if include in parent:
                    idx = list(parent).index(include)
                    parent.remove(include)
                    for i, child in enumerate(children):
                        parent.insert(idx + i, child)
                    break

            # Recursively resolve includes in inserted content
            for child in children:
                XMLResolver._resolve_includes(child, base_dir)

    @staticmethod
    def _inject_mesh_sizes(root: ET.Element, base_dir: Path):
        """Inject bounding box sizes into mesh geoms - OFFENSIVE!

        For mesh geoms without size attribute, calculate bbox from .obj file
        and add size attribute to XML. This enables compile-time dimension extraction!
        """
        # Find all mesh geoms
        for geom in root.findall(".//geom"):
            if geom.get("type") != "mesh":
                continue

            # Skip if already has size attribute
            if geom.get("size") is not None:
                continue

            # Find mesh reference
            mesh_name = geom.get("mesh")
            if not mesh_name:
                continue

            # Find mesh definition in <asset> section
            mesh_elem = root.find(f".//mesh[@name='{mesh_name}']")
            if mesh_elem is None:
                continue

            # Get mesh file path
            mesh_file = mesh_elem.get("file")
            if not mesh_file:
                continue

            # Resolve relative path
            mesh_path = base_dir / mesh_file
            if not mesh_path.exists():
                continue

            # Calculate bounding box from mesh file
            bbox = _calculate_mesh_bbox(mesh_path)
            if bbox is None:
                continue

            # Inject size attribute (MuJoCo uses HALF-sizes!)
            size_str = f"{bbox[0]} {bbox[1]} {bbox[2]}"
            geom.set("size", size_str)

    @staticmethod
    def _add_names(elem: ET.Element, prefix: str):
        """Add names to ALL geoms, joints, bodies that don't have them"""

        # Track counts for unique names
        counts = {
            "geom": 0,
            "joint": 0,
            "body": 0,
            "site": 0
        }

        # Add names to bodies
        for body in elem.findall(".//body"):
            if not body.get("name"):
                body.set("name", f"{prefix}_body{counts['body']}")
                counts['body'] += 1

        # Add names to geoms
        for geom in elem.findall(".//geom"):
            if not geom.get("name"):
                # Try to use mesh/material as hint
                mesh = geom.get("mesh", "")
                material = geom.get("material", "")
                geom_type = geom.get("type", "mesh")

                if mesh:
                    geom.set("name", f"{prefix}_{mesh}_geom")
                elif material:
                    geom.set("name", f"{prefix}_{material}_geom{counts['geom']}")
                else:
                    geom.set("name", f"{prefix}_{geom_type}{counts['geom']}")
                counts['geom'] += 1

        # Add names to joints
        for joint in elem.findall(".//joint"):
            if not joint.get("name"):
                joint_type = joint.get("type", "free")
                axis = joint.get("axis", "")
                if axis:
                    axis_name = axis.replace(" ", "").replace("1", "x").replace("0", "")
                    joint.set("name", f"{prefix}_{joint_type}_{axis_name}")
                else:
                    joint.set("name", f"{prefix}_{joint_type}{counts['joint']}")
                counts['joint'] += 1

        # Add names to sites
        for site in elem.findall(".//site"):
            if not site.get("name"):
                site.set("name", f"{prefix}_site{counts['site']}")
                counts['site'] += 1

    @staticmethod
    def get_full_xml(asset_name: str, category: str) -> str:
        """Get fully resolved XML with all names - OFFENSIVE"""

        base_dir = Path(__file__).parent / "mujoco_assets"
        xml_path = base_dir / category / asset_name / f"{asset_name}.xml"

        if not xml_path.exists():
            raise ValueError(f"No XML for {category}/{asset_name}")

        return XMLResolver.resolve_and_name(xml_path, asset_name)

    @staticmethod
    def extract_components(xml_str: str) -> dict:
        """Extract all named components from resolved XML

        Modal-Oriented: Self-discovery of all structural elements
        """

        root = ET.fromstring(xml_str)

        components = {
            "bodies": [],
            "geoms": [],
            "joints": [],
            "sites": []
        }

        for body in root.findall(".//body"):
            name = body.get("name")
            if name:
                components["bodies"].append(name)

        for geom in root.findall(".//geom"):
            name = geom.get("name")
            if name:
                components["geoms"].append(name)

        for joint in root.findall(".//joint"):
            name = joint.get("name")
            if name:
                components["joints"].append(name)

        for site in root.findall(".//site"):
            name = site.get("name")
            if name:
                components["sites"].append(name)

        return components

    @staticmethod
    def infer_behaviors_from_sites(site_names: List[str], existing_behaviors: List[str]) -> List[str]:
        """Self-discovery: Infer behaviors from site naming patterns

        Modal-Oriented: Assets discover their own capabilities from semantic markers.

        Site patterns (semantic naming):
            grasp_* → graspable behavior
            place_* → surface behavior
            attach_* → attachable behavior
            sensor_* → sensory behavior
            actuator_* → actuated behavior
        """
        behaviors = list(existing_behaviors)

        for site_name in site_names:
            site_lower = site_name.lower()

            if site_lower.startswith("grasp_") and "graspable" not in behaviors:
                behaviors.append("graspable")

            if site_lower.startswith("place_") and "surface" not in behaviors:
                behaviors.append("surface")

            if site_lower.startswith("attach_") and "attachable" not in behaviors:
                behaviors.append("attachable")

            if site_lower.startswith("sensor_") and "sensory" not in behaviors:
                behaviors.append("sensory")

            if site_lower.startswith("actuator_") and "actuated" not in behaviors:
                behaviors.append("actuated")

        return behaviors

    @staticmethod
    def build_scene_xml(scene, registry) -> str:
        """
        Build complete scene XML from room + placements - OFFENSIVE & ELEGANT

        Moved from scene_modal.py for modal-oriented architecture

        Args:
            scene: Scene object with room, placements
            registry: Asset registry for loading assets

        Returns:
            Complete MuJoCo XML string
        """
        lines = [f'<mujoco model="{scene.room.name}">']
        lines.append('  <compiler angle="radian"/>')

        # Physics options - CRITICAL for gripper to work properly!
        # Must include integrator settings from robot XML
        # timestep=0.005 = 200 Hz (STABLE! Save rate controlled separately at 30 FPS)
        # NOTE: Tried iterations=4, noslip_iterations=3 but got SLOWER (9.20:1 vs 9.86:1)
        # MuJoCo defaults are already well-tuned - don't override them
        lines.append('  <option integrator="implicitfast" impratio="20" cone="elliptic" timestep="0.005">')
        lines.append('    <flag multiccd="enable"/>')
        lines.append('  </option>')
        lines.append('  <size njmax="5000" nconmax="5000" />')

        # Visual settings - offscreen rendering buffer for high-res cameras
        lines.append('  <visual>')
        lines.append('    <global offwidth="1920" offheight="1080"/>')
        lines.append('    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>')
        lines.append('    <rgba haze="0.15 0.25 0.35 1"/>')
        lines.append('  </visual>')
        lines.append('')

        # Resolve all furniture XMLs (expand includes)
        resolved_furniture = XMLResolver._resolve_furniture_xmls(scene, registry)

        # Render room and merge furniture assets/defaults
        room_xml = scene.room.render_xml(registry)
        room_lines = room_xml.split('\n')

        # Process room XML line by line
        for line in room_lines:
            if '</asset>' in line:
                # Merge furniture assets (with path fixes) before closing
                XMLResolver._merge_asset_sections(scene, resolved_furniture, registry, lines)
                lines.append('  </asset>')
                lines.append('')

                # Merge furniture defaults (class definitions) - ELEGANT with XmlSection!
                XmlSection("default").merge(resolved_furniture, lines)
            else:
                lines.extend(['  ' + line])

        # Merge furniture bodies into worldbody and track freejoint objects for keyframe
        freejoint_objects = XMLResolver._merge_worldbody_placements(scene, resolved_furniture, registry, lines)

        # Close worldbody
        lines.append('  </worldbody>')
        lines.append('')

        # Merge sections from all assets (including robots) - ELEGANT with XmlSection!
        # Replaces 6 individual _merge_X_sections() methods (200+ lines → 6 lines!)
        XmlSection("actuator").merge(resolved_furniture, lines)
        XmlSection("sensor").merge(resolved_furniture, lines)
        XmlSection("tendon").merge(resolved_furniture, lines)
        XmlSection("equality").merge(resolved_furniture, lines)
        XmlSection("contact").merge(resolved_furniture, lines)

        # Get robot modal from scene.robots dict (if any robots exist)
        robot_modal = None
        robot_position = (0, 0, 0)  # Default position
        robot_orientation = None  # Default orientation (identity quaternion)
        if scene.robots:
            # Get first robot's robot_modal
            first_robot_info = next(iter(scene.robots.values()))
            robot_modal = first_robot_info.get("robot_modal")

            # FIX MOP VIOLATION: Get robot position AND orientation from placement (SSOT)
            # Find robot placement to get actual position + orientation (not from robot.initial_*)
            robot_placement = next((p for p in scene.placements if p.asset in registry.ROBOTS), None)
            if robot_placement:
                robot_position = robot_placement.get_xyz(scene)
                robot_orientation = robot_placement.get_quat(scene)  # NEW: Get orientation from Placement!


        # Merge keyframe section + robot's SELF-GENERATED initial state + object positions
        XmlSection("keyframe").merge_with_robot(resolved_furniture, lines, robot_modal, robot_position, robot_orientation, freejoint_objects)

        # Close mujoco
        lines.append('</mujoco>')

        return '\n'.join(lines)

    @staticmethod
    def _resolve_furniture_xmls(scene, registry):
        """Parse and resolve all furniture XMLs with includes expanded - OFFENSIVE"""
        resolved = []
        for placement in scene.placements:
            asset_path = XMLResolver._get_asset_path(placement.asset, registry)
            if asset_path is None:
                continue

            # For OBJECTS: add names to geoms (critical for contact tracking)
            # For FURNITURE: only resolve includes (furniture uses body names for position tracking)
            if placement.asset in registry.OBJECTS:
                resolved_xml = XMLResolver.resolve_and_name(asset_path, placement.asset)
                asset_root = ET.fromstring(resolved_xml)
            else:
                # Furniture/Robots: just resolve includes
                asset_tree = ET.parse(asset_path)
                asset_root = asset_tree.getroot()
                XMLResolver._resolve_includes(asset_root, asset_path.parent)

            resolved.append(asset_root)

        return resolved

    @staticmethod
    def _get_asset_path(asset_name: str, registry):
        """Get asset XML path - OFFENSIVE"""
        if asset_name in registry.FURNITURE:
            return Path(registry.ASSETS_DIR) / "furniture" / asset_name / f"{asset_name}.xml"
        elif asset_name in registry.OBJECTS:
            return Path(registry.ASSETS_DIR) / "objects" / asset_name / f"{asset_name}.xml"
        elif asset_name in registry.ROBOTS:
            return Path(registry.ASSETS_DIR) / "robots" / asset_name / f"{asset_name}.xml"
        return None

    @staticmethod
    def _merge_asset_sections(scene, resolved_furniture, registry, lines):
        """Merge asset sections from furniture with path fixes - OFFENSIVE + PURE MOP!

        MOP: Asset definitions (mesh, texture, material) are SHARED resources.
        Only add each asset TYPE once - multiple instances reference the same definitions.
        """
        # MOP: Deduplicate asset definitions by asset type!
        seen_assets = set()

        for idx, asset_root in enumerate(resolved_furniture):
            asset_section = asset_root.find(".//asset")
            if asset_section is None:
                continue

            # Get asset directory for fixing relative paths
            placement = scene.placements[idx]
            asset_type = placement.asset  # e.g., "wood_block"

            # MOP: Skip duplicate asset types - definitions are SHARED!
            if asset_type in seen_assets:
                continue
            seen_assets.add(asset_type)

            # Get asset directory for path resolution
            if asset_type in registry.FURNITURE:
                asset_dir = Path(registry.ASSETS_DIR) / "furniture" / asset_type
            elif asset_type in registry.OBJECTS:
                asset_dir = Path(registry.ASSETS_DIR) / "objects" / asset_type
            elif asset_type in registry.ROBOTS:
                # Robot meshes are in robot/assets subdirectory
                asset_dir = Path(registry.ASSETS_DIR) / "robots" / asset_type / "robot" / "assets"
            else:
                asset_dir = None

            # Add asset section ONCE per type (PURE MOP!)
            for child in asset_section:
                # Fix relative texture paths to absolute
                if child.tag == 'texture' and child.get('file') and asset_dir:
                    file_path = child.get('file')
                    if not file_path.startswith('/'):
                        # Relative path - make absolute
                        abs_path = asset_dir / file_path
                        child.set('file', str(abs_path))

                # Fix relative mesh paths to absolute
                if child.tag == 'mesh' and child.get('file') and asset_dir:
                    file_path = child.get('file')
                    if not file_path.startswith('/'):
                        # Relative path - make absolute
                        abs_path = asset_dir / file_path
                        child.set('file', str(abs_path))

                # NO renaming needed! Mesh definitions are SHARED by all instances!
                # Bodies/geoms get unique names, but they all reference the same mesh.

                child_str = ET.tostring(child, encoding='unicode')
                lines.append(f'    {child_str.strip()}')

    # NOTE: Old _merge_X_sections() methods deleted - replaced by XmlSection dataclass!
    # Lines saved: 160+ lines of repetitive code eliminated!

    @staticmethod
    def _resolve_orientation(orientation: Union[Tuple[float, float, float, float], str]) -> Tuple[float, float, float, float]:
        """Convert orientation to MuJoCo quaternion (w,x,y,z) - OFFENSIVE

        Args:
            orientation: Quaternion tuple (w,x,y,z) or preset string like "upright"

        Returns:
            Quaternion tuple (w,x,y,z) in MuJoCo format

        MOP: Presets for common orientations (upright, sideways, inverted)
        """
        if isinstance(orientation, tuple):
            # Already a quaternion - validate and return
            if len(orientation) != 4:
                raise ValueError(f"Quaternion must be (w,x,y,z), got {orientation}")
            return orientation

        # String preset - AUTO-DISCOVERABLE orientation presets!
        ORIENTATION_PRESETS = {
            # Object states
            "upright": (1, 0, 0, 0),           # Identity quaternion - no rotation
            "sideways": (0.707, 0, 0, 0.707),  # 90° around Z axis
            "inverted": (0, 1, 0, 0),          # 180° upside down
            "upside_down": (0, 1, 0, 0),       # Alias for inverted

            # Cardinal directions (face direction in XY plane)
            "north": (1, 0, 0, 0),             # Face +Y (forward in MuJoCo)
            "south": (0, 0, 0, 1),             # Face -Y (180° rotation)
            "east": (0.707, 0, 0, 0.707),      # Face +X (90° CW around Z)
            "west": (0.707, 0, 0, -0.707),     # Face -X (90° CCW around Z)

            # Special orientations
            "facing_origin": None,             # Calculated dynamically based on position
        }

        if orientation not in ORIENTATION_PRESETS:
            raise ValueError(
                f"❌ Unknown orientation preset '{orientation}'!\n"
                f"\n✅ Valid presets: {list(ORIENTATION_PRESETS.keys())}\n"
                f"\n💡 Or provide quaternion tuple: (w, x, y, z)"
            )

        return ORIENTATION_PRESETS[orientation]

    @staticmethod
    def _rename_xml_tree(element: ET.Element, new_prefix: str, old_prefix: str):
        """Rename ALL elements in XML tree to use new prefix - OFFENSIVE MOP!

        This ensures multiple instances of the same asset type don't have naming collisions.

        Args:
            element: Root XML element to rename (will be modified in-place)
            new_prefix: New prefix to use (e.g., "block_red")
            old_prefix: Old prefix to replace (e.g., "wood_block")

        Example:
            Before: <body name="wood_block"><geom name="wood_block_mesh"/></body>
            After:  <body name="block_red"><geom name="block_red_mesh"/></body>
        """
        # Rename body name if present
        if element.get("name"):
            old_name = element.get("name")
            # Replace old prefix with new prefix
            new_name = old_name.replace(old_prefix, new_prefix)
            element.set("name", new_name)

        # Rename geom names but KEEP mesh references unchanged (meshes are SHARED!)
        for geom in element.findall(".//geom"):
            if geom.get("name"):
                old_name = geom.get("name")
                new_name = old_name.replace(old_prefix, new_prefix)
                geom.set("name", new_name)

            # MOP: Do NOT rename mesh references! Meshes are SHARED resources.
            # All instances reference the same mesh definition (e.g., "wood_block_mesh")

        # Rename joint names
        for joint in element.findall(".//joint"):
            if joint.get("name"):
                old_name = joint.get("name")
                new_name = old_name.replace(old_prefix, new_prefix)
                joint.set("name", new_name)

        # Rename site names
        for site in element.findall(".//site"):
            if site.get("name"):
                old_name = site.get("name")
                new_name = old_name.replace(old_prefix, new_prefix)
                site.set("name", new_name)

        # Rename child body names recursively
        for child_body in element.findall(".//body"):
            if child_body.get("name"):
                old_name = child_body.get("name")
                new_name = old_name.replace(old_prefix, new_prefix)
                child_body.set("name", new_name)

    @staticmethod
    def _merge_worldbody_placements(scene, resolved_furniture, registry, lines):
        """Merge furniture bodies into worldbody - OFFENSIVE

        Returns:
            List of (body_name, position, quaternion) for freejoint objects (for keyframe generation)
        """
        # Track freejoint objects for keyframe generation
        freejoint_objects = []

        for i, placement in enumerate(scene.placements):
            # NEW: Pass resolved_furniture for XML-based dimension extraction!
            pos = placement.get_xyz(scene, resolved_furniture=resolved_furniture)
            asset_name = placement.asset

            # DEBUG: Check robot placement position
            if asset_name in registry.ROBOTS:
                print(f"  [DEBUG] Robot placement: asset={asset_name}, pos={pos}, placement.position={placement.position}")

            # Use pre-resolved XML if available
            if i < len(resolved_furniture):
                asset_root = resolved_furniture[i]
                worldbody = asset_root.find(".//worldbody")

                if worldbody is not None:
                    # ROBOTS: Insert directly without extra wrapper (they have complete kinematic chains)
                    # This prevents breaking the robot's physics by double-wrapping bodies
                    if asset_name in registry.ROBOTS:
                        # Get robot modal for site injection
                        robot_info = scene.robots.get(placement.instance_name or asset_name, {})
                        robot_modal = robot_info.get("robot_modal")

                        for body in worldbody.findall("body"):
                            body_name = body.get("name", "")

                            # INJECT SITES for actuators (MOP!)
                            if robot_modal:
                                XMLResolver._inject_actuator_sites(body, body_name, robot_modal)

                            # Get existing pos or default to 0,0,0
                            existing_pos = body.get("pos", "0 0 0")
                            existing_coords = [float(x) for x in existing_pos.split()]

                            # Add placement offset to existing position
                            new_pos = [existing_coords[j] + pos[j] for j in range(3)]
                            body.set("pos", f"{new_pos[0]} {new_pos[1]} {new_pos[2]}")

                            # Output the body directly (no wrapper)
                            body_str = ET.tostring(body, encoding='unicode')
                            lines.append(f'    {body_str.strip()}')
                    else:
                        # FURNITURE/OBJECTS: Wrap in body tag for positioning
                        for body in worldbody.findall("body"):
                            # MOP: Use instance_name for unique identifiers!
                            effective_name = placement.instance_name if placement.instance_name else asset_name
                            body_name = body.get("name", asset_name)

                            # If instance_name provided, rename ALL elements in tree to avoid collisions
                            if placement.instance_name and placement.instance_name != asset_name:
                                XMLResolver._rename_xml_tree(body, placement.instance_name, asset_name)
                                body_name = placement.instance_name

                            # Build body tag with position + optional orientation (NEW!)
                            body_attrs = f'name="{body_name}" pos="{pos[0]} {pos[1]} {pos[2]}"'
                            if placement.orientation:
                                # Convert orientation to MuJoCo quat format
                                quat = XMLResolver._resolve_orientation(placement.orientation)
                                body_attrs += f' quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}"'

                            lines.append(f'    <body {body_attrs}>')

                            # Objects need freejoint for physics (gravity, collisions, movement)
                            if asset_name in registry.OBJECTS:
                                lines.append(f'      <freejoint/>')

                                # Track this object for keyframe generation
                                quat = XMLResolver._resolve_orientation(placement.orientation) if placement.orientation else (1, 0, 0, 0)
                                freejoint_objects.append((body_name, pos, quat))

                            # Add all body contents (geoms, joints, etc)
                            for child in body:
                                child_str = ET.tostring(child, encoding='unicode')
                                lines.append(f'      {child_str.strip()}')

                            lines.append('    </body>')
                else:
                    # Fallback - simple geom
                    lines.append(f'    <body name="{asset_name}" pos="{pos[0]} {pos[1]} {pos[2]}">')
                    lines.append(f'      <geom name="{asset_name}_geom" type="box" size="0.05 0.05 0.05"/>')
                    lines.append('    </body>')
            else:
                # Fallback - simple geom for unknown assets
                lines.append(f'    <body name="{asset_name}" pos="{pos[0]} {pos[1]} {pos[2]}">')
                lines.append(f'      <geom name="{asset_name}_geom" type="box" size="0.05 0.05 0.05"/>')
                lines.append('    </body>')

        # Return tracked objects for keyframe generation
        return freejoint_objects

    @staticmethod
    def _inject_actuator_sites(body: ET.Element, body_name: str, robot_modal) -> None:
        """Inject missing actuator sites dynamically - MOP: Actuator declares needs!

        Sites added based on actuator_modals.py declarations.
        Keeps stretch.xml unchanged (base definition).

        Args:
            body: XML body element to potentially inject sites into
            body_name: Name of the body (e.g., "link_grasp_center")
            robot_modal: Robot modal instance with actuator declarations
        """
        if not robot_modal or not hasattr(robot_modal, 'actuators'):
            return

        # Site definitions per body (relative positions)
        # These match what actuator_modals.py expects but aren't in base XML
        SITE_DEFINITIONS = {
            "link_grasp_center": {  # Gripper sites
                "grasp_left": "-0.02 0 0",   # 2cm left of center
                "grasp_right": "0.02 0 0",   # 2cm right of center
                "grasp_center": "0 0 0",     # May already exist
            },
            "link_arm_l3": {  # Arm reach point
                "reach_point": "0 0 0.1",    # 10cm above arm tip
            },
            "link_head_tilt": {  # Head gaze point
                "gaze_forward": "0.1 0 0",   # 10cm forward from camera
            }
        }

        if body_name not in SITE_DEFINITIONS:
            return

        # Check which sites already exist in this body
        existing_sites = {site.get("name") for site in body.findall("site")}

        # Inject missing sites
        for site_name, pos in SITE_DEFINITIONS[body_name].items():
            if site_name not in existing_sites:
                site_elem = ET.SubElement(body, "site")
                site_elem.set("name", site_name)
                site_elem.set("pos", pos)
                site_elem.set("size", "0.01")  # Small marker


# Test
if __name__ == "__main__":
    print("=== RESOLVING KITCHEN COUNTER ===")
    xml = XMLResolver.get_full_xml("kitchen_counter", "furniture")

    # Check what we got
    components = XMLResolver.extract_components(xml)
    print(f"Bodies: {components['bodies']}")
    print(f"Geoms: {len(components['geoms'])} total")
    print(f"  First 5: {components['geoms'][:5]}")
    print(f"Joints: {components['joints']}")

    # Save to see
    with open("/tmp/kitchen_counter_resolved.xml", "w") as f:
        f.write(xml)
    print("\nFull XML saved to /tmp/kitchen_counter_resolved.xml")

    print("\n=== RESOLVING MICROWAVE ===")
    xml = XMLResolver.get_full_xml("microwave", "furniture")
    components = XMLResolver.extract_components(xml)
    print(f"Bodies: {components['bodies']}")
    print(f"Geoms: {len(components['geoms'])} total")
    print(f"  First 5: {components['geoms'][:5]}")
    print(f"Joints: {components['joints']}")

    print("\n=== RESOLVING BOWL ===")
    xml = XMLResolver.get_full_xml("bowl", "objects")
    components = XMLResolver.extract_components(xml)
    print(f"Bodies: {components['bodies']}")
    print(f"Geoms: {components['geoms']}")
    print(f"Joints: {components['joints']}")
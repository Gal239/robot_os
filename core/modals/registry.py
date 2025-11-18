"""
REGISTRY - Asset discovery and loading
OFFENSIVE - crashes if assets don't exist
"""

from pathlib import Path
from typing import Dict, Optional
import json
from .asset_modals import Asset, BEHAVIORS

# Base directory for all assets
ASSETS_DIR = Path(__file__).parent / "mujoco_assets"

# Auto-discover all assets - OFFENSIVE (crashes if dirs don't exist)
TEXTURES = {f.stem: str(f) for f in (ASSETS_DIR / "textures").glob("*.png")}
FURNITURE = {d.name for d in (ASSETS_DIR / "furniture").iterdir() if d.is_dir()}
OBJECTS = {d.name for d in (ASSETS_DIR / "objects").iterdir() if d.is_dir()}
ROOMS = {f.stem: str(f) for f in (ASSETS_DIR / "rooms").glob("*.xml")}
ROBOTS = {d.name for d in (ASSETS_DIR / "robots").iterdir() if d.is_dir()}
ROOM_PARTS = {d.name for d in (ASSETS_DIR / "room_parts").iterdir() if d.is_dir()} if (ASSETS_DIR / "room_parts").exists() else set()

# Asset cache
ASSET_CACHE: Dict[str, Asset] = {}


def list_available(asset_type: str = None) -> list:
    """List all available assets of a given type - INFORMATIVE

    Args:
        asset_type: "texture", "furniture", "object", "room", "robot", "room_parts" (None = all)

    Returns:
        List of available asset names (or dict of all if asset_type is None)
    """
    if asset_type == "texture":
        return sorted(TEXTURES.keys())
    elif asset_type == "furniture":
        return sorted(FURNITURE)
    elif asset_type == "object":
        return sorted(OBJECTS)
    elif asset_type == "room":
        return sorted(ROOMS.keys())
    elif asset_type == "robot":
        return sorted(ROBOTS)
    elif asset_type == "room_parts":
        return sorted(ROOM_PARTS)
    elif asset_type is None:
        # Return all asset types
        return {
            "textures": sorted(TEXTURES.keys()),
            "furniture": sorted(FURNITURE),
            "objects": sorted(OBJECTS),
            "rooms": sorted(ROOMS.keys()),
            "robots": sorted(ROBOTS),
            "room_parts": sorted(ROOM_PARTS)
        }
    else:
        return []


def load_asset_config(name: str) -> dict:
    """Load asset config JSON - OFFENSIVE"""
    # Look for config file
    config_paths = [
        ASSETS_DIR / f"furniture/{name}/config.json",
        ASSETS_DIR / f"objects/{name}/config.json",
        ASSETS_DIR / f"robots/{name}/config.json",
        ASSETS_DIR / f"room_parts/{name}/config.json"
    ]

    for path in config_paths:
        if path.exists():
            return json.loads(path.read_text())

    # No config found - create minimal one
    if name in FURNITURE:
        category = "furniture"
        xml_file = f"furniture/{name}/{name}.xml"
    elif name in OBJECTS:
        category = "object"
        xml_file = f"objects/{name}/{name}.xml"
    elif name in ROBOTS:
        category = "robot"
        xml_file = f"robots/{name}/{name}.xml"
    else:
        raise ValueError(f"Asset '{name}' not found")

    return {
        "name": name,
        "category": category,
        "xml_file": xml_file,
        "components": {}
    }


def get_asset_xml_path(asset_name: str, asset_type: str, xml_file: str) -> Path:
    """Get full path to asset XML file - SINGLE SOURCE OF TRUTH!

    MOP: Centralized path resolution - change file structure in ONE place!

    Args:
        asset_name: Asset name (e.g., "table", "apple")
        asset_type: Asset type category (e.g., "furniture", "objects")
        xml_file: XML filename (e.g., "table.xml")

    Returns:
        Path object pointing to XML file

    Raises:
        FileNotFoundError: If XML file doesn't exist in expected locations

    Example:
        path = get_asset_xml_path("table", "furniture", "table.xml")
        # Returns: Path(".../core/modals/mujoco_assets/furniture/table/table.xml")
    """
    # Search paths (in order of preference)
    search_paths = [
        ASSETS_DIR / asset_type / asset_name / xml_file,  # Standard: furniture/table/table.xml
        ASSETS_DIR / asset_type / xml_file,               # Fallback: furniture/table.xml
    ]

    for path in search_paths:
        if path.exists():
            return path

    # OFFENSIVE: Crash with helpful message!
    raise FileNotFoundError(
        f"XML file '{xml_file}' not found for asset '{asset_name}' (type: {asset_type})\n"
        f"Searched:\n" +
        "\n".join(f"  - {p}" for p in search_paths)
    )


def load_asset(name: str) -> Asset:
    """Load asset with components and behaviors - OFFENSIVE"""
    if name in ASSET_CACHE:
        return ASSET_CACHE[name]

    config = load_asset_config(name)
    asset = Asset(name, config)
    ASSET_CACHE[name] = asset
    return asset


def resolve(asset_type: str, name: str) -> str:
    """Resolve asset name to path - OFFENSIVE & INFORMATIVE

    Crashes with helpful message showing what's available when asset not found.
    """
    if asset_type == "texture":
        if name not in TEXTURES:
            available = list_available("texture")
            raise KeyError(f"Texture '{name}' not found. Available textures: {available}")
        return TEXTURES[name]
    elif asset_type == "furniture":
        if name not in FURNITURE:
            available = list_available("furniture")
            raise KeyError(f"Furniture '{name}' not found. Available furniture: {available}")
        return str(ASSETS_DIR / f"furniture/{name}/{name}.xml")
    elif asset_type == "object":
        if name not in OBJECTS:
            available = list_available("object")
            raise KeyError(f"Object '{name}' not found. Available objects: {available}")
        return str(ASSETS_DIR / f"objects/{name}/{name}.xml")
    elif asset_type == "room":
        if name not in ROOMS:
            available = list_available("room")
            raise KeyError(f"Room '{name}' not found. Available rooms: {available}")
        return ROOMS[name]
    elif asset_type == "robot":
        if name not in ROBOTS:
            available = list_available("robot")
            raise KeyError(f"Robot '{name}' not found. Available robots: {available}")
        return str(ASSETS_DIR / f"robots/{name}/{name}.xml")
    else:
        raise ValueError(f"Unknown asset type: {asset_type}")


def get_xml(name: str) -> str:
    """Get asset XML content - OFFENSIVE"""
    if name in FURNITURE:
        path = ASSETS_DIR / f"furniture/{name}/{name}.xml"
        return path.read_text()
    elif name in OBJECTS:
        path = ASSETS_DIR / f"objects/{name}/{name}.xml"
        return path.read_text()
    elif name in ROBOTS:
        path = ASSETS_DIR / f"robots/{name}/{name}.xml"
        return path.read_text()
    elif name in ROOMS:
        return Path(ROOMS[name]).read_text()
    else:
        raise ValueError(f"Asset '{name}' not found in registry")


def refresh():
    """Refresh registry to pick up new files"""
    global TEXTURES, FURNITURE, OBJECTS, ROOMS, ROBOTS, ASSET_CACHE

    TEXTURES = {f.stem: str(f) for f in (ASSETS_DIR / "textures").glob("*.png")}
    FURNITURE = {d.name for d in (ASSETS_DIR / "furniture").iterdir() if d.is_dir()}
    OBJECTS = {d.name for d in (ASSETS_DIR / "objects").iterdir() if d.is_dir()}
    ROOMS = {f.stem: str(f) for f in (ASSETS_DIR / "rooms").glob("*.xml")}
    ROBOTS = {d.name for d in (ASSETS_DIR / "robots").iterdir() if d.is_dir()}

    # Clear asset cache
    ASSET_CACHE = {}

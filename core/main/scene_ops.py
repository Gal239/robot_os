"""
Scene Operations - Clean API for scene creation
Hides modal implementation details from users
"""

from typing import Optional, Union, Dict, List
from pathlib import Path
import json

from ..modals.scene_modal import Scene
from ..modals.room_modal import RoomModal
from ..modals import registry


def create_scene(room_name: str, **room_overrides) -> Scene:
    """
    Create a scene with a room.

    Clean API - hides RoomModal implementation detail.

    Modal-Oriented: Tries to load generated room JSON, falls back to defaults.

    Args:
        room_name: Room name (e.g., "kitchen", "warehouse")
        **room_overrides: Override room properties (width, length, height, etc.)

    Returns:
        Scene instance ready for adding robots and objects

    Examples:
        # Try to load generated room, or create with defaults
        scene = create_scene("kitchen")

        # Override loaded/default values
        scene = create_scene("kitchen", width=10, length=8)

        # Create new room
        scene = create_scene("warehouse", width=20, length=20, height=5)

    To save a room for reuse:
        room = RoomModal("kitchen", width=5, length=5, height=3)
        room.save()  # Modal generates assets/rooms/kitchen.json

    Available rooms:
        Run: from core.main.scene_ops import list_rooms; list_rooms()
    """
    try:
        # Try to load room from generated JSON (Modal self-generated!)
        room = RoomModal.load(room_name)

        # Apply overrides if provided
        for key, value in room_overrides.items():
            setattr(room, key, value)

    except FileNotFoundError:
        # Create new room with defaults + overrides
        defaults = {
            "name": room_name,
            "width": 5,
            "length": 5,
            "height": 3,
            "floor_texture": "wood_floor",
            "wall_texture": "gray_wall"
        }
        defaults.update(room_overrides)

        room = RoomModal(**defaults)

    # Create scene (modal hidden from user!)
    return Scene(room)


def list_rooms() -> List[str]:
    """
    List available predefined room types.

    Returns:
        List of room names

    Example:
        rooms = list_rooms()
        print(f"Available rooms: {rooms}")
    """
    rooms_dir = Path(__file__).parent.parent.parent / "assets" / "rooms"
    if not rooms_dir.exists():
        return []

    return [f.stem for f in rooms_dir.glob("*.json")]


def list_objects() -> List[str]:
    """List available objects"""
    return registry.list_available("objects")


def list_furniture() -> List[str]:
    """List available furniture"""
    return registry.list_available("furniture")


def list_robots() -> List[str]:
    """List available robot types"""
    return registry.list_available("robot")
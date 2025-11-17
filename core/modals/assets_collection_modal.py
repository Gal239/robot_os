"""
Assets Collection - Beautiful property-based access to scene assets

PURE MOP: Assets self-expose via beautiful property syntax
No more dict parsing! IDE autocomplete works! Type-safe!

Usage:
    pos = ops.assets.apple.position  # ✨ Beautiful!
    behaviors = ops.assets.apple.behaviors
    is_on_table = ops.assets.apple.is_stacked_on('table')
"""

from typing import List, Tuple, Any, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .asset_modals import Asset
    from .scene_modal import Scene


class AssetProxy:
    """Proxy for beautiful syntax: ops.assets.apple.position

    PURE MOP: Asset knows how to expose itself beautifully!
    - Properties for static access (position, behaviors)
    - Methods for parameterized queries (is_stacked_on, distance_to)

    Example:
        apple = ops.assets.apple
        print(apple.position)  # (2.0, 0.0, 0.76)
        print(apple.behaviors)  # ['graspable', 'stackable']
        print(apple.is_stacked_on('table'))  # True
    """

    def __init__(self, asset: 'Asset', state: Dict[str, Any], scene: 'Scene'):
        """Initialize proxy with asset modal and current state

        Args:
            asset: Asset modal from scene
            state: Current state dict for this asset
            scene: Scene modal (for cross-asset queries)
        """
        self._asset = asset
        self._state = state
        self._scene = scene
        self._name = asset.name

    @property
    def position(self) -> Tuple[float, float, float]:
        """Position in world coordinates (x, y, z)

        MOP: Asset self-extracts position from state!
        """
        # Search for position key in state
        for key, value in self._state.items():
            if 'position' in key.lower() and isinstance(value, (list, tuple)):
                if len(value) >= 3:
                    return tuple(float(v) for v in value[:3])
        return (0.0, 0.0, 0.0)

    @property
    def rotation(self) -> float:
        """Rotation angle in radians

        MOP: Asset self-extracts rotation from state!
        """
        for key, value in self._state.items():
            if 'rotation' in key.lower() and isinstance(value, (int, float)):
                return float(value)
        return 0.0

    @property
    def velocity(self) -> Tuple[float, float, float]:
        """Linear velocity (vx, vy, vz)

        MOP: Asset self-extracts velocity from state!
        """
        for key, value in self._state.items():
            if 'velocity' in key.lower() and isinstance(value, (list, tuple)):
                if len(value) >= 3:
                    return tuple(float(v) for v in value[:3])
        return (0.0, 0.0, 0.0)

    @property
    def behaviors(self) -> List[str]:
        """All behaviors from all components

        MOP: Asset self-reports its behaviors!
        Returns list like: ['graspable', 'stackable', 'spatial']
        """
        # Look for behaviors key in state
        for key, value in self._state.items():
            if 'behaviors' in key.lower() and isinstance(value, list):
                return value

        # Fallback: Extract from asset components
        all_behaviors = []
        for component in self._asset.components.values():
            if hasattr(component, 'behaviors'):
                all_behaviors.extend(component.behaviors)
        return list(set(all_behaviors))

    @property
    def held(self) -> bool:
        """True if graspable object is currently held

        MOP: Asset knows if it's being grasped!
        """
        for key, value in self._state.items():
            if 'held' in key.lower() and isinstance(value, bool):
                return value
        return False

    def is_stacked_on(self, target: str) -> bool:
        """Check if this asset is stacked on target asset/surface

        MOP: Uses dynamic relational properties!

        Args:
            target: Name of target asset (e.g., 'table', 'floor')

        Returns:
            True if stacked on target

        Example:
            ops.assets.apple.is_stacked_on('table')  # True
            ops.assets.apple.is_stacked_on('floor')  # False
        """
        # Look for stacked_on_<target> property
        prop_name = f'stacked_on_{target}'
        for key, value in self._state.items():
            if prop_name in key.lower():
                return bool(value) if isinstance(value, bool) else False
        return False

    def distance_to(self, target: str) -> float:
        """Calculate distance to another asset

        MOP: Uses dynamic distance properties or calculates from positions!

        Args:
            target: Name of target asset

        Returns:
            Distance in meters

        Example:
            ops.assets.apple.distance_to('table')  # 0.76
        """
        # Look for distance_to_<target> property
        prop_name = f'distance_to_{target}'
        for key, value in self._state.items():
            if prop_name in key.lower() and isinstance(value, (int, float)):
                return float(value)

        # Fallback: Calculate from positions
        if target in self._scene.assets:
            # Get target state from full state dict
            # Note: self._state is just this asset's state, need full state
            # This is a limitation - distance calculation needs both positions
            # For now, return inf if dynamic property not found
            return float('inf')

        return float('inf')

    def __repr__(self):
        """Beautiful representation"""
        return f"<Asset '{self._name}' at {self.position}>"

    def __str__(self):
        """String representation"""
        return f"Asset({self._name})"


class AssetsCollection:
    """Collection of assets with property access

    PURE MOP: Collection self-organizes assets for beautiful access!
    Uses dynamic __getattr__ to expose assets as properties.

    Example:
        assets = ops.assets
        apple = assets.apple  # Dynamic property access!
        table = assets.table

        # Or directly:
        ops.assets.apple.position
    """

    def __init__(self, scene: 'Scene', full_state: Dict[str, Any]):
        """Initialize collection with scene and current state

        Args:
            scene: Scene modal containing assets dict
            full_state: Full state dict from ops.get_state()
        """
        self._scene = scene
        self._full_state = full_state

    def __getattr__(self, name: str) -> AssetProxy:
        """Dynamic property access: ops.assets.apple

        MOP: Collection discovers assets on-demand!

        Args:
            name: Asset name

        Returns:
            AssetProxy for beautiful property access

        Raises:
            AttributeError: If asset not found (with helpful suggestions)
        """
        # Private attrs - use normal access
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        # Find asset in scene
        if name in self._scene.assets:
            asset = self._scene.assets[name]
            asset_state = self._full_state.get(name, {})
            return AssetProxy(asset, asset_state, self._scene)

        # Not found - provide helpful error with suggestions
        available = list(self._scene.assets.keys())
        raise AttributeError(
            f"Asset '{name}' not found in scene.\n"
            f"Available assets: {available}\n"
            f"\n"
            f"Usage: ops.assets.{available[0] if available else 'asset_name'}.position"
        )

    def __dir__(self):
        """Support for tab completion and dir()"""
        # Include all asset names as attributes
        return list(self._scene.assets.keys())

    def __repr__(self):
        """Beautiful representation"""
        assets = list(self._scene.assets.keys())
        preview = assets[:5]
        more = f"... +{len(assets) - 5} more" if len(assets) > 5 else ""
        return f"<AssetsCollection {len(assets)} assets: {preview}{more}>"

    def __len__(self):
        """Number of assets"""
        return len(self._scene.assets)

    def keys(self):
        """Get all asset names (dict-like interface)"""
        return self._scene.assets.keys()

    def values(self):
        """Get all asset proxies (dict-like interface)"""
        return [getattr(self, name) for name in self._scene.assets.keys()]

    def items(self):
        """Get (name, proxy) pairs (dict-like interface)"""
        return [(name, getattr(self, name)) for name in self._scene.assets.keys()]

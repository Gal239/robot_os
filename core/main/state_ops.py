"""
StateOps - Clean state extraction API
Following VideoOps pattern for consistent design
"""

from typing import Dict, Any, Optional


class StateOps:
    """State extraction & access - OFFENSIVE & MOP

    Clean API for all state operations!
    Wraps StateExtractor complexity.

    Usage:
        # In ExperimentOps:
        self.state_ops = StateOps(self.engine.backend)

        # In tests:
        state = ops.get_state()  # Clean API!
        apple_state = state["apple"]
        is_on_floor = apple_state.get("stacked_on_floor", False)
    """

    def __init__(self, backend):
        """Initialize with physics backend

        Args:
            backend: PhysicsBackend instance (MuJoCo wrapper)
        """
        from core.runtime.runtime_engine import StateExtractor
        self.state_extractor = StateExtractor(backend)
        self._cached_state = {}

    def get(self, scene, robot=None) -> Dict[str, Dict[str, Any]]:
        """Get current state - MAIN API

        Extracts state from MuJoCo and caches it.

        Args:
            scene: Scene instance
            robot: Robot instance (optional)

        Returns:
            Dict[asset_name, Dict[property, value]]

        Example:
            state = ops.state_ops.get(ops.scene, ops.robot)
            apple_state = state["apple"]
            is_on_floor = apple_state.get("stacked_on_floor", False)
        """
        state = self.state_extractor.extract(scene, robot)
        self._cached_state = state
        return state

    def get_asset(self, asset_name: str, scene=None, robot=None) -> Dict[str, Any]:
        """Get single asset state - OFFENSIVE

        Uses cached state if available, otherwise extracts fresh.

        Args:
            asset_name: Name of asset to get state for
            scene: Scene instance (for fresh extraction if needed)
            robot: Robot instance (optional, for fresh extraction)

        Returns:
            Dict of properties for this asset

        Raises:
            KeyError: If asset not found in state (OFFENSIVE!)
        """
        # Use cache if available, otherwise extract fresh
        if not self._cached_state and scene:
            self.get(scene, robot)

        if asset_name not in self._cached_state:
            available = list(self._cached_state.keys())
            raise KeyError(
                f"❌ Asset '{asset_name}' not in state!\n"
                f"\n📚 Available assets: {available}\n"
                f"\n💡 FIX: Check asset name spelling or ensure asset added to scene"
            )
        return self._cached_state[asset_name]

    def get_property(self, asset_name: str, property_name: str,
                     scene=None, robot=None) -> Any:
        """Get single property - OFFENSIVE

        Args:
            asset_name: Name of asset
            property_name: Name of property
            scene: Scene instance (for fresh extraction if needed)
            robot: Robot instance (optional)

        Returns:
            Property value

        Raises:
            KeyError: If asset or property not found (OFFENSIVE!)
        """
        asset_state = self.get_asset(asset_name, scene, robot)

        if property_name not in asset_state:
            available = list(asset_state.keys())
            raise KeyError(
                f"❌ Property '{property_name}' not found for '{asset_name}'!\n"
                f"\n📚 Available properties: {available}\n"
                f"\n💡 FIX: Check property name or ensure behavior generates this property"
            )
        return asset_state[property_name]

    def get_contacts(self, asset_name: str, scene=None, robot=None) -> Dict[str, bool]:
        """Get all contact/stacking properties for asset

        Returns only relational contact properties (stacked_on_*, supporting_*, etc.)

        Args:
            asset_name: Name of asset
            scene: Scene instance (for fresh extraction if needed)
            robot: Robot instance (optional)

        Returns:
            Dict of contact properties: {"stacked_on_table": True, ...}
        """
        asset_state = self.get_asset(asset_name, scene, robot)

        # Filter for relational properties
        contacts = {
            k: v for k, v in asset_state.items()
            if k.startswith(("stacked_on_", "supporting_", "contact_", "held_by_"))
        }
        return contacts

    def clear_cache(self):
        """Clear cached state - force fresh extraction on next get()"""
        self._cached_state = {}

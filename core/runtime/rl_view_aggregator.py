"""
ViewAggregator - Minimal observation aggregator for RL

For now: Simple sensor + actuator concatenation
TODO: Add VLM integration later
"""

import numpy as np
from typing import Dict, Optional, List


class ViewAggregator:
    """Minimal observation aggregator

    Aggregates observations from:
    - Sensors (odometry, lidar, imu, etc.)
    - Actuators (joint positions, velocities)

    TODO: Add VLM vision-language observations
    """

    def __init__(
        self,
        ops: 'ExperimentOps',
        config: str = 'baseline',
        obs_views: Optional[Dict[str, bool]] = None,
        device: str = 'cuda'
    ):
        """Initialize view aggregator

        Args:
            ops: ExperimentOps instance
            config: 'baseline' (sensors+actuators only)
            obs_views: Optional view configuration (ignored for now)
            device: Device for VLM models (ignored for now)
        """
        self.ops = ops
        self.config = config
        self.obs_views = obs_views or {}
        self.device = device

    def get_obs_dim(self) -> int:
        """Get observation dimension

        Returns:
            Observation vector dimension
        """
        # Get current state
        state = self.ops.get_state()

        # Count scalar values in state
        obs_dim = 0
        for asset_name, asset_data in state.items():
            if isinstance(asset_data, dict):
                for key, value in asset_data.items():
                    if isinstance(value, (int, float)):
                        obs_dim += 1
                    elif isinstance(value, (list, np.ndarray)):
                        obs_dim += len(value)

        return obs_dim if obs_dim > 0 else 1  # At least 1D

    def get_obs(self) -> np.ndarray:
        """Get observation vector

        Returns:
            Flattened observation vector (normalized to [-1, 1])
        """
        # Get current state
        state = self.ops.get_state()

        # Flatten all scalar values
        obs_values = []
        for asset_name, asset_data in state.items():
            if isinstance(asset_data, dict):
                for key, value in asset_data.items():
                    if isinstance(value, (int, float)):
                        obs_values.append(float(value))
                    elif isinstance(value, (list, np.ndarray)):
                        obs_values.extend([float(v) for v in value])

        if not obs_values:
            # Empty state - return zero vector
            return np.zeros(1, dtype=np.float32)

        # Convert to numpy array
        obs = np.array(obs_values, dtype=np.float32)

        # Normalize to [-1, 1] (simple clipping for now)
        # TODO: Proper normalization using BEHAVIORS.json ranges
        obs = np.clip(obs / 10.0, -1.0, 1.0)  # Rough normalization

        return obs

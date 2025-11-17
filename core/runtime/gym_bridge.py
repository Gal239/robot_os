"""
GYM BRIDGE - Wrap ExperimentOps as gym.Env for RL training
OFFENSIVE & MODAL-ORIENTED: Auto-discovers action/obs spaces from modals

THE KEY INNOVATIONS:
1. actuators_active: FOCUSED action space training (1-2D vs 15D)
2. obs_config: USER-DECIDES quality vs quantity at runtime!

Observation Configs:
- 'quality': Max quality, fewer sims (15 parallel, Qwen2-VL-7B, 14GB VRAM)
- 'balanced': Good quality, more sims (25 parallel, Qwen2-VL-2B, 4GB VRAM)
- 'lean': Fast, most sims (50 parallel, BLIP-base, 1GB VRAM)

Usage:
    # Quality mode - rich vision-language observations
    env = GymBridge(
        ops,
        actuators_active=['base'],  # Focused action space
        obs_config='quality',       # Max quality VLM
        obs_views={
            'vision_vlm': True,     # Vision → text → embedding
            'vision_depth': True,   # Depth map
            'sensors': True,
            'actuators': True,
            'text_instruction': True,
            'action_history': True,
        }
    )

    # Lean mode - fast for 50+ parallel sims
    env = GymBridge(
        ops,
        actuators_active=['base'],
        obs_config='lean',  # Lightweight models
        obs_views={
            'vision_vlm': True,  # Still get vision-text!
            'sensors': True,
            'actuators': True,
        }
    )

    # Backward compatible - no vision (sensors + actuators only)
    env = GymBridge(ops, actuators_active=['base'])
"""

import gymnasium as gym
import numpy as np
from typing import Optional, List, Dict, Tuple, Any
from ..main.experiment_ops_unified import ExperimentOps


class GymBridge(gym.Env):
    """Gym environment wrapper for ExperimentOps - OFFENSIVE & MODAL-ORIENTED

    Auto-discovers action/obs spaces from robot modals.
    Supports FOCUSED action space training (THE KEY INNOVATION).
    """

    def __init__(
        self,
        ops: ExperimentOps,
        actuators_active: Optional[List[str]] = None,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
        obs_config: str = 'baseline',  # 'baseline', 'lean', 'balanced', or 'quality'
        obs_views: Optional[Dict[str, bool]] = None,
        device: str = 'cuda'  # 'cuda' or 'cpu' for VLM models
    ):
        """Initialize gym environment from ExperimentOps

        Args:
            ops: ExperimentOps instance (scene + robot + rewards already set up)
            actuators_active: None (all actuators, 15D) or ['base'] (focused, 1-2D)
            max_steps: Max steps per episode
            render_mode: 'human', 'rgb_array', or None
            obs_config: 'baseline' (no VLM), 'lean' (50+ sims), 'balanced' (25 sims), 'quality' (15 sims)
            obs_views: Optional dict of view flags (overrides config defaults)
            device: 'cuda' or 'cpu' for VLM models

        Example:
            # Baseline mode (no VLM, 80+ parallel sims)
            env = GymBridge(ops, actuators_active=['base'], obs_config='baseline')

            # VLM mode - THE INNOVATION! (50 parallel sims)
            env = GymBridge(
                ops,
                actuators_active=['base'],
                obs_config='lean',  # BLIP + MiniLM
                obs_views={'vision_vlm': True, 'sensors': True}
            )

            # Quality mode (15 parallel sims)
            env = GymBridge(ops, obs_config='quality')  # Qwen2.5-VL-7B
        """
        super().__init__()

        self.ops = ops
        self.actuators_active = actuators_active
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obs_config = obs_config
        self.obs_views = obs_views
        self.device = device

        self.current_step = 0
        self.episode_reward = 0.0

        # OFFENSIVE: Crash if not compiled
        assert ops.robot is not None, "Call ops.add_robot() first!"

        # Ensure compiled and synced - MODAL-ORIENTED (need real data to discover spaces)
        if ops.engine is None:
            ops.compile()
        ops.sync_from_mujoco()

        # Auto-discover action space from robot actuators - MODAL-ORIENTED
        self.action_space = self._build_action_space()

        # PURE MOP INSTANCE FLOW: Create ViewAggregator (LEGO COMPOSITION!)
        # ViewAggregator discovers sensors, creates VLM pipeline, composes everything!
        from .rl_view_aggregator import ViewAggregator

        self.view_aggregator = ViewAggregator(
            ops=ops,
            config=obs_config,
            obs_views=obs_views,
            device=device
        )

        # Auto-discover observation space from ViewAggregator - MODAL-ORIENTED!
        self.observation_space = self._build_obs_space()

        # Track actuator indices for focused action space
        self._actuator_indices = self._build_actuator_indices()

    def _build_action_space(self) -> gym.spaces.Box:
        """Auto-discover action space from robot actuators - MODAL-ORIENTED

        THE KEY INNOVATION: Only includes actuators_active for FOCUSED training!

        Returns:
            Box space with continuous actions for each active actuator
        """
        robot = self.ops.robot

        # Get active actuators (focused or all)
        if self.actuators_active is not None:
            # FOCUSED action space (curriculum) - only specified actuators
            active_actuators = {
                name: actuator
                for name, actuator in robot.actuators.items()
                if name in self.actuators_active
            }

            if not active_actuators:
                raise ValueError(
                    f"No actuators found! actuators_active={self.actuators_active}, "
                    f"available={list(robot.actuators.keys())}"
                )
        else:
            # FULL action space (baseline) - all actuators
            active_actuators = robot.actuators

        # Build action space from actuator ranges - MODAL-ORIENTED
        low = []
        high = []

        for name, actuator in active_actuators.items():
            # Each actuator has .range = (min, max)
            min_val, max_val = actuator.range
            low.append(min_val)
            high.append(max_val)

        if not low:
            # No actuators? Default to single action
            low = [-1.0]
            high = [1.0]

        return gym.spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

    def _build_obs_space(self) -> gym.spaces.Box:
        """Auto-discover observation space from ViewAggregator - MODAL-ORIENTED!

        Returns:
            Box space with all observations normalized to [-1, 1]
        """
        # PURE MOP: Ask ViewAggregator for observation dimension!
        # ViewAggregator handles VLM, sensors, actuators, EVERYTHING!
        obs_dim = self.view_aggregator.get_obs_dim()

        # Observations are normalized to [-1, 1] by ViewAggregator
        return gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _build_actuator_indices(self) -> Dict[str, int]:
        """Build mapping from actuator name to action index

        Returns:
            Dict mapping actuator name -> action index
        """
        robot = self.ops.robot

        if self.actuators_active is not None:
            # FOCUSED: Only active actuators
            active_names = [
                name for name in robot.actuators.keys()
                if name in self.actuators_active
            ]
        else:
            # FULL: All actuators
            active_names = list(robot.actuators.keys())

        return {name: i for i, name in enumerate(active_names)}

    def _get_obs(self) -> np.ndarray:
        """Extract observation from current state - MODAL-ORIENTED!

        Uses ViewAggregator to collect observations from:
        - Vision (VLM embeddings if enabled)
        - Sensors (lidar, imu, etc.)
        - Actuators (joint states)
        - Text instructions
        - Action history

        Returns:
            Observation vector (normalized to [-1, 1])
        """
        # PURE MOP: Ask ViewAggregator for observation!
        # ViewAggregator handles VLM, sensors, actuators, composition, EVERYTHING!
        return self.view_aggregator.get_obs()

    def _apply_action(self, action: np.ndarray):
        """Apply gym action to robot actuators - OFFENSIVE

        Maps gym action vector to actuator commands.
        Only applies to active actuators (focused or all).

        Args:
            action: Numpy array of continuous actions
        """
        robot = self.ops.robot

        # Get active actuators
        if self.actuators_active is not None:
            active_actuators = {
                name: actuator
                for name, actuator in robot.actuators.items()
                if name in self.actuators_active
            }
        else:
            active_actuators = robot.actuators

        # Map action indices to actuators
        for i, (name, actuator) in enumerate(active_actuators.items()):
            if i < len(action):
                # Set actuator target position
                target = float(action[i])
                actuator.move_to(target)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset episode - recompile scene

        Args:
            seed: Random seed
            options: Optional reset parameters

        Returns:
            (observation, info)
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Reset episode counters
        self.current_step = 0
        self.episode_reward = 0.0

        # PURE MOP: Modals reset themselves (10-30x FASTER than recompile!)
        self.ops.reset()

        # Get initial observation
        obs = self._get_obs()

        info = {
            "episode": 0,
            "step": 0,
            "focused_actuators": self.actuators_active,
            "action_space_dim": self.action_space.shape[0]
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step with gym action - PURE MOP!

        Args:
            action: Numpy array of continuous actions

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # PURE MOP: Modals execute actions (NO MANUAL MAPPING!)
        self.ops.apply_action(action, self.actuators_active)

        # Execute physics step (includes sync)
        self.ops.step()

        # PURE MOP: RewardModals compute their own rewards!
        reward, reward_info = self.ops.evaluate_rewards()
        self.episode_reward += reward

        # Get next observation
        obs = self._get_obs()

        # Check termination
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # PURE MOP: RewardModals know when task complete!
        terminated = self.ops.check_termination()

        info = {
            "episode_reward": self.episode_reward,
            "step": self.current_step,
            "focused_actuators": self.actuators_active,
            "action_space_dim": self.action_space.shape[0],
            "reward_info": reward_info
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render environment using VIEW SYSTEM - MODAL-ORIENTED!

        The view system is the PROPER way to get ALL data:
        - Camera views (vision)
        - Sensor views (lidar, odometry, etc.)
        - Actuator views (joint states, etc.)
        - Works in headless mode (offscreen rendering)
        - Works with viewer (display rendering)
        - Modals self-describe their render capabilities!

        Returns ALL views for RL observations, video, recording, EVERYTHING!
        """
        if self.render_mode == 'human':
            # Viewer already showing (if headless=False)
            return None
        elif self.render_mode == 'rgb_array':
            # Get ALL views from view system (MODAL-ORIENTED!)
            # This uses offscreen rendering - works even in headless mode!
            # Views auto-discover themselves - cameras, sensors, actuators, ALL!
            all_views = self.ops.get_views()

            # Return ALL views - let the consumer decide what they need!
            # This is SELF-DESCRIPTIVE - each view tells you what it is!
            return all_views if all_views else None
        return None

    def close(self):
        """Cleanup resources"""
        if self.ops.backend:
            # Close viewer if open
            if hasattr(self.ops.backend, 'viewer') and self.ops.backend.viewer:
                self.ops.backend.viewer.close()

"""
RL Operations - Modal Oriented Programming for RL Training

ARCHITECTURE:
- LEGO COMPOSITION: Composes ExperimentOps (doesn't reinvent!)
- THIN LAYER: Delegates to ExperimentOps and RL modals
- STEP BUNDLING: Handles 30Hz RL rate vs 200Hz physics rate HERE
- MODAL-TO-MODAL: Modals communicate directly (no glue code)

This is where RL-specific logic lives:
- Step bundling (multiple physics steps per RL action)
- Episode lifecycle management
- Action/observation space auto-discovery
- Training session tracking

RuntimeEngine stays PURE (just physics stepping).
ExperimentOps stays GENERAL (scene setup, rewards, etc.).
RLOps adds RL-SPECIFIC functionality.

Usage:
    from core.main.experiment_ops_unified import ExperimentOps
    from core.main.rl_ops import RLOps

    # Setup scene with ExperimentOps
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl")
    ops.create_scene("training", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_reward(...)
    ops.compile()

    # Wrap with RLOps for training
    rl_ops = RLOps(ops, actuators_active=["base", "arm"])

    # RL training loop
    for episode in range(num_episodes):
        obs = rl_ops.reset_episode()
        while not done:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, info = rl_ops.step(action)
            done = terminated or truncated
"""

from typing import Dict, Tuple, Optional, List, Any
import numpy as np
import time

from core.modals.rl_modal import (
    EpisodeModal,
    ActionSpaceModal,
    ObservationSpaceModal,
    TrainingSessionModal
)


class RLOps:
    """
    RL training operations - LEGO COMPOSITION with ExperimentOps

    Thin wrapper that:
    - Bundles physics steps to match RL rate (THIS IS WHERE BUNDLING LIVES!)
    - Auto-discovers action/observation spaces from modals
    - Tracks episodes and training sessions
    - Delegates scene/reward/robot logic to ExperimentOps
    """

    def __init__(self, experiment_ops, actuators_active: Optional[List[str]] = None):
        """
        Initialize RL operations

        Args:
            experiment_ops: ExperimentOps instance (already compiled!)
            actuators_active: List of actuator names for focused training (None = all)
        """
        # LEGO COMPOSITION: Use ExperimentOps (don't reinvent!)
        self.experiment_ops = experiment_ops

        # RL-specific modals
        self.action_space_modal = None
        self.observation_space_modal = None
        self.episode_modal = None
        self.training_session_modal = None

        # Step bundling configuration (calculated once)
        self._steps_per_rl_action = None

        # Auto-configure from experiment
        self._configure_training(actuators_active)

    def _configure_training(self, actuators_active: Optional[List[str]] = None):
        """
        AUTO-DISCOVERY: Configure RL training from experiment

        Discovers action/observation spaces from robot modals.
        Calculates step bundling rate from physics vs RL frequency.
        """
        print("DEBUG [RLOps]: Starting _configure_training...")
        import sys
        sys.stdout.flush()

        robot = self.experiment_ops.robot
        engine = self.experiment_ops.engine

        # OFFENSIVE validation
        if robot is None:
            raise ValueError(
                "Cannot configure RL training: No robot in experiment!\n"
                "Call experiment_ops.add_robot() before creating RLOps!"
            )

        if engine is None:
            raise ValueError(
                "Cannot configure RL training: Experiment not compiled!\n"
                "Call experiment_ops.compile() before creating RLOps!"
            )

        # AUTO-DISCOVER action space from actuator modals
        print("DEBUG [RLOps]: Discovering action space...")
        sys.stdout.flush()
        self.action_space_modal = ActionSpaceModal()
        self.action_space_modal.discover_from_robot(robot, actuators_active)

        print(f"🎮 Action space: {self.action_space_modal.action_dim}D")
        print(f"   Actuators: {self.action_space_modal.actuator_names}")
        print(f"   Bounds: [{self.action_space_modal.action_low[0]:.1f}, {self.action_space_modal.action_high[0]:.1f}]")

        # AUTO-DISCOVER observation space from experiment
        print("DEBUG [RLOps]: Discovering observation space (this calls get_state())...")
        sys.stdout.flush()
        self.observation_space_modal = ObservationSpaceModal()
        print("DEBUG [RLOps]: About to call discover_from_experiment...")
        sys.stdout.flush()
        self.observation_space_modal.discover_from_experiment(self.experiment_ops)
        print("DEBUG [RLOps]: discover_from_experiment completed!")
        sys.stdout.flush()

        print(f"👁️  Observation space: {self.observation_space_modal.obs_dim}D")
        print(f"   Sensors: {len(self.observation_space_modal.sensors_enabled)}")
        print(f"   Actuators: {len(self.observation_space_modal.actuators_enabled)}")

        # Calculate step bundling (30Hz RL rate vs 200Hz physics)
        print("DEBUG [RLOps]: Calculating step bundling...")
        sys.stdout.flush()
        self._calculate_step_bundling()
        print("DEBUG [RLOps]: _configure_training complete!")
        sys.stdout.flush()

    def _calculate_step_bundling(self):
        """
        Calculate how many physics steps to bundle per RL action

        THIS IS WHERE STEP BUNDLING LOGIC LIVES!

        Physics runs at 200Hz (timestep=0.005s)
        RL observes/acts at 30Hz (step_rate=30 from render_mode="rl")

        → Bundle 200/30 ≈ 7 physics steps per RL action
        → No lag, perfect credit assignment!
        """
        engine = self.experiment_ops.engine

        # Get physics frequency from MuJoCo
        physics_dt = engine.backend.model.opt.timestep  # e.g., 0.005s
        physics_hz = 1.0 / physics_dt  # e.g., 200Hz

        # Get RL frequency from engine config
        rl_hz = engine.step_rate  # e.g., 30Hz from render_mode="rl"

        # Calculate steps to bundle
        self._steps_per_rl_action = max(1, int(physics_hz / rl_hz))

        print(f"⚡ RL stepping: {rl_hz}Hz (bundling {self._steps_per_rl_action} physics steps @ {physics_hz:.0f}Hz)")
        print(f"   → RL observes/acts every {self._steps_per_rl_action} physics steps")
        print(f"   → State updates synchronized with RL frequency (no lag!)")

    def reset_episode(self, episode_id: Optional[str] = None,
                     curriculum_lesson_id: Optional[str] = None,
                     max_steps: int = 1000) -> np.ndarray:
        """
        Reset episode - MODAL-TO-MODAL communication

        Args:
            episode_id: Optional episode identifier
            curriculum_lesson_id: Optional curriculum lesson identifier
            max_steps: Maximum steps per episode

        Returns:
            initial_observation: Initial observation as numpy array
        """
        # ExperimentOps resets physics (DELEGATE!)
        self.experiment_ops.reset()

        # Episode modal resets itself (SELF-RESETTING!)
        if episode_id is None:
            episode_id = f"ep_{int(time.time() * 1000)}"

        self.episode_modal = EpisodeModal(
            episode_id=episode_id,
            curriculum_lesson_id=curriculum_lesson_id,
            max_steps=max_steps
        )
        self.episode_modal.reset()

        # Get initial observation
        obs = self._get_observation()
        return obs

    def step_bundled(self) -> Tuple[np.ndarray, float, Dict]:
        """
        Core RL operation: Bundle N physics steps → 1 RL observation

        THIS IS THE CORE OPERATION FOR BOTH:
        - Continuous control (called once per action)
        - Action blocks (called until action completes)

        For 30Hz RL rate with 200Hz physics:
        - Bundles 7 physics steps (200/30 ≈ 7)
        - Returns observation/reward synchronized with state updates
        - No lag between state and rewards!

        Returns:
            observation: Observation vector (30Hz rate)
            reward: Scalar reward (30Hz rate)
            info: Reward details dict
        """
        # STEP BUNDLING: Execute multiple physics steps per RL observation
        for _ in range(self._steps_per_rl_action):
            self.experiment_ops.step()

        # Get observation (DELEGATE to ViewAggregator via ExperimentOps!)
        obs = self._get_observation()

        # Evaluate rewards (DELEGATE to RewardModal!)
        reward, reward_info = self.experiment_ops.evaluate_rewards()

        return obs, reward, reward_info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute RL step - BUNDLES multiple physics steps

        THIS IS WHERE STEP BUNDLING HAPPENS!

        For 30Hz RL rate with 200Hz physics:
        - Calls experiment_ops.step() 7 times (bundled)
        - Returns observation/reward after all physics steps complete
        - RL agent observes at 30Hz (synchronized with state updates)

        Args:
            action: Action vector (shape: action_dim,)

        Returns:
            observation: Observation vector
            reward: Scalar reward
            terminated: Goal reached (from reward modal)
            truncated: Max steps exceeded
            info: Additional info dict
        """
        # OFFENSIVE validation
        if self.episode_modal is None:
            raise RuntimeError(
                "Cannot step: Episode not initialized!\n"
                "Call reset_episode() before stepping!"
            )

        # Validate action shape
        if action.shape[0] != self.action_space_modal.action_dim:
            raise ValueError(
                f"Action dimension mismatch!\n"
                f"Expected: {self.action_space_modal.action_dim}\n"
                f"Got: {action.shape[0]}\n"
                f"Action must match action space dimension!"
            )

        # Apply action to actuators (DELEGATE to ExperimentOps!)
        self.experiment_ops.apply_action(
            action,
            self.action_space_modal.actuators_active
        )

        # Use core bundled step operation (MINIMAL CODE!)
        obs, reward, reward_info = self.step_bundled()

        # Check termination (DELEGATE to RewardModal!)
        terminated = self.experiment_ops.check_termination()

        # Episode modal updates itself (SELF-UPDATING!)
        truncated = self.episode_modal.step_count >= self.episode_modal.max_steps
        self.episode_modal.step(action, reward, obs, terminated, truncated)

        # Build info dict
        info = {
            'step_count': self.episode_modal.step_count,
            'episode_reward': self.episode_modal.total_reward,
            **reward_info  # Include reward details
        }

        return obs, reward, terminated, truncated, info

    def execute_action_block(self, action_block, track_trajectory: bool = False):
        """
        Execute action block at RL frequency - MINIMAL WRAPPER!

        Action blocks can be:
        - High-level: spin(45°), move_forward(1.0m)
        - Low-level: set_wheel_velocity(0.5, 0.5)
        - Hybrid: complex behaviors with multiple actuators

        The block executes itself until completion.
        RLOps just provides 30Hz observations/rewards (no lag!).

        Args:
            action_block: Action block instance (from action_blocks_registry)
            track_trajectory: If True, return full trajectory (for debugging)

        Returns:
            If track_trajectory=False:
                final_obs, total_reward, info
            If track_trajectory=True:
                (final_obs, total_reward, info), trajectory dict
        """
        # OFFENSIVE validation
        if self.episode_modal is None:
            raise RuntimeError(
                "Cannot execute action: Episode not initialized!\n"
                "Call reset_episode() before executing actions!"
            )

        # Submit action block (DELEGATE to ExperimentOps!)
        print(f"DEBUG [execute_action_block]: Submitting action block...")
        import sys
        sys.stdout.flush()
        self.experiment_ops.submit_block(action_block)
        print(f"DEBUG [execute_action_block]: Action block submitted, status={action_block.status}")
        sys.stdout.flush()

        # Track trajectory if requested
        trajectory = {
            "observations": [],
            "rewards": [],
            "step_rewards": [],
            "rl_steps": 0
        } if track_trajectory else None

        # Step at RL frequency until action completes
        total_reward = 0.0
        obs, reward, reward_info = None, 0.0, {}

        step_count = 0
        while action_block.status != "completed":
            step_count += 1
            if step_count % 10 == 1:  # Print every 10 steps
                print(f"DEBUG [execute_action_block]: Step {step_count}, status={action_block.status}")
                sys.stdout.flush()

            # Use core bundled step operation (SAME as continuous control!)
            obs, reward, reward_info = self.step_bundled()
            total_reward += reward

            # Track trajectory
            if track_trajectory:
                trajectory["observations"].append(obs)
                trajectory["rewards"].append(total_reward)
                trajectory["step_rewards"].append(reward)
                trajectory["rl_steps"] += 1

        print(f"DEBUG [execute_action_block]: Action block completed after {step_count} RL steps!")
        sys.stdout.flush()

        # Check termination
        terminated = self.experiment_ops.check_termination()

        # Update episode modal (if tracking episodes)
        if self.episode_modal:
            # For action blocks, we track the FINAL result
            truncated = self.episode_modal.step_count >= self.episode_modal.max_steps
            # Use dummy action (action blocks don't have fixed-dim vectors)
            dummy_action = np.zeros(1)
            self.episode_modal.step(dummy_action, total_reward, obs, terminated, truncated)

        # Build info dict
        info = {
            'action_status': action_block.status,
            'total_reward': total_reward,
            'rl_steps': trajectory["rl_steps"] if track_trajectory else None,
            **reward_info
        }

        if track_trajectory:
            return (obs, total_reward, info), trajectory
        return obs, total_reward, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation - DELEGATE to ExperimentOps

        Returns:
            observation: Flattened observation vector
        """
        # DELEGATE to ExperimentOps (uses StateOps internally) - CLEAN API!
        obs_dict = self.experiment_ops.get_state()

        # Flatten to numpy array for RL
        if isinstance(obs_dict, dict):
            obs_list = []
            for key, value in obs_dict.items():
                if isinstance(value, (int, float)):
                    obs_list.append(value)
                elif isinstance(value, np.ndarray):
                    obs_list.extend(value.flatten())
            return np.array(obs_list, dtype=np.float32)
        elif isinstance(obs_dict, np.ndarray):
            return obs_dict.flatten().astype(np.float32)
        else:
            # OFFENSIVE: Unknown format
            raise TypeError(
                f"Unknown observation format: {type(obs_dict)}\n"
                f"Expected dict or np.ndarray!"
            )

    def get_episode_metrics(self) -> Dict[str, Any]:
        """
        Get current episode metrics

        Returns:
            metrics: Episode metrics dict
        """
        if self.episode_modal is None:
            return {}
        return self.episode_modal.get_data()

    def start_training_session(self, session_id: str, algorithm: str = "unknown",
                               num_episodes: int = 1000):
        """
        Start multi-episode training session

        Args:
            session_id: Training session identifier
            algorithm: RL algorithm name (e.g., "PPO", "SAC")
            num_episodes: Total episodes to train
        """
        self.training_session_modal = TrainingSessionModal(
            session_id=session_id,
            algorithm=algorithm,
            num_episodes=num_episodes
        )
        self.training_session_modal.start()
        print(f"🚀 Training session started: {session_id}")
        print(f"   Algorithm: {algorithm}")
        print(f"   Episodes: {num_episodes}")

    def end_training_session(self) -> Dict[str, Any]:
        """
        End training session and get summary

        Returns:
            summary: Training session summary
        """
        if self.training_session_modal is None:
            raise RuntimeError("No training session active!")

        self.training_session_modal.end()
        summary = self.training_session_modal.get_summary()

        print(f"🏁 Training session completed: {self.training_session_modal.session_id}")
        print(f"   Episodes: {summary['episodes_completed']}")
        print(f"   Total steps: {summary['total_steps']}")
        print(f"   Mean reward: {summary['mean_episode_reward']:.2f}")
        print(f"   Best reward: {summary['best_episode_reward']:.2f}")
        print(f"   Duration: {summary['duration']:.1f}s")

        return summary

    def log_episode(self):
        """Log completed episode to training session"""
        if self.training_session_modal and self.episode_modal:
            metrics = self.episode_modal.get_rl()
            self.training_session_modal.add_episode(metrics)

    # Convenience properties for gym-like interface
    @property
    def action_space(self):
        """Get action space modal"""
        return self.action_space_modal

    @property
    def observation_space(self):
        """Get observation space modal"""
        return self.observation_space_modal

    @property
    def episode(self):
        """Get current episode modal"""
        return self.episode_modal

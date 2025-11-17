"""
RL Modals - Modal Oriented Programming for Reinforcement Learning

ARCHITECTURE:
- EpisodeModal: SELF-TRACKING episode state and metrics
- ActionSpaceModal: AUTO-DISCOVERY of action space from robot actuators
- ObservationSpaceModal: AUTO-DISCOVERY of observation space from robot sensors
- TrainingSessionModal: SELF-TRACKING multi-episode training runs

PRINCIPLES:
1. AUTO-DISCOVERY: Modals discover their structure from robot/sensors/actuators
2. SELF-RESETTING: Modals reset themselves (episode.reset())
3. SELF-UPDATING: Modals update themselves (episode.step())
4. SELF-RENDERING: Modals render in multiple formats (get_data(), get_rl())
5. OFFENSIVE VALIDATION: Crash with helpful errors, not defensive .get()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time
import numpy as np


@dataclass
class EpisodeModal:
    """
    Episode state tracking - SELF-TRACKING

    Tracks episode metrics, history, and lifecycle.
    Resets itself, updates itself, renders itself.

    Usage:
        episode = EpisodeModal(episode_id="ep_001", max_steps=1000)
        episode.reset()
        for _ in range(max_steps):
            episode.step(action, reward, obs, terminated, truncated)
            if episode.is_done():
                break
        metrics = episode.get_metrics()
    """

    # Episode identification
    episode_id: str
    curriculum_lesson_id: Optional[str] = None

    # Episode configuration
    max_steps: int = 1000

    # Episode state (auto-updated)
    step_count: int = 0
    total_reward: float = 0.0
    terminated: bool = False  # Goal reached
    truncated: bool = False   # Max steps exceeded

    # Episode history (auto-tracked)
    reward_history: List[float] = field(default_factory=list)
    action_history: List[np.ndarray] = field(default_factory=list)
    observation_history: List[np.ndarray] = field(default_factory=list)

    # Timestamps
    start_time: float = 0.0
    end_time: Optional[float] = None

    def reset(self):
        """SELF-RESETTING: Episode resets itself"""
        self.step_count = 0
        self.total_reward = 0.0
        self.terminated = False
        self.truncated = False
        self.reward_history = []
        self.action_history = []
        self.observation_history = []
        self.start_time = time.time()
        self.end_time = None

    def step(self, action: np.ndarray, reward: float, obs: np.ndarray,
             terminated: bool, truncated: bool):
        """SELF-UPDATING: Track step data"""
        self.step_count += 1
        self.total_reward += reward
        self.terminated = terminated
        self.truncated = truncated

        # Track history
        self.action_history.append(action.copy() if isinstance(action, np.ndarray) else action)
        self.reward_history.append(reward)
        self.observation_history.append(obs.copy() if isinstance(obs, np.ndarray) else obs)

        # Check truncation
        if self.step_count >= self.max_steps:
            self.truncated = True

        # Mark end time if done
        if self.is_done() and self.end_time is None:
            self.end_time = time.time()

    def is_done(self) -> bool:
        """Check if episode is finished"""
        return self.terminated or self.truncated

    def get_duration(self) -> float:
        """Get episode duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def get_data(self) -> Dict[str, Any]:
        """SELF-RENDERING: Human-readable format"""
        return {
            "episode_id": self.episode_id,
            "curriculum_lesson": self.curriculum_lesson_id,
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "duration": self.get_duration(),
            "mean_reward": np.mean(self.reward_history) ,
            "max_reward": np.max(self.reward_history) ,
            "min_reward": np.min(self.reward_history)
        }

    def get_rl(self) -> Dict[str, float]:
        """SELF-RENDERING: RL-compatible metrics format"""
        return {
            "episode_length": self.step_count,
            "episode_reward": self.total_reward,
            "mean_reward": np.mean(self.reward_history) if self.reward_history else 0.0
        }


@dataclass
class ActionSpaceModal:
    """
    Action space configuration - AUTO-DISCOVERY

    Discovers action space bounds from robot actuators.
    No manual configuration needed - scans actuator modals.

    Usage:
        action_space = ActionSpaceModal()
        action_space.discover_from_robot(robot, actuators_active=["base", "arm"])
        print(f"Action dim: {action_space.action_dim}")
        print(f"Bounds: [{action_space.action_low}, {action_space.action_high}]")
    """

    # Configuration
    actuators_active: Optional[List[str]] = None  # None = all actuators

    # Auto-discovered space metadata
    action_dim: int = 0
    action_low: Optional[np.ndarray] = None
    action_high: Optional[np.ndarray] = None
    actuator_names: List[str] = field(default_factory=list)

    def discover_from_robot(self, robot, actuators_active: Optional[List[str]] = None):
        """
        AUTO-DISCOVERY: Extract action bounds from actuator modals

        Args:
            robot: Robot object with actuators dict
            actuators_active: List of actuator names to include (None = all)
        """
        self.actuators_active = actuators_active

        # Select active actuators
        if actuators_active:
            active = {k: v for k, v in robot.actuators.items() if k in actuators_active}
            # OFFENSIVE validation
            missing = set(actuators_active) - set(robot.actuators.keys())
            if missing:
                raise ValueError(
                    f"Actuators not found: {missing}\n"
                    f"Available actuators: {list(robot.actuators.keys())}\n"
                    f"Check your actuators_active list!"
                )
        else:
            active = robot.actuators

        # Extract bounds from actuator modals
        lows, highs = [], []
        self.actuator_names = []

        for name, actuator in active.items():
            # TRUST MODAL: Actuator knows its own range
            if hasattr(actuator, 'range'):
                low, high = actuator.range
            elif hasattr(actuator, 'ctrl_range'):
                low, high = actuator.ctrl_range
            else:
                # OFFENSIVE: Crash with helpful error
                raise AttributeError(
                    f"Actuator '{name}' missing 'range' or 'ctrl_range' attribute!\n"
                    f"Actuator type: {type(actuator)}\n"
                    f"Actuator modals must declare their control ranges!"
                )

            lows.append(low)
            highs.append(high)
            self.actuator_names.append(name)

        self.action_dim = len(lows)
        self.action_low = np.array(lows, dtype=np.float32)
        self.action_high = np.array(highs, dtype=np.float32)

        # OFFENSIVE validation
        if self.action_dim == 0:
            raise ValueError(
                f"No actuators found!\n"
                f"Robot actuators: {list(robot.actuators.keys())}\n"
                f"Actuators active: {actuators_active}\n"
                f"Cannot create action space with 0 dimensions!"
            )

    def get_data(self) -> Dict[str, Any]:
        """SELF-RENDERING: Human-readable format"""
        return {
            "action_dim": self.action_dim,
            "action_low": self.action_low.tolist() if self.action_low is not None else None,
            "action_high": self.action_high.tolist() if self.action_high is not None else None,
            "actuators_active": self.actuators_active,
            "actuator_names": self.actuator_names
        }


@dataclass
class ObservationSpaceModal:
    """
    Observation space configuration - AUTO-DISCOVERY

    Discovers observation space from robot sensors/actuators.
    Delegates to ViewAggregator for actual observation extraction.

    Usage:
        obs_space = ObservationSpaceModal()
        obs_space.discover_from_experiment(experiment_ops)
        print(f"Obs dim: {obs_space.obs_dim}")
    """

    # Configuration
    sensors_enabled: List[str] = field(default_factory=list)
    actuators_enabled: List[str] = field(default_factory=list)
    vision_enabled: bool = False

    # Auto-discovered space metadata
    obs_dim: int = 0
    obs_low: Optional[np.ndarray] = None
    obs_high: Optional[np.ndarray] = None

    def discover_from_experiment(self, experiment_ops):
        """
        AUTO-DISCOVERY: Extract observation space from experiment

        Args:
            experiment_ops: ExperimentOps instance with robot and scene
        """
        robot = experiment_ops.robot

        # OFFENSIVE validation
        if robot is None:
            raise ValueError(
                "Cannot discover observation space: No robot in experiment!\n"
                "Call experiment_ops.add_robot() before discovering observation space!"
            )

        # Auto-discover sensors and actuators
        self.sensors_enabled = list(robot.sensors.keys()) if hasattr(robot, 'sensors') else []
        self.actuators_enabled = list(robot.actuators.keys()) if hasattr(robot, 'actuators') else []

        # Get sample observation to determine dimensionality
        # Delegate to StateOps (MODAL-TO-MODAL communication)
        sample_obs = experiment_ops.get_state()

        if isinstance(sample_obs, dict):
            # Flatten dict observations
            obs_list = []
            for key, value in sample_obs.items():
                if isinstance(value, (int, float)):
                    obs_list.append(value)
                elif isinstance(value, np.ndarray):
                    obs_list.extend(value.flatten())
            self.obs_dim = len(obs_list)
        elif isinstance(sample_obs, np.ndarray):
            self.obs_dim = sample_obs.flatten().shape[0]
        else:
            # OFFENSIVE: Unknown observation format
            raise TypeError(
                f"Unknown observation format: {type(sample_obs)}\n"
                f"Expected dict or np.ndarray, got {type(sample_obs)}\n"
                f"ViewAggregator must return dict or np.ndarray!"
            )

        # Set bounds (generic -inf to +inf for now)
        self.obs_low = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        self.obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)

    def get_data(self) -> Dict[str, Any]:
        """SELF-RENDERING: Human-readable format"""
        return {
            "obs_dim": self.obs_dim,
            "sensors_enabled": self.sensors_enabled,
            "actuators_enabled": self.actuators_enabled,
            "vision_enabled": self.vision_enabled
        }


@dataclass
class TrainingSessionModal:
    """
    Training session tracking - SELF-TRACKING

    Tracks multi-episode training runs.
    Aggregates metrics across episodes.

    Usage:
        session = TrainingSessionModal(session_id="ppo_training_001", num_episodes=1000)
        session.start()
        for episode in range(num_episodes):
            # ... run episode ...
            session.add_episode(episode_metrics)
        session.end()
        summary = session.get_summary()
    """

    # Session identification
    session_id: str
    algorithm: str = "unknown"

    # Session configuration
    num_episodes: int = 1000

    # Session state
    episodes_completed: int = 0
    total_steps: int = 0

    # Episode metrics (aggregated)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)

    # Timestamps
    start_time: float = 0.0
    end_time: Optional[float] = None

    def start(self):
        """Start training session"""
        self.start_time = time.time()
        self.episodes_completed = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def add_episode(self, episode_metrics: Dict):
        """Add completed episode metrics"""
        self.episodes_completed += 1
        self.episode_rewards.append(episode_metrics.get('episode_reward', 0.0))
        self.episode_lengths.append(episode_metrics.get('episode_length', 0))
        self.total_steps += episode_metrics.get('episode_length', 0)

    def end(self):
        """End training session"""
        self.end_time = time.time()

    def get_duration(self) -> float:
        """Get session duration in seconds"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    def get_summary(self) -> Dict[str, Any]:
        """SELF-RENDERING: Training session summary"""
        return {
            "session_id": self.session_id,
            "algorithm": self.algorithm,
            "episodes_completed": self.episodes_completed,
            "total_steps": self.total_steps,
            "duration": self.get_duration(),
            "mean_episode_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "mean_episode_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "best_episode_reward": np.max(self.episode_rewards) if self.episode_rewards else 0.0
        }

    def get_data(self) -> Dict[str, Any]:
        """SELF-RENDERING: Full session data"""
        return {
            **self.get_summary(),
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths
        }

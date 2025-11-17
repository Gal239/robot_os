#!/usr/bin/env python3
"""
LEVEL 2E: SIMPLE PPO TRAINING - Learn to Spin to 45°
====================================================

Train a simple RL agent to spin to 45° from any starting position.

Agent learns the mapping:
  observation = [current_rotation, target_rotation, error]
  action = degrees_to_spin

Over ~100 episodes, agent should learn:
  action ≈ target - current

We'll use a simple policy gradient approach (like PPO but simpler).
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Enable EGL for GPU-accelerated rendering
os.environ['MUJOCO_GL'] = 'egl'

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.rl.rl_experiment_modal import RLExperimentModal, TrainingMode
from core.rl.rl_experiment_ops import RLExperimentOps


class SimplePolicy:
    """Simple neural network policy - learns to map observation to action"""

    def __init__(self, obs_dim=3, action_dim=1, learning_rate=0.01):
        """Initialize simple 1-layer network"""
        # Weights: obs_dim x action_dim
        self.W = np.random.randn(obs_dim, action_dim) * 0.1
        self.b = np.zeros((1, action_dim))
        self.lr = learning_rate

        # Experience buffer
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def predict(self, obs):
        """Predict action from observation (with exploration noise)"""
        # Linear policy: action = W * obs + b
        obs = obs.reshape(1, -1)
        action = obs @ self.W + self.b

        # Add exploration noise (decreases over time)
        noise = np.random.randn() * 50.0  # ±50° exploration
        action = action[0, 0] + noise

        # Clip to action space
        action = np.clip(action, -360, 360)
        return action

    def store_experience(self, obs, action, reward):
        """Store experience for training"""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def train(self):
        """Update policy using collected experience (simple policy gradient)"""
        if len(self.obs_buffer) == 0:
            return

        # Convert to arrays
        obs = np.array(self.obs_buffer)
        actions = np.array(self.action_buffer).reshape(-1, 1)
        rewards = np.array(self.reward_buffer).reshape(-1, 1)

        # Normalize rewards (helps learning)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Predict current actions
        predicted_actions = obs @ self.W + self.b

        # Compute gradient (simplified policy gradient)
        # Update weights to make successful actions more likely
        grad_W = obs.T @ (rewards * (actions - predicted_actions))
        grad_b = (rewards * (actions - predicted_actions)).sum(axis=0, keepdims=True)

        # Update weights
        self.W += self.lr * grad_W / len(obs)
        self.b += self.lr * grad_b / len(obs)

        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

        return grad_W.mean()


def train_simple_ppo(num_episodes=100, plot_results=True):
    """Train simple policy to spin to 45°"""

    print("=" * 70)
    print("LEVEL 2E: SIMPLE RL TRAINING - Learn to Spin to 45°")
    print("=" * 70)
    print("\nScenario:")
    print("  - Robot starts at random rotation")
    print("  - Goal: Reach exactly 45°")
    print("  - Agent learns: action ≈ target - current")
    print()
    print(f"Training for {num_episodes} episodes...")
    print()

    # ============================================
    # STEP 1: Create simulation environment
    # ============================================
    ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
    ops.create_scene("rl_training", width=5, length=5, height=3)
    ops.add_robot("stretch", position=(0, 0, 0))

    # ============================================
    # STEP 2: Define the task (reward)
    # ============================================
    # NEW API: Target-based rewards that penalize overshooting!
    #
    # OLD BUG: threshold=45.0 gave 100pts for ANY angle ≥45° (discrete mode)
    #          → Robot at 102.5° got 100pts! Doesn't teach precision!
    #
    # NEW FIX: reward_target=45.0 with reward_mode="convergent"
    #          → Within ±tolerance = 100pts
    #          → Overshooting decreases reward proportionally
    #          → Severe overshooting can give negative rewards!
    #
    ops.add_reward(
        tracked_asset="stretch.base",
        behavior="rotation",
        target=45.0,         # Target value
        reward=100.0,        # Points at target
        mode="convergent",   # Penalize overshooting
        id="spin_to_45"
    )

    # ============================================
    # STEP 3: Compile
    # ============================================
    ops.compile()

    # ============================================
    # STEP 4: Define RL experiment modal
    # ============================================
    modal = RLExperimentModal(
        name="learn_to_spin_45",
        mode=TrainingMode.FIXED_TARGET,
        fixed_target=45.0,
        total_episodes=num_episodes,
        max_steps_per_episode=300
    )

    # ============================================
    # STEP 5: Create RL environment
    # ============================================
    env = RLExperimentOps(modal, ops)

    # ============================================
    # STEP 6: Create policy
    # ============================================
    policy = SimplePolicy(obs_dim=3, action_dim=1, learning_rate=0.01)

    # ============================================
    # STEP 7: Training loop!
    # ============================================
    episode_rewards = []
    episode_successes = []
    episode_actions = []
    episode_errors = []

    print(f"{'Episode':>8} | {'Action':>8} | {'Final':>8} | {'Error':>8} | {'Reward':>8} | {'Success':>8}")
    print("-" * 70)

    for episode in range(num_episodes):
        # Reset environment
        obs, info = env.reset()

        # Get action from policy
        action = policy.predict(obs)

        # Execute action
        next_obs, reward, done, truncated, info = env.step(np.array([action]))

        # Store experience
        policy.store_experience(obs, action, info['total_reward'])

        # Record metrics
        episode_rewards.append(info['total_reward'])
        episode_successes.append(info['success'])
        episode_actions.append(action)

        error = abs(info['goal'] - info['final_rotation'])
        episode_errors.append(error)

        # Print progress
        if episode % 10 == 0 or episode < 5:
            print(f"{episode+1:8d} | {action:+8.1f}° | {info['final_rotation']:+8.1f}° | "
                  f"{error:8.1f}° | {info['total_reward']:8.1f}pts | "
                  f"{'✅' if info['success'] else '❌':>8}")

        # Train policy every 10 episodes
        if (episode + 1) % 10 == 0:
            grad = policy.train()

    # ============================================
    # STEP 8: Evaluate final policy
    # ============================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - Evaluating Final Policy")
    print("=" * 70)

    # Last 10 episodes stats
    recent_success_rate = sum(episode_successes[-10:]) / 10
    recent_avg_reward = np.mean(episode_rewards[-10:])
    recent_avg_error = np.mean(episode_errors[-10:])

    # Overall stats
    total_success_rate = sum(episode_successes) / num_episodes

    print(f"\nOverall Performance:")
    print(f"  Total episodes: {num_episodes}")
    print(f"  Success rate: {total_success_rate*100:.1f}%")
    print()
    print(f"Final 10 Episodes:")
    print(f"  Success rate: {recent_success_rate*100:.1f}%")
    print(f"  Average reward: {recent_avg_reward:.1f}pts")
    print(f"  Average error: {recent_avg_error:.1f}°")
    print()

    # ============================================
    # STEP 9: Plot learning curves
    # ============================================
    if plot_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Plot 1: Rewards over time
        axes[0, 0].plot(episode_rewards, alpha=0.3, label='Episode Reward')
        # Rolling average
        window = 10
        if len(episode_rewards) >= window:
            rolling_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(episode_rewards)), rolling_avg,
                           linewidth=2, label=f'{window}-Episode Average')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Learning Curve - Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Success rate over time
        success_rolling = np.convolve(episode_successes, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(episode_successes)), success_rolling * 100,
                       linewidth=2, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Learning Curve - Success Rate')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Rotation error over time
        axes[1, 0].plot(episode_errors, alpha=0.3, label='Episode Error')
        if len(episode_errors) >= window:
            error_rolling = np.convolve(episode_errors, np.ones(window)/window, mode='valid')
            axes[1, 0].plot(range(window-1, len(episode_errors)), error_rolling,
                           linewidth=2, label=f'{window}-Episode Average')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Rotation Error (degrees)')
        axes[1, 0].set_title('Learning Curve - Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Action distribution over time
        axes[1, 1].scatter(range(len(episode_actions)), episode_actions,
                          c=episode_rewards, cmap='RdYlGn', alpha=0.5)
        axes[1, 1].axhline(y=45, color='r', linestyle='--', label='Optimal (45°)')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Action (degrees)')
        axes[1, 1].set_title('Actions Over Time')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        save_path = "rl_training_results.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Learning curves saved to: {save_path}")
        plt.close()

    # ============================================
    # STEP 10: Test final policy
    # ============================================
    print("\n" + "=" * 70)
    print("TESTING FINAL POLICY - 5 Test Episodes")
    print("=" * 70)

    for test_ep in range(5):
        obs, info = env.reset()
        action = policy.predict(obs)
        next_obs, reward, done, truncated, info = env.step(np.array([action]))

        print(f"\nTest {test_ep + 1}:")
        print(f"  Start: {obs[0]*180:+6.1f}° → Goal: {info['goal']:+6.1f}°")
        print(f"  Action: {action:+6.1f}°")
        print(f"  Final: {info['final_rotation']:+6.1f}° (error: {abs(info['goal'] - info['final_rotation']):.1f}°)")
        print(f"  Success: {'✅' if info['success'] else '❌'}")

    env.close()

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

    return episode_rewards, episode_successes


if __name__ == "__main__":
    # Train for 100 episodes
    rewards, successes = train_simple_ppo(num_episodes=100, plot_results=True)

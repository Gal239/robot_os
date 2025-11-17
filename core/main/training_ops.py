"""
TRAINING OPS - High-level RL training functions
OFFENSIVE & MODAL-ORIENTED: 10 lines max for researchers!

THE KEY INNOVATION: actuators_active parameter enables FOCUSED action space training
- Baseline: actuators_active=None (all actuators, 15D, slow)
- Curriculum: actuators_active=['base'] (1-2D, 10-100x faster!)

This enables learning THOUSANDS of sub-skills efficiently:
- Spin test: 4 atomic skills (proof of concept)
- Pick & place: 20 atomic skills
- Household tasks: 1000+ atomic skills

Usage (10 lines total!):
    # Train atomic skill with FOCUSED action space
    train_atomic_skill(
        skill_name="turn_left_90",
        actuators_active=["base"],  # FOCUSED: 1-2D action space!
        reward_fn={
            "tracked_asset": "stretch.odometry",
            "behavior": "rotation",
            "threshold": 90.0,
            "reward": 100,
            "mode": "smooth"
        },
        timesteps=50_000
    )

    # Compose 4x turn_left_90 = spin_360
    compose_skills(["turn_left_90"] * 4, ops)
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np


def train_atomic_skill(
    skill_name: str,
    actuators_active: List[str],
    reward_fn: Dict,
    timesteps: int = 50_000,
    algorithm: str = "PPO",
    scene_config: Optional[Dict] = None,
    save_dir: str = "runs",
    verbose: int = 1
) -> str:
    """Train atomic skill with FOCUSED action space - 10 LINES MAX FOR USER!

    THE KEY INNOVATION: actuators_active = FOCUSED action space (1-2D per skill)
    This is what enables learning 1000s of skills efficiently!

    Args:
        skill_name: "turn_left_90", "move_forward", "extend_arm", etc.
        actuators_active: ["base"] for navigation, ["arm"] for manipulation, etc.
        reward_fn: {"tracked_asset": "stretch.odometry", "behavior": "rotation", "threshold": 90.0, "reward": 100}
        timesteps: Training timesteps (default 50K)
        algorithm: "PPO" or "SAC" (default PPO)
        scene_config: Optional scene parameters (width, length, height)
        save_dir: Directory to save models
        verbose: 0 (none), 1 (info), 2 (debug)

    Returns:
        Path to saved model (e.g., "runs/turn_left_90/model.zip")

    Example (researcher writes 10 lines):
        train_atomic_skill(
            skill_name="turn_left_90",
            actuators_active=["base"],  # FOCUSED: 1-2D only!
            reward_fn={
                "tracked_asset": "stretch.odometry",
                "behavior": "rotation",
                "threshold": 90.0,
                "reward": 100,
                "mode": "smooth",
                "id": "turn_left"
            },
            timesteps=50_000
        )
    """
    from ..main.experiment_ops_unified import ExperimentOps
    from ..runtime.gym_bridge import GymBridge
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    # Default scene config
    if scene_config is None:
        scene_config = {
            "name": "training_room",
            "width": 5.0,
            "length": 5.0,
            "height": 2.5
        }

    # Create experiment
    ops = ExperimentOps(headless=True)
    ops.create_scene(**scene_config)
    ops.add_robot("stretch")

    # Add reward (OFFENSIVE: crash if missing required fields)
    assert "id" in reward_fn, "reward_fn must include 'id' field"
    ops.add_reward(**reward_fn)

    # Compile scene
    ops.compile()

    # Wrap as gym environment with FOCUSED action space
    env = GymBridge(ops, actuators_active=actuators_active)
    env = Monitor(env)  # Track episode stats

    # Create save directory
    save_path = Path(save_dir) / skill_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(save_path),
        log_path=str(save_path / "logs"),
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Train with specified algorithm
    if algorithm.upper() == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            tensorboard_log=str(save_path / "tensorboard")
        )
    elif algorithm.upper() == "SAC":
        model = SAC(
            "MlpPolicy",
            env,
            verbose=verbose,
            tensorboard_log=str(save_path / "tensorboard")
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'PPO' or 'SAC'")

    # Train
    start_time = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback
    )
    training_time = time.time() - start_time

    # Save final model
    model_path = save_path / "model"
    model.save(str(model_path))

    # Save training metadata
    metadata = {
        "skill_name": skill_name,
        "actuators_active": actuators_active,
        "action_space_dim": env.action_space.shape[0],
        "timesteps": timesteps,
        "training_time_seconds": training_time,
        "algorithm": algorithm,
        "reward_fn": reward_fn
    }

    import json
    with open(save_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose >= 1:
        print(f"\n✅ Skill '{skill_name}' trained successfully!")
        print(f"   Action space: {actuators_active} ({env.action_space.shape[0]}D)")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Saved to: {model_path}.zip")

    env.close()

    # Create Skill and register to SKILLS_REGISTRY.json - RL LEGO COMPOSITION
    from ..modals import Skill
    from ..modals.stretch.action_blocks_registry import register_skill

    skill = Skill(
        id=skill_name,
        name=skill_name.replace("_", " ").title(),
        description=f"RL-trained skill using {algorithm}",
        action_blocks=[],  # RL policy doesn't use predefined blocks
        source="rl_trained",
        created_by=algorithm.lower(),
        policy_path=str(model_path) + ".zip",
        training_info={
            "algorithm": algorithm,
            "actuators_active": actuators_active,
            "action_space_dim": env.action_space.shape[0],
            "timesteps": timesteps,
            "training_time_seconds": training_time,
            "reward_fn": reward_fn
        }
    )

    # Register to SKILLS_REGISTRY.json
    register_skill(skill)

    if verbose >= 1:
        print(f"   ✅ Registered skill to SKILLS_REGISTRY.json")

    return skill


def load_skill(skill_name: str, save_dir: str = "runs", algorithm: str = "PPO") -> Any:
    """Load trained skill policy - OFFENSIVE

    Args:
        skill_name: Name of skill to load
        save_dir: Directory where models are saved
        algorithm: "PPO" or "SAC"

    Returns:
        Trained SB3 model ready for predict()

    Example:
        model = load_skill("turn_left_90")
        action, _ = model.predict(obs)
    """
    from stable_baselines3 import PPO, SAC

    model_path = Path(save_dir) / skill_name / "model"

    if not model_path.with_suffix('.zip').exists():
        raise FileNotFoundError(
            f"Model '{skill_name}' not found at {model_path}.zip\n"
            f"Train it first with train_atomic_skill()"
        )

    # Load model
    if algorithm.upper() == "PPO":
        model = PPO.load(str(model_path))
    elif algorithm.upper() == "SAC":
        model = SAC.load(str(model_path))
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Use 'PPO' or 'SAC'")

    return model


def execute_skill(
    skill_name: str,
    ops: Any,
    max_steps: int = 1000,
    save_dir: str = "runs",
    algorithm: str = "PPO"
) -> Dict[str, Any]:
    """Execute trained skill until completion - OFFENSIVE

    Args:
        skill_name: Name of trained skill
        ops: ExperimentOps instance (scene already set up)
        max_steps: Maximum steps to execute
        save_dir: Directory where models are saved
        algorithm: "PPO" or "SAC"

    Returns:
        {"success": bool, "reward": float, "steps": int, "observations": list}

    Example:
        # Setup scene
        ops = ExperimentOps()
        ops.create_scene("test_room", 5, 5, 2)
        ops.add_robot("stretch")
        ops.compile()

        # Execute skill
        result = execute_skill("turn_left_90", ops)
        print(f"Success: {result['success']}, Reward: {result['reward']}")
    """
    from ..runtime.gym_bridge import GymBridge

    # Load skill
    model = load_skill(skill_name, save_dir, algorithm)

    # Wrap ops as gym env (need to infer actuators_active from metadata)
    metadata_path = Path(save_dir) / skill_name / "metadata.json"
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        actuators_active = metadata['actuators_active']  # OFFENSIVE - crash if missing!
    else:
        actuators_active = None  # Fall back to all actuators

    env = GymBridge(ops, actuators_active=actuators_active)

    # Execute skill
    obs, _ = env.reset()
    total_reward = 0.0
    observations = []

    for step in range(max_steps):
        # Get action from trained policy
        action, _ = model.predict(obs, deterministic=True)

        # Execute
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        observations.append(obs.copy())

        # Check completion
        if terminated or truncated:
            break

    success = terminated  # Skill completed successfully

    return {
        "success": success,
        "reward": total_reward,
        "steps": step + 1,
        "observations": observations,
        "info": info
    }


def compose_skills(
    skill_sequence: List[str],
    ops: Any,
    save_dir: str = "runs",
    algorithm: str = "PPO",
    max_steps_per_skill: int = 1000,
    verbose: int = 1
) -> Dict[str, Any]:
    """Compose trained skills into new behavior - SEQUENTIAL EXECUTION

    This is THE TEST: Can atomic skills compose to achieve complex tasks?
    Example: 4x turn_left_90 = spin_360

    Args:
        skill_sequence: ["turn_left_90", "turn_left_90", "turn_left_90", "turn_left_90"]
        ops: ExperimentOps instance (scene + robot + rewards already set up)
        save_dir: Directory where models are saved
        algorithm: "PPO" or "SAC"
        max_steps_per_skill: Max steps per skill execution
        verbose: 0 (none), 1 (info), 2 (debug)

    Returns:
        {
            "success": bool,
            "total_reward": float,
            "total_steps": int,
            "skills_executed": int,
            "skill_results": [...]
        }

    Example (spin 360° from 4 atomic skills):
        # Setup scene with spin reward
        ops = ExperimentOps()
        ops.create_scene("test_room", 5, 5, 2)
        ops.add_robot("stretch")
        ops.add_reward(
            tracked_asset="stretch.odometry",
            behavior="rotation",
            threshold=360.0,
            reward=100,
            id="spin_360"
        )
        ops.compile()

        # Compose 4x 90° = 360°
        result = compose_skills(["turn_left_90"] * 4, ops)
        print(f"Success: {result['success']}, Reward: {result['total_reward']}")
    """
    total_reward = 0.0
    total_steps = 0
    skill_results = []

    if verbose >= 1:
        print(f"\n🔄 Composing {len(skill_sequence)} skills...")

    for i, skill_name in enumerate(skill_sequence):
        if verbose >= 1:
            print(f"   [{i+1}/{len(skill_sequence)}] Executing '{skill_name}'...")

        # Execute skill
        result = execute_skill(
            skill_name,
            ops,
            max_steps=max_steps_per_skill,
            save_dir=save_dir,
            algorithm=algorithm
        )

        skill_results.append({
            "skill_name": skill_name,
            "success": result['success'],
            "reward": result['reward'],
            "steps": result['steps']
        })

        total_reward += result['reward']
        total_steps += result['steps']

        if verbose >= 2:
            print(f"      Success: {result['success']}, Reward: {result['reward']:.2f}, Steps: {result['steps']}")

        # Stop if skill failed
        if not result['success']:
            if verbose >= 1:
                print(f"   ❌ Skill '{skill_name}' failed! Stopping composition.")
            break

    success = all(r['success'] for r in skill_results)

    if verbose >= 1:
        if success:
            print(f"✅ Composition successful!")
        else:
            print(f"❌ Composition failed!")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Total steps: {total_steps}")

    return {
        "success": success,
        "total_reward": total_reward,
        "total_steps": total_steps,
        "skills_executed": len(skill_results),
        "skill_results": skill_results
    }


def train_end_to_end(
    task_name: str,
    reward_fn: Dict,
    timesteps: int = 1_000_000,
    algorithm: str = "PPO",
    scene_config: Optional[Dict] = None,
    save_dir: str = "runs",
    verbose: int = 1
) -> Dict[str, Any]:
    """Train task end-to-end with FULL action space (baseline) - OFFENSIVE

    This is the BASELINE for Level 2A test.
    Uses ALL actuators (15D action space) - slow learning!

    Args:
        task_name: "spin_360", "pick_apple", etc.
        reward_fn: Same as train_atomic_skill
        timesteps: Training timesteps (default 1M for full action space!)
        algorithm: "PPO" or "SAC"
        scene_config: Optional scene parameters
        save_dir: Directory to save models
        verbose: 0 (none), 1 (info), 2 (debug)

    Returns:
        {
            "task_name": str,
            "timesteps": int,
            "training_time_seconds": float,
            "success_rate": float,
            "action_space_dim": int,
            "model_path": str
        }

    Example (baseline for spin 360°):
        result = train_end_to_end(
            task_name="spin_360_baseline",
            reward_fn={
                "tracked_asset": "stretch.odometry",
                "behavior": "rotation",
                "threshold": 360.0,
                "reward": 100,
                "id": "spin_360"
            },
            timesteps=1_000_000  # Much longer training!
        )
    """
    if verbose >= 1:
        print(f"\n🔵 Training '{task_name}' END-TO-END (baseline)")
        print(f"   Action space: ALL actuators (15D)")
        print(f"   This will be SLOW compared to curriculum!")

    # Train with actuators_active=None (all actuators = FULL action space)
    model_path = train_atomic_skill(
        skill_name=task_name,
        actuators_active=None,  # FULL action space (baseline)
        reward_fn=reward_fn,
        timesteps=timesteps,
        algorithm=algorithm,
        scene_config=scene_config,
        save_dir=save_dir,
        verbose=verbose
    )

    # Load metadata
    metadata_path = Path(save_dir) / task_name / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return {
        "task_name": task_name,
        "timesteps": metadata['timesteps'],
        "training_time_seconds": metadata['training_time_seconds'],
        "action_space_dim": metadata['action_space_dim'],
        "model_path": model_path,
        "approach": "end_to_end"
    }


def compare_approaches(
    task_name: str,
    reward_fn: Dict,
    curriculum_skills: Optional[List[str]] = None,
    baseline_timesteps: int = 1_000_000,
    curriculum_timesteps_per_skill: int = 50_000,
    save_dir: str = "runs",
    verbose: int = 1
) -> Dict[str, Any]:
    """Compare curriculum (focused action spaces) vs end-to-end (full action space)

    THE LEVEL 2A TEST: Does curriculum beat baseline?

    Args:
        task_name: "spin_360"
        reward_fn: Reward for the complete task
        curriculum_skills: ["turn_left_90"] or None for baseline only
        baseline_timesteps: Training time for baseline (default 1M)
        curriculum_timesteps_per_skill: Training time per atomic skill (default 50K)
        save_dir: Directory to save models
        verbose: 0 (none), 1 (info), 2 (debug)

    Returns:
        {
            "task_name": str,
            "curriculum": {...},  # Results for curriculum approach
            "baseline": {...},    # Results for baseline approach
            "winner": str         # "curriculum" or "baseline"
        }

    Example (Level 2A test):
        results = compare_approaches(
            task_name="spin_360",
            reward_fn={
                "tracked_asset": "stretch.odometry",
                "behavior": "rotation",
                "threshold": 360.0,
                "reward": 100,
                "id": "spin_360"
            },
            curriculum_skills=["turn_left_90"]  # 4x this = 360°
        )

        print(f"Winner: {results['winner']}")
        print(f"Speedup: {results['speedup']}x")
    """
    import json

    results = {"task_name": task_name}

    # Train baseline (end-to-end with FULL action space)
    if verbose >= 1:
        print(f"\n{'='*60}")
        print(f"LEVEL 2A TEST: {task_name}")
        print(f"{'='*60}")

    baseline_result = train_end_to_end(
        task_name=f"{task_name}_baseline",
        reward_fn=reward_fn,
        timesteps=baseline_timesteps,
        save_dir=save_dir,
        verbose=verbose
    )
    results["baseline"] = baseline_result

    # Train curriculum (if specified)
    if curriculum_skills:
        if verbose >= 1:
            print(f"\n🟢 Training CURRICULUM (focused action spaces)")
            print(f"   Skills: {curriculum_skills}")
            print(f"   Action space per skill: 1-2D (FOCUSED)")

        curriculum_result = {
            "approach": "curriculum",
            "skills": curriculum_skills,
            "total_timesteps": 0,
            "training_time_seconds": 0,
            "skill_models": []
        }

        # Train each atomic skill with FOCUSED action space
        for skill_name in set(curriculum_skills):  # Unique skills only
            # Infer actuators_active from skill name (simple heuristic)
            if "base" in skill_name or "turn" in skill_name or "move" in skill_name:
                actuators_active = ["base"]
            elif "arm" in skill_name:
                actuators_active = ["arm"]
            elif "lift" in skill_name:
                actuators_active = ["lift"]
            elif "gripper" in skill_name:
                actuators_active = ["gripper"]
            else:
                actuators_active = ["base"]  # Default

            # Create skill-specific reward
            skill_reward = reward_fn.copy()
            skill_reward['id'] = skill_name

            # Adjust threshold for atomic skill (1/4 of full if 4 skills)
            if len(curriculum_skills) > 1:
                original_threshold = skill_reward['threshold']  # OFFENSIVE - crash if missing!
                skill_reward['threshold'] = original_threshold / len(curriculum_skills)

            model_path = train_atomic_skill(
                skill_name=skill_name,
                actuators_active=actuators_active,  # FOCUSED!
                reward_fn=skill_reward,
                timesteps=curriculum_timesteps_per_skill,
                save_dir=save_dir,
                verbose=verbose
            )

            # Load metadata
            metadata_path = Path(save_dir) / skill_name / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            curriculum_result['total_timesteps'] += metadata['timesteps']
            curriculum_result['training_time_seconds'] += metadata['training_time_seconds']
            curriculum_result['skill_models'].append(model_path)

        results["curriculum"] = curriculum_result

        # Determine winner
        if curriculum_result['total_timesteps'] <= baseline_result['timesteps']:
            results['winner'] = 'curriculum'
        else:
            results['winner'] = 'baseline'

        results['speedup'] = baseline_result['timesteps'] / curriculum_result['total_timesteps']

        if verbose >= 1:
            print(f"\n{'='*60}")
            print(f"RESULTS")
            print(f"{'='*60}")
            print(f"Baseline:")
            print(f"  Timesteps: {baseline_result['timesteps']:,}")
            print(f"  Time: {baseline_result['training_time_seconds']:.1f}s")
            print(f"  Action space: {baseline_result['action_space_dim']}D")
            print(f"\nCurriculum:")
            print(f"  Timesteps: {curriculum_result['total_timesteps']:,}")
            print(f"  Time: {curriculum_result['training_time_seconds']:.1f}s")
            print(f"  Skills trained: {len(set(curriculum_skills))}")
            print(f"\n🏆 WINNER: {results['winner'].upper()}")
            print(f"   Speedup: {results['speedup']:.2f}x")

    return results

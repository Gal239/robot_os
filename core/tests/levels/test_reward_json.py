#!/usr/bin/env python3
"""
Test reward breakdown JSON output
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps

ops = ExperimentOps(headless=True)
ops.create_scene("test", width=5, length=5, height=3)
ops.add_robot("stretch", position=(0, 0, 0))
ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
ops.add_asset("banana", relative_to="table", relation="on_top", distance=0.75, surface_position="center")
ops.compile()

# Add rewards
ops.add_reward("table", "stable", target=True, reward=100, id="table_stable")
ops.add_reward("banana", "stacked_on", target=True, spatial_target="table", reward=100, id="banana_stacked")

print("\n🧪 Running 10 steps...\n")

for i in range(10):
    result = ops.step()

    # Print nice JSON for each step
    step_data = {
        "step": i,
        "step_reward": result.get('delta', 0),  # This step's reward
        "total_accumulated": result.get('total', 0),  # Cumulative total
        "rewards": {}
    }

    # Build per-reward details (PURE MOP: now includes cumulative total!)
    for reward_id, reward_data in result.get('rewards', {}).items():
        step_data["rewards"][reward_id] = {
            "delta": reward_data.get('delta', 0),
            "total": reward_data.get('total', 0),  # PURE MOP: Per-reward cumulative total!
            "multiplier": reward_data.get('multiplier', 0)
        }

    # Print all steps to see when rewards trigger
    print(f"📊 Step {i} Reward Breakdown:")
    print(json.dumps(step_data, indent=2))
    print()

print("\n📈 Final Summary:")
final_summary = {
    "total_accumulated_reward": result.get('total', 0),
    "rewards_breakdown": {}
}

# Access the reward modal to get original add_reward() parameters
reward_modal = ops.scene.reward_modal

for reward_id, reward_data in result.get('rewards', {}).items():
    # Get original condition from reward modal
    cond_info = reward_modal.conditions.get(reward_id, {})
    condition = cond_info.get("condition")
    reward_points = cond_info.get("reward", 0)

    # Build complete info showing original add_reward() call
    add_reward_params = {
        "asset": condition.asset if condition else None,
        "property": condition.prop if condition else None,
        "target": condition.val if condition else None,
        "reward": reward_points,
        "id": reward_id
    }

    # Add spatial_target if present (for spatial properties)
    if condition and condition.target:
        add_reward_params["spatial_target"] = condition.target

    # Add mode if not default
    if condition and condition.mode != "discrete":
        add_reward_params["mode"] = condition.mode

    reward_info = {
        "add_reward_call": add_reward_params,
        "results": {
            "total_accumulated": reward_data.get('total', 0),
            "last_delta": reward_data.get('delta', 0),
            "last_multiplier": reward_data.get('multiplier', 0)
        }
    }

    final_summary["rewards_breakdown"][reward_id] = reward_info

print(json.dumps(final_summary, indent=2))
#!/usr/bin/env python3
"""
DEBUG: Test spatial property reward
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps

ops = ExperimentOps(headless=True)
ops.create_scene("test", width=5, length=5, height=3)
ops.add_robot("stretch", position=(0, 0, 0))
ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
ops.add_asset("banana", relative_to="table", relation="on_top", distance=0.75, surface_position="center")
ops.compile()

# Test 1: Non-spatial property (should work)
ops.add_reward("table", "stable", target=True, reward=100, id="table_stable")

# Test 2: Spatial property
ops.add_reward("banana", "stacked_on", target=True, spatial_target="table", reward=100, id="banana_stacked")

print("\n🧪 Running 10 steps...")
for i in range(10):
    result = ops.step()
    if i == 0:
        state = ops.get_state()
        print(f"\n📊 State after first step:")
        print(f"   banana properties: {list(state.get('banana', {}).keys())[:10]}...")
        print(f"   stacked_on_table: {state.get('banana', {}).get('stacked_on_table', 'MISSING')}")
        print(f"   table.stable: {state.get('table', {}).get('stable', 'MISSING')}")

    rewards_dict = result.get('rewards', {})
    if any(r.get('delta', 0) > 0 for r in rewards_dict.values()):
        print(f"\nStep {i}: Rewards triggered!")
        for rid, rdata in rewards_dict.items():
            if rdata.get('delta', 0) > 0:
                print(f"   ✓ {rid}: +{rdata['delta']}pts")

print("\n📈 Final rewards:")
for rid, rdata in result.get('rewards', {}).items():
    print(f"   {rid}: delta={rdata.get('delta', 0)}, multiplier={rdata.get('multiplier', 0)}")

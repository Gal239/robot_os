#!/usr/bin/env python3
"""Test that scene_assets_state view is created and saved"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
ops.create_scene("test", width=10, length=10, height=3)
ops.add_asset("table", relative_to=(2, 0, 0))
ops.add_robot("stretch", position=(0, 0, 0))
ops.compile()
ops.step()

if hasattr(ops.engine, "last_views") and ops.engine.last_views:
    print("VIEWS CREATED:")
    for name in sorted(ops.engine.last_views.keys()):
        print(f"  {name}")

    if "scene_assets_state" in ops.engine.last_views:
        print("\nSCENE_ASSETS_STATE VIEW EXISTS! (FULL MOP STATE)")
        state = ops.engine.last_views["scene_assets_state"]
        state_clean = {k: v for k, v in state.items() if k != "__meta__"}
        for asset_name in sorted(state_clean.keys()):
            print(f"  {asset_name}: {list(state_clean[asset_name].keys())[:5]}...")
    else:
        print("\nERROR: scene_assets_state NOT FOUND!")

# Check if JSON data was collected
if hasattr(ops.engine, "timeline_saver"):
    ts = ops.engine.timeline_saver
    # Wait for async writer to finish
    import time
    time.sleep(0.5)
    print(f"\nJSON DATA COLLECTED:")
    json_keys = list(ts.json_data.keys())  # Copy keys to avoid race
    for view_name in sorted(json_keys):
        print(f"  {view_name}: {len(ts.json_data[view_name])} frames")

ops.close()

# Check if files saved
exp_dir = Path(ops.experiment_dir)
system_dir = exp_dir / "timeline" / "system"
print(f"\nFILES SAVED TO {system_dir}:")
if system_dir.exists():
    for f in system_dir.iterdir():
        print(f"  {f.name}")
else:
    print("  (directory missing)")

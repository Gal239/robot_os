"""Quick test of rl_core performance with RTX 3090"""
import sys
import os
import time
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

os.environ['MUJOCO_VIEWER_MODE'] = 'false'

from simulation_center.core.main.experiment_ops_unified import ExperimentOps

print("Testing rl_core with RTX 3090...")
ops = ExperimentOps(headless=True, render_mode='rl_core')
ops.create_scene('test', width=5, length=5, height=3)
ops.add_robot('stretch', position=(0, 0, 0))
ops.compile()

num_steps = 1000
t_start = time.perf_counter()
for i in range(num_steps):
    ops.step()
t_end = time.perf_counter()

elapsed = t_end - t_start
sim_time = num_steps * 0.005
rtf = sim_time / elapsed

ops.close()

print(f"\nrl_core Performance:")
print(f"  Before (Intel GPU): 2.68x")
print(f"  After (RTX 3090):   {rtf:.2f}x")
if rtf > 2.68:
    improvement = ((rtf - 2.68) / 2.68) * 100
    print(f"  Improvement:        +{improvement:.1f}%")

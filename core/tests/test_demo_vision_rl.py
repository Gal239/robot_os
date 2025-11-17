"""
Test tower stacking demo in vision_rl mode (real use case)
This is the actual RL training scenario with RTX 3090
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps

print("="*100)
print("TOWER STACKING DEMO - VISION_RL MODE (Real RL Training Use Case)")
print("="*100)
print("\nConfiguration:")
print("  - vision_rl mode (640x480, 10fps)")
print("  - Robot cameras: nav_camera, d405_camera")
print("  - Free cameras: overhead, tower_close, tower_side")
print("  - RTX 3090 GPU")
print()

# Create scene with vision_rl mode (ACTUAL RL TRAINING MODE)
ops = ExperimentOps(mode="simulated", headless=True, render_mode="vision_rl", save_fps=30)
ops.create_scene(name="tower_vision_rl", width=6, length=6, height=4)
ops.add_robot(robot_name="stretch", position=(0, 0, 0))
ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))

# Tower stacking
ops.add_asset(asset_name="wood_block", asset_id="block1", relative_to="table", relation="on_top", surface_position="center")
ops.add_asset(asset_name="wood_block", asset_id="block2", relative_to="block1", relation="on_top", surface_position="center")

# Rewards
ops.add_reward(tracked_asset="block1", behavior="stacked_on", target="table", reward=100, id="block1_table")
ops.add_reward(tracked_asset="block2", behavior="stacked_on", target="block1", reward=100, id="block2_on_block1")
ops.add_reward(tracked_asset="block1", behavior="stable", target=True, reward=50, id="block1_stable")
ops.add_reward(tracked_asset="block2", behavior="stable", target=True, reward=50, id="block2_stable")

# Cameras
ops.add_overhead_camera()
ops.add_free_camera(camera_id="tower_close", track_target="block1", distance=1.5, azimuth=45, elevation=-25)
ops.add_free_camera(camera_id="tower_side", lookat=(2.0, 0.0, 1.0), distance=3.0, azimuth=90, elevation=-15)

print("Compiling scene...")
ops.compile()

# SIMULATION SPEED TEST
num_steps = 1000
print(f"\nRunning {num_steps} steps (vision_rl mode - real RL training scenario)...")

t_start = time.perf_counter()

for i in range(num_steps):
    obs = ops.step()

    # This is what RL training would do - access camera observations
    if i == 0:
        print(f"\nObservation keys available: {list(obs.keys())}")
        if 'nav_camera_rgb' in obs:
            print(f"  nav_camera_rgb shape: {obs['nav_camera_rgb'].shape}")
        if 'd405_camera_rgb' in obs:
            print(f"  d405_camera_rgb shape: {obs['d405_camera_rgb'].shape}")

    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{num_steps}")

t_end = time.perf_counter()

# Get final reward
if 'reward' in obs:
    print(f"\nFinal total reward: {obs['reward']}")

print("\nClosing and encoding videos...")
close_start = time.perf_counter()
ops.close()
close_time = time.perf_counter() - close_start

# Calculate performance
elapsed = t_end - t_start
sim_time = num_steps * 0.005  # 5ms per step
real_time_factor = sim_time / elapsed

print("\n" + "="*100)
print("PERFORMANCE RESULTS - vision_rl MODE (RL TRAINING)")
print("="*100)
print(f"GPU: NVIDIA RTX 3090")
print(f"Mode: vision_rl (640x480, 10fps)")
print(f"Cameras: 2 robot + 3 free cameras")
print()
print(f"Pure simulation time: {elapsed:.2f}s")
print(f"Video encoding time:  {close_time:.2f}s")
print(f"Total time:           {elapsed + close_time:.2f}s")
print()
print(f"Simulation only:")
print(f"  Simulated: {sim_time:.2f}s ({num_steps} steps)")
print(f"  Real:      {elapsed:.2f}s")
print(f"  Real-time factor: {real_time_factor:.2f}x")
print()

# Camera data throughput
fps_nav = 10  # vision_rl uses 10fps for cameras
fps_free = 30  # free cameras at save_fps
robot_cameras = 2
free_cameras = 3

pixels_per_sec = (640 * 480 * robot_cameras * fps_nav) + (640 * 480 * free_cameras * fps_free)

print(f"Camera rendering:")
print(f"  Robot cameras: {robot_cameras} @ 640x480 @ {fps_nav}fps")
print(f"  Free cameras:  {free_cameras} @ 640x480 @ {fps_free}fps")
print(f"  Total throughput: {pixels_per_sec / 1e6:.1f}M pixels/sec")
print()

if real_time_factor >= 1.8:
    print(f"✅ EXCELLENT! Target achieved: {real_time_factor:.2f}x >= 1.8x")
    print(f"   Perfect for RL training!")
elif real_time_factor >= 1.5:
    print(f"✅ GOOD! Close to target: {real_time_factor:.2f}x (target: 1.8-2.2x)")
    print(f"   Usable for RL training")
else:
    print(f"⚠️  Below target: {real_time_factor:.2f}x (target: 1.8-2.2x)")

print()
print("COMPARISON:")
print(f"  Before GPU fix (Intel iGPU):  1.54x")
print(f"  After GPU fix (RTX 3090):     {real_time_factor:.2f}x")
improvement = ((real_time_factor - 1.54) / 1.54) * 100
print(f"  Improvement:                  +{improvement:.1f}%")
print("="*100)

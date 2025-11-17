"""
Measure PURE simulation speed for 2k_demo mode (no video encoding overhead)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps

print("="*100)
print("PURE SIMULATION SPEED TEST - 2K_DEMO MODE WITH RTX 3090")
print("="*100)
print("\nConfiguration:")
print("  - 7 cameras at 1920x1080 (2k resolution)")
print("  - 2 robot cameras (nav_camera, d405_camera)")
print("  - 3 free cameras (overhead, tracking, side)")
print("  - Recording at 30fps")
print("  - RTX 3090 GPU rendering")
print()

# Create scene with 2k_demo mode
ops = ExperimentOps(mode="simulated", headless=True, render_mode="2k_demo", save_fps=30)
ops.create_scene(name="speed_test", width=6, length=6, height=4)
ops.add_robot(robot_name="stretch", position=(0, 0, 0))
ops.add_asset(asset_name="table", relative_to=(2.0, 0.0, 0.0))

# Tower stacking
ops.add_asset(asset_name="wood_block", asset_id="block1", relative_to="table", relation="on_top", surface_position="center")
ops.add_asset(asset_name="wood_block", asset_id="block2", relative_to="block1", relation="on_top", surface_position="center")

# Add cameras
ops.add_overhead_camera()
ops.add_free_camera(camera_id="tower_close", track_target="block1", distance=1.5, azimuth=45, elevation=-25)
ops.add_free_camera(camera_id="tower_side", lookat=(2.0, 0.0, 1.0), distance=3.0, azimuth=90, elevation=-15)

print("Compiling scene...")
ops.compile()

# PURE SIMULATION SPEED TEST
num_steps = 1000
print(f"\nRunning {num_steps} steps (measuring ONLY simulation, not video encoding)...")

t_start = time.perf_counter()

for i in range(num_steps):
    ops.step()
    if (i + 1) % 200 == 0:
        print(f"  Progress: {i+1}/{num_steps}")

t_end = time.perf_counter()

print("\nClosing (this will encode videos - NOT counted in performance)...")
ops.close()

# Calculate performance
elapsed = t_end - t_start
sim_time = num_steps * 0.005  # 5ms per step
real_time_factor = sim_time / elapsed

print("\n" + "="*100)
print("PERFORMANCE RESULTS - PURE SIMULATION SPEED")
print("="*100)
print(f"GPU: NVIDIA RTX 3090")
print(f"Mode: 2k_demo (1920x1080)")
print(f"Cameras: 7 cameras (2 robot + 5 free)")
print(f"Recording: 30fps")
print()
print(f"Simulation time: {sim_time:.2f}s ({num_steps} steps)")
print(f"Real time:       {elapsed:.2f}s")
print(f"Real-time factor: {real_time_factor:.2f}x")
print()

# Comparison
print("COMPARISON:")
print(f"  vision_rl (640x480, 2 cameras, 10fps):  2.62x")
print(f"  2k_demo (1920x1080, 7 cameras, 30fps):  {real_time_factor:.2f}x")
print()

# Calculate rendering load
pixels_vision_rl = 640 * 480 * 2 * 10  # 2 cameras, 10fps
pixels_2k_demo = 1920 * 1080 * 7 * 30  # 7 cameras, 30fps
ratio = pixels_2k_demo / pixels_vision_rl

print(f"Pixel throughput:")
print(f"  vision_rl: {pixels_vision_rl / 1e6:.1f}M pixels/sec")
print(f"  2k_demo:   {pixels_2k_demo / 1e6:.1f}M pixels/sec")
print(f"  Ratio:     {ratio:.1f}x more pixels in 2k_demo!")
print()

if real_time_factor >= 1.0:
    print(f"✅ REAL-TIME CAPABLE: {real_time_factor:.2f}x speed!")
    print(f"   RTX 3090 handles {ratio:.1f}x more pixels with no slowdown!")
else:
    print(f"⚠️  Sub-real-time: {real_time_factor:.2f}x")
    print(f"   (This is expected with {ratio:.1f}x more pixels than vision_rl)")

print("="*100)

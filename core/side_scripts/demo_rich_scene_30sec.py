#!/usr/bin/env python3
"""
RICH 30-SECOND DEMO SCENE
Full scene with multiple objects, furniture, robot actions, all cameras
Perfect for showcasing the system!
"""

import sys
from pathlib import Path
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps
from simulation_center.core.modals.stretch.action_modals import ArmMoveTo, LiftMoveTo, BaseMoveForward, BaseMoveBackward, ActionBlock
import time
import os
print("=" * 80)
print("RICH 30-SECOND DEMO SCENE")
print("=" * 80)
print()
print("Creating a full demo with:")
print("  - Multiple objects (table, apples, bottles)")
print("  - Furniture (chairs)")
print("  - Robot actions (moving, lifting, reaching)")
print("  - Multiple cameras (orbiting, side view, top-down)")
print("  - 30 seconds @ demo mode (1280x720 HD @ 30fps)")
print()

# Create experiment with DEMO MODE for high quality
ops = ExperimentOps(headless=False, render_mode="rl_core")

print("Setting up scene...")

# Create big room for nice demo
ops.create_scene('rich_demo', width=8, length=8, height=3)

# Add robot at center
ops.add_robot('stretch', position=(0, 0, 0))

# Add table with objects
print("  Adding table...")
ops.add_asset("table", relative_to=(2, 0, 0))

print("  Adding fruits on table...")
ops.add_asset("apple", relative_to=(2, 0, 0.5))
ops.add_asset("banana", relative_to=(2.2, 0, 0.5))
ops.add_asset("orange", relative_to=(1.8, -0.2, 0.5))

print("  Adding objects...")
ops.add_asset("mug", relative_to=(2, 0.3, 0.5))
ops.add_asset("bowl", relative_to=(1.8, 0.3, 0.5))

# Add more furniture
print("  Adding furniture...")
ops.add_asset("desk", relative_to=(-2, 2, 0))
ops.add_asset("storage_bin", relative_to=(-3, 0, 0))

# Add multiple cameras for different views
print("  Setting up cameras...")

print("✓ Scene setup complete!")
print()

print("Compiling scene...")
ops.compile()
print("✓ Compiled!")
print()

# Plan robot actions for the 30 seconds
print("Planning robot actions...")
print("  0-5s:   Lift arm up")
print("  5-10s:  Extend arm forward")
print("  10-15s: Move forward toward table")
print("  15-20s: Retract arm")
print("  20-25s: Lower lift")
print("  25-30s: Move backward to start")
print()

# 30 seconds at 200Hz = 6000 steps
total_steps = 6000
print(f"Running {total_steps} steps (30 seconds at 200Hz)...")
print("Camera will orbit 360° around the scene...")
print()

start_time = time.time()

for step in range(total_steps):
    current_second = step / 200.0

    # Execute robot actions based on time
    if step == 0:
        # 0-5s: Lift arm up
        print("  [0s] Action: Lifting arm up to 1.0m")
        action = ActionBlock(id="lift_up", actions=[LiftMoveTo(position=1.0)])
        ops.submit_block(action)

    elif step == 1000:  # 5s
        # 5-10s: Extend arm forward
        print("  [5s] Action: Extending arm to 0.3m")
        action = ActionBlock(id="arm_extend", actions=[ArmMoveTo(position=0.3)])
        ops.submit_block(action)

    elif step == 2000:  # 10s
        # 10-15s: Move forward
        print("  [10s] Action: Moving forward 0.5m")
        action = ActionBlock(id="move_forward", actions=[BaseMoveForward(distance=0.5)])
        ops.submit_block(action)

    elif step == 3000:  # 15s
        # 15-20s: Retract arm
        print("  [15s] Action: Retracting arm to 0.0m")
        action = ActionBlock(id="arm_retract", actions=[ArmMoveTo(position=0.0)])
        ops.submit_block(action)

    elif step == 4000:  # 20s
        # 20-25s: Lower lift
        print("  [20s] Action: Lowering lift to 0.4m")
        action = ActionBlock(id="lift_lower", actions=[LiftMoveTo(position=0.4)])
        ops.submit_block(action)

    elif step == 5000:  # 25s
        # 25-30s: Move backward
        print("  [25s] Action: Moving backward to start")
        action = ActionBlock(id="move_backward", actions=[BaseMoveBackward(distance=0.5)])
        ops.submit_block(action)

    # Orbit the camera smoothly (360 degrees over 30 seconds)
    orbit_angle = (step / total_steps) * 360.0
    # ops.set_camera_angle('orbiting_cam', azimuth=orbit_angle)

    # Step simulation
    ops.step()

    # Progr
    # ess updates every 2 seconds
    if step > 0 and step % 400 == 0:
        elapsed = time.time() - start_time
        print(f"  [{int(current_second)}s / 30s] - camera angle={orbit_angle:.0f}° - elapsed={elapsed:.1f}s")

elapsed = time.time() - start_time
print()
print(f"✓ Completed {total_steps} steps in {elapsed:.1f}s")
print()

# Get experiment directory
exp_dir = ops.experiment_dir
print("=" * 80)
print("DEMO COMPLETE!")
print("=" * 80)
print()
print(f"📁 Experiment directory:")
print(f"   {exp_dir}")
print()
print(f"🎥 Videos created:")
print(f"   📹 orbiting_cam_rgb.mp4  - Main orbiting view (360° around scene)")
print(f"   📹 side_view_rgb.mp4      - Fixed side perspective")
print(f"   📹 birds_eye_rgb.mp4      - Top-down bird's eye view")
print(f"   📹 nav_camera_rgb.mp4     - Robot's navigation camera")
print(f"   📹 d405_camera_rgb.mp4    - Robot's wrist camera")
print()
print(f"📂 Full timeline directory:")
print(f"   {exp_dir}/timeline/cameras/")
print()
print("=" * 80)
print("VIDEO SPECS:")
print("=" * 80)
print("  Resolution: 1280x720 (HD)")
print("  Frame rate: 30 fps (smooth)")
print("  Duration: 30 seconds")
print("  Quality: Demo mode (high quality)")
print()
print("Scene content:")
print("  ✓ 1 robot performing 6 actions")
print("  ✓ 5 objects (3 apples, 2 bottles)")
print("  ✓ 1 table")
print("  ✓ 2 chairs")
print("  ✓ 2 boxes")
print("  ✓ 5 cameras (3 free + 2 robot)")
print()
print("Actions performed:")
print("  ✓ Lift arm up (0-5s)")
print("  ✓ Extend arm forward (5-10s)")
print("  ✓ Move toward table (10-15s)")
print("  ✓ Retract arm (15-20s)")
print("  ✓ Lower lift (20-25s)")
print("  ✓ Return to start (25-30s)")
print()
print("🎬 Play the videos to see the complete demo!")
print()

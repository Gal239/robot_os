#!/usr/bin/env python3
"""
SIM-TO-REAL RATIO BENCHMARK - All Render Modes
Tests performance of different render configurations
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps
import time

def test_render_mode(mode_name: str, render_mode: str, headless: bool):
    """
    Benchmark a specific render mode configuration

    Args:
        mode_name: Display name for results
        render_mode: "rl_core", "vision_rl", or "demo"
        headless: True (no viewer) or False (with viewer)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {mode_name}")
    print(f"  render_mode={render_mode}, headless={headless}")
    print('='*70)

    try:
        # Create experiment
        ops = ExperimentOps(
            headless=headless,
            render_mode=render_mode
        )

        # Simple scene
        ops.create_scene("benchmark", width=5, length=5, height=3)
        ops.add_robot("stretch", position=(0, 0, 0))
        ops.compile()

        # Benchmark 1000 steps
        steps = 1000
        sim_time = steps * 0.005  # 0.005s per step (200 Hz physics)

        print(f"\nRunning {steps} steps ({sim_time:.1f}s simulated time)...")
        start = time.time()

        for i in range(steps):
            ops.step()

        real_time = time.time() - start
        ratio = sim_time / real_time

        print(f"\n✓ Results:")
        print(f"  Simulated time: {sim_time:.2f}s")
        print(f"  Real time: {real_time:.2f}s")
        print(f"  Ratio: {ratio:.2f}x", end="")

        if ratio >= 1.0:
            print(f" ({'faster than real-time!' if ratio > 1.0 else 'real-time'})")
        else:
            print(f" (slower than real-time)")

        print(f"  Steps/sec: {steps/real_time:.1f}")

        # Cleanup
        ops.close()

        return ratio

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    """Run all benchmark tests"""
    print("="*70)
    print("SIM-TO-REAL RATIO BENCHMARK - ALL RENDER MODES")
    print("="*70)
    print()
    print("Testing 7 configurations:")
    print("  1. RL mode (no cameras, headless)")
    print("  2. Vision RL mode (10fps cameras, headless)")
    print("  3. Demo mode (30fps HD 720p cameras, headless)")
    print("  4. 2K Demo mode (30fps 2K 1080p cameras, headless)")
    print("  5. 4K Demo mode (30fps 4K 2160p cameras, headless)")
    print("  6. MuJoCo Demo mode (200Hz HD cameras, headless)")
    print("  7. RL mode with viewer (no cameras, viewer window)")
    print()

    results = {}

    # Test 1: RL mode (fastest - no cameras)
    results['rl_headless'] = test_render_mode(
        "RL Mode (No Cameras, Headless)",
        render_mode="rl_core",
        headless=True
    )

    # Test 2: Vision RL mode (medium - 10fps cameras)
    results['vision_rl_headless'] = test_render_mode(
        "Vision RL Mode (10fps Cameras, Headless)",
        render_mode="vision_rl",
        headless=True
    )

    # Test 3: Demo mode (slower - 30fps HD 720p cameras)
    results['demo_headless'] = test_render_mode(
        "Demo Mode (30fps HD 720p 1280x720, Headless)",
        render_mode="demo",
        headless=True
    )

    # Test 4: 2K Demo mode (slower - 30fps 2K 1080p cameras)
    results['2k_demo_headless'] = test_render_mode(
        "2K Demo Mode (30fps 2K 1080p 1920x1080, Headless)",
        render_mode="2k_demo",
        headless=True
    )

    # Test 5: 4K Demo mode (slowest - 30fps 4K 2160p cameras!)
    results['4k_demo_headless'] = test_render_mode(
        "4K Demo Mode (30fps 4K 2160p 3840x2160, Headless)",
        render_mode="4k_demo",
        headless=True
    )

    # Test 6: MuJoCo Demo mode (full rate HD cameras - 200Hz!)
    results['mujoco_demo_headless'] = test_render_mode(
        "MuJoCo Demo Mode (200Hz HD Cameras, Headless)",
        render_mode="mujoco_demo",
        headless=True
    )

    # Test 7: RL mode with viewer (viewer window overhead)
    results['rl_with_viewer'] = test_render_mode(
        "RL Mode (No Cameras, With Viewer)",
        render_mode="rl_core",
        headless=False
    )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - SIM-TO-REAL RATIOS")
    print('='*70)
    print()

    # Sort by speed (fastest first)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (mode, ratio) in enumerate(sorted_results, 1):
        status = "✓" if ratio >= 1.0 else "✗"
        real_time = "faster" if ratio > 1.0 else ("real-time" if ratio == 1.0 else "slower")

        # Format mode name
        mode_display = {
            'rl_headless': 'RL (headless)',
            'vision_rl_headless': 'Vision RL (headless)',
            'demo_headless': 'Demo HD 720p (headless)',
            '2k_demo_headless': '2K 1080p (headless)',
            '4k_demo_headless': '4K 2160p (headless)',
            'mujoco_demo_headless': 'MuJoCo Demo (headless)',
            'rl_with_viewer': 'RL (with viewer)'
        }.get(mode, mode)

        print(f"  {i}. {mode_display:25s}: {ratio:5.2f}x  {status} ({real_time})")

    print()
    print("Key Insights:")

    # Only show comparisons if both values are valid (avoid division by zero)
    if results.get('rl_with_viewer', 0) > 0 and results.get('rl_headless', 0) > 0:
        speedup = results['rl_headless'] / results['rl_with_viewer']
        print(f"  • Headless gives {speedup:.1f}x speedup (no viewer overhead)")
    elif results.get('rl_with_viewer', 0) == 0:
        print(f"  • Viewer mode test failed (GLFW/EGL conflict with cameras)")

    if results.get('vision_rl_headless', 0) > 0 and results.get('rl_headless', 0) > 0:
        slowdown = results['rl_headless'] / results['vision_rl_headless']
        print(f"  • Cameras cost: RL→Vision RL = {slowdown:.1f}x slower")

    if results.get('demo_headless', 0) > 0 and results.get('vision_rl_headless', 0) > 0:
        slowdown = results['vision_rl_headless'] / results['demo_headless']
        print(f"  • HD cameras: Vision RL→Demo = {slowdown:.1f}x slower")

    if results.get('mujoco_demo_headless', 0) > 0 and results.get('demo_headless', 0) > 0:
        slowdown = results['demo_headless'] / results['mujoco_demo_headless']
        print(f"  • Full-rate: Demo→MuJoCo Demo = {slowdown:.1f}x slower")

    print()
    print("Recommendation:")
    if results.get('mujoco_demo_headless', 0) >= 1.0:
        print("  ✓ MuJoCo Demo mode is fast enough for real-time streaming at 200Hz!")
    elif results.get('demo_headless', 0) >= 1.0:
        print("  ✓ Demo mode is fast enough for real-time streaming!")
    elif results.get('vision_rl_headless', 0) >= 1.0:
        print("  ✓ Vision RL mode is fast enough for real-time streaming!")
    else:
        print("  ⚠ Use vision_rl for real-time, demo for offline")

    print()
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

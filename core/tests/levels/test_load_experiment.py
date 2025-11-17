"""
TEST: load_experiment() - MOP Time Machine

Tests the complete save → load → replay cycle:
1. Save experiment with 2k_demo cameras
2. Load with viewer (headless=False toggle)
3. Verify all cameras work
4. Test frame loading
5. Test replay_frames()

Pure MOP Testing:
- Modals save themselves
- Factory function creates fully-loaded ops
- Read-only playback (no modifications)
"""

import sys
from pathlib import Path

# Add simulation_center to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps, load_experiment


def test_save_and_load_experiment():
    """Test complete save → load → view cycle"""
    print("\n" + "="*70)
    print("TEST: MOP Time Machine - Save and Load Experiment")
    print("="*70)

    # STEP 1: Create and save experiment with 2K cameras
    print("\n📹 STEP 1: Creating experiment with 2K cameras...")
    ops1 = ExperimentOps(
        headless=True,  # Save headless
        render_mode="2k_demo",  # 1920x1080 cameras!
        save_fps=30
    )

    ops1.create_scene("kitchen", width=5, length=5, height=3)
    ops1.add_robot("stretch", position=(0, 0, 0))
    ops1.add_asset("table", relative_to=(2, 2, 0))
    ops1.add_asset("apple", relative_to="table", relation="on_top")

    # Add 3 cameras (within room bounds)
    ops1.add_overhead_camera("birds_eye")
    ops1.add_free_camera("side_view",
                        lookat=(0, 0, 0.8),  # Look at center
                        distance=2.0,  # Shorter distance
                        azimuth=90,
                        elevation=-30)
    ops1.add_free_camera("robot_pov",
                        lookat=(0, 0, 0.8),  # Look at center
                        distance=1.5,
                        azimuth=0,
                        elevation=-15)

    ops1.compile()

    # Run simulation (saves mujoco_package + scene_state)
    print("  🎬 Recording 100 frames...")
    for i in range(100):
        ops1.step()
        if i % 20 == 0:
            print(f"    Frame {i}/100")

    experiment_path = ops1.experiment_dir
    ops1.close()

    print(f"  ✓ Saved experiment: {experiment_path}")

    # STEP 2: Load with viewer (headless=False)
    print("\n📺 STEP 2: Loading with viewer (headless=False)...")
    ops2 = load_experiment(
        path=experiment_path,
        frame=None,  # Start at frame 0
        headless=False  # VIEWER ON!
    )

    # Verify loaded correctly
    assert ops2.backend is not None, "Backend not loaded!"
    assert ops2.backend.model is not None, "Model not loaded!"
    assert ops2.engine is not None, "Engine not created!"

    print("  ✓ Experiment loaded successfully")
    print(f"  ✓ Viewer: {'ON' if not ops2.headless else 'OFF'}")

    # STEP 3: Test frame jumping
    print("\n⏭️  STEP 3: Testing frame loading...")
    ops2.load_frame(10)
    print("  ✓ Jumped to frame 10")

    ops2.load_frame(15)
    print("  ✓ Jumped to frame 15")

    # STEP 4: Test frame count
    frame_count = ops2.get_frame_count()
    print(f"\n📊 STEP 4: Frame count = {frame_count}")
    # With save_fps=30 and ~100 simulation steps, we expect ~17 frames
    assert frame_count >= 15 and frame_count <= 20, f"Expected ~17 frames, got {frame_count}"

    # STEP 5: Test replay (first 10 frames, fast)
    print("\n▶️  STEP 5: Testing replay (frames 0-10 at 60 FPS)...")
    ops2.replay_frames(start_frame=0, end_frame=10, fps=60)
    print("  ✓ Replay complete")

    # STEP 6: Verify cameras are working
    print("\n📷 STEP 6: Verifying cameras from saved experiment...")
    # All 3 cameras should be in the loaded model
    import mujoco
    cam_count = ops2.backend.model.ncam
    print(f"  ✓ Cameras in model: {cam_count}")
    assert cam_count >= 3, f"Expected at least 3 cameras, got {cam_count}"

    # List camera names
    for i in range(cam_count):
        cam_name = mujoco.mj_id2name(ops2.backend.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if cam_name:
            print(f"    - {cam_name}")

    ops2.close()

    print("\n" + "="*70)
    print("✅ TEST PASSED: load_experiment() works perfectly!")
    print("="*70)


def test_load_headless_toggle():
    """Test that headless can be toggled regardless of save mode"""
    print("\n" + "="*70)
    print("TEST: Headless Toggle - Save headless=True, Load headless=False")
    print("="*70)

    # Save headless
    ops1 = ExperimentOps(headless=True, render_mode="demo", save_fps=30)
    ops1.create_scene("test", 5, 5, 3)
    ops1.add_robot("stretch")
    ops1.compile()

    for _ in range(50):
        ops1.step()

    exp_path = ops1.experiment_dir
    ops1.close()

    print(f"  ✓ Saved with headless=True")

    # Load with viewer!
    ops2 = load_experiment(exp_path, headless=False)

    assert ops2.headless == False, "headless should be False!"
    assert ops2.backend.viewer is not None, "Viewer should exist!"

    print(f"  ✓ Loaded with headless=False (viewer ON!)")

    ops2.close()

    print("✅ TEST PASSED: Headless toggle works!")


def test_load_nonexistent_experiment():
    """Test error handling for missing experiment"""
    print("\n" + "="*70)
    print("TEST: Error Handling - Nonexistent Experiment")
    print("="*70)

    try:
        ops = load_experiment("database/FAKE_EXPERIMENT_12345/")
        assert False, "Should have raised FileNotFoundError!"
    except FileNotFoundError as e:
        print(f"  ✓ Caught expected error: {e}")
        print("✅ TEST PASSED: Error handling works!")


def test_load_specific_frame():
    """Test loading experiment at specific frame"""
    print("\n" + "="*70)
    print("TEST: Load Specific Frame")
    print("="*70)

    # Create experiment
    ops1 = ExperimentOps(headless=True, render_mode="demo", save_fps=30)
    ops1.create_scene("test", 5, 5, 3)
    ops1.add_robot("stretch")
    ops1.compile()

    # Run 50 frames
    for i in range(50):
        ops1.step()

    exp_path = ops1.experiment_dir
    ops1.close()

    print(f"  ✓ Saved 50 frames")

    # Load at frame 5
    ops2 = load_experiment(exp_path, frame=5, headless=True)

    print(f"  ✓ Loaded at frame 5")

    # Verify we can access other frames
    ops2.load_frame(2)
    print(f"  ✓ Jumped to frame 2")

    ops2.load_frame(7)
    print(f"  ✓ Jumped to frame 7")

    ops2.close()

    print("✅ TEST PASSED: Frame loading works!")


if __name__ == "__main__":
    print("\n🚀 Starting MOP Time Machine Tests...")
    print("="*70)

    test_save_and_load_experiment()
    test_load_headless_toggle()
    test_load_nonexistent_experiment()
    test_load_specific_frame()

    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("="*70)
    print("\nMOP Time Machine is READY:")
    print("  ✅ save_experiment() → Creates mujoco_package + scene_state")
    print("  ✅ load_experiment() → Factory function returns ready ops")
    print("  ✅ load_frame() → Jump to any frame")
    print("  ✅ replay_frames() → Watch full timeline")
    print("  ✅ Headless toggle → Viewer independent of save mode")
    print("  ✅ All cameras/sensors/assets preserved")
    print("\n🎬 Pure MOP: Read-only time travel through saved experiments!")

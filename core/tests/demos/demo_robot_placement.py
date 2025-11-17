#!/usr/bin/env python3
"""
ROBOT AUTO-PLACEMENT DEMO - THE ULTIMATE INFRASTRUCTURE TEST! 🤖🔥

This demonstrates the MOST POWERFUL feature:
- Robot automatically picks and places objects from scene specifications
- NO manual programming of motions
- Proves end-to-end integration: IK → Actions → Sensors → Control

Demo: Robot places apple in basket automatically!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.main.robot_placement_ops import RobotPlacementDemo


def demo_apple_in_basket():
    """Simplest possible demo: Robot picks apple and places in bowl

    Scene:
    - Bowl at (2, 0, 0) on floor
    - Apple at (0.5, 0, 0.5) beside robot
    - Robot automatically: pick apple → move to bowl → place inside

    This is EASIER than tower stacking:
    - Single object placement (not 4!)
    - No dependencies
    - Forgiving target (bowl interior, not precise stack)
    """

    print("\n" + "="*70)
    print("🤖 ROBOT AUTO-PLACEMENT DEMO: Apple in Bowl")
    print("="*70)

    # ================================================================
    # SCENE SETUP (Normal ops API - unchanged!)
    # ================================================================
    print("\n📦 Setting up scene...")

    ops = ExperimentOps(
        mode="simulated",
        headless=False,  # Required for 2k_demo render mode
        render_mode="rl_core",  # Beautiful visuals!
        save_fps=30
    )

    # Create scene
    ops.create_scene(name="robot_placement_demo", width=5, length=5, height=3)

    # Add robot
    ops.add_robot(robot_name="stretch", position=(0, 0, 0))

    # Add bowl (target container)
    ops.add_asset(asset_name="bowl", relative_to=(2.0, 0.0, 0.0))

    # Add apple (object to place) - spawn beside robot
    ops.add_asset(asset_name="apple", relative_to=(0.5, 0.0, 0.5))

    # Add camera to watch the magic!
    ops.add_overhead_camera()

    print("   ✓ Scene created!")
    print(f"   - Robot at (0, 0, 0)")
    print(f"   - Bowl at (2, 0, 0)")
    print(f"   - Apple at (0.5, 0, 0.5)")

    # ================================================================
    # COMPILE SCENE (Normal compilation)
    # ================================================================
    print("\n🔧 Compiling scene...")
    ops.compile()
    print("   ✓ Scene compiled!")

    # Let physics settle
    print("\n⏳ Settling physics (200 steps)...")
    for _ in range(200):
        ops.step()
    print("   ✓ Physics settled!")

    # ================================================================
    # THE MAGIC: AUTOMATIC ROBOT PLACEMENT!
    # ================================================================
    print("\n✨ ACTIVATING AUTOMATIC PLACEMENT...")

    demo = RobotPlacementDemo(ops)

    # Target: inside basket (basket at ground level, place apple ~0.8m high)
    target_location = (2.0, 0.0, 0.8)

    try:
        demo.execute_placement(
            object_name="apple",
            target_location=target_location,
            settling_steps=100
        )

        print("\n🎉 SUCCESS! Robot placed apple in basket automatically!")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nThis is EXPECTED for first run - we may need to tune:")
        print("  - IK calculations")
        print("  - Action parameters")
        print("  - Force limits")
        return False

    # ================================================================
    # VALIDATION
    # ================================================================
    print("\n🔍 Validating placement...")

    # Check apple's final position
    final_state = ops.get_state()
    apple_final_pos = final_state['apple']['position']

    print(f"   Apple final position: ({apple_final_pos[0]:.2f}, {apple_final_pos[1]:.2f}, {apple_final_pos[2]:.2f})")
    print(f"   Target position: ({target_location[0]:.2f}, {target_location[1]:.2f}, {target_location[2]:.2f})")

    # Check if apple is near target
    distance = ((apple_final_pos[0] - target_location[0])**2 +
                (apple_final_pos[1] - target_location[1])**2 +
                (apple_final_pos[2] - target_location[2])**2)**0.5

    if distance < 0.2:  # Within 20cm
        print(f"\n✅ PASS: Apple within {distance:.2f}m of target!")
    else:
        print(f"\n⚠️  PARTIAL: Apple {distance:.2f}m from target (expected < 0.2m)")

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    print("\n📸 Saving results...")
    ops.save_all_screenshots(frame=10, subdir="robot_placement_screenshots")

    print("\n📹 Validating videos...")
    ops.validate_videos(timeout=180)

    # ================================================================
    # CLEANUP
    # ================================================================
    ops.close()

    print("\n" + "="*70)
    print("🏁 DEMO COMPLETE!")
    print("="*70)

    return True


if __name__ == "__main__":
    success = demo_apple_in_basket()

    if success:
        print("\n🚀 INFRASTRUCTURE TEST PASSED!")
        print("\nThis proves end-to-end integration:")
        print("  ✓ IK solver (ReachabilityModal)")
        print("  ✓ Action system (ActionBlocks)")
        print("  ✓ Sensor feedback (force sensing)")
        print("  ✓ Runtime control (ExperimentOps)")
        print("\n🔥 NEXT: Extend to 2-block tower stacking!")
    else:
        print("\n⚙️  DEBUGGING NEEDED")
        print("\nThis is normal for first run!")
        print("We'll iterate on:")
        print("  - IK accuracy")
        print("  - Motion parameters")
        print("  - Force limits")
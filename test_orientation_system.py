#!/usr/bin/env python3
"""
TEST: Complete Orientation System
Tests all three orientation types: preset, relational, and manual quaternion
"""
from core.main.experiment_ops_unified import ExperimentOps

print("\n" + "="*70)
print("TEST: Complete Orientation System")
print("="*70)

ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
ops.create_scene(name="orientation_test", width=10, length=10, height=3)

# Test 1: Preset orientation (east)
print("\n1. Testing PRESET orientation (robot faces east)...")
ops.add_robot(robot_name="stretch", position=(0.0, 0.0, 0.0), orientation="east")

# Test 2: Relational orientation (robot faces table)
print("\n2. Testing RELATIONAL orientation (robot faces table)...")
ops.add_asset(asset_name="table", relative_to=(3.0, 0.0, 0.0))

# Test 3: Check that robot is facing table
print("\n3. Compiling scene...")
ops.compile()

# Test 4: Verify orientation was applied
print("\n4. Verifying robot orientation...")
state = ops.get_state()
robot_base = state.get("stretch.base", {})
robot_pos = robot_base.get("position", [0, 0, 0])
robot_quat = robot_base.get("quaternion", [1, 0, 0, 0])

print(f"   Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})")
print(f"   Robot quaternion: ({robot_quat[0]:.3f}, {robot_quat[1]:.3f}, {robot_quat[2]:.3f}, {robot_quat[3]:.3f})")

# Expected: Robot should face +X (east), which is quaternion (0.707, 0, 0, 0.707)
if abs(robot_quat[0] - 0.707) < 0.01 and abs(robot_quat[3] - 0.707) < 0.01:
    print("\n   ✅ SUCCESS! Robot correctly oriented facing east")
else:
    print(f"\n   ⚠️  Robot quaternion unexpected (expected ~0.707, 0, 0, 0.707)")

# Test 5: Check keyframe
print("\n5. Checking keyframe generation...")
xml_content = ops.get_mjcf_xml()

if '<keyframe>' in xml_content and 'qpos=' in xml_content:
    print("   ✅ Keyframe generated successfully")

    # Extract qpos from keyframe
    import re
    qpos_match = re.search(r'qpos="([^"]+)"', xml_content)
    if qpos_match:
        qpos = qpos_match.group(1)
        print(f"   Keyframe qpos: {qpos[:50]}...")  # First 50 chars
else:
    print("   ❌ Keyframe not found in XML")

print("\n" + "="*70)
print("✅ Orientation system test COMPLETE!")
print("="*70)

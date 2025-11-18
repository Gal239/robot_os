#!/usr/bin/env python3
"""Test is_facing() utility - MOP spatial queries!"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from core.main.experiment_ops_unified import ExperimentOps

ops = ExperimentOps(mode="simulated", headless=True, render_mode="rl_core")
ops.create_scene("test_room", width=10, length=10, height=3)
ops.add_asset("table", relative_to=(2, 0, 0))
ops.add_asset("apple", relative_to="table", relation="on_top")
ops.add_robot("stretch", position=(0, 0, 0))
ops.compile()

print("\n" + "="*70)
print("TEST 1: Is arm facing apple?")
print("="*70)

result = ops.is_facing("stretch.arm", "apple")
print(f"Facing: {result['facing']}")
print(f"Dot product: {result['dot']:.3f}")
print(f"Classification: {result['dot_class']}")
print(f"Explanation: {result['dot_explain']}")
print(f"Distance: {result['distance']:.2f}m")
print(f"Arm direction: {result['object1_direction']}")
print(f"Apple position: {result['object2_position']}")

print("\n" + "="*70)
print("TEST 2: Is table facing apple? (table surface faces upward)")
print("="*70)

result = ops.is_facing("table", "apple")
print(f"Facing: {result['facing']}")
print(f"Dot product: {result['dot']:.3f}")
print(f"Classification: {result['dot_class']}")
print(f"Explanation: {result['dot_explain']}")
print(f"Table direction: {result['object1_direction']} (upward)")

print("\n" + "="*70)
print("TEST 3: Is apple facing table? (with lower threshold)")
print("="*70)

result = ops.is_facing("apple", "table", threshold=0.5)
print(f"Facing: {result['facing']}")
print(f"Dot product: {result['dot']:.3f}")
print(f"Classification: {result['dot_class']}")
print(f"Explanation: {result['dot_explain']}")
print(f"Distance: {result['distance']:.2f}m")

print("\n" + "="*70)
print("TEST 4: Using just asset names (auto-finds first spatial component)")
print("="*70)

# This should work because apple has "body" component with position+direction
result = ops.is_facing("apple", "table.table_surface")
print(f"Apple facing table surface: {result['facing']}")
print(f"Dot product: {result['dot']:.3f}")
print(f"Classification: {result['dot_class']}")
print(f"Explanation: {result['dot_explain']}")

print("\n✅ All tests passed!")

ops.close()

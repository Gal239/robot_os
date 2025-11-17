"""
Run demo test with performance measurement using RTX 3090
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import the demo tests
from simulation_center.core.tests.demos.demo_1_ai_generated_scenes import test_1_tower_stacking

print("="*100)
print("RUNNING TOWER STACKING DEMO WITH RTX 3090")
print("="*100)
print("\nThis demo uses render_mode='2k_demo' with multiple high-res cameras")
print("Perfect test for RTX 3090 GPU performance!\n")

# Measure performance
start_time = time.perf_counter()

# Run the test
result = test_1_tower_stacking()

end_time = time.perf_counter()
elapsed = end_time - start_time

print("\n" + "="*100)
print("PERFORMANCE RESULTS")
print("="*100)
print(f"Total time: {elapsed:.2f}s")
print(f"Test result: {'✅ PASS' if result else '❌ FAIL'}")
print("\nGPU: NVIDIA RTX 3090")
print("Render mode: 2k_demo (high resolution)")
print("Cameras: Multiple tracking cameras with 60fps recording")
print("="*100)

"""
Test MuJoCo performance with RTX 3090 GPU
Test vision_rl mode to verify we reach target 1.8-2.2x performance
"""
import subprocess
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent.parent.parent

def test_vision_rl_performance():
    """Test vision_rl with RTX 3090"""
    script = f"""
import sys
import os
import time
from pathlib import Path

# Force EGL backend for headless GPU rendering
os.environ['MUJOCO_VIEWER_MODE'] = 'false'

project_root = Path('{project_root}')
sys.path.insert(0, str(project_root))

from simulation_center.core.main.experiment_ops_unified import ExperimentOps

# Create vision_rl experiment
print("Creating vision_rl experiment with RTX 3090...")
ops = ExperimentOps(headless=True, render_mode='vision_rl')
ops.create_scene('test', width=5, length=5, height=3)
ops.add_robot('stretch', position=(0, 0, 0))
ops.compile()

# Get GPU info
import mujoco
from OpenGL import GL

model = ops.backend.model
renderer = mujoco.Renderer(model, height=100, width=100)

vendor = GL.glGetString(GL.GL_VENDOR)
renderer_name = GL.glGetString(GL.GL_RENDERER)

if vendor:
    vendor = vendor.decode('utf-8') if isinstance(vendor, bytes) else vendor
if renderer_name:
    renderer_name = renderer_name.decode('utf-8') if isinstance(renderer_name, bytes) else renderer_name

print(f"GPU: {{vendor}} | {{renderer_name}}")
renderer.close()

# Performance test
num_steps = 1000
print(f"Running {{num_steps}} steps...")

t_start = time.perf_counter()
for i in range(num_steps):
    ops.step()
    if (i + 1) % 200 == 0:
        print(f"  Progress: {{i+1}}/{{num_steps}}")

t_end = time.perf_counter()
elapsed = t_end - t_start
sim_time = num_steps * 0.005
real_time_factor = sim_time / elapsed

ops.close()

print(f"\\nRESULT###{{renderer_name}}###{{real_time_factor:.3f}}###{{elapsed:.2f}}")
"""

    print("="*100)
    print("TESTING vision_rl PERFORMANCE WITH RTX 3090")
    print("="*100)
    print()

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(project_root)
    )

    # Print output
    for line in result.stdout.split('\n'):
        if line.strip() and not line.startswith('RESULT###'):
            print(line)

    # Parse result
    for line in result.stdout.split('\n'):
        if line.startswith('RESULT###'):
            parts = line.split('###')
            gpu = parts[1]
            rtf = float(parts[2])
            elapsed = float(parts[3])

            print()
            print("="*100)
            print("PERFORMANCE RESULTS")
            print("="*100)
            print(f"GPU: {gpu}")
            print(f"Elapsed: {elapsed:.2f}s")
            print(f"Real-time factor: {rtf:.2f}x")
            print()

            # Analysis
            if 'nvidia' in gpu.lower() or '3090' in gpu.lower():
                print("✅ Using NVIDIA RTX 3090 GPU!")
            else:
                print(f"⚠️  Not using NVIDIA GPU, using: {gpu}")

            if rtf >= 1.8:
                print(f"✅ EXCELLENT! Reached target performance: {rtf:.2f}x >= 1.8x")
            elif rtf >= 1.5:
                print(f"✅ GOOD! Close to target: {rtf:.2f}x (target: 1.8-2.2x)")
            else:
                print(f"⚠️  Below target: {rtf:.2f}x (target: 1.8-2.2x)")

            print()
            print("Comparison:")
            print(f"  Before (Intel GPU):  1.54x")
            print(f"  After (RTX 3090):    {rtf:.2f}x")
            if rtf > 1.54:
                improvement = ((rtf - 1.54) / 1.54) * 100
                print(f"  Improvement:         +{improvement:.1f}%")
            print("="*100)

            return rtf

    print("\n❌ Failed to parse results")
    if result.stderr:
        print("STDERR:")
        print(result.stderr[-500:])
    return None

if __name__ == '__main__':
    test_vision_rl_performance()

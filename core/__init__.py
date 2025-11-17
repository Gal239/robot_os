"""
Simulation Center Core - Dual Import Support

This module enables imports to work BOTH ways:
- Internal: from core.modals import X (within simulation_center)
- External: from simulation_center.core.modals import X (from rl_center)

MOP-compliant: Zero boilerplate, smart fallback.
"""

# ============================================================================
# MuJoCo's rendering backend MUST be set BEFORE MuJoCo is imported!
# ============================================================================
import os
import sys

if 'MUJOCO_GL' not in os.environ:
    # ALWAYS USE EGL (fastest for GPU rendering, works headless)
    os.environ['MUJOCO_GL'] = 'egl'

    # Detect GPU type
    import subprocess
    gpu_type = "CPU (llvmpipe)"

    # Check NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=1)
        if result.returncode == 0:
            gpu_type = "NVIDIA GPU"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Intel/AMD
    if gpu_type == "CPU (llvmpipe)":
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=1)
            if 'VGA' in result.stdout or 'Display' in result.stdout:
                if 'Intel' in result.stdout:
                    gpu_type = "Intel GPU"
                elif 'AMD' in result.stdout or 'Radeon' in result.stdout:
                    gpu_type = "AMD GPU"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    print(f"🎮 MuJoCo Rendering: EGL with {gpu_type}")
else:
    # User explicitly set MUJOCO_GL
    print(f"🎮 MuJoCo Rendering: User-specified ({os.environ['MUJOCO_GL'].upper()})")

# NOW safe to import other modules
from pathlib import Path

# Ensure core modules can be imported both ways
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Re-export main API for external use
try:
    from .main.robot_ops import create_robot
    from .main.experiment_ops_unified import ExperimentOps
except ImportError:
    # Fallback for internal use
    pass
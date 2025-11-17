"""
LEVEL 1F: CINEMATIC SCENE COMPOSITION WITH VIDEO
=================================================

Comprehensive cinematic tests with multi-angle camera views and video recording:
- Real-world scenarios: kitchen, dining, workspace, storage, market, etc.
- Multi-camera cinematography with camera tours
- High-quality 30 FPS video recording (demo mode)
- Multi-layer validation: Physics + Semantics + Vision + Reasoning + Video Quality

Each test creates a 10-second cinematic video showing the scene from multiple angles!

12 Tests Total - Cinematic Scene Composition! 🎬

Run with: PYTHONPATH=$PWD python3 core/tests/levels/level_1f_scene_operations.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import json
import numpy as np
import cv2
from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.action_modals import ArmMoveTo, LiftMoveTo, ActionBlock
from core.modals.stretch.action_modals import (
    ArmMoveTo, LiftMoveTo, GripperMoveTo, HeadPanMoveTo,
    ActionBlock, SensorCondition
)
from core.modals.stretch.action_blocks_registry import spin

# Load tolerances from discovered_tolerances.json (PURE MOP - single source of truth)
TOLERANCE_PATH = Path(__file__).parent.parent.parent / "modals" / "stretch" / "discovered_tolerances.json"
with open(TOLERANCE_PATH) as f:
    TOLERANCES = json.load(f)


# ============================================================================
# TEST 1: KITCHEN BREAKFAST SCENE
# ============================================================================

def test_1_kitchen_breakfast_scene():
    ops = ExperimentOps(mode="simulated",headless=False,render_mode="rl_core", save_fps=30)
    ops.create_scene("breakfast_scene", width=8, length=8, height=10)
    ops.add_robot("stretch", position=(0, 0, 0))
    ops.add_asset("table", relative_to=(2.0, 0.0, 0.0))
    ops.add_asset("apple", relative_to="table", relation="on_top",distance=0.75, surface_position="center")
    ops.add_asset("banana", relative_to="table", relation="on_top",distance=0.75, surface_position="center")
    ops.add_asset("mug", relative_to="table", relation="on_top",distance=0.75, surface_position="center")
    ops.add_asset("bowl", relative_to="table", relation="front", distance=0.75,surface_position="center")
    ops.compile()
    block = spin(degrees=360, speed=6.0)  # 90 degrees left
    ops.submit_block(block)
    for step in range(2000):
        result = ops.step()
        continue


#if main
if __name__ == "__main__":
    test_1_kitchen_breakfast_scene()
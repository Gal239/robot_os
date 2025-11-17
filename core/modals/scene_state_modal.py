"""
SceneStateModal - Per-frame physics state snapshot

PURE MOP:
- SELF-CAPTURING: Captures qpos/qvel/ctrl/time from MuJoCo
- SELF-RENDERING: Converts to MuJoCo keyframe XML
- SELF-SAVING: Writes frame_XXXX.xml to disk

This modal enables:
1. Frame-by-frame playback in Python MuJoCo
2. Loading specific simulation states
3. Creating trajectory JSON for web viewers
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class SceneStateModal:
    """PURE MOP: Self-contained physics state that saves itself

    Usage:
        # Capture state during simulation
        state = SceneStateModal(model, data, frame_num=42)

        # Save to disk
        state.save(scene_state_dir)
        # Creates: scene_state/frame_0042.xml

        # Later: Load and apply
        state = SceneStateModal.load(scene_state_dir / "frame_0042.xml")
        state.apply(model, data)
    """

    frame_num: int  # Frame number (0-indexed)
    time: float  # Simulation time
    qpos: np.ndarray  # Joint positions
    qvel: np.ndarray  # Joint velocities
    ctrl: np.ndarray  # Control inputs
    act: Optional[np.ndarray] = None  # Activations (for muscles/tendons)

    @classmethod
    def from_mujoco(cls, model, data, frame_num: int) -> 'SceneStateModal':
        """SELF-CAPTURING: Capture current state from MuJoCo - PURE MOP!

        Args:
            model: MuJoCo model (mujoco.MjModel)
            data: MuJoCo data (mujoco.MjData)
            frame_num: Frame number for naming

        Returns:
            SceneStateModal instance with captured state
        """
        return cls(
            frame_num=frame_num,
            time=float(data.time),
            qpos=data.qpos.copy(),
            qvel=data.qvel.copy(),
            ctrl=data.ctrl.copy(),
            act=data.act.copy() if model.na > 0 else None
        )

    def to_keyframe_xml(self) -> str:
        """SELF-RENDERING: Convert to MuJoCo keyframe XML - PURE MOP!

        Returns:
            XML string with keyframe definition

        Example output:
            <keyframe>
              <key name="frame_0042"
                   time="1.4"
                   qpos="1.523 2.012 0.843 1.0 0.0 0.0 0.0 ..."
                   qvel="0.01 0.02 -0.03 0.0 0.0 0.0 ..."
                   ctrl="0.0 0.0 0.5 0.3 ..."/>
            </keyframe>
        """
        # Create keyframe element
        keyframe_elem = ET.Element('keyframe')

        # Create key element
        key_elem = ET.SubElement(keyframe_elem, 'key')
        key_elem.set('name', f'frame_{self.frame_num:04d}')
        key_elem.set('time', f'{self.time:.6f}')

        # Convert numpy arrays to space-separated strings
        qpos_str = ' '.join(f'{x:.6f}' for x in self.qpos)
        qvel_str = ' '.join(f'{x:.6f}' for x in self.qvel)
        ctrl_str = ' '.join(f'{x:.6f}' for x in self.ctrl)

        key_elem.set('qpos', qpos_str)
        key_elem.set('qvel', qvel_str)
        key_elem.set('ctrl', ctrl_str)

        # Add activations if present
        if self.act is not None and len(self.act) > 0:
            act_str = ' '.join(f'{x:.6f}' for x in self.act)
            key_elem.set('act', act_str)

        # Convert to string with pretty formatting
        ET.indent(keyframe_elem, space='  ')
        xml_str = ET.tostring(keyframe_elem, encoding='unicode')

        # Add XML declaration
        return f'<?xml version="1.0" encoding="utf-8"?>\n{xml_str}'

    def save(self, scene_state_dir: Path):
        """SELF-SAVING: Write frame_XXXX.xml to disk - PURE MOP!

        Args:
            scene_state_dir: Directory to save frame (e.g., database/{exp_id}/scene_state/)

        Creates:
            scene_state/frame_0042.xml
        """
        # Ensure directory exists
        scene_state_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f'frame_{self.frame_num:04d}.xml'
        filepath = scene_state_dir / filename

        # Write XML
        xml_content = self.to_keyframe_xml()
        with open(filepath, 'w') as f:
            f.write(xml_content)

        # Quiet mode - no print (called at 30 FPS!)

    @staticmethod
    def load(filepath: Path) -> 'SceneStateModal':
        """Load state from frame_XXXX.xml

        Args:
            filepath: Path to frame XML file

        Returns:
            SceneStateModal instance
        """
        # Parse XML
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Find key element
        key_elem = root.find('.//key')
        if key_elem is None:
            raise ValueError(f"No <key> element found in {filepath}")

        # Extract attributes
        name = key_elem.get('name', 'unknown')
        time = float(key_elem.get('time', '0.0'))

        # Parse arrays
        qpos_str = key_elem.get('qpos', '')
        qvel_str = key_elem.get('qvel', '')
        ctrl_str = key_elem.get('ctrl', '')
        act_str = key_elem.get('act', '')

        qpos = np.array([float(x) for x in qpos_str.split()])
        qvel = np.array([float(x) for x in qvel_str.split()])
        ctrl = np.array([float(x) for x in ctrl_str.split()])
        act = np.array([float(x) for x in act_str.split()]) if act_str else None

        # Extract frame number from name (e.g., "frame_0042" → 42)
        frame_num = int(name.split('_')[1]) if '_' in name else 0

        return SceneStateModal(
            frame_num=frame_num,
            time=time,
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            act=act
        )

    def apply(self, model, data):
        """Apply this state to MuJoCo model/data

        Args:
            model: MuJoCo model (mujoco.MjModel)
            data: MuJoCo data (mujoco.MjData)
        """
        import mujoco

        # Set state
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.ctrl[:] = self.ctrl
        data.time = self.time

        if self.act is not None and model.na > 0:
            data.act[:] = self.act

        # Forward kinematics to update dependent quantities
        mujoco.mj_forward(model, data)

    def to_dict(self) -> dict:
        """Convert to dictionary (for trajectory JSON)

        Returns:
            Dict with time, qpos, qvel, ctrl
        """
        result = {
            'frame': self.frame_num,
            'time': float(self.time),
            'qpos': self.qpos.tolist(),
            'qvel': self.qvel.tolist(),
            'ctrl': self.ctrl.tolist()
        }

        if self.act is not None:
            result['act'] = self.act.tolist()

        return result

    @staticmethod
    def create_trajectory_json(scene_state_dir: Path, output_path: Optional[Path] = None) -> dict:
        """Create trajectory JSON from all frame XMLs - for mjc_viewer

        Args:
            scene_state_dir: Directory with frame_XXXX.xml files
            output_path: Optional path to save trajectory.json

        Returns:
            Dict with trajectory data

        Example output:
            {
              "frames": [
                {"frame": 0, "time": 0.0, "qpos": [...], "qvel": [...], "ctrl": [...]},
                {"frame": 1, "time": 0.033, "qpos": [...], "qvel": [...], "ctrl": [...]}
              ],
              "fps": 30,
              "total_frames": 1800,
              "duration": 60.0
            }
        """
        # Find all frame XMLs
        frame_files = sorted(scene_state_dir.glob('frame_*.xml'))

        if not frame_files:
            return {
                "frames": [],
                "fps": 0,
                "total_frames": 0,
                "duration": 0.0
            }

        # Load all frames
        frames = []
        for frame_file in frame_files:
            state = SceneStateModal.load(frame_file)
            frames.append(state.to_dict())

        # Compute stats
        total_frames = len(frames)
        duration = frames[-1]['time'] if frames else 0.0
        fps = total_frames / duration if duration > 0 else 0

        trajectory = {
            "frames": frames,
            "fps": float(fps),
            "total_frames": total_frames,
            "duration": float(duration)
        }

        # Save if output path provided
        if output_path:
            import json
            with open(output_path, 'w') as f:
                json.dump(trajectory, f, indent=2)
            print(f"  💾 Saved trajectory.json ({total_frames} frames, {duration:.2f}s)")

        return trajectory
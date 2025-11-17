"""
MUJOCO MODAL - MuJoCo simulation state as a MODAL!
PURE MOP: Self-saving, self-loading, self-executing

Pattern: MuJoCo IS A MODAL, not just a backend!
- OWNS: model, data, XML, state
- DOES: compile, step, save, load

This is the SINGLE SOURCE OF TRUTH for all MuJoCo state.
Backend delegates to this modal.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class MuJoCoModal:
    """MuJoCo simulation state - PURE MOP!

    OWNS:
    - model (MjModel) - the physics model
    - data (MjData) - the simulation state
    - compiled_xml (str) - the XML that created the model
    - Viewer - the visualization window

    DOES:
    - compile_xml() - create model/data from XML
    - step() - advance physics simulation
    - to_xml() - I know my XML!
    - to_state() - I know my state!
    - from_xml() - I can load from XML!
    - from_state() - I can restore state!
    - to_json/from_json - full serialization

    OFFENSIVE: Crashes loudly if used incorrectly!
    """

    # MuJoCo objects (runtime only, not serialized)
    model: Optional[Any] = None  # mujoco.MjModel
    data: Optional[Any] = None   # mujoco.MjData
    viewer: Optional[Any] = None # mujoco.viewer

    # Serializable state
    compiled_xml: Optional[str] = None
    state_dict: Optional[Dict[str, Any]] = None

    # Configuration
    enable_viewer: bool = False
    is_headless: bool = True

    def compile_xml(self, xml: str):
        """Compile XML to MuJoCo model - OFFENSIVE!

        Args:
            xml: Complete MuJoCo XML string

        CRASHES if XML is invalid!
        """
        import mujoco

        # Create model and data
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.compiled_xml = xml

        # Launch viewer if requested
        if self.enable_viewer and not self.is_headless and not self.viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            # Disable LIDAR visualization (annoying yellow lines!)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    def step(self):
        """Step physics forward one timestep - OFFENSIVE!

        CRASHES if model/data not initialized!
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first!"

        import mujoco
        mujoco.mj_step(self.model, self.data)

        # Sync viewer if active
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()

    def reset_to_keyframe(self, keyframe_name: str):
        """Reset to saved keyframe - OFFENSIVE & FAST!

        10-30x faster than recompiling!

        Args:
            keyframe_name: Name of keyframe (e.g., 'initial')

        CRASHES if keyframe not found!
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first!"

        import mujoco

        # Find keyframe
        keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)

        if keyframe_id < 0:
            available = self.list_keyframes()
            raise ValueError(f"Keyframe '{keyframe_name}' not found! Available: {available}")

        # Reset (FAST!)
        mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)

    def list_keyframes(self) -> list:
        """List all available keyframes - OFFENSIVE

        Returns:
            List of keyframe names

        CRASHES if model not initialized!
        """
        assert self.model is not None, "Call compile_xml() first!"

        import mujoco

        keyframes = []
        for i in range(self.model.nkey):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name:
                keyframes.append(name)

        return keyframes

    # === MODAL INTERFACE - I KNOW HOW TO SAVE/LOAD MYSELF! ===

    def to_xml(self) -> str:
        """I know my XML - OFFENSIVE!

        Returns the compiled MuJoCo XML string.
        CRASHES if model not compiled!
        """
        assert self.model is not None, "Call compile_xml() first!"

        import mujoco
        import tempfile
        import os

        # Save to temp file then read back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            temp_path = f.name

        try:
            mujoco.mj_saveLastXML(temp_path, self.model)
            with open(temp_path, 'r') as f:
                xml = f.read()
            return xml
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def to_state(self) -> dict:
        """I know my state - OFFENSIVE!

        Returns complete MuJoCo state as JSON-serializable dict.
        CRASHES if model/data not ready!
        """
        assert self.model is not None and self.data is not None, "Call compile_xml() first!"

        return {
            "qpos": self.data.qpos.tolist(),
            "qvel": self.data.qvel.tolist(),
            "ctrl": self.data.ctrl.tolist(),
            "time": float(self.data.time),
        }

    def from_xml(self, xml: str):
        """I can load from XML - OFFENSIVE!

        Compiles model from saved XML string.
        FASTER than regenerating XML from scene!

        Args:
            xml: Complete MuJoCo XML string
        """
        self.compile_xml(xml)

    def from_state(self, state: dict):
        """I can restore my state - OFFENSIVE!

        Restores MuJoCo state from saved dict.
        CRASHES if model/data not ready!

        Args:
            state: Dict with qpos, qvel, ctrl, time
        """
        assert self.model is not None and self.data is not None, "Call from_xml() or compile_xml() first!"

        import mujoco

        # Restore state arrays
        self.data.qpos[:] = np.array(state["qpos"])
        self.data.qvel[:] = np.array(state["qvel"])
        self.data.ctrl[:] = np.array(state["ctrl"])
        self.data.time = state["time"]

        # Forward kinematics to sync positions
        mujoco.mj_forward(self.model, self.data)

    def to_json(self) -> dict:
        """I know how to serialize myself - PURE MOP!

        Returns complete MuJoCo state as JSON-serializable dict.
        """
        return {
            "compiled_xml": self.to_xml() if self.model else None,
            "state": self.to_state() if self.data else None,
            "enable_viewer": self.enable_viewer,
            "is_headless": self.is_headless,
        }

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself - PURE MOP!

        Creates MuJoCoModal from saved JSON data.

        Args:
            data: Dict from to_json()

        Returns:
            MuJoCoModal with restored state
        """
        modal = cls(
            enable_viewer=data["enable_viewer"],  # OFFENSIVE - crash if missing!
            is_headless=data["is_headless"]  # OFFENSIVE - crash if missing!
        )

        # Load compiled XML if available
        if data.get("compiled_xml"):
            modal.from_xml(data["compiled_xml"])

            # Restore state if available
            if data.get("state"):
                modal.from_state(data["state"])

        return modal

    def cleanup(self):
        """Close viewer and cleanup - OFFENSIVE"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# === USAGE EXAMPLE ===

if __name__ == "__main__":
    print("=" * 80)
    print("MUJOCO MODAL - Pure MOP Example")
    print("=" * 80)

    # Simple XML
    xml = """
    <mujoco>
        <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
            <body pos="0 0 1">
                <joint type="free"/>
                <geom type="box" size=".1 .1 .1" rgba="0 .9 0 1"/>
            </body>
        </worldbody>
    </mujoco>
    """

    # Create modal
    print("\n1️⃣  Creating MuJoCoModal...")
    modal = MuJoCoModal(is_headless=True)
    print("   ✓ Modal created")

    # Compile
    print("\n2️⃣  Compiling XML...")
    modal.compile_xml(xml)
    print(f"   ✓ Compiled: {modal.model.nq} DOFs")

    # Step physics
    print("\n3️⃣  Stepping physics...")
    for i in range(10):
        modal.step()
    print(f"   ✓ Stepped 10 times, time: {modal.data.time:.3f}s")

    # Save state
    print("\n4️⃣  Saving state...")
    saved_xml = modal.to_xml()
    saved_state = modal.to_state()
    print(f"   ✓ XML: {len(saved_xml)} chars")
    print(f"   ✓ State: {len(saved_state['qpos'])} qpos values")

    # Step more
    print("\n5️⃣  Stepping more...")
    for i in range(10):
        modal.step()
    print(f"   ✓ Time now: {modal.data.time:.3f}s")

    # Restore state
    print("\n6️⃣  Restoring state...")
    modal.from_state(saved_state)
    print(f"   ✓ Time restored: {modal.data.time:.3f}s")

    # Full serialization
    print("\n7️⃣  Full serialization...")
    json_data = modal.to_json()
    print(f"   ✓ JSON keys: {list(json_data.keys())}")

    # Deserialize
    print("\n8️⃣  Deserializing...")
    modal2 = MuJoCoModal.from_json(json_data)
    print(f"   ✓ Loaded: {modal2.model.nq} DOFs")
    print(f"   ✓ Time: {modal2.data.time:.3f}s")

    print("\n✅ MUJOCO MODAL WORKING PERFECTLY!")
    print("   PURE MOP - Modal saves/loads itself!")

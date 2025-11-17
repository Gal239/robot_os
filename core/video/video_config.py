"""
VIDEO CONFIGURATION - Single Source of Truth for Video Settings
================================================================

MOP: Centralized video configuration eliminates duplication!
Previously scattered across ExperimentOps init and compile methods.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoConfig:
    """Video configuration for simulation recording - OFFENSIVE!

    Single source of truth for all video parameters.
    Replaces scattered configuration in ExperimentOps.

    Attributes:
        render_mode: Video quality mode (rl_core, vision_rl, demo, 2k_demo, 4k_demo)
        save_fps: Timeline recording rate (frames per second)
        camera_width: Camera resolution width in pixels
        camera_height: Camera resolution height in pixels
        camera_fps: Camera rendering rate (0 = disabled)
        enable_depth: Whether to record depth images
        codec: Video codec (h264, mjpeg)
        quality: Video quality (CRF: 0=lossless, 23=default, 51=worst)
    """

    render_mode: str
    save_fps: float
    camera_width: int
    camera_height: int
    camera_fps: int
    enable_depth: bool = True
    codec: str = "h264"
    quality: int = 20  # Lower = better quality

    @classmethod
    def from_render_mode(cls, render_mode: str, save_fps: float = 30.0) -> 'VideoConfig':
        """Create video configuration from render mode - OFFENSIVE!

        Single source of truth for video settings.
        Replaces duplicate logic in ExperimentOps.__init__ and compile().

        Args:
            render_mode: Video quality mode
            save_fps: Timeline recording rate

        Returns:
            VideoConfig with appropriate settings

        Raises:
            ValueError: If render_mode is unknown

        Render Modes:
            - "rl_core": No cameras (camera_fps=0, 640x480)
            - "vision_rl": Low res for RL (10fps, 640x480)
            - "demo": HD video (30fps, 1280x720)
            - "2k_demo": Full HD (30fps, 1920x1080)
            - "4k_demo": 4K UHD (30fps, 3840x2160)
            - "mujoco_demo": Full rate (200fps, 1280x720)
            - "slow": Full rate low res (200fps, 640x480)
        """
        # Validate render_mode
        valid_modes = ["rl_core", "rl_core_no_timeline", "vision_rl", "demo", "2k_demo", "4k_demo", "mujoco_demo", "slow"]
        if render_mode not in valid_modes:
            raise ValueError(
                f"❌ Unknown render_mode: '{render_mode}'!\n"
                f"   Valid modes: {valid_modes}\n"
                f"   FIX: Use one of the valid render modes"
            )

        # Resolution mapping (from ExperimentOps.__init__ lines 128-143)
        if render_mode == "4k_demo":
            width, height = 3840, 2160
        elif render_mode == "2k_demo":
            width, height = 1920, 1080
        elif render_mode in ["demo", "mujoco_demo"]:
            width, height = 1280, 720
        else:  # rl_core, rl_core_no_timeline, vision_rl, slow
            width, height = 640, 480

        # FPS mapping (from ExperimentOps.compile() lines 752-789)
        # step_rate: RL agent observation + action frequency (default = state_hz for sim-to-real)
        render_mode_config = {
            "rl_core": {"camera_fps": 0, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "rl_core_no_timeline": {"camera_fps": 0, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},  # BENCHMARK!
            "vision_rl": {"camera_fps": 10, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "demo": {"camera_fps": 30, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "2k_demo": {"camera_fps": 30, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "4k_demo": {"camera_fps": 30, "sensor_hz": 30, "state_hz": 30, "step_rate": 30},
            "mujoco_demo": {"camera_fps": 200, "sensor_hz": 200, "state_hz": 200, "step_rate": 200},
            "slow": {"camera_fps": 200, "sensor_hz": 200, "state_hz": 200, "step_rate": 200},
        }

        config = render_mode_config[render_mode]
        camera_fps = config["camera_fps"]

        return cls(
            render_mode=render_mode,
            save_fps=save_fps,
            camera_width=width,
            camera_height=height,
            camera_fps=camera_fps,
            enable_depth=True,
            codec="h264",
            quality=20
        )

    def __repr__(self) -> str:
        """Human-readable representation"""
        return (
            f"VideoConfig({self.render_mode}, "
            f"{self.camera_width}x{self.camera_height}@{self.camera_fps}fps, "
            f"save@{self.save_fps}fps)"
        )

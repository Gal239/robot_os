"""
VIDEO MODULE - Unified Video Operations for Simulation Center
==============================================================

MOP: Single source of truth for ALL video operations!

Public API:
- VideoOps: All video conversion, validation, thumbnails
- VideoConfig: Centralized video configuration

Usage:
    from core.video import VideoOps, VideoConfig

    # Create config
    config = VideoOps.create_config(render_mode="2k_demo", save_fps=30)

    # Convert videos
    results = VideoOps.convert_all_videos(timeline_dir)
    VideoOps.validate_conversion(results, timeout=180)

    # Generate thumbnails
    VideoOps.generate_all_thumbnails(timeline_dir)
"""

from .video_ops import VideoOps, ConversionResults
from .video_config import VideoConfig

__all__ = [
    'VideoOps',
    'VideoConfig',
    'ConversionResults'
]

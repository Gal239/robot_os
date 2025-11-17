#!/usr/bin/env python3
"""
TIMELINE VIDEO COMPRESSOR - Offline video compression tool using VideoOps!
CRASH-SAFE: Compresses saved videos anytime after experiment

MOP: Now uses VideoOps - NO duplicate FFmpeg code!
Reduced from 224 lines to ~80 lines (64% reduction)!

Usage:
    python timeline_compress_videos.py experiment_dir/
    python timeline_compress_videos.py experiment_dir/ --quality 18

Features:
    - Uses VideoOps for ALL conversion logic
    - Compresses AVI videos to MP4
    - Generates thumbnails automatically
    - Can run ANYTIME (even after crash!)
    - Re-compressible with different settings
    - Shows compression ratios

Design:
    - Independent of runtime simulation
    - Scans timeline/cameras/ for AVI files
    - Delegates to VideoOps.convert_all_videos()
    - OFFENSIVE: Crashes if conversion fails
"""

import argparse
from pathlib import Path
import sys


def compress_timeline_videos(experiment_dir: Path, quality: int = 20):
    """Compress all videos in experiment timeline using VideoOps!

    Args:
        experiment_dir: Path to experiment directory
        quality: CRF quality (0=lossless, 20=high, 23=default, 51=worst)

    Returns:
        Number of videos successfully compressed

    Raises:
        RuntimeError: If timeline directory doesn't exist or conversion fails
    """
    timeline_dir = experiment_dir / "timeline"

    if not timeline_dir.exists():
        raise RuntimeError(
            f"❌ No timeline directory found!\n"
            f"   Expected: {timeline_dir}\n"
            f"   FIX: Check that experiment ran successfully"
        )

    cameras_dir = timeline_dir / "cameras"

    if not cameras_dir.exists():
        raise RuntimeError(
            f"❌ No cameras directory found!\n"
            f"   Expected: {cameras_dir}\n"
            f"   FIX: Check that experiment ran with cameras enabled"
        )

    # Use VideoOps for ALL conversion logic! - MOP
    from ..video import VideoOps

    print(f"🎬 Compressing videos using VideoOps...")
    print(f"   Quality: CRF={quality} (0=lossless, 20=high, 23=default)")
    print(f"   Location: {cameras_dir}")

    # Convert all videos (synchronous mode)
    results = VideoOps.convert_all_videos(
        cameras_dir,
        quality=quality,
        async_mode=False  # Synchronous for CLI tool
    )

    # Validate results
    try:
        VideoOps.validate_conversion(results, timeout=300)
    except RuntimeError as e:
        print(f"\n❌ Conversion failed!")
        raise

    # Success!
    print(f"\n✅ Compressed {results.completed}/{results.expected} videos successfully!")
    return results.completed


def main():
    parser = argparse.ArgumentParser(
        description="Compress timeline videos to MP4 using VideoOps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default quality)
  python timeline_compress_videos.py experiments/exp_001/

  # High quality compression
  python timeline_compress_videos.py experiments/exp_001/ --quality 18

  # Lower quality (smaller files)
  python timeline_compress_videos.py experiments/exp_001/ --quality 28

Quality Guide:
  --quality 18 = High quality, larger files (visually lossless)
  --quality 20 = Default, good balance
  --quality 23 = Standard, smaller files
  --quality 28 = Low quality, smallest files
        """
    )

    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to experiment directory (contains timeline/ subdirectory)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=20,
        help="CRF quality: 0=lossless, 20=high, 23=default, 51=worst (default: 20)"
    )

    args = parser.parse_args()

    # Validate experiment directory
    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    # Validate quality range
    if not (0 <= args.quality <= 51):
        print(f"❌ Quality must be between 0-51 (you provided: {args.quality})")
        sys.exit(1)

    # Compress videos using VideoOps!
    try:
        success_count = compress_timeline_videos(experiment_dir, args.quality)

        if success_count == 0:
            print("\n⚠️  No videos found to compress!")
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

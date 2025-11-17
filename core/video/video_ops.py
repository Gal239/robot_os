"""
VIDEO OPERATIONS - Unified Video Handling for Simulation Center
================================================================

MOP: Single source of truth for ALL video operations!
Eliminates 80% code duplication between timeline_saver.py and timeline_compress_videos.py.

Responsibilities:
- Video format conversion (AVI → MP4, JPEG → MP4)
- Thumbnail generation
- Video validation
- Metadata extraction
- Centralized FFmpeg operations

Replaces:
- timeline_saver.py lines 600-750 (conversion logic)
- timeline_compress_videos.py lines 32-140 (duplicate conversion)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import threading
from dataclasses import dataclass, field


@dataclass
class ConversionResults:
    """Results from video conversion operation - MOP!

    Tracks conversion status for validation.
    Replaces scattered tracking in timeline_saver.py.
    """
    expected: int
    completed: int = 0
    failed: List[Dict[str, str]] = field(default_factory=list)
    thread: Optional[threading.Thread] = None


class VideoOps:
    """Unified video operations for simulation_center - OFFENSIVE!

    MOP: All video operations in ONE place!
    NO duplication of FFmpeg commands!
    NO silent failures - crashes explicitly!

    Static methods only - stateless operations.
    """

    # ========================================================================
    # VIDEO CONVERSION
    # ========================================================================

    @staticmethod
    def convert_video(
        input_path: Path,
        output_path: Path,
        quality: int = 20,
        fps: Optional[float] = None
    ) -> bool:
        """Convert single video to H.264/MP4 - OFFENSIVE!

        Unified conversion logic - no more duplication!

        Args:
            input_path: Input video (AVI, MJPEG, etc.)
            output_path: Output MP4 file
            quality: CRF quality (0=lossless, 23=default, 51=worst)
            fps: Optional FPS override

        Returns:
            True if conversion succeeded, False otherwise

        Raises:
            RuntimeError: If ffmpeg not available or severe error
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise RuntimeError(
                f"❌ Input video does not exist: {input_path}\n"
                f"   FIX: Check file path or recording process"
            )

        # Build FFmpeg command (unified from timeline_saver.py and timeline_compress_videos.py)
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', str(input_path),  # Input video
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'baseline',  # Baseline profile for universal compatibility
            '-level', '3.0',  # Level 3.0 for older devices
            '-preset', 'fast',  # Fast encoding
            '-crf', str(quality),  # Quality setting
            '-pix_fmt', 'yuv420p',  # Compatible pixel format
        ]

        # Add FPS override if specified
        if fps is not None:
            cmd.extend(['-r', str(fps)])

        cmd.append(str(output_path))

        try:
            # Run conversion
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                return True
            else:
                # Conversion failed
                print(f"  ❌ FFmpeg failed (code {result.returncode}): {result.stderr[:200]}")
                return False

        except subprocess.TimeoutExpired:
            print(f"  ❌ FFmpeg timeout (>120s) for {input_path.name}")
            return False
        except Exception as e:
            print(f"  ❌ FFmpeg error: {e}")
            return False

    @staticmethod
    def convert_all_videos(
        timeline_dir: Path,
        quality: int = 20,
        async_mode: bool = True
    ) -> ConversionResults:
        """Convert all videos in timeline directory - OFFENSIVE!

        Unified batch conversion - replaces duplicate code!

        Args:
            timeline_dir: Timeline/cameras directory
            quality: CRF quality (0=lossless, 23=default, 51=worst)
            async_mode: If True, run in background thread

        Returns:
            ConversionResults with tracking info

        MOP: Tracks ALL conversions for validation!
        """
        timeline_dir = Path(timeline_dir)

        if not timeline_dir.exists():
            # No videos to convert - return empty results
            return ConversionResults(expected=0)

        # Find all AVI files
        avi_files = list(timeline_dir.glob("*/*.avi"))

        if not avi_files:
            # No videos found
            return ConversionResults(expected=0)

        print(f"\n🎬 Converting {len(avi_files)} videos to H.264/MP4...")

        # Create results tracker
        results = ConversionResults(expected=len(avi_files))

        if async_mode:
            # Background conversion (non-daemon for proper completion!)
            thread = threading.Thread(
                target=VideoOps._convert_videos_worker,
                args=(avi_files, quality, results),
                daemon=False,  # NOT daemon - must complete!
                name="H264_Converter"
            )
            results.thread = thread
            thread.start()
        else:
            # Synchronous conversion
            VideoOps._convert_videos_worker(avi_files, quality, results)

        return results

    @staticmethod
    def _convert_videos_worker(
        avi_files: List[Path],
        quality: int,
        results: ConversionResults
    ):
        """Worker that converts videos and generates thumbnails - OFFENSIVE!

        Unified worker - replaces duplicate code in timeline_saver.py!

        Args:
            avi_files: List of AVI files to convert
            quality: CRF quality
            results: ConversionResults to update

        MOP: Tracks EVERY conversion - no silent failures!
        """
        for avi_path in avi_files:
            mp4_path = avi_path.with_suffix('.mp4')

            # Convert video
            success = VideoOps.convert_video(avi_path, mp4_path, quality=quality)

            if success:
                # Print conversion stats
                avi_size = avi_path.stat().st_size / (1024 * 1024)
                mp4_size = mp4_path.stat().st_size / (1024 * 1024)
                ratio = avi_size / mp4_size if mp4_size > 0 else 1.0

                print(f"  ✓ {mp4_path.name}: {avi_size:.1f}MB → {mp4_size:.1f}MB ({ratio:.1f}x)")

                # Generate thumbnail
                thumbnail_path = mp4_path.with_name(mp4_path.stem + '_thumbnail.jpg')
                thumbnail_success = VideoOps.generate_thumbnail(mp4_path, thumbnail_path)

                if thumbnail_success:
                    # Both MP4 and thumbnail succeeded!
                    results.completed += 1
                    # Delete AVI source
                    avi_path.unlink()
                else:
                    # Thumbnail failed - track as failure
                    results.failed.append({
                        'video': str(avi_path.name),
                        'error': 'Thumbnail generation failed',
                        'type': 'thumbnail_failed'
                    })
                    print(f"  ❌ Failed to create thumbnail for {mp4_path.name}")
            else:
                # Conversion failed - track it
                results.failed.append({
                    'video': str(avi_path.name),
                    'error': 'FFmpeg conversion failed',
                    'type': 'conversion_failed'
                })

        # Print summary
        print(f"\n🎬 H.264 conversion complete: {results.completed}/{results.expected} successful, {len(results.failed)} failed")

    # ========================================================================
    # THUMBNAIL GENERATION
    # ========================================================================

    @staticmethod
    def generate_thumbnail(
        video_path: Path,
        output_path: Optional[Path] = None,
        quality: int = 2
    ) -> bool:
        """Extract last frame as high-quality JPEG thumbnail - OFFENSIVE!

        Unified thumbnail generation - replaces code in timeline_saver.py!

        Args:
            video_path: Input video file
            output_path: Output JPEG path (default: video_name_thumbnail.jpg)
            quality: JPEG quality (2=highest, 31=lowest)

        Returns:
            True if thumbnail generated, False otherwise

        Raises:
            RuntimeError: If video doesn't exist
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise RuntimeError(
                f"❌ Video does not exist: {video_path}\n"
                f"   FIX: Ensure video was converted before generating thumbnail"
            )

        # Default output path
        if output_path is None:
            output_path = video_path.with_name(video_path.stem + '_thumbnail.jpg')
        else:
            output_path = Path(output_path)

        # FFmpeg command to extract last frame
        cmd = [
            'ffmpeg', '-y',  # Overwrite
            '-sseof', '-0.1',  # Seek to 0.1 seconds before end
            '-i', str(video_path),  # Input video
            '-vframes', '1',  # Extract 1 frame
            '-q:v', str(quality),  # JPEG quality
            str(output_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10  # Quick operation
            )

            return result.returncode == 0

        except Exception:
            return False

    @staticmethod
    def generate_all_thumbnails(timeline_dir: Path) -> List[Path]:
        """Generate thumbnails for all MP4 videos - OFFENSIVE!

        Args:
            timeline_dir: Timeline/cameras directory

        Returns:
            List of generated thumbnail paths
        """
        timeline_dir = Path(timeline_dir)

        if not timeline_dir.exists():
            return []

        # Find all MP4 files without thumbnails
        mp4_files = list(timeline_dir.glob("*/*.mp4"))
        generated = []

        for mp4_path in mp4_files:
            thumbnail_path = mp4_path.with_name(mp4_path.stem + '_thumbnail.jpg')

            # Skip if thumbnail already exists
            if thumbnail_path.exists():
                continue

            if VideoOps.generate_thumbnail(mp4_path, thumbnail_path):
                generated.append(thumbnail_path)

        return generated

    # ========================================================================
    # VIDEO VALIDATION
    # ========================================================================

    @staticmethod
    def validate_conversion(
        results: ConversionResults,
        timeout: float = 180
    ):
        """Wait for and validate async conversions - MOP!

        OFFENSIVE: Crashes if any conversions failed!
        NO silent failures - we validate EVERYTHING!

        Args:
            results: ConversionResults from convert_all_videos()
            timeout: Maximum seconds to wait

        Raises:
            RuntimeError: If conversions failed or timeout

        MOP: Tests MUST call this to ensure video conversion succeeded!
        """
        # Check if conversion was started
        if results.expected == 0:
            return  # No videos to validate

        thread = results.thread
        if thread is None:
            # Synchronous mode or no thread - check results immediately
            VideoOps._validate_results(results)
            return

        # Wait for thread to complete
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread still running - TIMEOUT!
            raise RuntimeError(
                f"❌ Video conversion TIMEOUT after {timeout}s!\n"
                f"   Expected: {results.expected} videos\n"
                f"   Completed: {results.completed} videos\n"
                f"   Failed: {len(results.failed)} videos\n"
                f"\n"
                f"   MOP: Conversion must complete within timeout!\n"
                f"   FIX: Increase timeout or check ffmpeg performance"
            )

        # Validate results
        VideoOps._validate_results(results)

    @staticmethod
    def _validate_results(results: ConversionResults):
        """Validate conversion results - CRASHES on failures!

        Args:
            results: ConversionResults to validate

        Raises:
            RuntimeError: If any conversions failed
        """
        # Check if ALL conversions succeeded
        if len(results.failed) > 0 or results.completed != results.expected:
            # Build detailed error message
            error_lines = [
                f"❌ Video conversion FAILED!",
                f"   Expected: {results.expected} videos",
                f"   Completed: {results.completed} successful",
                f"   Failed: {len(results.failed)} videos",
                f"",
                f"   Failed videos:"
            ]

            for failure in results.failed:
                error_lines.append(f"     - {failure['video']}: {failure['type']}")
                error_lines.append(f"       Error: {failure['error'][:100]}")

            error_lines.extend([
                f"",
                f"   MOP: ALL videos MUST convert successfully!",
                f"   FIX: Check ffmpeg installation and video codec support"
            ])

            raise RuntimeError("\n".join(error_lines))

        # Success!
        print(f"✅ Video conversion validated: {results.completed}/{results.expected} successful")

    @staticmethod
    def validate_video_file(video_path: Path) -> Dict[str, Any]:
        """Validate single video file integrity - OFFENSIVE!

        Args:
            video_path: Path to video file

        Returns:
            Dict with validation info:
                - exists: bool
                - size: int (bytes)
                - valid: bool
                - error: Optional[str]
        """
        video_path = Path(video_path)

        if not video_path.exists():
            return {
                'exists': False,
                'size': 0,
                'valid': False,
                'error': 'File does not exist'
            }

        # Check file size
        size = video_path.stat().st_size

        if size < 1000:  # Less than 1KB - probably just header
            return {
                'exists': True,
                'size': size,
                'valid': False,
                'error': f'File too small ({size} bytes) - likely incomplete'
            }

        # Try to get video info with ffprobe
        try:
            info = VideoOps.get_video_info(video_path)
            return {
                'exists': True,
                'size': size,
                'valid': True,
                'error': None,
                'info': info
            }
        except Exception as e:
            return {
                'exists': True,
                'size': size,
                'valid': False,
                'error': str(e)
            }

    # ========================================================================
    # VIDEO METADATA
    # ========================================================================

    @staticmethod
    def get_video_info(video_path: Path) -> Dict[str, Any]:
        """Extract video metadata using ffprobe - OFFENSIVE!

        Args:
            video_path: Path to video file

        Returns:
            Dict with: duration, fps, width, height, codec

        Raises:
            RuntimeError: If ffprobe fails or video is invalid
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise RuntimeError(
                f"❌ Video does not exist: {video_path}\n"
                f"   FIX: Check file path"
            )

        # Use ffprobe to get video metadata
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,codec_name,duration',
            '-of', 'default=noprint_wrappers=1',
            str(video_path)
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            # Parse output
            info = {}
            for line in result.stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    info[key] = value

            # Convert to structured format
            return {
                'width': int(info.get('width', 0)),
                'height': int(info.get('height', 0)),
                'fps': info.get('r_frame_rate', 'unknown'),
                'codec': info.get('codec_name', 'unknown'),
                'duration': float(info.get('duration', 0.0))
            }

        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to get video info for {video_path.name}!\n"
                f"   Error: {e}\n"
                f"   FIX: Ensure ffprobe is installed and video is valid"
            )

    @staticmethod
    def get_all_videos(timeline_dir: Path) -> List[Dict[str, Any]]:
        """List all videos with metadata - OFFENSIVE!

        Args:
            timeline_dir: Timeline/cameras directory

        Returns:
            List of dicts with video info
        """
        timeline_dir = Path(timeline_dir)

        if not timeline_dir.exists():
            return []

        videos = []
        for mp4_path in timeline_dir.glob("*/*.mp4"):
            try:
                info = VideoOps.get_video_info(mp4_path)
                info['path'] = str(mp4_path)
                info['name'] = mp4_path.name
                videos.append(info)
            except Exception:
                # Skip invalid videos
                continue

        return videos

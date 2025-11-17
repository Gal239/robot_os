"""
TIME TRAVELER - FPS-based timeline saving for complete experiment replay
OFFENSIVE & ELEGANT: Each view saves at its own FPS rate

Pattern: ViewAggregator consumer that saves views to disk
- Cameras → CSV data (render videos LATER on demand!)
- Sensors → CSV (timestamped rows)
- MuJoCo/Rewards/Actions → JSON (snapshots)

FAST: NO Matplotlib rendering during simulation!
- Rendering blocks physics and creates thread safety issues
- Save raw CSV data, generate videos later when needed
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import csv
import numpy as np
import cv2
import subprocess
import threading

from .async_writer import AsyncWriter


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types - OFFENSIVE"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class TimelineSaver:
    """TIME TRAVELER - Save views at configured FPS rates

    Each view team saves in its own format:
    - Cameras → JPEG images (later compress to H.264 video)
    - Sensors → CSV files (timestamped rows)
    - MuJoCo → JSON (state snapshots)
    - Rewards → JSON (progress tracking)
    - Actions → JSON (queue state)

    Design:
    - Registered as ViewAggregator consumer
    - Receives all views every step
    - Saves each view at its configured FPS
    - Tracks last save step per view

    OFFENSIVE: Crashes if timeline dir creation fails
    """

    def __init__(self, experiment_dir: str, global_fps: float = 10.0, timestep: float = 0.005,
                 backend=None, db_ops=None, experiment_ops=None):
        """Initialize timeline saver with ASYNC worker thread

        Args:
            experiment_dir: Experiment directory path
            global_fps: Default FPS for views without specific config
            timestep: Physics timestep (default 0.005s = 200Hz)
            backend: MuJoCo backend (for accessing model/data for scene state) - NEW
            db_ops: DatabaseOps instance (for saving package/state) - NEW
            experiment_ops: ExperimentOps instance (for package creation flag) - NEW
        """
        self.experiment_dir = Path(experiment_dir)
        self.timeline_dir = self.experiment_dir / "timeline"
        self.global_fps = global_fps
        self.timestep = timestep

        # NEW: MuJoCo package and scene state support
        self.backend = backend
        self.db_ops = db_ops
        self.experiment_ops = experiment_ops
        self.frame_count = 0  # Frame counter for scene_state naming
        self.last_saved_step = -9999  # Track last saved step for scene_state

        # Create timeline directory structure
        self._create_timeline_structure()

        # Tracking
        self.step = 0
        self.sim_time = 0.0
        self.last_save_step: Dict[str, int] = {}

        # Format-specific writers (simple, fast!)
        self.csv_files: Dict[str, Any] = {}  # {view_name: csv.writer}
        self.csv_handles: Dict[str, Any] = {}  # {view_name: file_handle}
        self.json_data: Dict[str, Dict] = {}  # {view_name: {step: data}}

        # Camera video writers (raw pixels → MP4, NO Matplotlib!)
        self.video_writers: Dict[str, Any] = {}  # {view_name: cv2.VideoWriter}
        self.camera_frame_counts: Dict[str, int] = {}  # {view_name: frame_count}

        # ASYNC I/O: Background writer for non-blocking disk writes
        # Prevents blocking simulation for slow disk/video operations
        self.async_writer = AsyncWriter(maxsize=200, name="TimelineSaver")

    def _create_timeline_structure(self):
        """Create timeline directory structure - OFFENSIVE"""
        # Main timeline dir
        self.timeline_dir.mkdir(parents=True, exist_ok=True)

        # Team directories
        (self.timeline_dir / "cameras").mkdir(exist_ok=True)
        (self.timeline_dir / "sensors").mkdir(exist_ok=True)
        (self.timeline_dir / "actuators").mkdir(exist_ok=True)
        (self.timeline_dir / "mujoco").mkdir(exist_ok=True)
        (self.timeline_dir / "rewards").mkdir(exist_ok=True)
        (self.timeline_dir / "actions").mkdir(exist_ok=True)
        (self.timeline_dir / "system").mkdir(exist_ok=True)

    def save_frame(self, views: Dict[str, Any]):
        """Save views to CSV/JSON - BATCHED ASYNC (no rendering!)

        Called by ViewAggregator when views are distributed (already throttled by RuntimeEngine!)

        PERFORMANCE: All writes batched into ONE async task to reduce lock contention!
        Before: 16 async tasks/frame = 16 lock acquisitions (1.071s wasted)
        After: 1 async task/frame = 1 lock acquisition (0.2s)

        Args:
            views: Dict of view_name -> view_data from ViewAggregator
        """
        # Extract PHYSICS step from runtime_status view
        physics_step = views["runtime_status"]["step_count"]  # OFFENSIVE - crash if missing!

        # Calculate SIM TIME from physics step (not elapsed_time which is REAL time!)
        # sim_time = step_count × timestep
        physics_time = physics_step * self.timestep

        # Update tracking
        self.step = physics_step
        self.sim_time = physics_time

        # BATCHED ASYNC: Collect all write operations on main thread
        # Then submit ONE task that does ALL writes in background
        write_operations = []

        # Collect all view save operations
        for view_name, view_data in views.items():
            # Collect writes for this view
            ops = self._collect_view_writes(view_name, view_data)
            write_operations.extend(ops)
            self.last_save_step[view_name] = self.step

        # Submit ONE batched async task if we have writes
        if write_operations:
            def batch_write():
                """Execute all writes in one background task"""
                for write_func in write_operations:
                    try:
                        write_func()
                    except Exception as e:
                        print(f"⚠️  Write operation failed: {e}")

            self.async_writer.submit(batch_write)

        # NEW: MuJoCo package and scene state saving
        self._save_mujoco_package_and_state()

    def _save_mujoco_package_and_state(self):
        """Save MuJoCo package (first frame) and scene state (at save_fps) - PURE MOP!

        Called every frame, but:
        - MuJoCo package created LAZILY on first frame only
        - Scene state saved at save_fps rate (30 FPS)

        PURE MOP: Modals save themselves!
        - AssetPackageModal discovers assets, copies files, rewrites XML
        - SceneStateModal captures qpos/qvel/ctrl, generates keyframe XML
        - DatabaseOps provides paths (thin layer)
        """
        # Skip if no backend/db_ops/experiment_ops
        if not self.backend or not self.db_ops or not self.experiment_ops:
            return

        # LAZY: Create mujoco_package/ on FIRST call only
        if not self.experiment_ops._package_created:
            self._create_asset_package()
            self.experiment_ops._package_created = True

        # Save scene state at save_fps rate (30 FPS)
        if self._should_save_frame():
            self._save_scene_state()
            # Update tracking
            self.last_saved_step = self.step
            self.frame_count += 1

    def _should_save_frame(self) -> bool:
        """Check if we should save a scene state frame at save_fps rate

        Returns:
            True if enough physics steps have passed for next frame
        """
        # Calculate steps between frames at save_fps
        steps_per_frame = int((1.0 / self.global_fps) / self.timestep)

        # Check if enough steps have passed since last save
        steps_since_last = self.step - self.last_saved_step
        return steps_since_last >= steps_per_frame

    def _create_asset_package(self):
        """Create mujoco_package/ on first frame - ASYNC! PURE MOP!"""
        # PERFORMANCE: Run in background thread to avoid blocking simulation!
        # Copying 90+ mesh files (80MB+) would freeze the sim for ~0.5-1.0s

        # MAIN THREAD: Capture data references (must be synchronous for thread safety!)
        try:
            # Get compiled XML from backend
            if not hasattr(self.backend, 'modal') or not hasattr(self.backend.modal, 'compiled_xml'):
                print("  ⚠️  Backend has no compiled_xml - skipping mujoco_package creation")
                return

            # Capture references on main thread for thread safety
            compiled_xml = self.backend.modal.compiled_xml
            experiment_id = self.experiment_ops.experiment_id
            db_ops = self.db_ops

        except Exception as e:
            print(f"  ⚠️  Failed to prepare mujoco_package data: {e}")
            return

        def create_package_async():
            """Background thread: Create package without blocking simulation"""
            try:
                from ..modals.asset_package_modal import AssetPackageModal

                # Create package modal (auto-discovers assets!)
                package = AssetPackageModal(
                    compiled_xml=compiled_xml,
                    experiment_id=experiment_id
                )

                # Modal saves itself! (Pure MOP)
                # All file copying happens here - in background thread!
                db_ops.save_asset_package(package)

            except Exception as e:
                print(f"  ⚠️  Failed to create mujoco_package: {e}")
                # Don't crash simulation - package creation is optional

        # Submit to background thread - returns immediately!
        self.async_writer.submit(create_package_async)

    def _save_scene_state(self):
        """Save current physics state as keyframe XML - ASYNC! PURE MOP!"""
        # PERFORMANCE: Capture state on main thread, save to disk in background!
        # THREAD SAFETY: MuJoCo model/data accessed on main thread only
        try:
            from ..modals.scene_state_modal import SceneStateModal

            # MAIN THREAD: Capture state from MuJoCo (must be synchronous!)
            # This is fast (~1-2ms) - just copies qpos/qvel arrays
            state = SceneStateModal.from_mujoco(
                self.backend.model,
                self.backend.data,
                frame_num=self.frame_count
            )

            # BACKGROUND THREAD: Save to disk (file I/O is slow!)
            def save_state_async():
                """Background thread: Write state to disk without blocking simulation"""
                try:
                    # Modal saves itself! (Pure MOP)
                    # File I/O happens in background thread!
                    self.db_ops.save_scene_state(state)

                except Exception as e:
                    print(f"  ⚠️  Failed to save scene state frame {self.frame_count}: {e}")
                    # Don't crash simulation - state saving is optional

            # Submit file I/O to background thread - returns immediately!
            self.async_writer.submit(save_state_async)

        except Exception as e:
            print(f"  ⚠️  Failed to capture scene state frame {self.frame_count}: {e}")
            # Don't crash simulation - state saving is optional

    def _collect_view_writes(self, view_name: str, view_data: Dict[str, Any]):
        """Collect all write operations for this view WITHOUT executing them

        BATCHING: Returns list of functions to execute in ONE async task
        This eliminates lock contention from 16 separate async_writer.submit() calls

        Args:
            view_name: Name of view
            view_data: View data dict with __meta__ field

        Returns:
            List of functions to execute (each function does one write operation)
        """
        write_ops = []

        # Extract metadata
        meta = view_data.get("__meta__")
        if not meta:
            return write_ops  # No metadata, skip

        view_type = meta.get("view_type")
        if not view_type:
            return write_ops  # No view type, skip

        # Collect write operations based on view type
        if view_type == "video":
            # Camera video writes
            ops = self._collect_camera_video_writes(view_name, view_data)
            write_ops.extend(ops)

        elif view_type == "video_and_data":
            # CSV writes for sensors/actuators
            ops = self._collect_csv_writes(view_name, view_data, view_type)
            write_ops.extend(ops)

        elif view_type == "data":
            # JSON writes for system/rewards/actions
            ops = self._collect_json_writes(view_name, view_data)
            write_ops.extend(ops)

        return write_ops

    def _collect_camera_video_writes(self, view_name: str, view_data: Dict[str, Any]):
        """Collect camera video write operations WITHOUT executing"""
        write_ops = []

        rgb = view_data.get('rgb')
        depth = view_data.get('depth')

        if rgb is None:
            return write_ops

        clean_name = view_name.replace("_view", "")
        video_dir = self.timeline_dir / "cameras" / clean_name

        # PERFORMANCE FIX: Move color conversion to background thread!
        # Only copy the RGB data reference on main thread, do cvtColor in background
        rgb_key = f"{view_name}_rgb"
        if rgb_key not in self.video_writers:
            # Create writer (must be on main thread for thread safety)
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{clean_name}_rgb.avi"
            height, width = rgb.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(str(video_path), fourcc, self.global_fps, (width, height))
            if not writer.isOpened():
                return write_ops  # Failed to create writer
            self.video_writers[rgb_key] = writer
            self.camera_frame_counts[rgb_key] = 0

        # RGB video write - OPTIMIZED: cvtColor + copy in background thread
        writer = self.video_writers[rgb_key]
        rgb_data = rgb.astype(np.uint8).copy()  # Shallow copy to detach from view
        write_ops.append(lambda w=writer, data=rgb_data: w.write(cv2.cvtColor(data, cv2.COLOR_RGB2BGR)))
        self.camera_frame_counts[rgb_key] += 1

        # Depth video write - OPTIMIZED: normalization + cvtColor in background thread
        if depth is not None:
            depth_key = f"{view_name}_depth"
            if depth_key not in self.video_writers:
                video_path = video_dir / f"{clean_name}_depth.avi"
                height, width = depth.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(str(video_path), fourcc, self.global_fps, (width, height))
                if not writer.isOpened():
                    return write_ops
                self.video_writers[depth_key] = writer
                self.camera_frame_counts[depth_key] = 0

            depth_writer = self.video_writers[depth_key]
            depth_data = depth.copy()  # Copy for thread safety

            # Define background processing function
            def process_and_write_depth(w, data):
                # Normalize depth to 0-255
                depth_min, depth_max = data.min(), data.max()
                if depth_max > depth_min:
                    normalized = ((data - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(data, dtype=np.uint8)
                # Convert to BGR for video
                depth_bgr = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
                w.write(depth_bgr)

            write_ops.append(lambda w=depth_writer, data=depth_data: process_and_write_depth(w, data))
            self.camera_frame_counts[depth_key] += 1

        return write_ops

    def _collect_csv_writes(self, view_name: str, view_data: Dict[str, Any], view_type: str):
        """Collect CSV write operations WITHOUT executing"""
        write_ops = []

        data_to_save = {k: v for k, v in view_data.items() if k != "__meta__"}
        modal_category = view_data["__meta__"].get("modal_category")

        if view_type == "video":
            subdir = "cameras"
        elif modal_category == "sensor":
            subdir = "sensors"
        elif modal_category == "actuator":
            subdir = "actuators"
        else:
            return write_ops  # Unknown category

        clean_name = view_name.replace("_view", "").replace("_sensor", "").replace("_actuator", "")

        # Create CSV file if needed (must be on main thread)
        if view_name not in self.csv_files:
            filepath = self.timeline_dir / subdir / f"{clean_name}.csv"
            handle = open(filepath, 'w', newline='')
            fieldnames = ['step', 'time'] + list(data_to_save.keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            self.csv_handles[view_name] = handle
            self.csv_files[view_name] = writer

        # Prepare write operation
        data_copy = data_to_save.copy()
        current_step = self.step
        current_time = self.sim_time
        writer = self.csv_files[view_name]

        def csv_write():
            row = {'step': current_step, 'time': current_time}
            row.update(data_copy)
            for key, value in row.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    row[key] = str(value)
            writer.writerow(row)

        write_ops.append(csv_write)
        return write_ops

    def _collect_json_writes(self, view_name: str, view_data: Dict[str, Any]):
        """Collect JSON write operations WITHOUT executing"""
        write_ops = []

        view_data_copy = {k: v for k, v in view_data.items() if k != "__meta__"}
        current_step = self.step
        current_time = self.sim_time

        # Ensure JSON storage exists (must be on main thread)
        if view_name not in self.json_data:
            self.json_data[view_name] = {}

        json_storage = self.json_data[view_name]

        def json_write():
            json_storage[str(current_step)] = {
                "time": current_time,
                "data": view_data_copy
            }

        write_ops.append(json_write)
        return write_ops

    def _should_save(self, view_name: str, fps: float) -> bool:
        """Check if view should be saved this step - OFFENSIVE

        Args:
            view_name: Name of view
            fps: Target FPS for this view

        Returns:
            True if enough steps have passed since last save
        """
        if view_name not in self.last_save_step:
            return True  # First save

        # Calculate steps between saves at this FPS
        steps_per_save = int((1.0 / fps) / self.timestep)

        # Check if enough steps have passed
        steps_since_last = self.step - self.last_save_step[view_name]
        return steps_since_last >= steps_per_save

    def _save_view(self, view_name: str, view_data: Dict[str, Any]):
        """Save view data ONLY - NO rendering!

        Args:
            view_name: Name of view (e.g., "nav_camera_view", "lidar_view")
            view_data: View data dict with __meta__ field

        Design:
            - Cameras → CSV data (save camera info, render videos LATER!)
            - Sensors/Actuators → CSV data
            - System/Rewards/Actions → JSON

        FAST: NO Matplotlib calls during simulation!
        """
        # Extract metadata
        meta = view_data["__meta__"]  # OFFENSIVE - crash if missing!

        # OFFENSIVE: Crash if no view_type (shows which view is broken!)
        if "view_type" not in meta:
            raise RuntimeError(
                f"❌ View '{view_name}' has no view_type in __meta__!\n"
                f"   Available keys in view_data: {list(view_data.keys())}\n"
                f"   __meta__ contents: {meta}\n"
                f"   FIX: ViewAggregator must set view_type for all views!"
            )

        view_type = meta["view_type"]

        # OFFENSIVE: Crash if view_type unknown
        VALID_VIEW_TYPES = ["video", "video_and_data", "data"]
        if view_type not in VALID_VIEW_TYPES:
            raise RuntimeError(
                f"❌ View '{view_name}' has unknown view_type: '{view_type}'!\n"
                f"   Valid types: {VALID_VIEW_TYPES}\n"
                f"   FIX: Add '{view_type}' to VALID_VIEW_TYPES or fix ViewAggregator!"
            )

        # === SAVE DATA ===

        # TYPE 1: VIDEO-ONLY (cameras) → Save raw pixels to MP4 (NO Matplotlib!)
        # Cameras have rgb/depth arrays - write directly to video file
        if view_type == "video":
            self._save_camera_video(view_name, view_data)

        # TYPE 2: DATA + VIDEO (sensors, actuators) → CSV only (NO rendering!)
        elif view_type == "video_and_data":
            self._save_csv(view_name, view_data, view_type)

        # TYPE 3: DATA-ONLY (system, rewards, assets, scenes) → JSON
        elif view_type == "data":
            self._save_json(view_name, view_data)

    def _save_camera_video(self, view_name: str, view_data: Dict[str, Any]):
        """Save camera raw pixels directly to MP4 - FAST (no Matplotlib!)

        Saves both RGB and depth (if available) as separate video files.

        Args:
            view_name: Camera view name (e.g., "nav_camera_view")
            view_data: View data with 'rgb' and optionally 'depth' arrays
        """
        # Get RGB and depth arrays from camera view
        rgb = view_data.get('rgb')  # LEGITIMATE - optional (not all views have rgb)
        depth = view_data.get('depth')  # LEGITIMATE - optional (not all views have depth)

        if rgb is None:
            return  # No image data

        clean_name = view_name.replace("_view", "")
        video_dir = self.timeline_dir / "cameras" / clean_name
        video_dir.mkdir(parents=True, exist_ok=True)

        # === SAVE RGB VIDEO ===
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Create RGB video writer on first frame
        rgb_key = f"{view_name}_rgb"
        if rgb_key not in self.video_writers:
            # Use AVI container with MJPG codec (always works!)
            # Will be converted to MP4/H.264 async after simulation completes
            video_path = video_dir / f"{clean_name}_rgb.avi"
            height, width = rgb.shape[:2]

            # MJPG codec (always available, built into OpenCV)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.global_fps,
                (width, height)
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"❌ Failed to create RGB video writer for '{view_name}'!\n"
                    f"   Path: {video_path}\n"
                    f"   FIX: Check cv2.VideoWriter codec availability"
                )

            self.video_writers[rgb_key] = writer
            self.camera_frame_counts[rgb_key] = 0

        # Write RGB frame to video - ASYNC to prevent blocking!
        # CRITICAL: Copy frame data before queuing (numpy array might be reused)
        bgr_copy = bgr.copy()
        writer = self.video_writers[rgb_key]
        self.async_writer.submit(lambda: writer.write(bgr_copy))
        self.camera_frame_counts[rgb_key] += 1

        # === SAVE DEPTH VIDEO (if available) ===
        if depth is not None:
            # Normalize depth to 0-255 grayscale
            depth_normalized = depth.copy()
            depth_min, depth_max = depth_normalized.min(), depth_normalized.max()
            if depth_max > depth_min:
                depth_normalized = ((depth_normalized - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)

            # Convert grayscale to BGR (OpenCV VideoWriter needs 3 channels)
            depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)

            # Create depth video writer on first frame
            depth_key = f"{view_name}_depth"
            if depth_key not in self.video_writers:
                # Use AVI container with MJPG codec (always works!)
                # Will be converted to MP4/H.264 async after simulation completes
                video_path = video_dir / f"{clean_name}_depth.avi"
                height, width = depth.shape[:2]

                # MJPG codec (always available, built into OpenCV)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    self.global_fps,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"❌ Failed to create depth video writer for '{view_name}'!\n"
                        f"   Path: {video_path}\n"
                        f"   FIX: Check cv2.VideoWriter codec availability"
                    )

                self.video_writers[depth_key] = writer
                self.camera_frame_counts[depth_key] = 0

            # Write depth frame to video - ASYNC to prevent blocking!
            # CRITICAL: Copy frame data before queuing (numpy array might be reused)
            depth_bgr_copy = depth_bgr.copy()
            depth_writer = self.video_writers[depth_key]
            self.async_writer.submit(lambda: depth_writer.write(depth_bgr_copy))
            self.camera_frame_counts[depth_key] += 1

    def _create_ffmpeg_writer(self, video_path: Path, width: int, height: int, fps: float):
        """Create FFmpeg subprocess video writer - GUARANTEED TO WORK!

        Uses direct ffmpeg subprocess instead of broken cv2.VideoWriter.
        System ffmpeg has working H.264 encoders even though OpenCV doesn't.

        Args:
            video_path: Output video file path
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second

        Returns:
            FFmpegWriter: Writer with .write(frame_bgr) and .release() methods

        Raises:
            RuntimeError: If ffmpeg subprocess fails to start
        """
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-f', 'rawvideo',  # Input format: raw video frames
            '-vcodec', 'rawvideo',  # Input codec: uncompressed
            '-s', f'{width}x{height}',  # Frame size
            '-pix_fmt', 'bgr24',  # Pixel format (OpenCV uses BGR)
            '-r', str(fps),  # Input framerate
            '-i', '-',  # Read from stdin
            '-an',  # No audio
            '-vcodec', 'libx264',  # Output codec: H.264
            '-preset', 'veryfast',  # Fast but better quality than ultrafast
            '-crf', '20',  # Good quality (23=default, lower=better)
            '-pix_fmt', 'yuv420p',  # Standard format for compatibility
            str(video_path)
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                bufsize=10485760  # 10MB buffer to prevent blocking
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Failed to start ffmpeg subprocess for '{video_path}'!\n"
                f"   Command: {' '.join(cmd)}\n"
                f"   Error: {e}\n"
                f"   FIX: Ensure ffmpeg is installed ('which ffmpeg')"
            )

        class FFmpegWriter:
            """Wrapper that mimics cv2.VideoWriter interface"""

            def __init__(self, proc, path):
                self.process = proc
                self.path = path
                self.closed = False

            def write(self, frame_bgr: np.ndarray):
                """Write BGR frame to video"""
                if self.closed:
                    return
                try:
                    self.process.stdin.write(frame_bgr.tobytes())
                except BrokenPipeError:
                    # FFmpeg process died - get error message
                    stderr = self.process.stderr.read().decode('utf-8')
                    raise RuntimeError(
                        f"❌ FFmpeg process died while writing frame!\n"
                        f"   Video: {self.path}\n"
                        f"   FFmpeg error:\n{stderr}"
                    )

            def release(self):
                """Close video writer and finalize video file"""
                if self.closed:
                    return

                self.closed = True

                try:
                    # Close stdin to signal end of input
                    self.process.stdin.close()

                    # Wait for ffmpeg to finish encoding (max 30 seconds)
                    returncode = self.process.wait(timeout=30)

                    if returncode != 0:
                        stderr = self.process.stderr.read().decode('utf-8')
                        raise RuntimeError(
                            f"❌ FFmpeg encoding failed for '{self.path}'!\n"
                            f"   Return code: {returncode}\n"
                            f"   FFmpeg error:\n{stderr}"
                        )

                except subprocess.TimeoutExpired:
                    # Kill process if it takes too long
                    self.process.kill()
                    stderr = self.process.stderr.read().decode('utf-8')
                    raise RuntimeError(
                        f"❌ FFmpeg encoding timeout for '{self.path}'!\n"
                        f"   Process took >30 seconds to finalize video\n"
                        f"   FFmpeg error:\n{stderr}"
                    )

        return FFmpegWriter(process, video_path)

    def _save_csv(self, view_name: str, view_data: Dict[str, Any], view_type: str):
        """Save view data to CSV - cameras, sensors, actuators

        Args:
            view_name: View name
            view_data: View data dict with __meta__ field
            view_type: Type of view ("video", "video_and_data")
        """
        # Remove metadata before saving
        data_to_save = {k: v for k, v in view_data.items() if k != "__meta__"}

        # Determine subdirectory from modal_category
        modal_category = view_data["__meta__"]["modal_category"]  # OFFENSIVE - crash if missing!

        # For cameras (view_type="video"), use "cameras" subdirectory
        if view_type == "video":
            subdir = "cameras"
        # For sensors/actuators, use modal_category
        elif modal_category == "sensor":
            subdir = "sensors"
        elif modal_category == "actuator":
            subdir = "actuators"
        else:
            # OFFENSIVE: Unknown category
            raise RuntimeError(
                f"❌ Unknown modal_category for '{view_name}': '{modal_category}'!\n"
                f"   View type: {view_type}\n"
                f"   Valid categories: ['sensor', 'actuator'] or view_type='video'\n"
                f"   FIX: Check Robot.get_viewable_components() implementation"
            )

        clean_name = view_name.replace("_view", "").replace("_sensor", "").replace("_actuator", "")

        # Create CSV file if needed
        if view_name not in self.csv_files:
            filepath = self.timeline_dir / subdir / f"{clean_name}.csv"

            # Open CSV file
            handle = open(filepath, 'w', newline='')

            # Create writer with header from view_data keys
            fieldnames = ['step', 'time'] + list(data_to_save.keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            self.csv_handles[view_name] = handle
            self.csv_files[view_name] = writer

        # FULLY ASYNC: Move ALL data prep to background thread!
        # Main thread just copies data and returns immediately

        # Fast shallow copy of data (just dict references, not deep copy)
        data_copy = data_to_save.copy()

        # Capture current step/time (primitives are fast)
        current_step = self.step
        current_time = self.sim_time

        # Get writer reference
        writer = self.csv_files[view_name]

        # REMOVED async_writer.submit() - now handled by batching in save_frame()
        # This function is DEAD CODE - kept for reference but never called
        # The batching code in _collect_csv_writes() does the same thing
        pass

    def _save_json(self, view_name: str, view_data: Dict[str, Any]):
        """Save view data to JSON - FULLY ASYNC!

        Handles: rewards, system, scene, assets, and any other complex data

        Args:
            view_name: View name
            view_data: View data dict with __meta__ field
        """
        # FULLY ASYNC: Move dict operations to background thread

        # Fast shallow copy (just references, not deep copy)
        view_data_copy = {k: v for k, v in view_data.items() if k != "__meta__"}

        # Capture current step/time
        current_step = self.step
        current_time = self.sim_time

        # Ensure JSON storage exists (must be on main thread for dict safety)
        if view_name not in self.json_data:
            self.json_data[view_name] = {}

        # Get reference to storage
        json_storage = self.json_data[view_name]

        # REMOVED async_writer.submit() - now handled by batching in save_frame()
        # This function is DEAD CODE - kept for reference but never called
        # The batching code in _collect_json_writes() does the same thing
        pass

    def close(self):
        """Close all files - SIMPLE and FAST!

        Only catches harmless cleanup errors (already closed files).
        Real bugs will crash with full traceback!
        """

        # ASYNC: Wait for all queued writes to complete (CRITICAL!)
        # Must happen BEFORE releasing video writers (they're still being used!)
        self.async_writer.close(timeout=10)

        # Release camera video writers
        for view_name, writer in self.video_writers.items():
            try:
                writer.release()
            except Exception as e:
                if "released" not in str(e).lower():
                    print(f"  ❌ Video writer error for '{view_name}': {e}")
                    raise

        # Close CSV files
        for view_name, handle in self.csv_handles.items():
            try:
                handle.close()
            except ValueError:
                pass  # Already closed - harmless
            except Exception as e:
                print(f"  ❌ Failed to close CSV for '{view_name}': {e}")
                raise  # Real error - CRASH!

        # Save JSON data
        self._save_all_json()

        # Convert MJPG videos to H.264 async using VideoOps!
        # MOP: Unified video operations - no duplicate code!
        from ..video import VideoOps
        self.conversion_results = VideoOps.convert_all_videos(
            self.timeline_dir / "cameras",
            quality=20,
            async_mode=True
        )

        # Quiet close - no prints

    def _save_all_json(self):
        """Save all JSON data to files - OFFENSIVE (with numpy support!)

        OFFENSIVE: Crashes if directory doesn't exist (shows broken cleanup order!)
        """
        for view_name, data in self.json_data.items():
            # Determine output directory
            if "mujoco" in view_name:
                filepath = self.timeline_dir / "mujoco" / f"{view_name}.json"
            elif "reward" in view_name:
                filepath = self.timeline_dir / "rewards" / f"{view_name}.json"
            elif "action" in view_name:
                filepath = self.timeline_dir / "actions" / f"{view_name}.json"
            elif "runtime" in view_name or "status" in view_name:
                filepath = self.timeline_dir / "system" / f"{view_name}.json"
            else:
                filepath = self.timeline_dir / f"{view_name}.json"

            # OFFENSIVE: Check parent directory exists (crash if cleanup order wrong!)
            if not filepath.parent.exists():
                raise RuntimeError(
                    f"❌ Cannot save {view_name}.json - parent directory doesn't exist!\n"
                    f"   Expected: {filepath.parent}\n"
                    f"   Timeline dir: {self.timeline_dir}\n"
                    f"   Timeline exists: {self.timeline_dir.exists()}\n"
                    f"   FIX: Timeline directory was deleted before cleanup! Check cleanup order."
                )

            # Save JSON with NumpyEncoder (handles np.bool_, np.int64, etc.)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)

    def validate_videos(self, timeout: float = 180):
        """Wait for and validate async video conversion - MOP!

        Args:
            timeout: Maximum seconds to wait for conversion (default 3 minutes)

        Raises:
            RuntimeError: If any videos failed to convert or timeout occurs

        MOP: Tests MUST call this to ensure video conversion succeeded!
              NO silent failures - we crash explicitly!

        Usage:
            ops.close()  # Starts async conversion
            ops.engine.timeline_saver.validate_videos()  # Validates completion
        """
        if not hasattr(self, 'conversion_results'):
            return  # No conversions started

        # Use VideoOps for validation!
        from ..video import VideoOps
        VideoOps.validate_conversion(self.conversion_results, timeout=timeout)

"""
POST-SIMULATION VIDEO GENERATOR
Generate visualization videos from CSV timeline data

FAST: Run AFTER simulation completes
- During simulation: Save only CSV/JSON data (fast!)
- After simulation: Generate Matplotlib visualization videos (slow but doesn't block physics)

Usage:
    from core.tools.generate_timeline_videos import generate_timeline_videos
    generate_timeline_videos(experiment_dir)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pandas as pd
import json


class TimelineVideoGenerator:
    """Generate visualization videos from timeline CSV data"""

    def __init__(self, experiment_dir: str, fps: float = 10.0):
        """
        Args:
            experiment_dir: Path to experiment directory
            fps: Frame rate for generated videos
        """
        self.experiment_dir = Path(experiment_dir)
        self.timeline_dir = self.experiment_dir / "timeline"
        self.fps = fps

        if not self.timeline_dir.exists():
            raise FileNotFoundError(f"Timeline directory not found: {self.timeline_dir}")

    def generate_all_videos(self):
        """Generate all visualization videos from CSV data"""
        print(f"\n🎬 Generating timeline videos from CSV data...")
        print(f"📂 Timeline: {self.timeline_dir}")

        # Generate sensor visualization videos
        sensor_csvs = list((self.timeline_dir / "sensors").glob("*.csv"))
        print(f"\n📊 Generating {len(sensor_csvs)} sensor visualization videos...")
        for csv_file in sensor_csvs:
            self._generate_sensor_video(csv_file)

        # Generate actuator visualization videos
        actuator_csvs = list((self.timeline_dir / "actuators").glob("*.csv"))
        print(f"\n🎮 Generating {len(actuator_csvs)} actuator visualization videos...")
        for csv_file in actuator_csvs:
            self._generate_actuator_video(csv_file)

        # Generate grid view video
        print(f"\n📺 Generating grid view video...")
        self._generate_grid_view()

        print(f"\n✅ All visualization videos generated!")

    def _generate_sensor_video(self, csv_file: Path):
        """Generate visualization video for a sensor from CSV data"""
        sensor_name = csv_file.stem
        print(f"  📹 Generating {sensor_name}.mp4...")

        # Read CSV data
        df = pd.read_csv(csv_file)

        # Skip if no data
        if len(df) == 0:
            print(f"    ⚠️  No data in {sensor_name}.csv, skipping")
            return

        # Determine visualization type based on sensor name
        if "lidar" in sensor_name.lower():
            self._generate_lidar_video(df, sensor_name)
        elif "imu" in sensor_name.lower():
            self._generate_imu_video(df, sensor_name)
        elif "force" in sensor_name.lower():
            self._generate_force_video(df, sensor_name)
        elif "odometry" in sensor_name.lower():
            self._generate_odometry_video(df, sensor_name)
        else:
            # Generic line plot for unknown sensors
            self._generate_generic_plot_video(df, sensor_name, "sensors")

    def _generate_actuator_video(self, csv_file: Path):
        """Generate visualization video for an actuator from CSV data"""
        actuator_name = csv_file.stem
        print(f"  📹 Generating {actuator_name}.mp4...")

        # Read CSV data
        df = pd.read_csv(csv_file)

        # Skip if no data
        if len(df) == 0:
            print(f"    ⚠️  No data in {actuator_name}.csv, skipping")
            return

        # Generic position/velocity plot
        self._generate_generic_plot_video(df, actuator_name, "actuators")

    def _generate_generic_plot_video(self, df: pd.DataFrame, name: str, category: str):
        """Generate generic line plot video from CSV data"""
        # Create output directory
        output_dir = self.timeline_dir / category / name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{name}.mp4"

        # Setup video writer with MJPG codec (always available)
        fig, ax = plt.subplots(figsize=(10, 6))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (1000, 600))

        # Get numeric columns (skip step and time)
        numeric_cols = [col for col in df.columns if col not in ['step', 'time']]

        for i in range(len(df)):
            ax.clear()

            # Plot data up to current frame
            for col in numeric_cols:
                try:
                    values = pd.to_numeric(df[col][:i+1], errors='coerce')
                    ax.plot(df['time'][:i+1], values, label=col, marker='o', markersize=3)
                except:
                    pass  # Skip non-numeric columns

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Value')
            ax.set_title(f'{name.upper()} - t={df["time"].iloc[i]:.2f}s')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

            # Convert matplotlib figure to OpenCV image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            writer.write(img_bgr)

        writer.release()
        plt.close(fig)
        print(f"    ✅ Saved: {output_path.relative_to(self.timeline_dir)}")

    def _generate_lidar_video(self, df: pd.DataFrame, name: str):
        """Generate radar plot video for LIDAR data"""
        print(f"    (LIDAR visualization - TODO)")
        # TODO: Implement radar plot for lidar ranges

    def _generate_imu_video(self, df: pd.DataFrame, name: str):
        """Generate IMU gauges video"""
        print(f"    (IMU visualization - TODO)")
        # TODO: Implement gauge displays for IMU orientation

    def _generate_force_video(self, df: pd.DataFrame, name: str):
        """Generate force plot video"""
        print(f"    (Force visualization - TODO)")
        # TODO: Implement force vector visualization

    def _generate_odometry_video(self, df: pd.DataFrame, name: str):
        """Generate odometry trajectory video"""
        print(f"    (Odometry visualization - TODO)")
        # TODO: Implement 2D trajectory plot

    def _generate_grid_view(self):
        """Generate grid view video combining all cameras and sensor plots"""
        print(f"    (Grid view - TODO)")
        # TODO: Implement grid layout with cameras + sensor plots


def generate_timeline_videos(experiment_dir: str, fps: float = 10.0):
    """
    Generate all visualization videos from timeline CSV data

    Args:
        experiment_dir: Path to experiment directory
        fps: Frame rate for generated videos (default: 10.0)
    """
    generator = TimelineVideoGenerator(experiment_dir, fps=fps)
    generator.generate_all_videos()

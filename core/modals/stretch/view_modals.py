"""
ATOMIC VIEW SYSTEM - Complete modular implementation
render_data() is the single source of truth - everything flows from it
"""
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from .sensors_modals import NavCamera, D405Camera, Lidar2D, IMU, GripperForceSensor, ReSpeakerArray, Odometry
from .action_modals import Action


@dataclass
class AtomicView:
    """Base atomic view - everything flows from render_data()"""
    modal: Any  # The sensor/actuator/action object
    view_id: str
    view_type: str = "default"

    def render_data(self, data: Dict = None) -> Dict:
        """THE SOURCE OF TRUTH - returns raw data - OFFENSIVE!"""
        return self.modal.get_data()  # TRUST - crash if missing!

    def render_image(self, data: Dict = None) -> np.ndarray:
        """Create visualization FROM render_data() - OFFENSIVE!"""
        # Ask modal for data if not provided - MODAL-ORIENTED
        if data is None:
            data = self.modal.get_data()  # TRUST - crash if missing!

        raw_data = self.render_data(data)

        # Camera modals pass RGB directly for H.264 - OFFENSIVE duck typing!
        if data and 'nav_rgb' in data:  # Duck typing - if it has nav_rgb, use it!
            return data['nav_rgb'].astype(np.uint8)
        elif data and 'd405_rgb' in data:  # Duck typing - if it has d405_rgb, use it!
            return data['d405_rgb'].astype(np.uint8)

        # Generate visualization from raw_data
        return self._visualize_from_data(raw_data)

    def render_text(self, data: Dict = None) -> str:
        """Create description FROM render_data() - NO FAKE DATA!"""
        raw_data = self.render_data(data)
        # OFFENSIVE - generate from ACTUAL data, not fake hardcoded strings!
        return self._describe_from_data(raw_data)

    def render_rl(self, data: Dict = None) -> np.ndarray:
        """Create normalized vector FROM render_data() - OFFENSIVE!"""
        return self.modal.get_rl()  # TRUST - crash if missing!

    def render_ui(self, data: Dict = None) -> Dict:
        """Send render_data() to UI component"""
        raw_data = self.render_data(data)
        return {
            "component": f"ui_comp/{self.view_id}",
            "data": raw_data,  # UI gets the raw data!
            "type": self.view_type
        }

    def render_scene(self, data: Dict = None, extracted_state: Dict = None) -> Dict:
        """MODAL-ORIENTED: Trust modals to know themselves!

        Args:
            data: Legacy parameter (unused)
            extracted_state: Pre-extracted state to avoid duplicate extraction (PERFORMANCE!)

        NO type checks! NO translations! Modals are self-aware.
        Just ask them for their data - they know what to return.

        OFFENSIVE: Crashes if modal lacks get_data() - GOOD! Teaches developers.
        """
        # ================================================================
        # TRUST MODALS - They know themselves!
        # ================================================================
        # All modals (ActuatorComponent, sensors, assets) implement get_data()
        # They return their own correct state - no translation needed!

        # Guard: Skip if modal is None (can happen during cleanup)
        if self.modal is None:
            return {}

        # PERFORMANCE: Pass extracted_state to avoid duplicate extraction!
        if extracted_state is not None and hasattr(self.modal, 'get_data'):
            # Check if get_data() accepts extracted_state parameter
            import inspect
            sig = inspect.signature(self.modal.get_data)
            if 'extracted_state' in sig.parameters:
                return self.modal.get_data(extracted_state=extracted_state)

        return self.modal.get_data()

    # ============================================
    # DELETED: render_action() method
    # ============================================
    # REASON: VIEWS ARE FOR VIEWING, NOT COMMANDING!
    # Views should ONLY display/visualize data, not translate actions to commands.
    # That's the job of the action system or runtime engine, not views!
    # MOP VIOLATION: Views mixing display logic with command logic.

    def _visualize_from_data(self, raw_data: Dict) -> np.ndarray:
        """Generate type-appropriate visualization from raw data"""
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Lidar: polar plot - OFFENSIVE!
        if 'ranges' in raw_data:
            ranges = raw_data['ranges']  # OFFENSIVE - crash if missing!
            angles = raw_data['angles']  # OFFENSIVE - crash if missing!
            if ranges and angles:
                cx, cy = w//2, h//2
                for r, a in zip(ranges[:360], angles[:360]):
                    if r > 0:
                        x = int(cx + r * 40 * np.cos(a))
                        y = int(cy + r * 40 * np.sin(a))
                        if 0 <= x < w and 0 <= y < h:
                            brightness = min(255, int(255 * (1 - r/10)))
                            img[y, x] = [0, brightness, brightness//2]

        # Force sensors: dual bars - OFFENSIVE!
        elif 'force_left' in raw_data:
            left_force = raw_data['force_left']  # OFFENSIVE - crash if missing!
            right_force = raw_data['force_right']  # OFFENSIVE - crash if missing!
            left_height = int(h * min(left_force / 10, 1))
            right_height = int(h * min(right_force / 10, 1))

            # Left bar
            img[h-left_height:, :w//2-10] = [0, 255, 100] if left_force < 5 else [255, 200, 0] if left_force < 8 else [255, 50, 50]
            # Right bar
            img[h-right_height:, w//2+10:] = [0, 255, 100] if right_force < 5 else [255, 200, 0] if right_force < 8 else [255, 50, 50]

        # Actuator: position bar - OFFENSIVE!
        elif 'position' in raw_data and 'limits' in raw_data:
            pos = raw_data['position']  # OFFENSIVE - crash if missing!
            limits = raw_data['limits']  # OFFENSIVE - crash if missing!
            normalized = (pos - limits[0]) / (limits[1] - limits[0])
            filled = int(w * normalized)
            img[h//2-20:h//2+20, :filled] = [0, 150, 255]

        # Action: progress bar - OFFENSIVE!
        elif 'progress' in raw_data:
            progress = raw_data['progress']  # OFFENSIVE - crash if missing!
            filled = int(w * progress / 100)
            img[h//2-15:h//2+15, :filled] = [0, 255, 100]

        return img

    def _describe_from_data(self, raw_data: Dict) -> str:
        """Generate rich text description from raw data - OFFENSIVE!"""

        # Lidar description - OFFENSIVE!
        if 'ranges' in raw_data:
            ranges = raw_data['ranges']  # OFFENSIVE - crash if missing!
            if ranges:
                min_r = min(r for r in ranges if r > 0)
                obstacles = sum(1 for r in ranges if 0 < r < 3)
                return f"Lidar: {obstacles} obstacles within 3m, closest at {min_r:.2f}m, path clear forward"

        # Force description - OFFENSIVE!
        elif 'force_left' in raw_data:
            fl = raw_data['force_left']  # OFFENSIVE - crash if missing!
            fr = raw_data['force_right']  # OFFENSIVE - crash if missing!
            avg = (fl + fr) / 2
            contact = "both fingers" if raw_data['contact_left'] and raw_data['contact_right'] else "partial"
            return f"Gripper: {avg:.1f}N average force (L:{fl:.1f}N, R:{fr:.1f}N), {contact} contact, {'secure' if avg > 1 else 'light'} grasp"

        # Actuator description - OFFENSIVE!
        elif 'position' in raw_data and 'velocity' in raw_data:
            pos = raw_data['position']  # OFFENSIVE - crash if missing!
            vel = raw_data['velocity']  # OFFENSIVE - crash if missing!
            effort = raw_data['effort']  # OFFENSIVE - crash if missing!
            limits = raw_data['limits']  # OFFENSIVE - crash if missing!
            percent = ((pos - limits[0]) / (limits[1] - limits[0])) * 100
            return f"{self.view_id}: at {pos:.3f} ({percent:.0f}% range), velocity {vel:.3f}, effort {effort:.1f}%"

        # Action description - OFFENSIVE!
        elif 'target' in raw_data or 'target_position' in raw_data:
            target = raw_data['target'] if 'target' in raw_data else raw_data['target_position']
            current = raw_data['current'] if 'current' in raw_data else raw_data['current_position']
            progress = raw_data['progress']  # OFFENSIVE - crash if missing!
            eta = raw_data['eta']  # OFFENSIVE - crash if missing!
            return f"{self.view_id}: moving to {target:.3f} (at {current:.3f}), {progress:.0f}% done, {eta:.1f}s remaining"

        # IMU description - OFFENSIVE!
        elif 'linear_acceleration' in raw_data:
            accel = raw_data['linear_acceleration']  # OFFENSIVE - crash if missing!
            gyro = raw_data['angular_velocity']  # OFFENSIVE - crash if missing!
            return f"IMU: accel [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}] m/s², gyro [{gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}] rad/s"

        # Odometry description - OFFENSIVE!
        elif 'x' in raw_data and 'y' in raw_data:
            x = raw_data['x']  # OFFENSIVE - crash if missing!
            y = raw_data['y']  # OFFENSIVE - crash if missing!
            theta = raw_data['theta']  # OFFENSIVE - crash if missing!
            vx = raw_data['vx']  # OFFENSIVE - crash if missing!
            speed = np.sqrt(vx**2 + raw_data['vy']**2)  # OFFENSIVE - crash if vy missing!
            return f"Position: ({x:.2f}m, {y:.2f}m), heading {np.degrees(theta):.0f}°, speed {speed:.2f}m/s"

        # Audio description - OFFENSIVE!
        elif 'voice_activity' in raw_data:
            intensity = raw_data['sound_intensity']  # OFFENSIVE - crash if missing!
            direction = raw_data['direction_of_arrival']  # OFFENSIVE - crash if missing!
            speech = raw_data['detected_speech']  # OFFENSIVE - crash if missing!
            return f"Audio: {intensity:.0f}dB from {np.degrees(direction):.0f}°, speech: \"{speech}\"" if speech else f"Audio: {intensity:.0f}dB, no speech"

        return str(raw_data)[:100]

    def _normalize_for_rl(self, raw_data: Dict) -> np.ndarray:
        """Normalize raw data for RL training"""
        values = []

        # Extract numeric values and normalize
        for key, val in raw_data.items():
            if isinstance(val, (int, float)):
                values.append(float(val))
            elif isinstance(val, (list, tuple)) and len(val) > 0 and isinstance(val[0], (int, float)):
                values.extend([float(v) for v in val[:10]])  # Limit array size

        if not values:
            return np.zeros(10, dtype=np.float32)

        # Pad or truncate to fixed size
        if len(values) < 10:
            values.extend([0.0] * (10 - len(values)))
        else:
            values = values[:10]

        return np.array(values, dtype=np.float32)


# ============================================
# FACTORY FUNCTIONS
# ============================================

def create_sensor_view(sensor: Any, sensor_id: str) -> AtomicView:
    """Create view for any sensor modal - DUCK TYPING!

    View Types (3-type system):
    - "video" = Type 1 (cameras: video only, no CSV)
    - "video_and_data" = Type 2 (sensors: CSV + video)
    - "data" = Type 3 (system: JSON only)
    """
    # TYPE 1: Cameras are video-only (pixels ARE the data)
    # DUCK TYPING: If it has camera_type, it's a camera!
    if hasattr(sensor, 'camera_type'):
        return AtomicView(sensor, sensor_id, "video")

    # TYPE 2: All other sensors get data + video
    return AtomicView(sensor, sensor_id, "video_and_data")


def create_actuator_view(actuator: Any, actuator_id: str) -> AtomicView:
    """Create view for any actuator modal - MODAL-ORIENTED (trusts modals)

    TYPE 2: Actuators save both CSV data + visualization graphs
    """
    return AtomicView(actuator, actuator_id, "video_and_data")


def create_reward_view(reward_modal: Any, reward_id: str = "reward") -> AtomicView:
    """Create view for reward modal - MODAL-ORIENTED (for RL training!)

    TYPE 3: Rewards are data-only (JSON)
    """
    return AtomicView(reward_modal, reward_id, "data")


# NOTE: create_action_view() DELETED - actions tracked in ExecutionQueueModal now


# ============================================
# ANALYSIS VIEWS (process sensor data)
# ============================================

# ============================================
# ALL FAKE ANALYSIS & COMPOSITE VIEWS DELETED!
# ============================================
# REASON: These views compute fake analysis (edge detection, surface normals,
# occupancy grids) that don't exist as methods on camera/sensor modals.
# They were DEMO/PLACEHOLDER code with hardcoded logic.
#
# DELETED CLASSES:
# - NavEdgesView (edge detection)
# - NavDepthView (depth statistics)
# - D405SurfaceView (surface normals)
# - LidarOccupancyView (occupancy grid)
# - LidarHeatmapView (density heatmap)
# - LidarTrajectoryView (trajectory planning)
# - GridView (grid layout)
# - ActionStatusView (action status)
# - CollisionIndicatorView (collision detection)
# - ManipulationGridView (manipulation suite)
# - NavigationGridView (navigation suite)
# - get_view_dependencies() (dependency function)
#
# PURE MOP: If we need these features, implement them in the MODALS first!

# ============================================
# VIEW REGISTRY FUNCTION - DELETED
# ============================================
# get_all_views() DELETED - MOP violation!
# Views are created on-demand by ViewAggregator from modal instances.
# NO factory functions that create new modals!
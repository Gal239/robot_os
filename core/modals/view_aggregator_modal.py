"""
VIEW AGGREGATOR - Unified view distribution system for RuntimeEngine
OFFENSIVE & ELEGANT: Create views once, distribute to multiple consumers

Pattern: One view creation → many consumers (RL agent, LLM, streaming, UI, logging)
Views come from robot._view_factory() + system views (runtime status, action queue, rewards)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from ..modals.robot_modal import Robot


@dataclass
class ViewConsumer:
    """Consumer that receives view updates - OFFENSIVE

    Consumers can be:
    - RL agents (need specific sensor views)
    - LLMs (need scene understanding + action feedback)
    - Streaming servers (need camera views)
    - UI dashboards (need all views)
    - Loggers (need everything for replay)

    Filter controls which views consumer receives:
    - None: Receive all views
    - List[str]: Receive only views with names in list
    - Callable: Custom filter function (view_name, view_data) -> bool
    """
    name: str  # Consumer identifier (e.g., "rl_agent", "stream_server", "ui_dashboard")
    callback: Callable[[Dict[str, Any]], None]  # Function called with views dict
    filter: Optional[Any] = None  # None, List[str], or Callable[[str, Any], bool]

    def should_receive(self, view_name: str, view_data: Any) -> bool:
        """Check if consumer should receive this view - OFFENSIVE

        Args:
            view_name: Name of view (e.g., "nav_camera_view", "gripper_force_view")
            view_data: View data dict from view.render_scene()

        Returns:
            True if consumer wants this view
        """
        if self.filter is None:
            return True  # No filter = receive everything

        if isinstance(self.filter, list):
            return view_name in self.filter  # Whitelist

        if callable(self.filter):
            return self.filter(view_name, view_data)  # Custom filter

        return True  # Unknown filter type = pass through


class ViewAggregator:
    """Unified view distribution system - OFFENSIVE

    Design:
    1. RuntimeEngine calls create_views(robot, runtime_state) once per step
    2. ViewAggregator calls robot._view_factory() to create robot views
    3. ViewAggregator creates system views (runtime status, action queue, rewards)
    4. ViewAggregator calls distribute(views) to send to all consumers with filtering

    Benefits:
    - Views created once per step (not once per consumer!)
    - Consumers get filtered subset (RL agent doesn't get camera views if not needed)
    - Easy to add new consumers (stream server, logger, etc.)
    - System views included alongside robot views (complete observability)
    """

    def __init__(self):
        self.consumers: List[ViewConsumer] = []
        self.view_cache: Dict[str, Any] = {}  # Cache views between updates (async rates!)

    def add_consumer(self, name: str, callback: Callable[[Dict[str, Any]], None],
                     filter: Optional[Any] = None):
        """Add view consumer - OFFENSIVE

        Args:
            name: Consumer identifier
            callback: Function to call with views dict
            filter: None (all), List[str] (whitelist), or Callable (custom)

        Example:
            # RL agent only needs specific sensors
            aggregator.add_consumer(
                "rl_agent",
                lambda views: agent.observe(views),
                filter=["nav_camera_view", "lidar_view", "odometry_view"]
            )

            # Streaming server needs cameras
            aggregator.add_consumer(
                "stream_server",
                lambda views: stream.send(views),
                filter=lambda name, data: "camera" in name
            )

            # UI dashboard wants everything
            aggregator.add_consumer(
                "ui_dashboard",
                lambda views: ui.update(views),
                filter=None
            )
        """
        consumer = ViewConsumer(name=name, callback=callback, filter=filter)
        self.consumers.append(consumer)

    def has_consumers(self) -> bool:
        """Check if any consumers registered - OFFENSIVE"""
        return len(self.consumers) > 0

    def _add_metadata(self, view_data: Dict[str, Any], view_type: str, modal_ref: Any = None, modal_category: str = None) -> Dict[str, Any]:
        """Add metadata to view data - PURE MOP

        Args:
            view_data: Raw view data from modal.get_data()
            view_type: Type from AtomicView ("video", "video_and_data", "data")
            modal_ref: Reference to the modal object (for render_visualization())
            modal_category: Category of modal ("sensor", "actuator", or None for system views)

        Returns:
            View data with __meta__ field including modal reference and category
        """
        return {
            "__meta__": {
                "view_type": view_type,
                "modal_ref": modal_ref,  # For TimelineSaver to call render_visualization()
                "modal_category": modal_category  # For directory organization (sensors/ vs actuators/)
            },
            **view_data
        }

    def create_views(self, robot: Optional[Robot] = None,
                     scene: Optional[Any] = None,
                     runtime_state: Optional[Dict[str, Any]] = None,
                     model: Optional[Any] = None,
                     data: Optional[Any] = None,
                     update_cameras: bool = True,
                     update_sensors: bool = True,
                     event_log: Optional[Any] = None,
                     snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create all views (robot + scene + system) - OFFENSIVE & MODAL-ORIENTED

        Args:
            robot: Robot modal with _view_factory (None = no robot views)
            scene: Scene modal with room, assets, rewards (None = no scene views)
            runtime_state: Runtime state dict with step_count, reward, etc.
            model: MuJoCo model (for syncing asset positions)
            data: MuJoCo data (for syncing asset positions)
            event_log: EventLog modal (optional, for event tracking)
            snapshot: Optional MuJoCo data snapshot (for async view creation!)

        Returns:
            Complete views dict: {view_name: view_data}

        View sources:
        1. Robot views: From robot._view_factory() (sensor, actuator, action views)
        2. Scene views: AtomicView(room), AtomicView(asset), AtomicView(reward)
        3. System views: runtime_status, action_queue, event_log

        OFFENSIVE: Crashes if robot has no _view_factory
        MODAL-ORIENTED: Wraps modals in AtomicView, trusts their get_data()
        ASYNC: Use snapshot for thread-safe background view creation!
        """
        from ..modals.stretch.view_modals import AtomicView
        from ..modals.behavior_extractors import SnapshotData

        views = {}

        # ASYNC MODE: Use snapshot instead of live data (thread-safe!)
        if snapshot is not None:
            data_source = SnapshotData(snapshot, model)
            # MOP: PIGGYBACK snapshot to all components (like extraction_cache!)
            if scene:
                for asset in scene.assets.values():
                    for comp in asset.components.values():
                        comp._mujoco_snapshot = snapshot
            if robot:
                # Robot has actuators and sensors, not components
                for actuator in robot.actuators.values():
                    actuator._mujoco_snapshot = snapshot
                for sensor in robot.sensors.values():
                    sensor._mujoco_snapshot = snapshot
        else:
            data_source = data

        # 1. Robot views (if robot provided) - MOP: Robot provides components!
        if robot is not None:
            # MOP-CORRECT: Robot tells us what's viewable AND what type!
            # NO digging into robot internals (no `if comp_id in robot.sensors` checks)
            robot_views = {}

            # Get viewable components from robot: {comp_id: (modal, view_type, modal_category)}
            robot_view_metadata = {}  # Store modal_category for each view
            for comp_id, (comp_modal, comp_type, modal_category) in robot.get_viewable_components().items():
                view_name = f"{comp_id}_view"
                robot_views[view_name] = AtomicView(comp_modal, comp_id, comp_type)
                robot_view_metadata[view_name] = modal_category  # Store for later

            # Render each view to scene data (with async caching!)
            for view_name, view_modal in robot_views.items():
                # MOP-CORRECT: Auto-detect from view properties (NO robot access!)
                is_camera = 'camera' in view_name.lower()
                is_sensor = view_modal.view_type == "video_and_data" and robot_view_metadata[view_name] == "sensor"
                is_actuator = view_modal.view_type == "video_and_data" and robot_view_metadata[view_name] == "actuator"

                # Decide whether to render or reuse cached
                should_render = True
                if is_camera and not update_cameras:
                    should_render = False  # Skip camera rendering (expensive!)
                elif is_sensor and not update_sensors:
                    should_render = False  # Skip sensor rendering (medium cost)
                elif is_actuator:
                    should_render = True  # Always update actuators (cheap, change every step)

                if should_render:
                    # Render fresh view with metadata + modal reference + category (DISCOVERABLE!)
                    view_data = view_modal.render_scene()
                    modal_category = robot_view_metadata.get(view_name)
                    views[view_name] = self._add_metadata(view_data, view_modal.view_type, modal_ref=view_modal.modal, modal_category=modal_category)
                    self.view_cache[view_name] = views[view_name]  # Cache it!
                else:
                    # Reuse cached view
                    if view_name in self.view_cache:
                        views[view_name] = self.view_cache[view_name]
                    else:
                        # First time - must render even if flag says skip
                        view_data = view_modal.render_scene()
                        modal_category = robot_view_metadata.get(view_name)
                        views[view_name] = self._add_metadata(view_data, view_modal.view_type, modal_ref=view_modal.modal, modal_category=modal_category)
                        self.view_cache[view_name] = views[view_name]

        # 2. Scene views (if scene provided) - CACHE THESE TOO!
        if scene is not None:
            # Room view - cheap but still cache it
            room_view_name = "room_view"
            if room_view_name not in self.view_cache:
                room_view = AtomicView(modal=scene.room, view_id="room", view_type="data")  # TYPE 3
                room_data = room_view.render_scene()
                self.view_cache[room_view_name] = self._add_metadata(room_data, "data", modal_ref=scene.room)
            views[room_view_name] = self.view_cache[room_view_name]

            # Asset views - wrap each asset modal in AtomicView + merge extracted state
            # Assets MUST update every step because extracted_state changes!
            # FILTER: Skip room components and robot duplicates (already saved as robot views)
            SKIP_ASSETS = {
                'wall_north', 'wall_south', 'wall_east', 'wall_west',
                'floor', 'ceiling'
            }

            for asset_name, asset in scene.assets.items():
                # Skip room components (walls, floor, ceiling)
                if asset_name in SKIP_ASSETS:
                    continue

                # Skip robot component duplicates (e.g., stretch.arm, stretch.lift)
                # These are already saved as robot actuator/sensor views
                if '.' in asset_name and asset_name.split('.')[0] in ['stretch', 'fetch', 'pr2']:
                    continue

                asset_view = AtomicView(modal=asset, view_id=asset_name, view_type="data")  # TYPE 3

                # PERFORMANCE FIX: Pass extracted_state to render_scene() to skip duplicate extraction!
                # extracted_state already contains all component states extracted at 30Hz
                # Without this, get_data() re-extracts state on background thread (duplicate work!)
                asset_extracted_state = None
                if runtime_state and "extracted_state" in runtime_state:
                    extracted_state = runtime_state["extracted_state"]
                    if asset_name in extracted_state:
                        asset_extracted_state = extracted_state[asset_name]

                view_data = asset_view.render_scene(extracted_state=asset_extracted_state)

                views[f"asset_{asset_name}_view"] = self._add_metadata(view_data, "data", modal_ref=asset)

            # NEW: Camera views (from scene.cameras) - CameraModal support!
            if hasattr(scene, 'cameras') and scene.cameras:
                for camera_id, camera_modal in scene.cameras.items():
                    view_name = f"{camera_id}_view"

                    # SKIP if already created by robot.sensors (backward compat)
                    # This prevents duplicate views when camera is in both places
                    if view_name in views:
                        continue

                    # Only render if update_cameras=True (expensive!)
                    if update_cameras:
                        # NOTE: Camera rendering is done on MAIN THREAD by sync_sensors_from_backend()
                        # DO NOT render here - OpenGL contexts are NOT thread-safe!
                        # Just read pre-rendered image from camera_modal.rgb_image

                        # Get camera data (includes pre-rendered rgb_image)
                        camera_data = camera_modal.get_data()

                        # Add view with metadata
                        # NOTE: Use modal_category="sensor" for timeline compatibility
                        # (TimelineSaver expects "sensor" or "actuator", not "camera")
                        views[view_name] = self._add_metadata(
                            camera_data,
                            view_type="video",  # Cameras are video-only (TYPE 1)
                            modal_ref=camera_modal,
                            modal_category="sensor"  # Use "sensor" for timeline compatibility
                        )
                        self.view_cache[view_name] = views[view_name]
                    else:
                        # Reuse cached view (performance optimization)
                        if view_name in self.view_cache:
                            views[view_name] = self.view_cache[view_name]

            # Reward view - wrap reward modal in AtomicView (cheap, just condition state)
            if scene.reward_modal:
                reward_view = AtomicView(modal=scene.reward_modal, view_id="rewards", view_type="data")  # TYPE 3
                reward_data = reward_view.render_scene()
                views["rewards_view"] = self._add_metadata(reward_data, "data", modal_ref=scene.reward_modal)

        # 3. System views (if runtime_state provided)
        # NOTE: System views don't have modal_ref yet (Phase 6 TODO)
        if runtime_state is not None:
            # Runtime status view
            if "step_count" in runtime_state or "current_reward" in runtime_state:
                status_data = self._create_runtime_status_view(runtime_state)
                views["runtime_status"] = self._add_metadata(status_data, "data", modal_ref=None)  # TYPE 3

            # Action queue view
            if "action_queue" in runtime_state:
                queue_data = self._create_action_queue_view(runtime_state)
                views["action_queue"] = self._add_metadata(queue_data, "data", modal_ref=None)  # TYPE 3

        # 4. Event log view (if event_log provided) - ELEGANT
        if event_log is not None:
            # Event log is a modal - extract its data
            if hasattr(event_log, '__dict__'):
                # EventLog has events list - extract it
                event_data = {"events": [vars(e) if hasattr(e, '__dict__') else e for e in event_log.events]}
            else:
                event_data = {"event_log": str(event_log)}
            views["event_log"] = self._add_metadata(event_data, "data", modal_ref=None)  # TYPE 3

        return views

    def distribute(self, views: Dict[str, Any]):
        """Distribute views to all consumers with filtering - OFFENSIVE

        Args:
            views: Complete views dict from create_views()

        Each consumer receives filtered subset based on their filter
        """
        for consumer in self.consumers:
            # Filter views for this consumer
            filtered_views = {}

            for view_name, view_data in views.items():
                if consumer.should_receive(view_name, view_data):
                    filtered_views[view_name] = view_data

            # Call consumer callback with filtered views
            if filtered_views:  # Only call if consumer got any views
                try:
                    consumer.callback(filtered_views)
                except Exception as e:
                    # Don't crash RuntimeEngine if consumer fails
                    print(f"WARNING: Consumer '{consumer.name}' failed: {e}")

    # === SYSTEM VIEW CREATORS ===

    def _create_runtime_status_view(self, runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create runtime status view - OFFENSIVE

        Shows: step_count, current_reward, elapsed_time, physics_fps, etc.
        """
        status = {}

        if "step_count" in runtime_state:
            status["step_count"] = runtime_state["step_count"]

        if "current_reward" in runtime_state:
            status["current_reward"] = runtime_state["current_reward"]

        if "elapsed_time" in runtime_state:
            status["elapsed_time"] = runtime_state["elapsed_time"]

        if "start_time" in runtime_state:
            status["start_time"] = runtime_state["start_time"]

        return status

    def _create_action_queue_view(self, runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create action queue view - OFFENSIVE

        Shows: currently_executing (with details), pending_blocks (with actions), num_pending
        """
        queue_view = {}

        action_queue = runtime_state["action_queue"]  # OFFENSIVE - crash if missing!

        if "currently_executing" in action_queue:
            queue_view["currently_executing"] = action_queue["currently_executing"]

        if "pending_blocks" in action_queue:
            queue_view["pending_blocks"] = action_queue["pending_blocks"]

        if "num_pending" in action_queue:
            queue_view["num_pending"] = action_queue["num_pending"]

        # Backward compatibility
        if "pending_actions" in action_queue:
            queue_view["pending_actions"] = action_queue["pending_actions"]

        if "last_completed" in action_queue:
            queue_view["last_completed"] = action_queue["last_completed"]

        return queue_view

    def _create_reward_progress_view(self, runtime_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create reward progress view - OFFENSIVE

        Shows: conditions timeline, which conditions met, current scores
        """
        reward_view = {}

        conditions = runtime_state["reward_conditions"]  # OFFENSIVE - crash if missing!

        # Conditions status
        reward_view["conditions"] = []
        for condition_id, condition_data in conditions.items():
            reward_view["conditions"].append({
                "id": condition_id,
                "description": condition_data["description"],  # OFFENSIVE - crash if missing!
                "is_met": condition_data["is_met"],  # OFFENSIVE - crash if missing!
                "reward_value": condition_data["reward_value"],  # OFFENSIVE - crash if missing!
                "timestamp": condition_data.get("timestamp", None)  # LEGITIMATE - optional
            })

        # Total reward
        if "current_reward" in runtime_state:
            reward_view["total_reward"] = runtime_state["current_reward"]

        return reward_view

    # === CAMERA VIEW UTILITIES (For UI Snapshot Saving) ===

    @staticmethod
    def is_camera_view(view_name: str, view_data: dict) -> bool:
        """Check if view is a camera view - OFFENSIVE! (No fallbacks!)

        Args:
            view_name: View name string
            view_data: View data dict (REQUIRED - no optionals!)

        Returns:
            True if this is a camera view with RGB data

        MOP: Runtime layer owns view classification, not database layer!
        OFFENSIVE: Crashes if view lacks metadata (educational error!)
        """
        # OFFENSIVE: Views MUST have metadata!
        if "__meta__" not in view_data:
            raise KeyError(
                f"View '{view_name}' missing __meta__!\n"
                f"Views created by ViewAggregator MUST have metadata.\n"
                f"Did you create a view manually without using ViewAggregator?\n"
                f"Fix: Use ViewAggregator.create_views() to create all views!"
            )

        view_type = view_data["__meta__"].get("view_type")

        # OFFENSIVE: Metadata MUST have view_type!
        if view_type is None:
            raise KeyError(
                f"View '{view_name}' metadata missing 'view_type'!\n"
                f"ViewAggregator._add_metadata() MUST include view_type.\n"
                f"Fix: Check ViewAggregator._add_metadata() implementation!"
            )

        # Cameras are TYPE 1 (video only) - trust the metadata!
        return view_type == "video"

    def get_camera_views(self, views: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to camera views only - RUNTIME OPERATION

        Args:
            views: Complete views dict

        Returns:
            Filtered dict with only camera views

        MOP: ViewAggregator (runtime) classifies views, not external layers!
        """
        return {name: data for name, data in views.items()
                if self.is_camera_view(name, data)}

    def extract_camera_images(self, views: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract camera RGB images from views - RUNTIME OPERATION

        Args:
            views: Complete views dict

        Returns:
            Dict mapping camera view names to RGB numpy arrays

        MOP: Runtime layer extracts images, database layer just writes files!
        """
        import numpy as np

        camera_images = {}
        camera_views = self.get_camera_views(views)

        for view_name, view_data in camera_views.items():
            # Normalize image key access (handles "rgb", "rgb_image", "image")
            # OFFENSIVE: Check explicitly (can't use 'or' with numpy arrays!)
            rgb_data = view_data.get("rgb")
            if rgb_data is None:
                rgb_data = view_data.get("rgb_image")
            if rgb_data is None:
                rgb_data = view_data.get("image")

            # Verify it's a numpy array with data
            if rgb_data is not None and isinstance(rgb_data, np.ndarray) and rgb_data.size > 0:
                camera_images[view_name] = rgb_data

        return camera_images


# === USAGE PATTERN ===

"""
# Setup
aggregator = ViewAggregator()

# Add consumers
aggregator.add_consumer(
    "rl_agent",
    lambda views: print(f"RL got {len(views)} views"),
    filter=["nav_camera_view", "lidar_view"]
)

aggregator.add_consumer(
    "stream_server",
    lambda views: print(f"Stream got {len(views)} views"),
    filter=lambda name, data: "camera" in name
)

# In RuntimeEngine.step():
runtime_state = {
    "step_count": self.step_count,
    "current_reward": reward,
    "elapsed_time": time.time() - self.start_time,
    "action_queue": {
        "currently_executing": self.action_executor.currently_executing,
        "pending_actions": len(self.action_executor.queue_modal.blocks)
    },
    "reward_conditions": {...}
}

# Create and distribute views
views = aggregator.create_views(self.robot, runtime_state)
aggregator.distribute(views)
"""


# === TEST ===

if __name__ == "__main__":
    from ..main.robot_ops import create_robot

    print("=== Testing ViewAggregator ===")

    # Create aggregator
    aggregator = ViewAggregator()

    # Add mock consumers
    rl_views = []
    stream_views = []
    ui_views = []

    aggregator.add_consumer(
        "rl_agent",
        lambda views: rl_views.append(views),
        filter=["nav_camera_view", "lidar_view", "odometry_view"]
    )

    aggregator.add_consumer(
        "stream_server",
        lambda views: stream_views.append(views),
        filter=lambda name, data: "camera" in name
    )

    aggregator.add_consumer(
        "ui_dashboard",
        lambda views: ui_views.append(views),
        filter=None  # Everything
    )

    print(f"Added {len(aggregator.consumers)} consumers")

    # Create test robot
    robot = create_robot("stretch")
    print(f"Robot has {len(robot.views)} views")

    # Create runtime state
    runtime_state = {
        "step_count": 42,
        "current_reward": 150.5,
        "elapsed_time": 5.2,
        "action_queue": {
            "currently_executing": "arm_move_to(0.5)",
            "pending_actions": 3
        }
    }

    # Create and distribute views
    print("\nCreating views...")
    views = aggregator.create_views(robot, runtime_state)
    print(f"Created {len(views)} total views")

    print("\nDistributing views...")
    aggregator.distribute(views)

    print(f"\nRL agent received: {len(rl_views[0])} views")
    print(f"  View names: {list(rl_views[0].keys())}")

    print(f"\nStream server received: {len(stream_views[0])} views")
    print(f"  View names: {list(stream_views[0].keys())[:5]}...")

    print(f"\nUI dashboard received: {len(ui_views[0])} views")
    print(f"  Includes runtime_status: {'runtime_status' in ui_views[0]}")
    print(f"  Includes action_queue: {'action_queue' in ui_views[0]}")

    print("\n✓ ViewAggregator test complete!")

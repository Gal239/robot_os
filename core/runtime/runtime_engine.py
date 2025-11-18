"""
RUNTIME ENGINE - Unified orchestrator for robotics simulation/execution
OFFENSIVE & ELEGANT: One engine, all systems coordinated

Pattern: RuntimeEngine owns all subsystems, coordinates everything
Components: Backend, EventBus, ViewAggregator, 4 subsystems (Scene, Action, State, Reward)
"""

import time
import threading
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from .backend_interface import PhysicsBackend
from .event_bus import EventBus
from .async_writer import AsyncWriter
from ..modals.view_aggregator_modal import ViewAggregator
from ..modals.scene_modal import Scene
from ..modals.robot_modal import Robot
from ..modals.execution_queue_modal import ExecutionQueueModal
from ..modals.xml_resolver import XMLResolver
from ..modals.behavior_extractors import extract_component_state, SnapshotData
from ..modals.event_log_modal import EventLog
from ..modals.experiment_artifact_modal import ExperimentArtifactModal
from ..modals import registry


# === SUBSYSTEMS (extracted from ops) ===

class SceneCompiler:
    """Compiles scene to physics backend - OFFENSIVE

    Extracted from: scene_ops.compile() + _apply_states()
    """

    def __init__(self, backend: PhysicsBackend, registry):
        self.backend = backend
        self.registry = registry
        self.model = None
        self.data = None

    def compile(self, scene: Scene):
        """Compile scene XML to physics backend - OFFENSIVE"""
        # Generate XML
        xml = XMLResolver.build_scene_xml(scene, self.registry)

        # Compile to backend
        self.model, self.data = self.backend.compile_xml(xml)

        # Apply initial states (only for MuJoCo backends with real model/data)
        # MockBackend and other test backends skip this
        try:
            import mujoco
            if isinstance(self.model, mujoco.MjModel):
                self._apply_initial_states(scene)
                self._apply_dynamic_placements(scene)
        except (ImportError, AttributeError):
            pass  # Not a MuJoCo backend, skip initial states

        return self.model, self.data

    def _apply_initial_states(self, scene: Scene):
        """Apply initial joint states from placements - OFFENSIVE

        Extracted from: scene_ops._apply_states()
        """
        import mujoco

        assert self.model is not None and self.data is not None, "Compile first"

        for placement in scene.placements:
            if placement.initial_state is None:
                continue

            # Apply each joint value from initial_state
            for joint_name, joint_value in placement.initial_state.items():
                # Find joint ID in model
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id < 0:
                    raise ValueError(f"Joint '{joint_name}' not found in model for asset '{placement.asset}'")

                # Set qpos
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = joint_value

        # Forward kinematics to update positions
        mujoco.mj_forward(self.model, self.data)

    def _apply_dynamic_placements(self, scene: Scene):
        """Apply dynamic placements using actual MuJoCo positions - OFFENSIVE

        Handles both robot components (via placement_site) and furniture relations (via surface height)
        CRASHES if placement_site missing or site not in model. No fallbacks!
        """
        import mujoco
        import numpy as np

        for placement in scene.placements:
            # Only handle relation-based placements
            if not isinstance(placement.position, dict):
                continue

            rel = placement.position
            base_name = rel["relative_to"]  # OFFENSIVE - crash if missing!
            relation = rel["relation"]  # OFFENSIVE - crash if missing!

            if not base_name:
                continue

            # Determine target position based on base type
            target_pos = None

            # CASE 1: Robot component (has dot, e.g., "stretch.gripper")
            if "." in base_name:
                # Get component asset - OFFENSIVE, crashes if not found
                component_asset = scene.assets[base_name]
                placement_site = component_asset.config["placement_site"]  # CRASH if missing!

                # Query ACTUAL position from MuJoCo - OFFENSIVE
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, placement_site)
                assert site_id >= 0, f"Site '{placement_site}' not in model for '{base_name}'"

                # Get site position in world frame
                target_pos = self.data.site_xpos[site_id].copy()

                # Adjust for relation
                if relation == "in_gripper":
                    # IN_GRIPPER: Full whole-body IK solution (MOP!)
                    # ===============================================

                    # Get object body to measure width
                    object_body_name = placement.instance_name or placement.asset
                    object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)

                    # Discover object width from geoms (MOP - physics-based!)
                    object_width = self._get_object_width(object_body_id)

                    # Solve IK for whole-body grasp (MOP - geometry-based!)
                    from ..modals.reachability_modal import ReachabilityModal
                    ik_modal = ReachabilityModal()
                    joint_values = ik_modal.solve_for_grasp(
                        target_pos=target_pos,
                        object_width=object_width,
                        robot_config=component_asset.config
                    )

                    # Apply joint values to robot placement (set initial_state!)
                    # Find robot placement in scene (look for robot name, e.g., "stretch")
                    robot_name = base_name.split('.')[0]  # "stretch.gripper" → "stretch"
                    for p in scene.placements:
                        if p.asset == robot_name:
                            # Merge with existing initial_state
                            if p.initial_state is None:
                                p.initial_state = {}
                            p.initial_state.update(joint_values)

                            # Special: Handle base position (not a joint!)
                            if "_base_position" in joint_values:
                                base_pos = joint_values["_base_position"]
                                p.position = base_pos  # Update robot placement position

                            break

                    # Object position stays at gripper center (target_pos already correct)

            # CASE 2: Furniture/asset (no dot, e.g., "table")
            else:
                # Find base object body
                base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_name)
                if base_body_id < 0:
                    continue  # Base not found, skip

                # Get base body position
                base_pos = self.data.xpos[base_body_id].copy()

                # Calculate target based on relation
                if relation == "on_top":
                    # Query actual surface height from geoms
                    surface_height = self._get_surface_height(base_body_id)
                    target_pos = base_pos.copy()
                    target_pos[2] = surface_height + 0.05  # 5cm above surface

                elif relation in ["next_to", "front", "back", "left", "right"]:
                    # For horizontal relations, use table surface height as Z
                    surface_height = self._get_surface_height(base_body_id)
                    target_pos = base_pos.copy()
                    target_pos[2] = surface_height + 0.05

                    # Apply horizontal offset (use distance from rel or default)
                    distance = rel.get("distance", 0.2)  # LEGITIMATE - has default
                    if relation == "front":
                        target_pos[1] += distance
                    elif relation == "back":
                        target_pos[1] -= distance
                    elif relation == "left":
                        target_pos[0] -= distance
                    elif relation == "right":
                        target_pos[0] += distance

                else:
                    # Default: use base position
                    target_pos = base_pos

            if target_pos is None:
                continue

            # Find object's freejoint - OFFENSIVE
            object_body_name = placement.instance_name or placement.asset
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
            assert body_id >= 0, f"Body '{object_body_name}' not found"

            # Get first joint attached to this body
            jnt_adr = self.model.body_jntadr[body_id]
            if jnt_adr < 0:
                continue  # No joint (fixed body like table/wall)

            # Check if it's a freejoint (only reposition free objects, not fixed furniture)
            jnt_type = self.model.jnt_type[jnt_adr]
            if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
                continue  # Not a freejoint, skip

            # Set freejoint qpos (7 DOF: 3 XYZ + 4 quaternion)
            qpos_addr = self.model.jnt_qposadr[jnt_adr]
            self.data.qpos[qpos_addr:qpos_addr + 3] = target_pos  # Position
            self.data.qpos[qpos_addr + 3:qpos_addr + 7] = [1, 0, 0, 0]  # Identity quaternion (w,x,y,z)

            # Zero velocity to prevent flying away
            qvel_addr = self.model.jnt_dofadr[jnt_adr]
            self.data.qvel[qvel_addr:qvel_addr + 6] = 0  # 6 DOF velocity (3 linear + 3 angular)

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def _get_object_width(self, body_id: int) -> float:
        """Get object X dimension from geoms (MOP - discovered!)

        Measures bounding box width in X direction.
        Used for gripper opening calculation.

        Args:
            body_id: MuJoCo body ID

        Returns:
            Width in meters (X dimension)
        """
        import mujoco
        import numpy as np

        min_x = np.inf
        max_x = -np.inf

        # Find all geoms attached to this body
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] == body_id:
                geom_pos = self.data.geom_xpos[geom_id]
                geom_size = self.model.geom_size[geom_id]
                geom_type = self.model.geom_type[geom_id]

                # Calculate X extent based on geom type
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    # Box: size[0] is half-width in X
                    min_x = min(min_x, geom_pos[0] - geom_size[0])
                    max_x = max(max_x, geom_pos[0] + geom_size[0])
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    # Sphere: size[0] is radius
                    min_x = min(min_x, geom_pos[0] - geom_size[0])
                    max_x = max(max_x, geom_pos[0] + geom_size[0])
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    # Cylinder: size[0] is radius
                    min_x = min(min_x, geom_pos[0] - geom_size[0])
                    max_x = max(max_x, geom_pos[0] + geom_size[0])
                else:
                    # Other geom types: use position as approximate
                    min_x = min(min_x, geom_pos[0])
                    max_x = max(max_x, geom_pos[0])

        # Return width (or default if no geoms found)
        return max_x - min_x if max_x > min_x else 0.05  # Default 5cm

    def _get_surface_height(self, body_id: int) -> float:
        """Get the maximum Z position of all geoms in a body and its children (surface height)"""
        import mujoco
        import numpy as np

        max_z = -np.inf

        # Get all bodies in this subtree (body + all descendants)
        body_ids = self._get_body_subtree(body_id)

        # Iterate through all geoms in the subtree
        for geom_id in range(self.model.ngeom):
            if self.model.geom_bodyid[geom_id] in body_ids:
                # Get geom position and size
                geom_pos = self.data.geom_xpos[geom_id]
                geom_size = self.model.geom_size[geom_id]
                geom_type = self.model.geom_type[geom_id]

                # Calculate top surface Z based on geom type
                if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                    top_z = geom_pos[2] + geom_size[2]  # Box: size[2] is half-height
                elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                    top_z = geom_pos[2] + geom_size[1]  # Cylinder: size[1] is half-length
                elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    top_z = geom_pos[2] + geom_size[0]  # Sphere: size[0] is radius
                else:
                    top_z = geom_pos[2]  # Default: just position

                max_z = max(max_z, top_z)

        return max_z if max_z > -np.inf else 0.0

    def _get_body_subtree(self, body_id: int) -> set:
        """Get all body IDs in subtree (body + all descendants)"""
        result = {body_id}

        # Find all children recursively
        for child_id in range(self.model.nbody):
            if self.model.body_parentid[child_id] == body_id:
                result.update(self._get_body_subtree(child_id))

        return result


class ActionExecutor:
    """Executes robot actions with real actuator feedback - OFFENSIVE

    Extracted from: execution_ops.py
    """

    def __init__(self):
        self.queue_modal = ExecutionQueueModal()
        # MOP-CORRECT: Queue modal IS the single source of truth!
        # NO duplicate state - read from queue_modal.actuators[id].current_action

    def init_from_robot(self, robot: Robot):
        """Initialize with robot's actuators"""
        self.queue_modal.init_from_actuators(robot.actuators)

    def submit_block(self, block) -> int:
        """Submit action block for execution"""
        return self.queue_modal.submit_block(block)

    def tick(self, robot: Robot, event_log=None) -> Dict[str, Any]:
        """PURE MOP: Queue modal self-coordinates, runtime is DUMB!

        This is the ELEGANT MOP solution:
        - Queue modal calls each action.tick() ONCE
        - Actions return Dict[str, float] (uniform interface!)
        - Runtime just adds hold commands for idle actuators

        NO isinstance checks, NO glue code, PURE modal-to-modal communication!

        Args:
            robot: Robot with actuators and sensors
            event_log: EventLog modal instance for action event emission (MOP!)

        Returns:
            Dict of commands to send to actuators
        """
        # MOP: Queue modal self-coordinates! Get all commands from executing actions
        commands = self.queue_modal.get_next_commands(robot, event_log)



        # Add STOP commands for idle velocity actuators (wheels)
        # Position actuators (arm, lift, gripper) don't need HOLD - MuJoCo maintains last position automatically
        for actuator_id, actuator in robot.actuators.items():
            # MOP: Auto-discover virtual actuators (no joints = not in MuJoCo!)
            if not actuator.joint_names:
                continue  # Skip virtual actuators (speaker, etc.)

            # Only send commands to velocity actuators (wheels) when idle
            if actuator_id not in commands:
                # CRITICAL: Velocity actuators (wheels) MUST send 0.0 to stop!
                # Position actuators hold automatically (MuJoCo remembers last ctrl value)
                if 'wheel' in actuator_id or 'robot_wheel' in str(actuator.behaviors):
                    commands[actuator_id] = 0.0  # STOP wheels
                # Position actuators (arm, lift, gripper, wrist): No command = hold last position

        return commands

    def get_status(self) -> Dict:
        """Get complete system status"""
        return self.queue_modal.get_summary()


class StateExtractor:
    """Extracts semantic state for rewards - OFFENSIVE

    Extracted from: scene_ops.get_current_state_for_rewards()
    """

    def __init__(self, backend: PhysicsBackend):
        self.backend = backend

    def extract(self, scene: Scene, robot: Optional[Robot] = None, event_log=None, snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Extract MuJoCo state using behavior_extractors - OFFENSIVE

        Returns state dict in format: state[asset_name][property_name] = value

        UNIFORM: All entries in scene.assets are Assets with .components dict

        Args:
            scene: Scene with assets to extract state from
            robot: Robot with sensors (for self-validation!)
            event_log: EventLog to track behavior changes (ALWAYS-ON!)
            snapshot: Optional MuJoCo data snapshot (for thread-safe async extraction)
        """
        state = {}

        # Check if this is a real MuJoCo backend (not MockBackend)
        # MockBackend can't handle behavior extraction
        import mujoco
        if not isinstance(self.backend.model, mujoco.MjModel):
            return state  # Return empty state for non-MuJoCo backends

        # Use snapshot if provided (async extraction), otherwise use live data
        if snapshot is not None:
            data_source = SnapshotData(snapshot, self.backend.model)
        else:
            data_source = self.backend.data

        # ================================================================
        # BUILD EXTRACTION CACHE - ONCE per extraction (30Hz)
        # ================================================================
        # MOP PRINCIPLE: Auto-discover ONCE, reuse everywhere!
        # - Contacts: O(ncon) scan → reused for all O(n²) contact queries
        # - Name→ID: Build ALL mappings ONCE → eliminates 6.15M mj_name2id() calls!
        # - Body→Geoms: Reverse mapping → eliminates 132K list comprehensions!
        # - Geom sizes: Cached → eliminates 850K size lookups!
        # Rebuilt at 30Hz (state extraction rate), not 200Hz (physics rate) → 6x fewer builds!
        from simulation_center.core.modals.behavior_extractors import build_extraction_cache
        extraction_cache = build_extraction_cache(self.backend.model, data_source)

        # PERFORMANCE: Extract ONLY tracked assets (selective state extraction)!
        # Assets become tracked when:
        #   1. add_reward() is called on them (marks _is_tracked=True)
        #   2. add_asset(..., is_tracked=True) is used
        #   3. They're the robot (always tracked)
        # This prevents extracting state for 100+ background objects when only 5 are reward-relevant
        tracked_assets = scene.get_tracked_assets()  # Only tracked assets

        # MOP: PIGGYBACK cache to ALL asset components (auto-discover ONCE, reuse everywhere!)
        # PERFORMANCE: Must give cache to ALL assets (not just tracked), because sync_assets_from_backend()
        # syncs ALL assets at throttled rate. Non-tracked assets without cache do expensive mj_name2id() lookups!
        # This prevents 2,000+ extra name lookups per 500 steps (walls, floor, ceiling, table, etc.)
        for asset in scene.assets.values():
            for comp in asset.components.values():
                comp._extraction_cache = extraction_cache

        # ================================================================
        # EXTRACT STATE - SELECTIVE for tracked assets only!
        # ================================================================
        # Only process assets that are tracked (robot + reward targets + tracked objects)
        # ALL assets have .components - no special cases!

        for asset_name, asset in tracked_assets.items():
            asset_state = {}

            # Get asset config for unit conversion - OFFENSIVE!
            # For robot components, extract robot type from name
            # For room parts (walls, floor), use asset.config (already has everything)
            if asset.config.get('category') == 'room_parts':
                # Room parts: use existing config (no registry lookup needed)
                asset_config = asset.config
            elif "." in asset_name:
                # Robot components: load robot config
                robot_name = asset_name.split(".")[0]
                asset_config = registry.load_asset_config(robot_name)
            else:
                # Regular assets: load from registry using asset TYPE, not instance name!
                # MOP: asset.name = asset type (e.g., "wood_block")
                #      asset_name = instance ID (e.g., "block_red")
                asset_config = registry.load_asset_config(asset.name)

            # Extract state for EACH component - UNIFORM + SELF-VALIDATING!
            for comp_name, comp in asset.components.items():
                comp_state = extract_component_state(
                    self.backend.model,
                    data_source,  # Use snapshot or live data (thread-safe!)
                    comp,
                    tracked_assets,  # PERFORMANCE: Pass tracked assets only (not all assets!)
                    asset_config,  # Pass config for unit conversion
                    event_log=event_log,  # Pass event log for tracking (optional)
                    asset_id=asset_name,  # Pass asset name for event source
                    robot=robot,  # Pass robot for sensor access (self-validation!)
                    contact_cache=extraction_cache  # Pass extraction cache (contacts + name→ID + sizes)!
                )
                # MOP: Preserve component hierarchy! Asset contains components.
                # state["table"]["table_surface"] = {...}
                # state["apple"]["body"] = {...}
                asset_state[comp_name] = comp_state

            if asset_state:
                state[asset_name] = asset_state

        return state


class RewardComputer:
    """Computes rewards from conditions - OFFENSIVE

    Extracted from: scene_ops.get_total_reward()
    """

    def compute(self, state: Dict[str, Any], reward_modal,
                current_time: float = None, start_time: float = 0.0, scene=None) -> Dict[str, Any]:
        """Get reward dict for this step - PURE MOP!

        OFFENSIVE FIX: RewardModal.step() returns rich dict with delta + total + breakdown!
        Don't reimplement - call the modal's method (MOP principle).

        Returns:
            Dict with:
                "delta": Points earned THIS step
                "total": Cumulative total
                "rewards": Per-reward breakdown
        """
        # PURE MOP: RewardModal knows how to compute its own step reward!
        # It handles both discrete (one-time) and smooth (delta) correctly
        # Returns dict with delta, total, and per-reward breakdown
        return reward_modal.step(state, current_time, scene=scene)


# === RUNTIME ENGINE ===

class RuntimeEngine:
    """Unified orchestrator for robotics simulation/execution - OFFENSIVE

    Design:
    - Owns backend (swappable: MuJoCo/Real/Isaac Gym)
    - Owns event bus (publish-subscribe)
    - Owns view aggregator (create once, serve many)
    - Owns 4 subsystems (compile, execute, extract, reward)
    - Coordinates everything through step() loop

    Usage:
        engine = RuntimeEngine(backend=MuJoCoBackend())
        engine.load_experiment(experiment)
        engine.add_consumer("stream", stream_ops.write_views)
        engine.on("step", lambda e: print(f"Step {e.data['step_count']}"))
        engine.run(steps=1000)
    """

    def __init__(self, backend: PhysicsBackend, camera_fps: int = 30, sensor_hz: int = 30, enable_recording: bool = True, timeline_fps: float = 30.0, state_extraction_hz: int = 200, step_rate: int = None, fast_mode: bool = None, camera_width: int = 640, camera_height: int = 480, camera_shadows: bool = True, camera_reflections: bool = True):
        """Initialize RuntimeEngine with physics backend - OFFENSIVE

        Args:
            backend: PhysicsBackend implementation (MuJoCo, Real, Isaac Gym, etc.)
            camera_fps: Camera update rate (Hz) - default 30fps (unified with physics for humanoids!)
            sensor_hz: Sensor update rate (Hz) - default 30Hz (unified with physics!)
            enable_recording: Enable experiment artifact recording (default True)
            timeline_fps: Enable TIME TRAVELER timeline saving at FPS rate (30 = default unified rate)
            state_extraction_hz: State extraction rate (Hz) - 10 for fast RL, 200 for debugging
            step_rate: RL agent observation + action frequency (Hz) - defaults to state_extraction_hz (sim-to-real!)
            fast_mode: DEPRECATED - ignored (use state_extraction_hz instead)
            camera_width: Camera render width in pixels (default 640, 1280 for demo mode)
            camera_height: Camera render height in pixels (default 480, 720 for demo mode)
            camera_shadows: Enable shadows in camera rendering (default True, disable for 2x faster!)
            camera_reflections: Enable reflections in camera rendering (default True, disable for faster!)
        """
        self.backend = backend
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_shadows = camera_shadows
        self.camera_reflections = camera_reflections

        # BACKWARD COMPATIBILITY: Warn if fast_mode is used
        if fast_mode is not None:
            import warnings
            warnings.warn(
                "RuntimeEngine fast_mode is deprecated. Use state_extraction_hz instead:\n"
                "  fast_mode=True  → state_extraction_hz=10\n"
                "  fast_mode=False → state_extraction_hz=200",
                DeprecationWarning,
                stacklevel=2
            )

        # Core systems
        self.event_bus = EventBus()
        self.view_aggregator = ViewAggregator()
        self.event_log = EventLog()  # Event tracking system

        # ASYNC I/O: Background writer for non-blocking view extraction
        # Prevents blocking simulation for expensive data collection operations
        self.async_writer = AsyncWriter(maxsize=200, name="ViewExtractor")

        # View update rates (async like reality!)
        self.camera_fps = camera_fps
        self.sensor_hz = sensor_hz
        self._camera_update_interval = None  # Calculated after compile (need physics_dt)
        self._sensor_update_interval = None

        # PERFORMANCE: State extraction throttling based on render_mode
        self.state_extraction_hz = state_extraction_hz  # Controlled by ExperimentOps render_mode
        self._state_update_interval = None
        self._last_state_update_step = -1
        self._last_state = {}  # Cache last extracted state
        self._state_lock = threading.Lock()  # Protect _last_state updates from async thread

        # SIM-TO-REAL: RL agent step rate (observation + action frequency)
        # Defaults to state_extraction_hz if not specified (agent steps when observations available)
        self.step_rate = step_rate if step_rate is not None else state_extraction_hz

        # Experiment recording (TIME-TRAVEL DEBUGGING!)
        self.enable_recording = enable_recording
        self.artifact = None  # ExperimentArtifactModal instance

        # TIME TRAVELER - Timeline saving
        self.timeline_fps = timeline_fps
        self.timeline_saver = None  # TimelineSaver instance (created in load_experiment)

        # FPS throttling for view rendering (only render when saving!)
        self.last_view_save_step = -1
        self.steps_per_view_save = 1  # Will be set after backend compiled

        # State
        self.experiment = None
        self.scene = None
        self.robot = None
        self.step_count = 0
        self.start_time = None
        self.current_reward_dict = {}  # Reward dict from modal (delta + total + breakdown)
        self.last_views = {}  # Cache views for query access

        # Subsystems
        self.action_executor = ActionExecutor()  # Runtime infrastructure, NOT experiment-specific!

        # Experiment-specific subsystems (initialized on load_experiment)
        self.scene_compiler = None
        self.state_extractor = None
        self.reward_computer = None

        # Camera backends for sensors (created after model compilation)
        # Each sensor gets its own backend - clean abstraction for sim-to-real
        self.camera_backends = {}

    def __del__(self):
        """Automatic cleanup when RuntimeEngine is destroyed - OFFENSIVE!

        Ensures timeline videos are finalized and resources released.
        Called automatically by Python garbage collector.
        """
        # SKIP async writer close - it's a daemon thread for view extraction (not critical)
        # Waiting for it can cause hangs with large queues
        # Just let it die naturally when process exits

        # CRITICAL: Close all camera backends to release resources!
        # Backends may hold EGL contexts, renderers, or real camera handles
        if hasattr(self, 'camera_backends') and self.camera_backends:
            try:
                for sensor_name, backend in self.camera_backends.items():
                    backend.close()
            except Exception as e:
                print(f"⚠️  Camera backend cleanup warning: {e}")

        # Close timeline saver to finalize videos
        if hasattr(self, 'timeline_saver') and self.timeline_saver is not None:
            try:
                self.timeline_saver.close()
            except Exception as e:
                # Silent fail in destructor (bad practice to raise in __del__)
                print(f"⚠️  Timeline cleanup warning: {e}")

        # Close frame snapshot saver to finalize snapshots
        if hasattr(self, 'frame_snapshot_saver') and self.frame_snapshot_saver is not None:
            try:
                self.frame_snapshot_saver.close()
            except Exception as e:
                print(f"⚠️  Frame snapshot cleanup warning: {e}")

        # Close backend viewer if any
        if hasattr(self, 'backend') and self.backend is not None:
            try:
                if hasattr(self.backend, 'cleanup'):
                    self.backend.cleanup()
            except Exception:
                pass  # Silent fail

    def load_experiment(self, experiment, experiment_dir: Optional[Path] = None,
                       db_ops=None, experiment_ops=None):
        """Load experiment and compile scene - OFFENSIVE

        Args:
            experiment: Experiment object with scene, robot, rewards
            experiment_dir: Optional experiment directory for artifact recording
            db_ops: DatabaseOps instance (for mujoco package saving) - NEW
            experiment_ops: ExperimentOps instance (for package creation flag) - NEW
        """
        self.experiment = experiment
        self.scene = experiment.scene
        self.robot = experiment.robot if hasattr(experiment, 'robot') else None

        # Initialize experiment artifact recording (EXECUTION HISTORY!)
        # Note: Configuration saved by GOD Modal (ExperimentModal)
        # This only records execution history (what happens each step)
        if self.enable_recording and experiment_dir:
            self.artifact = ExperimentArtifactModal(
                experiment_dir=Path(experiment_dir),
                enabled=True
            )
            # No record_initial_state - GOD Modal owns configuration!

        # Initialize TIME TRAVELER timeline saving
        # ALWAYS save timeline (even in fast_mode) for debugging - we need the data!
        if self.timeline_fps > 0 and experiment_dir:
            from .timeline_saver import TimelineSaver

            # Get physics timestep from backend
            timestep = self.backend.model.opt.timestep if hasattr(self.backend, 'model') and self.backend.model else 0.005

            # Calculate FPS throttling for view rendering
            # Only render views when we're saving (not every step!)
            self.steps_per_view_save = int((1.0 / self.timeline_fps) / timestep) if self.timeline_fps > 0 else 1

            self.timeline_saver = TimelineSaver(
                experiment_dir=str(experiment_dir),
                global_fps=self.timeline_fps,
                timestep=timestep,
                backend=self.backend,  # NEW: For scene state access
                db_ops=db_ops,  # NEW: For saving package/state
                experiment_ops=experiment_ops  # NEW: For package creation flag
            )

            # Register as ViewAggregator consumer
            self.view_aggregator.add_consumer(
                "timeline_saver",
                self.timeline_saver.save_frame,
                filter=None  # Save all views!
            )

            # Initialize Frame Snapshot Saver (TRUE TIME TRAVELER!)
            # Saves complete per-frame snapshots (all modals + MuJoCo + videos)
            # Note: Will be fully initialized after subsystems are created
            self.frame_snapshot_saver = None
            self._experiment_dir_for_snapshots = experiment_dir

        # Initialize experiment-specific subsystems
        self.scene_compiler = SceneCompiler(self.backend, registry)
        self.state_extractor = StateExtractor(self.backend)
        self.reward_computer = RewardComputer()

        # Compile scene
        print(f"Compiling scene '{self.scene.room.name}'...")
        self.scene_compiler.compile(self.scene)
        # Quiet mode - removed verbose print

        # Create shared renderer for all cameras (ONE renderer, ONE EGL context)
        # This prevents "Failed to make the EGL context current" errors
        # Use camera resolution from render_mode (1280x720 for demo, 640x480 for others)
        import mujoco

        # OFFENSIVE: Check for viewer/camera conflict BEFORE creating renderer!
        # Viewer (GLFW) and cameras (EGL) cannot coexist in the same process
        if self.backend.enable_viewer and self.robot:
            # Check robot sensors for cameras (behavior-based detection)
            # Any sensor with 'rgb_image' attribute is a camera sensor
            has_robot_cameras = any(
                hasattr(sensor, 'rgb_image')  # Camera = has images!
                for sensor in self.robot.sensors.values()
            )

            # Check scene for virtual MuJoCo cameras (CameraModal)
            has_scene_cameras = bool(self.scene.cameras) if hasattr(self.scene, 'cameras') else False

            has_cameras = has_robot_cameras or has_scene_cameras

            if has_cameras:
                raise RuntimeError(
                    "❌ Cannot use camera sensors with viewer mode!\n"
                    "\n"
                    "   CONFLICT: Viewer uses GLFW rendering context.\n"
                    "             Cameras need EGL rendering context.\n"
                    "             MuJoCo only allows ONE context per process.\n"
                    "\n"
                    "   WHY: The viewer window IS your camera - you see the simulation!\n"
                    "        Camera sensors are for headless mode (recording/streaming).\n"
                    "\n"
                    "   FIX 1: Use headless=True for camera sensors (recommended)\n"
                    "   FIX 2: Remove camera sensors if you just want viewer\n"
                    "\n"
                    "   Example: ops = ExperimentOps(headless=True)  # Cameras work!\n"
                    "            ops = ExperimentOps(headless=False) # Viewer works, no cameras!\n"
                )

        # AUTODISCOVER camera backends - NO HARDCODING! PURE MOP!
        # Each sensor camera gets its own backend matched to MuJoCo camera

        self.camera_backends = {}  # Maps sensor_name -> CameraBackend

        # PERFORMANCE OPTIMIZATION: Skip expensive camera autodiscovery if no cameras exist
        # This is critical for rl_core mode performance (2.3-2.6x real-time)
        import inspect

        # PRE-INSPECT ALL SENSORS ONCE (200Hz → 1Hz optimization!)
        # Cache signature inspection results in backend to avoid inspect.signature() every step
        # This saves ~1.3ms per step = 1.3s per 1000 steps!
        if self.robot:
            for sensor_name, sensor in self.robot.sensors.items():
                if hasattr(sensor, 'sync_from_mujoco'):
                    sig = inspect.signature(sensor.sync_from_mujoco)
                    # Cache both checks at once
                    self.backend._sensor_supports_event_log[sensor_name] = ('event_log' in sig.parameters)
                    self.backend._sensor_supports_camera_backend[sensor_name] = ('camera_backend' in sig.parameters)

        # Check for robot camera sensors (using cached results!)
        has_robot_cameras = False
        if self.robot:
            has_robot_cameras = any(
                self.backend._sensor_supports_camera_backend.get(name, False)
                for name in self.robot.sensors.keys()
            )

        # Check for scene free cameras
        has_scene_cameras = hasattr(self.scene, 'cameras') and bool(self.scene.cameras)

        # Early exit if NO cameras at all - saves ~50% time in rl_core mode!
        if not has_robot_cameras and not has_scene_cameras:
            print(f"⚡ PERFORMANCE: Skipping camera autodiscovery (no cameras in rl_core mode)")
        else:
            # Only import expensive camera backends if we actually need them
            from .camera_backends import MuJoCoRenderingBackend
            import mujoco

            # First, discover ALL cameras in MuJoCo model
            available_cameras = []
            for i in range(self.scene_compiler.model.ncam):
                cam_name = mujoco.mj_id2name(self.scene_compiler.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    available_cameras.append(cam_name)

            print(f"🔍 Autodiscovered {len(available_cameras)} MuJoCo cameras: {available_cameras}")

            # Now autodiscover camera sensors from robot and match them
            # NO HARDCODED CLASS NAMES - pure autodiscovery via interface inspection!
            if self.robot:
                for sensor_name, sensor in self.robot.sensors.items():
                    # Check if this is a camera sensor (has camera backend support)
                    # PERFORMANCE: Use cached signature inspection (already done at line 765!)
                    is_camera_sensor = self.backend._sensor_supports_camera_backend.get(sensor_name, False)

                    if is_camera_sensor:
                        print(f"🎥 Found camera sensor: {sensor_name}")

                        # Try to find matching MuJoCo camera
                        # Pattern: sensor_name might be 'nav_camera', MuJoCo camera is 'nav_camera_rgb'
                        possible_names = [
                            f"{sensor_name}_rgb",
                            sensor_name,
                            sensor_name.replace('_camera', '_rgb'),  # 'nav_camera' -> 'nav_rgb'
                            f"d435i_camera_rgb" if 'nav' in sensor_name else None,  # Legacy nav name
                            f"d405_rgb" if 'd405' in sensor_name else None,  # Short d405 name
                        ]
                        possible_names = [n for n in possible_names if n]  # Remove None

                        matched_camera = None
                        for cam_name in possible_names:
                            if cam_name in available_cameras:
                                matched_camera = cam_name
                                break

                        if matched_camera:
                            # Create backend for this sensor!
                            try:
                                backend = MuJoCoRenderingBackend.create_sensor_camera(
                                    self.scene_compiler.model,
                                    self.scene_compiler.data,
                                    camera_name=matched_camera,
                                    width=self.camera_width,
                                    height=self.camera_height,
                                    enable_depth=True,
                                    shadows=self.camera_shadows,
                                    reflections=self.camera_reflections
                                )
                                self.camera_backends[sensor_name] = backend
                                quality = "HIGH QUALITY" if (self.camera_shadows and self.camera_reflections) else "FAST (no shadows/reflections)"
                                print(f"✅ Created backend for '{sensor_name}' → MuJoCo camera '{matched_camera}' ({self.camera_width}x{self.camera_height}, {quality})")
                            except Exception as e:
                                raise RuntimeError(
                                    f"❌ FAILED to create backend for sensor '{sensor_name}'!\n"
                                    f"   Matched MuJoCo camera: '{matched_camera}'\n"
                                    f"   Error: {e}\n"
                                    f"   Available cameras: {available_cameras}"
                                )
                        else:
                            # NO MATCH FOUND - CRASH WITH CLEAR ERROR!
                            raise RuntimeError(
                                f"❌ Camera sensor '{sensor_name}' has NO matching MuJoCo camera!\n"
                                f"\n"
                                f"   Tried names: {possible_names}\n"
                                f"   Available MuJoCo cameras: {available_cameras}\n"
                                f"\n"
                                f"   FIX: Add camera '{sensor_name}_rgb' to robot XML, OR\n"
                                f"        Rename sensor to match one of: {available_cameras}\n"
                                f"\n"
                                f"   REASON: Modal-Oriented Programming requires explicit camera mapping!\n"
                            )

            # AUTODISCOVER free cameras (virtual viewpoints) from scene
            # Free cameras are added via ops.add_camera() and stored in scene.cameras
            if hasattr(self.scene, 'cameras') and self.scene.cameras:
                print(f"🎥 Autodiscovering {len(self.scene.cameras)} free cameras from scene...")

                for camera_id, camera in self.scene.cameras.items():
                    # Create free camera backend
                    try:
                        backend = MuJoCoRenderingBackend.create_free_camera(
                            self.scene_compiler.model,
                            self.scene_compiler.data,
                            lookat=tuple(camera.lookat),
                            distance=camera.distance,
                            azimuth=camera.azimuth,
                            elevation=camera.elevation,
                            width=camera.width,
                            height=camera.height,
                            enable_depth=False,  # Free cameras don't need depth by default
                            shadows=self.camera_shadows,
                            reflections=self.camera_reflections
                        )

                        # Connect backend to camera
                        camera.connect(
                            model=self.scene_compiler.model,
                            data=self.scene_compiler.data,
                            scene=self.scene,
                            camera_backend=backend
                        )

                        # Store backend for cleanup
                        self.camera_backends[camera_id] = backend

                        print(f"✅ Created backend for free camera '{camera_id}' (virtual, {camera.width}x{camera.height})")

                    except Exception as e:
                        raise RuntimeError(
                            f"❌ FAILED to create backend for free camera '{camera_id}'!\n"
                            f"   Error: {e}\n"
                            f"   Camera config: lookat={camera.lookat}, distance={camera.distance}, "
                            f"azimuth={camera.azimuth}, elevation={camera.elevation}"
                        )

        # Auto-discover geom_names for all assets from compiled MuJoCo model - OFFENSIVE
        self._auto_discover_geom_names()

        # Load robot initial state from keyframe - SEAMLESS SIM-TO-REAL!
        if self.robot:
            self._load_robot_keyframe()

        # Calculate view update intervals from physics_dt - ASYNC like reality!
        physics_dt = self.scene_compiler.model.opt.timestep
        physics_hz = 1.0 / physics_dt

        # Calculate how many physics steps between view updates
        if self.camera_fps > 0:
            self._camera_update_interval = max(1, int(physics_hz / self.camera_fps))
        else:
            self._camera_update_interval = 999999  # Never update (disabled)

        if self.sensor_hz > 0:
            self._sensor_update_interval = max(1, int(physics_hz / self.sensor_hz))
        else:
            self._sensor_update_interval = 999999

        # PERFORMANCE: Calculate state extraction interval (10Hz in fast_mode, 200Hz normally)
        if self.state_extraction_hz > 0:
            self._state_update_interval = max(1, int(physics_hz / self.state_extraction_hz))
        else:
            self._state_update_interval = 1  # Always extract if hz is 0 (disabled throttling)

        # Quiet mode - removed verbose print

        # Initialize action executor with robot
        if self.robot:
            self.action_executor.init_from_robot(self.robot)
            # Quiet mode - removed verbose print
            pass

        # Reset state
        self.step_count = 0
        self.start_time = time.time()
        self.current_reward_dict = {}
        self.event_log.clear()  # Clear event log for new experiment

        # CRITICAL: Sync robot state from backend ONCE before first step!
        # This connects actuators/sensors to MuJoCo so views can render
        if self.robot:
            self.backend.sync_actuators_from_backend(self.robot)
            self.backend.sync_sensors_from_backend(self.robot, update_cameras=False, update_sensors=False, event_log=self.event_log, camera_backends=self.camera_backends)

        # Initialize Frame Snapshot Saver now that all subsystems are ready
        if self.timeline_fps > 0 and hasattr(self, '_experiment_dir_for_snapshots'):
            from .frame_snapshot_saver import FrameSnapshotSaver
            from ..modals.experiment_modal import ExperimentModal
            from datetime import datetime

            # Create ExperimentModal (God Modal) for complete snapshots
            experiment_modal = ExperimentModal(
                scene=self.scene,  # The actual scene modal
                config={},
                description="Runtime experiment snapshot",
                created_at=datetime.fromtimestamp(time.time()).isoformat(),
                experiment_dir=Path(self._experiment_dir_for_snapshots)
            )

            self.frame_snapshot_saver = FrameSnapshotSaver(
                timeline_dir=Path(self._experiment_dir_for_snapshots) / "timeline",
                backend=self.backend,
                experiment=experiment_modal,  # ExperimentModal with to_json()!
                robot=self.robot,
                scene=self.scene,
                action_queue=self.action_executor.queue_modal if hasattr(self.action_executor, 'queue_modal') else None,
                reward_computer=self.reward_computer,
                global_fps=self.timeline_fps
            )

            # Register as ViewAggregator consumer
            self.view_aggregator.add_consumer(
                "frame_snapshot_saver",
                self.frame_snapshot_saver.save_frame,
                filter=None  # Save complete frame every step
            )

        # Initialize views ASYNC (don't block load_experiment!)
        # Views will be created on first step anyway, no need to block here
        # PERFORMANCE: This was blocking for 1.768s during initialization!
        runtime_state = {
            "step_count": 0,
            "current_reward": 0.0,
            "reward_dict": {},  # Empty reward dict initially
            "elapsed_time": 0.0,
            "action_queue": {"currently_executing": [], "pending_actions": 0},
            "extracted_state": {}
        }

        # ASYNC: Create initial views in background (don't block!)
        if self.view_aggregator.has_consumers():
            snapshot = self._snapshot_mujoco_data()
            robot_ref = self.robot
            scene_ref = self.scene
            model_ref = self.backend.model
            event_log_ref = self.event_log

            def create_initial_views():
                """Background: Create initial views without blocking load"""
                try:
                    views = self.view_aggregator.create_views(
                        robot_ref, scene_ref, runtime_state,
                        model_ref,
                        update_cameras=True,
                        update_sensors=True,
                        event_log=event_log_ref,
                        snapshot=snapshot
                    )
                    self.last_views = views
                except Exception as e:
                    print(f"Warning: Initial view creation failed: {e}")

            self.async_writer.submit(create_initial_views)
        else:
            # No consumers, just set empty views
            self.last_views = {}

        # Quiet mode - removed verbose print

    def _snapshot_mujoco_data(self) -> Optional[Dict[str, Any]]:
        """Fast snapshot of MuJoCo data for background processing - MOP STYLE

        Returns None if backend not ready (MOP principle: explicit failure!)
        Returns complete snapshot dict if backend ready.

        Takes ~1-2ms - much faster than 500ms blocking state extraction!

        Returns:
            Dict with copied numpy arrays, or None if backend not ready
        """
        # MOP FIX: Return None if backend not ready (explicit, not hidden!)
        # Caller MUST check for None - this prevents hiding bugs!
        if not self.backend or not hasattr(self.backend, 'data') or self.backend.data is None:
            return None

        data = self.backend.data

        # Copy contact data (CRITICAL for thread safety!)
        # Background thread will access contact.geom1, contact.geom2
        contact_list = []
        for i in range(data.ncon):
            contact_list.append({
                'geom1': int(data.contact[i].geom1),
                'geom2': int(data.contact[i].geom2)
            })

        # Fast array copies (~1-2ms total for typical robot)
        return {
            # Joint state
            'qpos': data.qpos.copy(),
            'qvel': data.qvel.copy(),

            # Body state
            'xpos': data.xpos.copy(),
            'xmat': data.xmat.copy(),
            'xquat': data.xquat.copy(),  # Quaternions for IMU-based rotation (CRITICAL for robot_base!)
            'cvel': data.cvel.copy(),  # Body velocities (linear + angular)

            # Geom state (for surface height calculations in state extraction)
            'geom_xpos': data.geom_xpos.copy(),

            # Contact state (FIXES IndexError race condition!)
            'ncon': int(data.ncon),
            'contact': contact_list,

            # Site state
            'site_xpos': data.site_xpos.copy(),

            # Metadata
            'time': data.time,
            'step': self.step_count
        }

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step - OFFENSIVE

        Returns:
            Dict with step results: step_count, reward, state, etc.
        """
        assert self.scene is not None, "Call load_experiment() first"

        # ARTIFACT RECORDING: Mark step start
        if self.artifact:
            self.artifact.record_step_start(self.step_count)

        # 1. Action execution (get commands) - MODAL-ORIENTED: No views needed!
        commands = {}
        if self.robot:
            commands = self.action_executor.tick(self.robot, event_log=self.event_log)

        # ARTIFACT RECORDING: Record controls sent to actuators
        if self.artifact and commands:
            self.artifact.record_controls(commands)

        # 2. Send controls to backend
        if commands:
            self.backend.set_controls(commands)

        # 3. Step physics
        self.backend.step()

        # Decide what to update THIS step (async rates!) - MUST calculate BEFORE sync
        update_cameras = (self.step_count % self._camera_update_interval == 0)
        update_sensors = (self.step_count % self._sensor_update_interval == 0)

        # 4. CRITICAL: Sync robot state from backend (with async rates!)
        if self.robot:
            # PERFORMANCE: Build extraction cache ONCE for name→ID lookups (eliminates 47.5x mj_name2id per step!)
            # Name→ID mappings never change during runtime - build once and reuse!
            if not hasattr(self, '_sync_cache'):
                from simulation_center.core.modals.behavior_extractors import build_extraction_cache
                self._sync_cache = build_extraction_cache(self.backend.model, self.backend.data)

            self.backend.sync_actuators_from_backend(self.robot, cache=self._sync_cache)
            self.backend.sync_sensors_from_backend(self.robot, update_cameras=update_cameras, update_sensors=update_sensors, event_log=self.event_log, camera_backends=self.camera_backends)

        # 4b. Sync FREE cameras (virtual viewpoints) - MAIN THREAD ONLY (OpenGL not thread-safe!)
        # Free cameras are in scene.cameras, NOT robot.sensors (they're virtual viewpoints, not robot sensors)
        if update_cameras and hasattr(self.scene, 'cameras') and self.scene.cameras:
            for camera_id, camera_modal in self.scene.cameras.items():
                if hasattr(camera_modal, 'sync_from_mujoco'):
                    # Get camera backend if exists (for rendering)
                    backend = self.camera_backends.get(camera_id)
                    if backend:
                        camera_modal.sync_from_mujoco(self.backend.model, self.backend.data, camera_backend=backend)

        # Increment step count BEFORE throttling check (fixes off-by-one bug)
        self.step_count += 1

        # FPS THROTTLING: Determine if we should do EXPENSIVE operations this step
        # Only sync assets, extract state, compute rewards, emit events at save FPS (10Hz), not every step (200Hz)!
        steps_since_last = self.step_count - self.last_view_save_step
        should_update_expensive = (self.last_view_save_step < 0 or steps_since_last >= self.steps_per_view_save)

        # Sync all assets (objects, furniture, walls) - THROTTLED! Objects don't move that fast
        if should_update_expensive:
            self.backend.sync_assets_from_backend(self.scene)

        # 5. Extract state - THROTTLED in fast_mode!
        # PERFORMANCE: In fast_mode, state extraction is throttled to 10Hz (was 200Hz)
        # Profiling showed state extraction takes 14% of time (1.105s/7.723s)
        # Actions need fresh ACTUATOR positions every step, but full state (contacts, spatial) can be throttled
        steps_since_last_state = self.step_count - self._last_state_update_step
        should_extract_state = (self._last_state_update_step < 0 or
                               steps_since_last_state >= self._state_update_interval)

        if should_extract_state:
            # FIRST extraction MUST be synchronous to populate cache
            # Subsequent extractions can be async for performance
            if self._last_state_update_step < 0:
                # FIRST extraction - blocking to properly initialize cache
                state = self.state_extractor.extract(self.scene, self.robot, event_log=self.event_log)
                with self._state_lock:
                    self._last_state = state
                    self._last_state_update_step = self.step_count
            else:
                # ASYNC STATE EXTRACTION: Make view extraction non-blocking!
                # 1. Snapshot MuJoCo data on main thread (~1-2ms)
                # 2. Submit extraction work to background thread with snapshot
                # 3. Use cached state immediately (will be updated by background thread)
                # This makes view extraction async so it doesn't block physics (0.5s -> 0ms overhead!)

                # CRITICAL: Snapshot MuJoCo data FIRST (main thread, ~1-2ms)
                # This fixes IndexError race conditions on data.contact[i]
                snapshot = self._snapshot_mujoco_data()

                # MOP FIX: If snapshot is None, use cached state but WARN about it!
                # This should ONLY happen during GOD MODAL restore when backend isn't ready.
                # If it happens elsewhere, the warning will expose the bug!
                if snapshot is None:
                    # EXPLICIT WARNING: This should only happen during init!
                    if self.step_count > 100:  # After 100 steps, backend MUST be ready
                        print(f"WARNING: Snapshot is None at step {self.step_count}! Backend not ready? This may be a bug!")
                    # Use cached state (schema for "no state yet")
                    with self._state_lock:
                        state = self._last_state.copy()
                else:
                    # Snapshot ready - normal async extraction path
                    # Capture current scene/robot/event_log refs for closure
                    scene_ref = self.scene
                    robot_ref = self.robot
                    event_log_ref = self.event_log
                    current_step = self.step_count

                    def extract_async():
                        """Background thread: Extract state from SNAPSHOT (thread-safe!)

                        THREAD SAFETY: Uses snapshot instead of live MuJoCo data
                        - Snapshot copied on main thread before mj_step() modifies it
                        - Background thread only accesses immutable snapshot arrays
                        - No race conditions on contact[i], qpos[i], etc.
                        - Extraction is 100% thread-safe!
                        """
                        try:
                            extracted_state = self.state_extractor.extract(
                                scene_ref, robot_ref,
                                event_log=event_log_ref,
                                snapshot=snapshot  # Thread-safe snapshot (verified NOT None!)
                            )
                            # Thread-safe update of cached state (not the step counter - that's on main thread!)
                            with self._state_lock:
                                self._last_state = extracted_state
                        except Exception as e:
                            print(f"Warning: Async state extraction failed: {e}")
                            import traceback
                            traceback.print_exc()
                            # Keep using old cached state on error

                    # CRITICAL FIX: Update step counter on MAIN THREAD to prevent race condition!
                    # Before: Updated in background thread → multiple steps submit before update
                    # After: Updated immediately → proper 30Hz throttling!
                    self._last_state_update_step = current_step

                    self.async_writer.submit(extract_async)
                    # Thread-safe read of cached state
                    with self._state_lock:
                        state = self._last_state.copy()
        else:
            # PERFORMANCE: Use cached state (10Hz instead of 200Hz)
            # This works for both fast_mode AND timeline recording!
            with self._state_lock:
                state = self._last_state.copy()

        # 6. Compute reward - ONLY when state is fresh (30Hz)!
        # MOP-CORRECT: Rewards depend on STATE → compute when state updates
        # PERFORMANCE: Saves 170 reward computations/sec (200Hz - 30Hz = 85% reduction)
        current_time = time.time() - self.start_time
        if should_extract_state:
            # Fresh state extracted - compute reward on it
            self.current_reward_dict = self.reward_computer.compute(
                state, self.scene.reward_modal, current_time, 0.0, scene=self.scene
            )
            reward_computed_this_step = True
            # OFFENSIVE: Crash if "delta" missing - exposes reward_modal bugs!
            rl_reward_delta = self.current_reward_dict["delta"]  # No .get()!
        else:
            # Using cached state - skip reward computation
            reward_computed_this_step = False
            rl_reward_delta = 0.0  # Explicit 0 (no computation, not missing data)

        # 7. Emit step event - THROTTLED! Don't spam events at 200Hz
        if should_update_expensive:
            self.event_bus.emit("step", {
                "step_count": self.step_count,
                "reward_step": self.current_reward_dict.get("delta", 0.0),
                "reward_total": self.current_reward_dict.get("total", 0.0),
                "state": state,
                "elapsed_time": current_time
            })

        # 8. Emit reward event if changed - THROTTLED!
        reward_total = self.current_reward_dict.get("total", 0.0)
        reward_delta = self.current_reward_dict.get("delta", 0.0)
        if reward_delta != 0:  # Emit when reward changes
            self.event_bus.emit("reward", {
                "reward_step": reward_delta,
                "reward_total": reward_total,
                "step_count": self.step_count
            })

        # 9. Create views at different rates (ASYNC like reality!) - VIEW-CENTRIC
        # Create views ONLY if external consumers exist - EXTERNAL-ONLY!
        # NOTE: update_cameras/update_sensors already calculated above (before sync)
        if self.view_aggregator.has_consumers():
            # FPS THROTTLING: Only render views when saving (not every step!)
            # This prevents wasting 95% of rendering when save_fps=10 but physics runs at 200Hz
            steps_since_last = self.step_count - self.last_view_save_step
            should_save = (self.last_view_save_step < 0 or steps_since_last >= self.steps_per_view_save)

            if should_save:
                # PERFORMANCE FIX: Build runtime_state ONLY when saving (10Hz), not every step (200Hz)!
                # This eliminates 20X redundant action graph traversals
                # MOP-CORRECT: Read action queue state from queue modal
                currently_executing = []
                pending_blocks = []

                # Get currently executing actions with details
                for actuator_id, queue_state in self.action_executor.queue_modal.actuators.items():
                    if queue_state.current_action is not None:
                        action = queue_state.current_action
                        action_repr = f"{action.__class__.__name__}(target={getattr(action, 'target', getattr(action, 'position', 'N/A'))})"
                        currently_executing.append({
                            "actuator": actuator_id,
                            "action": action_repr,
                            "progress": getattr(queue_state, 'progress', 0.0) if hasattr(queue_state, 'progress') else 0.0
                        })

                # Get pending action blocks with details
                for block_id, block_record in self.action_executor.queue_modal.blocks.items():
                    if block_record.status in ["queued", "executing"]:
                        block_info = {
                            "id": block_record.block_id,
                            "name": block_record.block_name,
                            "status": block_record.status,
                            "actions_total": block_record.actions_total,
                            "actions_completed": block_record.actions_completed,
                            "progress": f"{block_record.actions_completed}/{block_record.actions_total}"
                        }
                        pending_blocks.append(block_info)

                runtime_state = {
                    "step_count": self.step_count,
                    "current_reward": self.current_reward_dict.get("total", 0.0),
                    "reward_dict": self.current_reward_dict,  # Full reward dict
                    "elapsed_time": current_time,
                    "action_queue": {
                        "currently_executing": currently_executing,
                        "pending_blocks": pending_blocks,
                        "num_pending": len(self.action_executor.queue_modal.blocks)
                    },
                    "extracted_state": state  # Pass extracted state (with asset positions from behavior extractors)
                }

                # ASYNC VIEW CREATION: Snapshot + background thread (0ms blocking!)
                # =================================================================
                # 1. Snapshot MuJoCo data on main thread (~1-2ms, thread-safe!)
                # 2. Move create_views() + distribute() to background thread
                # 3. Main thread continues physics immediately (0ms blocking!)
                snapshot = self._snapshot_mujoco_data()

                # Capture references for async closure (avoid race conditions)
                robot_ref = self.robot
                scene_ref = self.scene
                runtime_state_ref = runtime_state
                model_ref = self.backend.model
                event_log_ref = self.event_log

                def create_and_distribute_views():
                    """Background thread: Create and distribute views from snapshot (thread-safe!)"""
                    try:
                        views = self.view_aggregator.create_views(
                            robot_ref, scene_ref, runtime_state_ref,
                            model_ref,
                            update_cameras=update_cameras,
                            update_sensors=update_sensors,
                            event_log=event_log_ref,
                            snapshot=snapshot  # ASYNC MODE: Use snapshot instead of live data!
                        )
                        self.view_aggregator.distribute(views)
                        # Update cached views (thread-safe - single write)
                        self.last_views = views
                    except Exception as e:
                        print(f"Warning: Async view creation failed: {e}")
                        import traceback
                        traceback.print_exc()

                self.async_writer.submit(create_and_distribute_views)

                self.last_view_save_step = self.step_count
        else:
            # No consumers - skip view creation for performance!
            pass


        # ARTIFACT RECORDING: Record state and rewards
        if self.artifact:
            self.artifact.record_state(state)
            # Record reward dict with full breakdown
            self.artifact.record_rewards(self.current_reward_dict)
            self.artifact.record_step_end()

        # MOP-CORRECT: Pass through reward dict from modal!
        # OFFENSIVE: No .get() with defaults - crash if reward_modal returns wrong structure!
        step_delta = rl_reward_delta if reward_computed_this_step else 0.0

        return {
            "step_count": self.step_count,
            "reward_step": step_delta,                          # Delta at state extraction boundaries (30Hz)
            "reward_total": self.current_reward_dict["total"],  # OFFENSIVE: Crash if missing!
            "rewards": self.current_reward_dict["rewards"],     # OFFENSIVE: Crash if missing!
            "reward_vector": self.current_reward_dict["reward_vector"],  # OFFENSIVE: Crash if missing!
            "state": state,
            "elapsed_time": current_time
        }

    def run(self, steps: Optional[int] = None, until: Optional[Callable] = None):
        """Run simulation loop - OFFENSIVE

        Args:
            steps: Number of steps to run (None = infinite)
            until: Callable that returns True when done (overrides steps)
        """
        assert self.scene is not None, "Call load_experiment() first"

        print(f"\nStarting simulation: {steps or 'infinite'} steps")

        if steps is None and until is None:
            # Infinite loop
            while True:
                result = self.step()
        elif until is not None:
            # Run until condition
            while not until(self):
                result = self.step()
        else:
            # Fixed steps
            for _ in range(steps):
                result = self.step()

        final_reward = self.current_reward_dict.get("total", 0.0)
        print(f"\n✓ Simulation complete: {self.step_count} steps, final reward: {final_reward:.2f}")

        # ARTIFACT RECORDING: Save artifact at end of experiment
        if self.artifact:
            artifact_path = self.artifact.save()
            print(f"💾 Experiment artifact saved: {artifact_path.name}")

    # === PUBLIC API ===

    def submit_action(self, action):
        """Submit single action - OFFENSIVE"""
        from ..modals.stretch.action_modals import ActionBlock
        block = ActionBlock(
            id=f"single_action_{action.id}",
            description=f"Single action: {action.__class__.__name__}",
            execution_mode="sequential",
            actions=[action]
        )
        return self.action_executor.submit_block(block)

    def submit_block(self, block):
        """Submit action block - OFFENSIVE"""
        return self.action_executor.submit_block(block)

    def add_consumer(self, name: str, callback: Callable[[Dict[str, Any]], None],
                     filter: Optional[Any] = None):
        """Add view consumer - OFFENSIVE"""
        self.view_aggregator.add_consumer(name, callback, filter)

    def on(self, event_type: str, callback: Callable):
        """Subscribe to event - OFFENSIVE"""
        self.event_bus.on(event_type, callback)

    # === HELPERS ===

    def _get_reward_conditions_status(self) -> Dict:
        """Get status of all reward conditions"""
        conditions = {}

        for condition_id, condition_data in self.scene.reward_modal.conditions.items():
            conditions[condition_id] = {
                "description": str(condition_data["condition"]),
                "reward_value": condition_data["reward"],
                "is_met": condition_id in self.scene.reward_modal.event_timestamps,
                "timestamp": self.scene.reward_modal.event_timestamps.get(condition_id, None)
            }

        return conditions

    def _auto_discover_geom_names(self):
        """Auto-discover geom_names for all assets from compiled MuJoCo model - OFFENSIVE

        After XMLResolver adds prefixes, we need to find the actual geom names in MuJoCo.
        This REPLACES component.geom_names for ALL assets (objects, furniture, robots, walls).

        UNIFORM: All entries in scene.assets are Assets with .components dict
        """
        import mujoco

        model = self.scene_compiler.model
        if not isinstance(model, mujoco.MjModel):
            return  # Not MuJoCo backend, skip

        # ================================================================
        # AUTO-DISCOVER GEOMS - UNIFORM for all assets!
        # ================================================================

        for asset_name, asset in self.scene.assets.items():
            # ALL assets have .components - no special cases!
            for comp_name, comp in asset.components.items():
                # Find geoms in MuJoCo model that match this component
                # For robot components, search using robot name prefix
                search_prefix = asset_name.split(".")[0] if "." in asset_name else asset_name

                discovered_geoms = []
                for i in range(model.ngeom):
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                    if geom_name and search_prefix in geom_name:
                        discovered_geoms.append(geom_name)

                # REPLACE geom_names with discovered names
                if discovered_geoms:
                    comp.geom_names = discovered_geoms

    def _load_robot_keyframe(self):
        """Load robot 'home' keyframe - sets MuJoCo qpos/ctrl

        Works with auto-discovered modal positions:
        1. Modals created with positions from keyframe (auto-discovery!)
        2. MuJoCo loads same keyframe (runtime!)
        3. Values match perfectly (validated!)

        SEAMLESS SIM-TO-REAL: Real robot will skip this (uses encoders)
        OFFENSIVE: Crashes if keyframe missing
        """
        import mujoco

        model = self.scene_compiler.model
        data = self.scene_compiler.data

        if not isinstance(model, mujoco.MjModel):
            return  # Not MuJoCo (e.g., real robot)

         # Find 'initial' keyframe (SELF-GENERATED by robot modal!)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "initial")

        if key_id < 0:
            # OFFENSIVE: Educational error
            available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
                         for i in range(model.nkey)
                         if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)]
            raise KeyError(
                f"Robot keyframe 'initial' not found!\n"
                f"Available: {available or ['(none)']}\n"
                f"\n"
                f"Robot modal should generate 'initial' keyframe via render_initial_state_xml().\n"
                f"This is MOP Principle #2 (SELF-GENERATION).\n"
                f"Real robot uses encoders instead."
            )

        # Load keyframe (sets ctrl values)
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # SIM BRIDGE: Set qpos directly from ctrl (SIM ADVANTAGE!)
        # ========================================================
        # REAL ROBOT: Can't teleport - encoders report actual position after movement
        # SIMULATION: Can teleport - we SET position instantly to match ctrl target
        #
        # This is DOCUMENTED in MODAL_ORIENTED_PROGRAMMING.md as sim-vs-real difference.
        # Real robot will SKIP this bridge and read from encoders.
        #
        # Why needed: MuJoCo keyframe sets ctrl but qpos stays 0 - controllers
        # need 500+ steps to integrate. Real robot already at position!

        # Map ctrl → qpos for position actuators
        lift_qpos_id = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_lift")]
        data.qpos[lift_qpos_id] = data.ctrl[2]  # lift

        # Arm tendon - split across 4 joints
        arm_ctrl = data.ctrl[3]
        for joint_name in ["joint_arm_l0", "joint_arm_l1", "joint_arm_l2", "joint_arm_l3"]:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_id = model.jnt_qposadr[joint_id]
                data.qpos[qpos_id] = arm_ctrl / 4.0

        # Other position actuators
        actuator_to_joint = {
            4: "joint_wrist_yaw", 5: "joint_wrist_pitch", 6: "joint_wrist_roll",
            7: "joint_gripper_slide", 8: "joint_head_pan", 9: "joint_head_tilt",
        }
        for ctrl_idx, joint_name in actuator_to_joint.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_id = model.jnt_qposadr[joint_id]
                data.qpos[qpos_id] = data.ctrl[ctrl_idx]

        mujoco.mj_forward(model, data)

        # Quiet mode - no keyframe load print

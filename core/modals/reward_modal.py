"""
REWARD MODAL - Conditions and rewards
Pure data structures for tracking - OFFENSIVE
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional, Literal
import mujoco
import os
import json
import numpy as np
from pathlib import Path


def discover_natural_range(behavior: str, property_name: str) -> float:
    """Discover natural range from BEHAVIORS.json - OFFENSIVE!

    Single source of truth: XML → ActuatorComponent.range → ROBOT_BEHAVIORS.json → here

    NO MAGIC NUMBERS! All ranges discovered from schemas.
    """
    # Load behavior schemas
    try:
        behaviors_path = Path(__file__).parent.parent / "behaviors" / "ROBOT_BEHAVIORS.json"
        with open(behaviors_path) as f:
            robot_behaviors = json.load(f)

        # Also load object behaviors
        obj_behaviors_path = Path(__file__).parent.parent / "behaviors" / "BEHAVIORS.json"
        with open(obj_behaviors_path) as f:
            object_behaviors = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"SCHEMA DISCOVERY FAILED!\n"
            f"  Could not load behavior schemas: {e}\n"
            f"  Behaviors must be regenerated: python3 core/tools/config_generator.py\n"
        )

    # Check robot behaviors first (robot_base, robot_arm, etc.)
    if behavior in robot_behaviors:
        props = robot_behaviors[behavior].get("properties", {})
        if property_name in props:
            natural_range = props[property_name].get("natural_range")
            if natural_range is not None:
                return float(natural_range)

    # Check object behaviors (hinged, graspable, etc.)
    if behavior in object_behaviors:
        props = object_behaviors[behavior].get("properties", {})
        if property_name in props:
            natural_range = props[property_name].get("natural_range")
            if natural_range is not None:
                return float(natural_range)

    # CRASH if not found - force explicit definition!
    raise RuntimeError(
        f"╔══════════════════════════════════════════════════════════════╗\n"
        f"║ NATURAL RANGE NOT FOUND!                                     ║\n"
        f"╚══════════════════════════════════════════════════════════════╝\n"
        f"\n"
        f"  Behavior:  {behavior}\n"
        f"  Property:  {property_name}\n"
        f"\n"
        f"  Natural range is REQUIRED for target-based rewards!\n"
        f"\n"
        f"  WHY: Natural range defines the maximum distance from target,\n"
        f"       used to calculate reward penalties for overshooting.\n"
        f"\n"
        f"  FIX OPTIONS:\n"
        f"  1. For robot behaviors: Regenerate ROBOT_BEHAVIORS.json\n"
        f"     → python3 core/tools/config_generator.py\n"
        f"\n"
        f"  2. For object behaviors: Add to BEHAVIORS.json\n"
        f"     → behaviors/{behavior}/properties/{property_name}/natural_range\n"
        f"\n"
        f"  3. For scene-dependent properties (position, distance_to):\n"
        f"     → These don't have fixed natural ranges!\n"
        f"     → Use discrete mode or provide override in add_reward()\n"
        f"\n"
        f"  NOTE: spatial.position and spatial.distance_to are scene-dependent!\n"
        f"        They can't have fixed natural ranges (depends on room size).\n"
    )


@dataclass
class ConditionResult:
    """
    Uniform return type for all conditions - ELEGANT

    Separates two concerns:
    1. is_met: Boolean - for sequences, events, dependencies (is threshold reached?)
    2. multiplier: Float 0-1 - for reward scaling (how much progress?)

    Examples:
        Discrete: ConditionResult(is_met=True, multiplier=1.0) or (False, 0.0)
        Convergent/Achievement: ConditionResult(is_met=(progress >= 1.0), multiplier=progress)
        Sequence: ConditionResult(is_met=all_steps_done, multiplier=steps_completed/total_steps)
    """
    is_met: bool        # True if condition fully satisfied (for event tracking, sequences)
    multiplier: float   # 0.0-1.0 reward scaling (partial progress for convergent/achievement rewards)


@dataclass
class Condition:
    """
    Unified condition - all features as optional fields (no wrappers!) - ELEGANT & OFFENSIVE

    Like stretch modals: ONE class with optional fields, not wrapper classes!

    Features:
        - Three reward modes: discrete, convergent, achievement
        - Time constraints (within/after from start or event)
        - Dependencies (requires)
        - Speed bonuses (calculated as math, not separate class)

    Usage:
        # Basic discrete
        Condition("door", "open@1.4", ">=", 1.4)

        # Convergent (smooth gradient with penalties)
        Condition("door", "open@1.4", ">=", 1.4, mode="convergent")

        # Achievement (smooth gradient, forgiving)
        Condition("door", "open@1.4", ">=", 1.4, mode="achievement")

        # With time
        Condition("door", "open@1.4", ">=", 1.4, within=60)

        # With dependency
        Condition("door", "open@1.4", ">=", 1.4, requires=["grabbed"])

        # Combined
        Condition("door", "open@1.4", ">=", 1.4, mode="convergent", within=50,
                 requires=["grabbed"], after_event="grabbed")
    """
    asset: str
    prop: str
    op: str  # ">", ">=", "<", "<=", "==", "!="
    val: Any

    # Optional features (not wrappers!)
    mode: Literal["discrete", "convergent", "achievement"] = "discrete"  # UNIFIED! One syntax!
    min_val: float = 0.0
    within: Optional[float] = None  # Seconds from reference (must complete WITHIN this time)
    after: Optional[float] = None   # Seconds from reference (only check AFTER this time)
    after_event: Optional[str] = None  # Event ID that starts the timer
    requires: Optional[List[str]] = None  # Prerequisites (condition IDs)
    speed_bonus: Optional[float] = None  # Extra points for faster completion (requires 'within')
    target: Optional[str] = None  # Target asset for spatial relations (e.g., "apple" for "holding")

    # TRUE MOP: Explicit behavior name (no string parsing!)
    behavior_name: Optional[str] = None  # Behavior name from scene (e.g., "robot_base", "door")

    # Advanced overrides (optional)
    tolerance_override: Optional[float] = None  # Override discovered tolerance
    natural_range_override: Optional[float] = None  # Override discovered natural range

    # Internal: Property unit from schema (set in __post_init__)
    _prop_unit: Optional[str] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Modal self-validation - TRUE MOP!

        I validate MYSELF! No one else needs to know my rules!
        All validation logic lives here - single source of truth!
        """
        # Load behavior schemas for validation
        from pathlib import Path
        import json

        behaviors_file = Path(__file__).parent.parent / "behaviors" / "BEHAVIORS.json"
        robot_behaviors_file = Path(__file__).parent.parent / "behaviors" / "ROBOT_BEHAVIORS.json"
        all_behaviors = {}

        if behaviors_file.exists():
            all_behaviors.update(json.loads(behaviors_file.read_text()))
        if robot_behaviors_file.exists():
            all_behaviors.update(json.loads(robot_behaviors_file.read_text()))

        # TRUE MOP: Use behavior_name from scene (NO STRING PARSING!)
        # scene_modal ALWAYS sets behavior_name (crashes if not found) - OFFENSIVE!
        # For "distance_to_table", behavior_name="distance_to" for schema lookup
        schema_lookup_name = self.behavior_name  # OFFENSIVE - crash if None!
        property_name = schema_lookup_name  # Alias for spatial validation

        behavior_def = None
        for beh_name, beh_spec in all_behaviors.items():
            if schema_lookup_name in beh_spec.get("properties", {}):
                behavior_def = beh_spec["properties"][schema_lookup_name]
                break

        # Validate if behavior found (skip validation for dynamic properties)
        if behavior_def:
            prop_unit = behavior_def.get("unit", "")

            # TRUE MOP: Store property unit for later use in check()
            self._prop_unit = prop_unit

            # Validate type: boolean vs numeric
            if prop_unit == "boolean" and self.val is not None and not isinstance(self.val, bool):
                raise TypeError(
                    f"❌ TYPE ERROR: Property '{self.prop}' expects BOOLEAN value!\n"
                    f"\n📚 Property unit: '{prop_unit}'\n"
                    f"   You provided: {self.val} (type: {type(self.val).__name__})\n"
                    f"\n✅ CORRECT: target=True or target=False\n"
                )

            elif prop_unit != "boolean" and self.val is not None and not isinstance(self.val, (int, float)):
                raise TypeError(
                    f"❌ TYPE ERROR: Property '{self.prop}' expects NUMERIC value!\n"
                    f"\n📚 Property unit: '{prop_unit}'\n"
                    f"   You provided: {self.val} (type: {type(self.val).__name__})\n"
                    f"\n✅ CORRECT: target=90.0 (number in {prop_unit})\n"
                )

        # Validate spatial properties require target asset
        spatial_props = ["holding", "looks_at", "reaching_toward", "in_view", "at_position",
                        "held_by", "contact", "contains", "distance_to",
                        "stacked_on", "supporting"]  # MOP: Stackable as spatial!
        is_spatial = property_name in spatial_props

        # Check if target is embedded in property name (e.g., "distance_to_table")
        # For "distance_to_table", property_name="distance_to", self.prop="distance_to_table"
        target_embedded = is_spatial and self.prop != property_name

        if is_spatial and not self.target and not target_embedded:
            raise ValueError(
                f"❌ MISSING TARGET: Spatial property '{self.prop}' requires 'target' parameter!\n"
                f"\n📚 Spatial relations check position relative to another asset.\n"
                f"   You MUST specify which asset to check against.\n"
                f"\n✅ CORRECT USAGE:\n"
                f"   Condition(asset='{self.asset}', prop='{self.prop}', op='{self.op}',\n"
                f"            val={self.val}, target='apple')  # ← Required!\n"
            )

    def check(self, state: Dict, event_timestamps: Dict = None, current_time: float = None,
              start_time: float = None, initial_state: Dict = None) -> ConditionResult:
        """
        Check condition - uniform interface - OFFENSIVE

        Returns ConditionResult with:
            - is_met: bool (for sequences, events, dependencies)
            - multiplier: float 0-1 (for reward scaling)

        Args:
            state: Current simulation state
            event_timestamps: Dict of condition ID -> timestamp (for dependencies, time tracking)
            current_time: Current time (for time constraints)
            start_time: When tracking started (for time constraints)
        """
        # 1. Check dependencies
        if self.requires:
            if not event_timestamps:
                return ConditionResult(is_met=False, multiplier=0.0)
            for req_id in self.requires:
                if req_id not in event_timestamps:
                    return ConditionResult(is_met=False, multiplier=0.0)

        # 2. Check time constraints
        if (self.within or self.after) and current_time is not None and start_time is not None:
            # Determine reference time
            if self.after_event:
                if not event_timestamps or self.after_event not in event_timestamps:
                    return ConditionResult(is_met=False, multiplier=0.0)
                reference_time = event_timestamps[self.after_event]
            else:
                reference_time = start_time

            elapsed = current_time - reference_time

            # Check time window
            if self.within is not None and elapsed > self.within:
                return ConditionResult(is_met=False, multiplier=0.0)
            if self.after is not None and elapsed < self.after:
                return ConditionResult(is_met=False, multiplier=0.0)

        # 3. Get actual value from state (component-prefixed keys: "body.position")
        # For relational properties with target, construct full property name: "contact_apple"
        prop_name = f"{self.prop}_{self.target}" if self.target else self.prop

        # MOP: Spatial boolean properties default to False if missing!
        # If apple isn't touching table, stacked_on_table=False (not missing!)
        spatial_boolean_prefixes = ('stacked_on_', 'supporting_', 'held_by_')
        is_spatial_boolean = any(prop_name.startswith(prefix) for prefix in spatial_boolean_prefixes)

        if prop_name not in state[self.asset]:
            # GRACEFUL: Spatial boolean properties default to False (no contact)
            if is_spatial_boolean:
                actual = False  # No contact detected
            else:
                # OFFENSIVE: Other properties MUST exist in state!
                raise RuntimeError(
                    f"╔══════════════════════════════════════════════════════════════╗\n"
                    f"║ PROPERTY NOT FOUND IN STATE!                                  ║\n"
                    f"╚══════════════════════════════════════════════════════════════╝\n"
                    f"\n"
                    f"  Asset:    {self.asset}\n"
                    f"  Property: {prop_name}\n"
                    f"  Unit:     {self._prop_unit}\n"
                    f"\n"
                    f"  This property should exist in state but doesn't!\n"
                    f"\n"
                    f"  POSSIBLE CAUSES:\n"
                    f"  1. Behavior extractor not running for this property\n"
                    f"  2. Target object '{self.target}' not in scene (if spatial property)\n"
                    f"  3. Property not defined in BEHAVIORS.json\n"
                    f"  4. Asset '{self.asset}' doesn't support this behavior\n"
                    f"\n"
                    f"  Available properties for '{self.asset}':\n"
                    f"  {sorted(state[self.asset].keys())}\n"
                    f"\n"
                    f"  FIX:\n"
                    f"  - If spatial property: Ensure target object exists in scene\n"
                    f"  - If robot property: Check behavior is enabled in robot config\n"
                    f"  - If object property: Check BEHAVIORS.json for this behavior\n"
                )
        else:
            actual = state[self.asset][prop_name]

        # 3a. Convert position vectors to scalar distance for progress calculation
        # When behavior="position", actual is [x,y,z] list - convert to magnitude
        if isinstance(actual, (list, tuple)) and len(actual) == 3:
            import math
            actual = math.sqrt(sum(x**2 for x in actual))

        # 3b. Get tolerance metadata if available (MODAL-TO-MODAL!)
        # Tolerance from ActuatorComponent → StateExtractor → here!
        # BOOLEAN PROPERTIES: No tolerance needed (exact True/False match)
        # OBJECTS: Don't have _tolerance (only robot actuators with physics compliance)
        if isinstance(actual, bool):
            # Boolean properties (stacked_on, held_by, etc) → no tolerance
            tolerance = 0.0
        elif '_tolerance' in state[self.asset]:
            # Robot actuators have tolerance from physics (gravity droop, compliance, etc)
            # PROPERTY-SPECIFIC: rotation uses IMU tolerance, position uses body tolerance
            tolerance = state[self.asset]['_tolerance']
        else:
            # Objects (apple, banana, table) don't have tolerance → 0.0
            # They use exact physics-based checks (contact detection, distance, etc)
            tolerance = 0.0

        # DEBUG: Log ALL tolerance checks (even 0.0)
        if self.prop in ["height", "extension"] and "_parallel" in str(getattr(self, 'id', '')):
            print(f"  🎯 TOLERANCE CHECK: asset={self.asset}, prop={self.prop}, val={self.val}")
            print(f"     actual={actual:.4f}, tolerance={tolerance*1000:.1f}mm, adjusted={self.val-tolerance:.4f}")
            print(f"     state keys: {list(state.get(self.asset, {}).keys())}")


        # 4. Check condition based on mode
        if self.mode == "discrete":
            # Discrete: boolean result (target reached or not)
            is_met = self._compare(actual, tolerance)
            return ConditionResult(is_met=is_met, multiplier=1.0 if is_met else 0.0)

        elif self.mode == "convergent":
            # CONVERGENT: Target-based with penalties for overshooting
            # TRUE MOP: Use explicit behavior_name (NO STRING PARSING!)
            behavior = self.behavior_name
            property_name = self.prop.split('.')[-1] if '.' in self.prop else self.prop

            # Discover natural range (from ROBOT_BEHAVIORS.json ← XML joint limits!)
            natural_range = self.natural_range_override
            if natural_range is None:
                natural_range = discover_natural_range(behavior, property_name)

            # Use discovered or overridden tolerance
            used_tolerance = self.tolerance_override if self.tolerance_override is not None else tolerance

            # TOLERANCE ZONE: target ± tolerance = 100pts
            distance = abs(actual - self.val)  # UNIFIED! self.val is the target!

            if distance <= used_tolerance:
                # Within tolerance zone → 100% progress!
                progress = 1.0
            else:
                # Outside tolerance - penalize based on distance
                excess = distance - used_tolerance
                max_excess = (natural_range / 2) - used_tolerance

                # CONVERGENT: Cumulative tracking - penalize overshooting (can go negative!)
                progress = 1.0 - (excess / max_excess)

            return ConditionResult(is_met=(progress >= 1.0), multiplier=progress)

        elif self.mode == "achievement":
            # ACHIEVEMENT: Target-based, forgiving (shortest path)
            # TRUE MOP: Use explicit behavior_name (NO STRING PARSING!)
            behavior = self.behavior_name
            property_name = self.prop.split('.')[-1] if '.' in self.prop else self.prop

            # Discover natural range (from ROBOT_BEHAVIORS.json ← XML joint limits!)
            natural_range = self.natural_range_override
            if natural_range is None:
                natural_range = discover_natural_range(behavior, property_name)

            # Use discovered or overridden tolerance
            used_tolerance = self.tolerance_override if self.tolerance_override is not None else tolerance

            # TOLERANCE ZONE: target ± tolerance = 100pts
            distance = abs(actual - self.val)  # UNIFIED! self.val is the target!

            if distance <= used_tolerance:
                # Within tolerance zone → 100% progress!
                progress = 1.0
            else:
                # Outside tolerance - calculate shortest path (forgiving!)
                excess = distance - used_tolerance
                max_excess = (natural_range / 2) - used_tolerance

                # ACHIEVEMENT: Shortest angular/physical distance
                if property_name == "rotation" and natural_range == 360.0:
                    # Shortest angular distance (wrapping)
                    shortest = min(distance, natural_range - distance)
                    excess_shortest = max(0, shortest - used_tolerance)
                    progress = 1.0 - (excess_shortest / max_excess)
                else:
                    # For non-rotation: same as convergent (no wrapping)
                    progress = 1.0 - (excess / max_excess)

                # Clamp achievement mode to [0, 1] (no negative rewards)
                progress = max(0.0, progress)

            return ConditionResult(is_met=(progress >= 1.0), multiplier=progress)

    def _compare(self, actual: Any, tolerance: float = 0.0) -> bool:
        """Apply comparison operator with physics tolerance - OFFENSIVE + PHYSICS-AWARE

        Tolerance adjusts thresholds for >= and > operators to account for:
        - Gravity droop (lift: ±14.3mm)
        - Spring compliance (gripper: ±1.3mm)
        - Motor inaccuracy (wrist_yaw: ±108mm)

        Args:
            actual: Actual value from physics
            tolerance: Physics tolerance from actuator modal (0.0 if not actuator)
        """
        if self.op == ">":
            return actual > (self.val - tolerance)  # Physics-aware!
        elif self.op == ">=":
            return actual >= (self.val - tolerance)  # Physics-aware!
        elif self.op == "<":
            return actual < (self.val + tolerance)  # Physics-aware (flip direction!)
        elif self.op == "<=":
            return actual <= (self.val + tolerance)  # Physics-aware (flip direction!)
        elif self.op == "==":
            # For equality, check if within tolerance
            return abs(actual - self.val) <= tolerance
        elif self.op == "!=":
            return actual != self.val  # No tolerance for inequality
        elif self.op == "boolean":
            return bool(actual) == self.val  # No tolerance for booleans
        else:
            raise ValueError(f"Unknown operator: {self.op}")

    def is_complete(self, state: Dict) -> bool:
        """Check if reward goal achieved - PURE MOP for RL termination!

        Used by ExperimentOps.check_termination() to know when task is done.
        Modal knows its own completion condition!

        Args:
            state: Current simulation state

        Returns:
            True if goal achieved (for rotation: reached threshold, for position: within tolerance)

        Examples:
            Rotation behavior: abs(rotation - threshold) < tolerance
            Position behavior: distance < tolerance
        """
        result = self.check(state)
        return result.is_met


@dataclass
class SequenceCondition:
    """
    Strict sequence of events - OFFENSIVE & ELEGANT

    Usage:
        # Define sequence of events (discrete - all-or-nothing)
        scene.add_reward_sequence(["grabbed", "opened", "walked_through"], reward=200)

        # Convergent mode (partial credit with normalization)
        scene.add_reward_sequence(["grabbed", "opened", "walked_through"], reward=200, mode="convergent")

        # With time constraint
        scene.add_reward_sequence(["grabbed", "opened"], reward=100, within=10)
    """
    sequence: List[str]  # Ordered list of condition IDs
    within: float = None  # Optional: all must happen within this time from first event
    mode: str = "discrete"  # "discrete" or "convergent"

    def __post_init__(self):
        """Validate - OFFENSIVE"""
        assert isinstance(self.sequence, list), \
            f"sequence must be a list, got {type(self.sequence)}"
        assert len(self.sequence) >= 2, \
            f"sequence must have at least 2 events, got {len(self.sequence)}"

    def check(self, state: Dict, event_timestamps: Dict) -> ConditionResult:
        """
        Check if sequence condition is met - UNIFORM INTERFACE

        Supports two modes:
        - discrete: All-or-nothing (multiplier 1.0 or 0.0)
        - smooth: Partial credit based on # of completed steps (0.0-1.0)

        Args:
            state: Current simulation state
            event_timestamps: Dict mapping condition IDs to their first True timestamp

        Returns:
            ConditionResult with is_met and multiplier based on mode
        """
        # Count how many steps completed in correct order
        # First check: if ANY later step happened before this step, sequence is broken
        completed_steps = 0
        for i, event_id in enumerate(self.sequence):
            if event_id not in event_timestamps:
                break  # This step not completed yet

            current_time = event_timestamps[event_id]

            # Check if this step happened after previous step
            if i > 0:
                prev_event_id = self.sequence[i - 1]
                if prev_event_id not in event_timestamps:
                    break  # Previous step not done, can't count this one
                if current_time <= event_timestamps[prev_event_id]:
                    break  # Out of order, stop counting

            # Check if any LATER step already happened before this step (breaks sequence)
            for j in range(i + 1, len(self.sequence)):
                later_event_id = self.sequence[j]
                if later_event_id in event_timestamps:
                    if event_timestamps[later_event_id] < current_time:
                        # Later step happened before this step - sequence broken
                        completed_steps = 0
                        return ConditionResult(is_met=False, multiplier=0.0)

            completed_steps += 1

        # Check time constraint if we have at least 2 completed steps
        if self.within is not None and completed_steps >= 2:
            first_time = event_timestamps[self.sequence[0]]
            last_completed_time = event_timestamps[self.sequence[completed_steps - 1]]
            elapsed = last_completed_time - first_time
            if elapsed > self.within:
                # Time expired - only count steps completed within time window
                # Find how many steps were within the time window
                valid_steps = 1  # First step always counts
                for i in range(1, completed_steps):
                    step_time = event_timestamps[self.sequence[i]]
                    if step_time - first_time <= self.within:
                        valid_steps += 1
                    else:
                        break
                completed_steps = valid_steps

        # Calculate results based on mode
        total_steps = len(self.sequence)
        is_met = (completed_steps == total_steps)

        if self.mode == "discrete":
            # Discrete: all-or-nothing
            multiplier = 1.0 if is_met else 0.0
        else:  # smooth
            # Smooth: partial credit
            multiplier = completed_steps / total_steps

        return ConditionResult(is_met=is_met, multiplier=multiplier)

    def is_complete(self, state: Dict, event_timestamps: Dict = None) -> bool:
        """Check if sequence goal achieved - PURE MOP for RL termination!

        Used by ExperimentOps.check_termination() to know when task is done.
        Modal knows its own completion condition!

        Args:
            state: Current simulation state
            event_timestamps: Dict mapping condition IDs to their first True timestamp

        Returns:
            True if all steps in sequence completed in correct order
        """
        if not event_timestamps:
            return False
        result = self.check(state, event_timestamps)
        return result.is_met


def display_condition(condition) -> str:
    """Display any condition type - OFFENSIVE"""
    if isinstance(condition, Condition):
        return f"{condition.asset}.{condition.prop} {condition.op} {condition.val}"
    if isinstance(condition, SequenceCondition):
        return f"sequence.ordered({len(condition.sequence)} events) == True"
    raise ValueError(f"Unknown condition type: {type(condition)}")


@dataclass
class RewardModal:
    """Reward tracking - pure data - CONTRIBUTION-BASED"""
    conditions: Dict = field(default_factory=dict)  # Dict of id -> {"condition": Condition, "reward": float}
    total: float = 0
    discrete_total: float = 0  # Accumulation of one-time rewards (discrete)
    smooth_total: float = 0  # Current value of smooth rewards (can decrease!)
    history: List = field(default_factory=list)  # History of rewards earned
    start_time: float = 0  # When reward tracking started (from simulation or real world)
    event_timestamps: Dict = field(default_factory=dict)  # Condition ID -> timestamp of first True
    initial_state: Dict = field(default_factory=dict)  # Baseline state (first step) - only reward CHANGES
    reward_vector: List[float] = field(default_factory=list)  # Cumulative reward at each step (for RL learning!)
    per_reward_totals: Dict[str, float] = field(default_factory=dict)  # PURE MOP: Cumulative total per reward ID
    initial_rewards: Dict[str, float] = field(default_factory=dict)  # MOP: Baseline rewards (old smooth mode) - only reward IMPROVEMENT
    initial_multipliers: Dict[str, float] = field(default_factory=dict)  # MOP: Baseline multipliers (convergent/achievement) - normalize to initial state
    per_reward_timeline: Dict[str, List[Dict]] = field(default_factory=dict)  # MOP: Per-reward timeline tracking (delta + total + multiplier per step)

    def add(self, asset: str, prop: str, op: str, val: Any, points: float, id: str):
        """Add basic reward condition - ID REQUIRED"""
        assert id is not None, "ID is REQUIRED for all reward conditions (for tracking, viz, debug)"
        condition = Condition(asset, prop, op, val)
        self.conditions[id] = {"condition": condition, "reward": points}
        self.per_reward_totals[id] = 0.0  # PURE MOP: Initialize per-reward tracking
        return self

    def add_condition(self, condition_or_proxy_or_id, reward: float,
                     within=None, after=None, after_event=None, requires=None, id: str = None,
                     mode="discrete", speed_bonus=None, conditions_registry=None):
        """
        Add advanced reward condition - ID REQUIRED

        Args:
            condition_or_proxy_or_id: Condition, PropertyProxy, or string ID to lookup
            reward: Points to award
            within: Seconds from reference - condition must be met WITHIN this time
            after: Seconds from reference - only check AFTER this time
            after_event: Condition/proxy/ID that starts the timer
            requires: Condition ID (or list of IDs) that must be met first
            id: REQUIRED - ID for this condition (for tracking, viz, debug)
            mode: "discrete", "convergent", or "achievement" - reward strategy
            speed_bonus: Extra points for faster completion
            conditions_registry: Dict of condition ID -> Condition (for lookups)

        Usage:
            # Named conditions
            reward_modal.add_condition(door.handle_grabbed(0.5), reward=0, id="grabbed")
            reward_modal.add_condition("grabbed", reward=10, id="grabbed_reward", conditions_registry=scene.conditions)

            # Dependencies
            reward_modal.add_condition("opened", reward=100, requires="grabbed", id="opened", conditions_registry=scene.conditions)

            # Convergent mode (smooth gradient with penalties)
            reward_modal.add_condition(door.open(1.4), reward=100, mode="convergent", id="door_convergent")

            # Achievement mode (smooth gradient, forgiving)
            reward_modal.add_condition(door.open(1.4), reward=100, mode="achievement", id="door_achievement")
        """
        assert id is not None, "ID is REQUIRED for all reward conditions (for tracking, viz, debug)"
        # 1. Resolve condition
        if isinstance(condition_or_proxy_or_id, str):
            if not conditions_registry or condition_or_proxy_or_id not in conditions_registry:
                raise ValueError(
                    f"Condition '{condition_or_proxy_or_id}' not defined. "
                    f"Available: {list(conditions_registry.keys()) if conditions_registry else []}"
                )
            condition = conditions_registry[condition_or_proxy_or_id]
        elif hasattr(condition_or_proxy_or_id, '__call__') and hasattr(condition_or_proxy_or_id, 'property_name'):
            # PropertyProxy object - auto-convert by calling it
            condition = condition_or_proxy_or_id()  # Call to get Condition
        elif hasattr(condition_or_proxy_or_id, 'to_condition'):
            # PropertyProxy-like object with to_condition method
            condition = condition_or_proxy_or_id.to_condition()
        else:
            condition = condition_or_proxy_or_id

        # 2. For Condition, ALWAYS copy (NO WRAPPERS!)
        # OFFENSIVE: reward_modal NEVER mutates registry objects
        # scene.conditions stays IMMUTABLE (clean templates)
        # reward_modal.conditions has INDEPENDENT copies
        if isinstance(condition, Condition):
            from dataclasses import replace
            condition = replace(condition)

            # Set mode for Condition
            if isinstance(condition, Condition):
                # Mode handling:
                # - "discrete" → all-or-nothing
                # - "convergent" → smooth gradient with penalties
                # - "achievement" → smooth gradient, forgiving (no negative)
                # - "auto" → lookup from BEHAVIORS.json unit_types
                # - None → default to "discrete"
                if mode == "auto":
                    # Auto-detect from BEHAVIORS.json
                    from .asset_modals import BEHAVIORS
                    prop_unit = None

                    # Find property unit by searching behaviors
                    for behavior_name, behavior_data in BEHAVIORS.items():
                        if behavior_name.startswith("_"):
                            continue
                        props = behavior_data["properties"]  # OFFENSIVE - crash if missing!
                        if condition.prop in props:
                            prop_unit = props[condition.prop]["unit"]  # OFFENSIVE - crash if missing!
                            break

                    # Check if unit supports smooth rewards (convergent mode)
                    if prop_unit:
                        unit_types = BEHAVIORS["_unit_types"]  # OFFENSIVE - crash if missing!
                        for unit_type_data in unit_types.values():
                            if unit_type_data["physical_units"] == prop_unit:  # OFFENSIVE!
                                if unit_type_data["smooth"]:  # OFFENSIVE - crash if missing!
                                    mode = "convergent"  # Smooth gradients with penalties
                                else:
                                    mode = "discrete"
                                break
                        else:
                            mode = "discrete"
                    else:
                        mode = "discrete"
                elif mode is None:
                    mode = "discrete"

                # Apply mode to Condition
                condition.mode = mode

            # Set time constraints
            condition.within = within
            condition.after = after

            # Handle after_event
            if after_event is not None:
                if isinstance(after_event, str):
                    condition.after_event = after_event
                elif hasattr(after_event, 'to_condition'):
                    after_event_cond = after_event.to_condition()
                    after_event_id = f"_after_event_{len(self.conditions)}"
                    if conditions_registry is not None:
                        conditions_registry[after_event_id] = after_event_cond
                    self.conditions[after_event_id] = {"condition": after_event_cond, "reward": 0}
                    condition.after_event = after_event_id
                elif isinstance(after_event, Condition):
                    after_event_id = f"_after_event_{len(self.conditions)}"
                    if conditions_registry is not None:
                        conditions_registry[after_event_id] = after_event
                    self.conditions[after_event_id] = {"condition": after_event, "reward": 0}
                    condition.after_event = after_event_id

            # Set dependencies
            if requires is not None:
                condition.requires = [requires] if isinstance(requires, str) else requires

            # Set speed bonus
            if speed_bonus is not None:
                if not within:
                    raise ValueError("speed_bonus requires 'within' parameter to define time window")
                condition.speed_bonus = speed_bonus

        # 3. Add to rewards
        self.conditions[id] = {"condition": condition, "reward": reward}
        self.per_reward_totals[id] = 0.0  # PURE MOP: Initialize per-reward tracking

        return self

    def add_sequence(self, sequence, reward: float, id: str, within=None, mode="discrete", conditions_registry=None):
        """
        Add reward for strict sequence of events - OFFENSIVE

        Args:
            sequence: List of condition IDs that must occur in order
            reward: Points to award when sequence completes
            id: REQUIRED - ID for the sequence (for tracking by ID)
            within: Optional time constraint (seconds from first to last event)
            mode: "discrete" (default) or "convergent"
                - discrete: All-or-nothing (complete sequence = full reward)
                - convergent: Partial credit with normalization (1/3 steps → 33% reward, 2/3 → 66%, etc.)
            conditions_registry: Dict of condition ID -> Condition (for validation)

        Usage:
            # Define named conditions first
            reward_modal.add_condition(door.handle_grabbed, reward=0, id="grabbed")
            reward_modal.add_condition(door.open(1.4), reward=0, id="opened")
            reward_modal.add_condition(robot.walked_through, reward=0, id="walked")

            # Reward for correct sequence (discrete)
            reward_modal.add_sequence(["grabbed", "opened", "walked"], reward=200, id="door_sequence")

            # Reward for correct sequence (convergent - partial credit)
            reward_modal.add_sequence(["grabbed", "opened", "walked"], reward=200, mode="convergent", id="door_sequence_smooth")
        """
        # Validate all IDs exist
        if conditions_registry:
            for cond_id in sequence:
                if cond_id not in conditions_registry:
                    raise ValueError(
                        f"Condition '{cond_id}' in sequence not defined. "
                        f"Available: {list(conditions_registry.keys())}"
                    )

        # Create sequence condition
        seq_condition = SequenceCondition(sequence=sequence, within=within, mode=mode)

        # Add to rewards
        self.conditions[id] = {"condition": seq_condition, "reward": reward}
        self.per_reward_totals[id] = 0.0  # PURE MOP: Initialize per-reward tracking

        return self
    def step(self, state: Dict, current_time: float = None, scene=None) -> float:
        """
        Check all conditions and return points earned this step - CONTRIBUTION-BASED

        Uses UNIFORM ConditionResult interface - no isinstance checks!
        Rewards only CHANGES from initial state (handles dirty XMLs elegantly)

        Args:
            state: Current simulation state
            current_time: Current time (from simulation or real world)
        """
        # Check if we have a baseline (for contribution tracking)
        has_baseline = bool(self.initial_state)

        # Capture initial state on first step (baseline for contribution tracking)
        if not has_baseline:
            from copy import deepcopy
            self.initial_state = deepcopy(state)

            # ================================================================
            # AUTO-DISCOVER NATURAL RANGES - TRUE MOP!
            # ================================================================
            # For spatial properties (distance_to, position), auto-calculate
            # natural range from initial state instead of requiring override
            for cond_id, data in self.conditions.items():
                condition = data["condition"]
                if isinstance(condition, Condition):
                    # Check if this is a spatial property needing auto-discovery
                    if condition.behavior_name == "spatial" and condition.natural_range_override is None:
                        property_name = condition.prop.split('.')[-1] if '.' in condition.prop else condition.prop

                        if property_name == "distance_to" and condition.target:
                            # Calculate initial distance between asset and target
                            # Get distance_to_{target} from initial state
                            prop_key = f"distance_to_{condition.target}"
                            if condition.asset in self.initial_state and prop_key in self.initial_state[condition.asset]:
                                initial_distance = self.initial_state[condition.asset][prop_key]
                                # Natural range = initial distance (how far from target at start)
                                condition.natural_range_override = initial_distance
                                # print(f"  [AUTO-DISCOVER] {condition.asset}.{prop_key}: natural_range={initial_distance:.2f}m")

                        elif property_name == "position" and condition.target:
                            # Calculate initial distance from target position
                            if condition.asset in self.initial_state and "position" in self.initial_state[condition.asset]:
                                asset_pos = self.initial_state[condition.asset]["position"]
                                # Target should be [x, y, z] array
                                if isinstance(condition.target, (list, tuple)) and len(condition.target) == 3:
                                    import math
                                    dx = asset_pos[0] - condition.target[0]
                                    dy = asset_pos[1] - condition.target[1]
                                    dz = asset_pos[2] - condition.target[2]
                                    initial_distance = math.sqrt(dx**2 + dy**2 + dz**2)
                                    condition.natural_range_override = initial_distance
                                    # print(f"  [AUTO-DISCOVER] {condition.asset}.position: natural_range={initial_distance:.2f}m")

        step_reward = 0  # Discrete rewards this step (accumulated)
        current_smooth_total = 0  # Smooth rewards (current state, can decrease)
        triggered = []

        # ================================================================
        # INJECT SCENE CONTEXT - for spatial relations
        # ================================================================
        # Before checking conditions, inject scene context into components
        # so they can resolve target positions for spatial properties
        if scene and hasattr(scene, 'assets'):
            for cond_id, data in self.conditions.items():
                condition = data["condition"]

                # Only inject for base Condition with target (not Sequence)
                if isinstance(condition, Condition) and condition.target:
                    # Find the tracked asset's components
                    asset_name = condition.asset
                    target_name = condition.target

                    if asset_name in scene.assets:
                        asset_or_component = scene.assets[asset_name]

                        # ================================================================
                        # INJECT SCENE CONTEXT - Handle both Asset and ActuatorComponent
                        # ================================================================
                        # OFFENSIVE: Trust uniform interface
                        # Assets have .components dict
                        # ActuatorComponents are stored as single-component assets

                        # All assets have .components - no checks needed!
                        for component in asset_or_component.components.values():
                            component._scene_context = {
                                'reward_target': target_name,
                                'assets': scene.assets,
                                'scene': scene
                            }

        # Check all conditions - UNIFORM INTERFACE!
        for cond_id, data in self.conditions.items():
            condition = data["condition"]
            points = data["reward"]

            try:
                # Check condition using uniform interface (ALL return ConditionResult)
                # SequenceCondition needs only state and event_timestamps
                if isinstance(condition, SequenceCondition):
                    result = condition.check(state, self.event_timestamps)
                else:
                    result = condition.check(state, self.event_timestamps, current_time, self.start_time)

                # Check mode (Condition and SequenceCondition have mode, default to discrete)
                mode = getattr(condition, 'mode', 'discrete')

                if mode == "discrete" and result.is_met:
                    # Check if already awarded using ID in event_timestamps
                    if cond_id not in self.event_timestamps:
                        reward = points * result.multiplier
                        step_reward += reward

                        # Calculate speed bonus if present
                        speed_bonus_points = 0
                        if hasattr(condition, 'speed_bonus') and condition.speed_bonus is not None:
                            if condition.within and current_time is not None and self.start_time is not None:
                                # Determine reference time
                                if condition.after_event:
                                    if condition.after_event in self.event_timestamps:
                                        reference_time = self.event_timestamps[condition.after_event]
                                    else:
                                        reference_time = self.start_time
                                else:
                                    reference_time = self.start_time

                                # Calculate elapsed and remaining time
                                elapsed = current_time - reference_time
                                if elapsed < condition.within:
                                    time_remaining = condition.within - elapsed
                                    bonus_multiplier = time_remaining / condition.within
                                    speed_bonus_points = condition.speed_bonus * bonus_multiplier
                                    step_reward += speed_bonus_points

                        # Track event timestamp (also marks as awarded!)
                        self.event_timestamps[cond_id] = current_time or 0.0

                        triggered.append({
                            "id": cond_id,  # For breakdown matching!
                            "condition": display_condition(condition),
                            "points": reward + speed_bonus_points,
                            "multiplier": result.multiplier  # FIX: Add multiplier! (should be 1.0 for discrete)
                        })

                elif mode != "discrete":
                    # TRUE MOP: convergent, achievement, smooth all use multipliers!
                    # Mode is read from condition.mode

                    # Step 1: Calculate absolute reward based on mode
                    if mode in ["convergent", "achievement"]:
                        # Convergent/achievement: Use multiplier from Condition.check()
                        absolute_reward = points * result.multiplier
                        multiplier = result.multiplier
                        condition_str = display_condition(condition)
                    elif isinstance(condition, SequenceCondition):
                        # Sequences: Use multiplier from result
                        absolute_reward = points * result.multiplier
                        multiplier = result.multiplier
                        condition_str = display_condition(condition)
                    elif has_baseline and condition.asset in self.initial_state and condition.prop in self.initial_state[condition.asset]:
                        # OLD smooth mode: baseline-based progress (deprecated)
                        initial_val = self.initial_state[condition.asset][condition.prop]
                        current_val = state[condition.asset][condition.prop]  # OFFENSIVE

                        # Convert position vectors to distance
                        if isinstance(initial_val, (list, tuple)) and len(initial_val) == 3:
                            import math
                            initial_val = math.sqrt(sum(x**2 for x in initial_val))
                        if isinstance(current_val, (list, tuple)) and len(current_val) == 3:
                            import math
                            current_val = math.sqrt(sum(x**2 for x in current_val))

                        # Calculate progress
                        contribution = current_val - initial_val
                        range_needed = condition.val - initial_val
                        if range_needed > 0:
                            multiplier = max(0.0, min(1.0, contribution / range_needed))
                            absolute_reward = points * multiplier
                            condition_str = f"{condition.asset}.{condition.prop} {condition.op} {condition.val}"
                        else:
                            continue  # Skip if range invalid
                    else:
                        # No baseline yet (first step) - skip
                        continue

                    # Step 2: MOP-CORRECT - SPLIT by semantic meaning!
                    # ACHIEVEMENT: Goal state (absolute position) - NO normalization
                    # CONVERGENT: Journey (improvement) - YES normalization

                    if mode == "achievement":
                        # ACHIEVEMENT: Reward absolute goal state (no normalization!)
                        # "Road doesn't matter, only destination"
                        # Robot at 40° toward 45° target gets 89pts immediately!
                        reward = points * multiplier  # Raw multiplier
                        # Forgiving: clamp to [0, points] (no negative)
                        reward = max(0.0, min(points, reward))

                    else:
                        # CONVERGENT/SEQUENCES: Reward journey (normalization!)
                        # Only reward IMPROVEMENT from initial state

                        # MULTIPLIER NORMALIZATION - preserves smooth gradients!
                        if cond_id not in self.initial_multipliers:
                            # First step: capture baseline multiplier, return 0 (no action yet)
                            self.initial_multipliers[cond_id] = multiplier
                            reward = 0.0
                        else:
                            initial_mult = self.initial_multipliers[cond_id]
                            # Handle edge case: already at target (initial_mult >= 0.99)
                            if initial_mult >= 0.99:
                                multiplier_normalized = 0.0  # No improvement possible
                            else:
                                # Normalize: how much did we improve from initial state?
                                multiplier_normalized = (multiplier - initial_mult) / (1.0 - initial_mult)
                                # Convergent/Sequences: allow penalties (clamped)
                                multiplier_normalized = max(-2.0, multiplier_normalized)
                            reward = points * multiplier_normalized

                    # Step 3: Add to smooth total and triggered list
                    current_smooth_total += reward
                    triggered.append({
                        "id": cond_id,
                        "condition": condition_str,
                        "points": reward,
                        "multiplier": multiplier
                    })

            except (KeyError, TypeError, AttributeError) as e:
                # OFFENSIVE: Crash with educational error showing available state
                asset = getattr(condition, 'asset', None)
                prop = getattr(condition, 'prop', None)

                available_assets = sorted(state.keys()) if state else []
                available_props = sorted(state.get(asset, {}).keys()) if asset and asset in state else []

                raise RuntimeError(
                    f"REWARD ERROR: Condition references missing asset/property!\n"
                    f"  Condition: {display_condition(condition)}\n"
                    f"  Looking for: {asset}.{prop if prop else '?'}\n"
                    f"\n"
                    f"  Available assets in state: {available_assets}\n"
                    f"  Available properties for '{asset}': {available_props}\n"
                    f"\n"
                    f"  Original error: {type(e).__name__}: {e}\n"
                    f"\n"
                    f"  FIX OPTIONS:\n"
                    f"  1. Check asset/property spelling in condition\n"
                    f"  2. Ensure asset is added to scene\n"
                    f"  3. Verify behavior property exists in BEHAVIORS.json\n"
                ) from e

        # Calculate smooth delta BEFORE updating
        smooth_delta = current_smooth_total - self.smooth_total

        # Update totals: discrete accumulates, smooth reflects current state
        self.discrete_total += step_reward
        self.smooth_total = current_smooth_total  # Can decrease!
        self.total = self.discrete_total + self.smooth_total

        # PURE MOP: Track reward trajectory for RL learning!
        # Append cumulative reward at each step (smooth trajectory!)
        self.reward_vector.append(self.total)

        # PURE MOP: Update per-reward totals (self-describing composition!)
        for triggered_info in triggered:
            cond_id = triggered_info.get("id")
            if cond_id:
                cond_data = self.conditions.get(cond_id, {})
                condition = cond_data.get("condition")
                mode = getattr(condition, 'mode', 'discrete') if condition else 'discrete'

                points = triggered_info.get("points", 0.0)

                if mode == "discrete":
                    # Discrete: accumulate (add delta)
                    self.per_reward_totals[cond_id] += points
                else:
                    # Smooth/convergent/achievement: set to current value (can decrease!)
                    self.per_reward_totals[cond_id] = points


        # MOP: Track per-reward timeline (delta + total + multiplier per step)
        current_step = len(self.reward_vector)  # Step number (before appending this step)
        for cond_id in self.conditions.keys():
            if cond_id not in self.per_reward_timeline:
                self.per_reward_timeline[cond_id] = []

            # Find triggered info for this reward (if it triggered this step)
            triggered_info = next((t for t in triggered if t.get("id") == cond_id), None)

            self.per_reward_timeline[cond_id].append({
                "step": current_step,
                "delta": triggered_info.get("points", 0.0) if triggered_info else 0.0,
                "total": self.per_reward_totals.get(cond_id, 0.0),
                "multiplier": triggered_info.get("multiplier", 0.0) if triggered_info else 0.0
            })

        # PURE MOP: Validate composability (global total = sum of parts)
        computed_total = sum(self.per_reward_totals.values())
        assert abs(computed_total - self.total) < 1e-6, \
            f"MOP VIOLATION: total ({self.total}) != sum of per-reward totals ({computed_total})"

        # Log this step if any rewards triggered
        if step_reward != 0 or current_smooth_total != 0:
            self.history.append({
                "time": current_time if current_time is not None else 0,
                "step_reward": step_reward,
                "smooth_reward": current_smooth_total,
                "total": self.total,
                "triggered": triggered
            })

        # Build per-reward breakdown (MOP style!)
        rewards_breakdown = {}
        for cond_id, cond_data in self.conditions.items():
            # Find if this condition triggered this step (match by ID!)
            triggered_info = next((t for t in triggered if t.get("id") == cond_id), None)

            # Get delta for this reward (from triggered list or 0)
            reward_delta = triggered_info.get("points", 0.0) if triggered_info else 0.0

            # PURE MOP: Include cumulative total per reward (self-describing composition!)
            rewards_breakdown[cond_id] = {
                "delta": reward_delta,
                "total": self.per_reward_totals.get(cond_id, 0.0),  # Cumulative total for this reward
                "multiplier": triggered_info.get("multiplier", 0.0) if triggered_info else 0.0
            }

        # Return RICH DICT (MOP style!): delta + total + per-reward breakdown + trajectory
        return {
            "delta": step_reward + smooth_delta,    # Points THIS step
            "total": self.total,                     # Cumulative total
            "discrete_total": self.discrete_total,   # Discrete only
            "smooth_total": self.smooth_total,       # Smooth only
            "rewards": rewards_breakdown,            # Per-reward breakdown
            "reward_vector": self.reward_vector      # Cumulative trajectory (for RL!)
        }

    def get_summary(self) -> Dict:
        """Get summary of rewards"""
        return {
            "total": self.total,
            "num_conditions": len(self.conditions),
            "history_length": len(self.history),
            "last_reward": self.history[-1] if self.history else None
        }

    def get_data(self) -> Dict[str, Any]:
        """I know my complete state - OFFENSIVE & MODAL-ORIENTED

        Returns detailed reward state for views.

        Returns:
            Dict with total, conditions, event timestamps, history
        """
        return {
            "total": self.total,
            "discrete_total": self.discrete_total,
            "smooth_total": self.smooth_total,
            "per_reward_totals": self.per_reward_totals,  # PURE MOP: Per-reward cumulative totals!
            "conditions": {
                cond_id: {
                    "reward": cond_data["reward"],
                    "cumulative_total": self.per_reward_totals.get(cond_id, 0.0),  # PURE MOP!
                    "is_met": cond_id in self.event_timestamps,
                    "timestamp": self.event_timestamps.get(cond_id),
                    "description": str(cond_data["condition"])
                }
                for cond_id, cond_data in self.conditions.items()
            },
            "history": self.history
        }

    def get_rl(self) -> 'np.ndarray':
        """I know my RL representation - OFFENSIVE & MODAL-ORIENTED

        Returns progress metrics for RL training.

        Returns:
            np.ndarray: [total_normalized, discrete_progress, smooth_progress, completion_rate]
        """
        import numpy as np

        # Total reward normalized (assuming max ~1000 points)
        total_norm = self.total / 1000.0

        # Discrete progress (fraction of discrete conditions met)
        num_discrete_met = sum(1 for cond_id in self.event_timestamps
                              if self.conditions[cond_id]["condition"].mode == "discrete")
        num_discrete_total = sum(1 for cond_data in self.conditions.values()
                                if cond_data["condition"].mode == "discrete")
        discrete_progress = num_discrete_met / num_discrete_total if num_discrete_total > 0 else 0.0

        # Smooth progress (normalized smooth total)
        smooth_progress = self.smooth_total / 500.0  # Assuming max ~500 smooth points

        # Completion rate (fraction of all conditions met)
        completion_rate = len(self.event_timestamps) / len(self.conditions) if self.conditions else 0.0

        return np.array([total_norm, discrete_progress, smooth_progress, completion_rate])

    def get_reward_timeline(self, format: str = "dict"):
        """Get reward progression over time - REUSABLE MOP UTILITY!

        Returns per-reward timeline showing delta (when triggered) and
        cumulative (total) for each reward at each step.

        Args:
            format: "dict" (JSON-serializable), "numpy" (structured array), or "table" (formatted string)

        Returns:
            If format="dict":
                {
                    "reward_ids": ["apple_on_table", "banana_on_table", ...],
                    "steps": [0, 1, 2, ...],
                    "timeline": {
                        "apple_on_table": [
                            {"step": 0, "delta": 100.0, "total": 100.0, "multiplier": 1.0},
                            {"step": 1, "delta": 0.0, "total": 100.0, "multiplier": 0.0},
                            ...
                        ],
                        ...
                    }
                }

            If format="numpy":
                np.ndarray with shape (num_rewards, num_steps, 4)
                where [:, :, 0] = delta, [:, :, 1] = total,
                      [:, :, 2] = multiplier, [:, :, 3] = step

            If format="table":
                Formatted string with total reward + all sub-rewards
        """
        import numpy as np

        if format == "dict":
            return {
                "reward_ids": list(self.conditions.keys()),
                "steps": list(range(len(self.reward_vector))) if self.reward_vector else [],
                "timeline": self.per_reward_timeline
            }

        elif format == "numpy":
            reward_ids = list(self.conditions.keys())
            num_rewards = len(reward_ids)
            num_steps = len(self.reward_vector) if self.reward_vector else 0

            # Create structured array: (num_rewards, num_steps, 4)
            timeline_array = np.zeros((num_rewards, num_steps, 4))

            for i, reward_id in enumerate(reward_ids):
                timeline = self.per_reward_timeline.get(reward_id, [])
                for step_data in timeline:
                    step = step_data["step"]
                    if step < num_steps:
                        timeline_array[i, step, 0] = step_data["delta"]
                        timeline_array[i, step, 1] = step_data["total"]
                        timeline_array[i, step, 2] = step_data["multiplier"]
                        timeline_array[i, step, 3] = step

            return timeline_array

        elif format == "table":
            # SINGLE UNIFIED TABLE - All rewards in one table!
            lines = []
            lines.append("=" * 120)
            lines.append("REWARD TIMELINE - UNIFIED TABLE (delta, total, multiplier per step)")
            lines.append("=" * 120)

            # Build header dynamically based on reward IDs
            reward_ids = list(self.conditions.keys())

            # Header line
            header = f"{'Step':<6} | {'Total_Δ':<8} {'Total_Σ':<8} {'Total_M':<6} |"
            for rid in reward_ids:
                # Shorten reward ID for display (max 12 chars)
                short_id = (rid[:9] + "...") if len(rid) > 12 else rid
                header += f" {short_id+'_Δ':<12} {short_id+'_Σ':<12} {short_id+'_M':<6} |"
            lines.append(header)
            lines.append("-" * len(header))

            # Build unified data - one row per step
            num_steps = len(self.reward_vector)
            for step in range(num_steps):
                # Total reward columns
                step_total = self.reward_vector[step]
                step_delta = step_total if step == 0 else (step_total - self.reward_vector[step - 1])
                step_mult = 1.0 if step_delta > 0 else 0.0

                row = f"{step:<6} | {step_delta:<8.2f} {step_total:<8.2f} {step_mult:<6.1f} |"

                # Sub-reward columns
                for rid in reward_ids:
                    timeline = self.per_reward_timeline.get(rid, [])
                    # Find entry for this step
                    entry = next((e for e in timeline if e["step"] == step), None)

                    if entry:
                        delta = entry["delta"]
                        total = entry["total"]
                        mult = entry["multiplier"]
                        row += f" {delta:<12.2f} {total:<12.2f} {mult:<6.1f} |"
                    else:
                        # No activity this step
                        row += f" {0.0:<12.2f} {'-':<12} {0.0:<6.1f} |"

                lines.append(row)

            lines.append("=" * 120)
            return "\n".join(lines)

        else:
            raise ValueError(f"Unknown format: {format}. Use 'dict', 'numpy', or 'table'")

    def timeline(self):
        """Print event timeline - CLEAN & ELEGANT"""
        if not self.history:
            print("No reward history to display")
            return

        print("\n" + "=" * 80)
        print("REWARD TIMELINE")
        print("=" * 80)

        for h in self.history:
            print(f"t={h['time']:6.2f}s: +{h['step_reward']:6.1f} pts → {h['total']:6.1f} total")
            for t in h['triggered']:
                mult_str = f" (×{t['multiplier']:.2f})" if 'multiplier' in t else ""
                print(f"  └─ {t['condition']} (+{t['points']:.1f} pts{mult_str})")

        print("=" * 80)
        print(f"FINAL TOTAL: {self.total:.1f} pts")
        print("=" * 80 + "\n")

    def export_csv(self, filename):
        """Export reward history to CSV - CLEAN"""
        import csv

        if not self.history:
            print("No reward history to export")
            return

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time", "step_reward", "total", "condition", "points", "multiplier"])

            for h in self.history:
                for t in h["triggered"]:
                    writer.writerow([
                        h["time"],
                        h["step_reward"],
                        h["total"],
                        t["condition"],
                        t["points"],
                        t.get("multiplier", 1.0)
                    ])

        print(f"✓ Exported {len(self.history)} events to {filename}")

    def to_json(self) -> dict:
        """I know how to serialize myself"""
        conditions_json = []
        for cond_id, data in self.conditions.items():
            condition = data["condition"]
            reward = data["reward"]

            if isinstance(condition, Condition):
                conditions_json.append({
                    "asset": condition.asset,
                    "prop": condition.prop,
                    "op": condition.op,
                    "val": condition.val,
                    "points": reward,
                    "id": cond_id
                })
            else:
                # SequenceCondition - store as display string
                conditions_json.append({
                    "display": display_condition(condition),
                    "points": reward,
                    "id": cond_id
                })

        return {
            "conditions": conditions_json,
            "total": self.total,
            "history": self.history
        }

    @classmethod
    def from_json(cls, data: dict):
        """I know how to deserialize myself"""
        modal = cls()
        modal.total = data.get("total", 0)
        modal.history = data.get("history", [])

        for cond_data in data.get("conditions", []):
            modal.add(
                cond_data["asset"],
                cond_data["prop"],
                cond_data["op"],
                cond_data["val"],
                cond_data["points"],
                cond_data.get("id")
            )

        return modal

    def reset(self):
        """Reset episode-specific state - PURE MOP!

        Resets all episode tracking (rewards, timestamps, history)
        while preserving configuration (conditions, initial_state).

        Used by ExperimentOps.reset() for FAST episode resets (10ms)
        without recompiling (300ms+).
        """
        import time

        # Reset episode tracking
        self.start_time = time.time()
        self.event_timestamps = {}
        self.total = 0
        self.discrete_total = 0
        self.smooth_total = 0
        self.history = []
        self.reward_vector = []  # Clear trajectory for new episode
        self.per_reward_totals = {cond_id: 0.0 for cond_id in self.conditions.keys()}  # PURE MOP: Reset per-reward tracking
        self.per_reward_timeline = {}  # Clear timeline for new episode

        # BUGFIX: Clear initial_state so it's captured fresh at start of new episode
        self.initial_state = {}

        # Clear baseline tracking (multiplier normalization)
        self.initial_multipliers = {}

        # Preserve conditions (configuration, not episode state)
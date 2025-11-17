"""
EXPERIMENT ARTIFACT MODAL - Execution History Recording
=======================================================

PURE MOP: Records EXECUTION HISTORY ONLY (not configuration)

This modal RECORDS what happened during experiment execution:
- Every action submitted
- Every state snapshot
- Every reward triggered
- Timing information

MOP PATTERN: The artifact IS A MODAL that:
1. SELF-BUILDS: Creates recording structure automatically
2. SELF-SYNCS: Records state at every step
3. SELF-RENDERING: Generates JSON artifacts
4. SELF-COMPOSING: Artifacts compose into experiment histories

SEPARATION OF CONCERNS:
- GOD MODAL (ExperimentModal): Saves CONFIGURATION (scene, agents, config)
- THIS MODAL: Records EXECUTION HISTORY (what happened each step)

KEY INSIGHT: "Time-Travel Debugging"
- Record execution step-by-step
- Replay exact sequence later
- Analyze failures without re-running

What Gets Recorded (Per Step):
- Actions submitted
- Controls sent to actuators
- Full state snapshot
- Rewards triggered
- Events fired
- Timing (timestamp, duration)

Use Cases:
- Debugging: "Why did the robot fail at step 543?"
- Analysis: "What was the reward progression?"
- Performance: "Which steps took longest?"
- Reproducibility: "Verify deterministic behavior"

Usage:
    # During experiment (automatic in RuntimeEngine!)
    artifact = ExperimentArtifactModal(experiment_dir=exp_dir)

    for step in range(1000):
        artifact.record_step_start(step)
        artifact.record_controls(controls)
        # ... execute step ...
        artifact.record_state(state)
        artifact.record_rewards(rewards)
        artifact.record_step_end()

    artifact.save()  # Writes experiment_artifact.json

    # Later: Analyze
    artifact = ExperimentArtifactModal.load(exp_dir)
    step_543 = artifact.get_step(543)  # Why did it fail here?
    print(f"State: {step_543.state}")
    print(f"Actions: {step_543.actions}")

Pattern: MODAL-TO-MODAL ARTIFACT
- RuntimeEngine WRITES artifacts (during execution)
- Analysis tools READ artifacts (for replay/debug)
- GOD Modal owns configuration, we own execution history
- Clean separation, no redundancy!
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class StepRecord:
    """Record of a single simulation step

    Contains everything that happened in one step:
    - Actions submitted
    - State snapshot
    - Rewards triggered
    - Events fired
    - Timing info
    """
    step_number: int
    timestamp: float  # Time since experiment start

    # What happened this step
    actions: Dict[str, Any] = field(default_factory=dict)
    controls: Dict[str, float] = field(default_factory=dict)  # Actual control signals
    state: Dict[str, Any] = field(default_factory=dict)
    rewards: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    step_duration: float = 0.0  # How long this step took (wall time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return asdict(self)


@dataclass
class ExperimentArtifactModal:
    """Modal that records experiment EXECUTION HISTORY (not configuration)

    PURE MOP - SINGLE RESPONSIBILITY: Execution history only!

    This IS A MODAL that self-manages execution recording:
    - Self-Building: Creates recording structure
    - Self-Syncing: Records every step automatically
    - Self-Rendering: Outputs JSON artifacts
    - Self-Composing: Can combine with other artifacts

    Key Pattern: EXECUTION HISTORY ONLY
    - GOD Modal (ExperimentModal): Owns configuration
    - THIS Modal: Owns execution history
    - No redundancy, clean separation!

    What We Record:
    - Step-by-step execution trace
    - Actions, controls, state, rewards per step
    - Timing information

    What We DON'T Record:
    - Initial configuration (GOD Modal has this!)
    - Scene setup (GOD Modal has this!)
    - Agent/robot config (GOD Modal has this!)
    """

    experiment_dir: Path
    enabled: bool = True  # Set False to disable recording

    # Minimal metadata (links to GOD Modal)
    experiment_id: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # EXECUTION HISTORY ONLY (no configuration data!)
    steps: List[StepRecord] = field(default_factory=list, init=False)

    # Current step tracking
    _current_step: Optional[StepRecord] = field(default=None, init=False)
    _step_start_time: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        """SELF-BUILDING: Initialize recording structure"""
        if not isinstance(self.experiment_dir, Path):
            self.experiment_dir = Path(self.experiment_dir)

        # Generate experiment ID from dir name
        if not self.experiment_id:
            self.experiment_id = self.experiment_dir.name

        # Start timer
        if self.start_time is None:
            self.start_time = time.time()

    # === RECORDING METHODS (EXECUTION HISTORY ONLY) ===

    def record_step_start(self, step_number: int):
        """Mark start of new step

        Args:
            step_number: Current step number
        """
        if not self.enabled:
            return

        # Calculate timestamp since experiment start
        timestamp = time.time() - self.start_time if self.start_time else 0.0

        # Create step record
        self._current_step = StepRecord(
            step_number=step_number,
            timestamp=timestamp
        )
        self._step_start_time = time.time()

    def record_actions(self, actions: Dict[str, Any]):
        """Record actions submitted this step

        Args:
            actions: Dict of actions by action name
        """
        if not self.enabled or self._current_step is None:
            return

        self._current_step.actions = actions.copy()

    def record_controls(self, controls: Dict[str, float]):
        """Record control signals sent to actuators

        Args:
            controls: Dict of control values by actuator name
        """
        if not self.enabled or self._current_step is None:
            return

        self._current_step.controls = controls.copy()

    def record_state(self, state: Dict[str, Any]):
        """Record state snapshot after step

        Args:
            state: Full state dict from state_sync
        """
        if not self.enabled or self._current_step is None:
            return

        self._current_step.state = state.copy()

    def record_rewards(self, rewards: Dict[str, float]):
        """Record rewards triggered this step

        Args:
            rewards: Dict of reward values by reward ID
        """
        if not self.enabled or self._current_step is None:
            return

        self._current_step.rewards = rewards.copy()

    def record_events(self, events: List[Dict[str, Any]]):
        """Record events fired this step

        Args:
            events: List of event dicts
        """
        if not self.enabled or self._current_step is None:
            return

        self._current_step.events = events.copy()

    def record_step_end(self):
        """Mark end of step and save record"""
        if not self.enabled or self._current_step is None:
            return

        # Calculate step duration
        if self._step_start_time:
            self._current_step.step_duration = time.time() - self._step_start_time

        # Save step record
        self.steps.append(self._current_step)

        # Clear current step
        self._current_step = None
        self._step_start_time = None

    # === ARTIFACT GENERATION (SELF-RENDERING) ===

    def save(self, filename: str = "experiment_artifact.json") -> Path:
        """SELF-RENDERING: Save artifact to JSON

        Args:
            filename: Name of artifact file

        Returns:
            Path to saved artifact
        """
        if not self.enabled:
            return self.experiment_dir / filename

        # Mark end time
        self.end_time = time.time()

        # Calculate duration
        duration = self.end_time - self.start_time if self.start_time and self.end_time else 0.0

        # Build artifact (EXECUTION HISTORY ONLY)
        artifact = {
            "_meta": {
                "version": "2.0",
                "created": datetime.now().isoformat(),
                "pattern": "EXECUTION-HISTORY-MODAL",
                "description": "Execution history recording for time-travel debugging",
                "note": "Configuration saved by GOD Modal (ExperimentModal) - this is execution history only!"
            },
            "experiment_info": {
                "id": self.experiment_id,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "duration": duration,
                "total_steps": len(self.steps)
            },
            # NO initial_state - GOD Modal owns configuration!
            "steps": [step.to_dict() for step in self.steps]
        }

        # Ensure directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON
        output_path = self.experiment_dir / filename
        with open(output_path, 'w') as f:
            json.dump(artifact, f, indent=2)

        return output_path

    @classmethod
    def load(cls, experiment_dir: Path, filename: str = "experiment_artifact.json") -> 'ExperimentArtifactModal':
        """Load artifact from JSON

        Args:
            experiment_dir: Directory containing artifact
            filename: Name of artifact file

        Returns:
            ExperimentArtifactModal instance loaded from file
        """
        artifact_path = Path(experiment_dir) / filename

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        with open(artifact_path) as f:
            data = json.load(f)

        # Create instance
        artifact = cls(
            experiment_dir=Path(experiment_dir),
            experiment_id=data["experiment_info"]["id"],
            start_time=data["experiment_info"]["start_time"],
            end_time=data["experiment_info"]["end_time"],
            enabled=False  # Don't record during replay!
        )

        # Load steps (execution history only!)
        # For configuration, load GOD Modal (ExperimentModal)
        for step_data in data["steps"]:
            step = StepRecord(
                step_number=step_data["step_number"],
                timestamp=step_data["timestamp"],
                actions=step_data["actions"],  # OFFENSIVE - crash if missing!
                controls=step_data["controls"],  # OFFENSIVE - crash if missing!
                state=step_data["state"],  # OFFENSIVE - crash if missing!
                rewards=step_data["rewards"],  # OFFENSIVE - crash if missing!
                events=step_data["events"],  # OFFENSIVE - crash if missing!
                step_duration=step_data["step_duration"]  # OFFENSIVE - crash if missing!
            )
            artifact.steps.append(step)

        return artifact

    # === ANALYSIS METHODS ===

    def get_step(self, step_number: int) -> Optional[StepRecord]:
        """Get specific step by number

        Args:
            step_number: Step number to retrieve

        Returns:
            StepRecord if found, None otherwise
        """
        for step in self.steps:
            if step.step_number == step_number:
                return step
        return None

    def get_state_at_step(self, step_number: int) -> Optional[Dict[str, Any]]:
        """Get state snapshot at specific step

        Args:
            step_number: Step number

        Returns:
            State dict if found, None otherwise
        """
        step = self.get_step(step_number)
        return step.state if step else None

    def get_reward_history(self, reward_id: Optional[str] = None) -> List[tuple]:
        """Get reward history over time

        Args:
            reward_id: Specific reward to track (None = all rewards)

        Returns:
            List of (step_number, reward_value) tuples
        """
        history = []
        for step in self.steps:
            if reward_id:
                if reward_id in step.rewards:
                    history.append((step.step_number, step.rewards[reward_id]))
            else:
                total = sum(step.rewards.values())
                if total > 0:
                    history.append((step.step_number, total))
        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get experiment statistics

        Returns:
            Dict with statistics (total steps, rewards, timing, etc.)
        """
        total_rewards = sum(sum(step.rewards.values()) for step in self.steps)
        avg_step_duration = sum(step.step_duration for step in self.steps) / len(self.steps) if self.steps else 0

        return {
            "total_steps": len(self.steps),
            "total_rewards": total_rewards,
            "avg_step_duration": avg_step_duration,
            "duration": self.end_time - self.start_time if self.start_time and self.end_time else 0.0
        }


# === HELPER FUNCTIONS ===

def create_artifact(experiment_dir: Path, enabled: bool = True) -> ExperimentArtifactModal:
    """Helper: Create new experiment artifact

    Args:
        experiment_dir: Directory for experiment
        enabled: Whether to enable recording

    Returns:
        ExperimentArtifactModal instance
    """
    return ExperimentArtifactModal(experiment_dir=experiment_dir, enabled=enabled)


def load_artifact(experiment_dir: Path) -> ExperimentArtifactModal:
    """Helper: Load experiment artifact

    Args:
        experiment_dir: Directory containing artifact

    Returns:
        Loaded ExperimentArtifactModal instance
    """
    return ExperimentArtifactModal.load(experiment_dir)


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("EXPERIMENT ARTIFACT MODAL - Example")
    print("=" * 80)

    # Create test artifact
    test_dir = Path("/tmp/test_artifact")
    test_dir.mkdir(exist_ok=True)

    artifact = create_artifact(test_dir)

    # Simulate recording (execution history only!)
    print("\n📝 Recording experiment execution...")
    print("   (Note: Configuration saved by GOD Modal - we only record execution)")

    # Record execution steps (no initial_state - GOD Modal owns that!)
    for i in range(10):
        artifact.record_step_start(i)
        artifact.record_actions({"move_forward": {"speed": 1.0}})
        artifact.record_controls({"wheel_left": 1.0, "wheel_right": 1.0})
        artifact.record_state({"robot": {"position": [i, 0, 0]}})
        artifact.record_rewards({"distance": i * 10})
        artifact.record_step_end()

    # Save artifact
    path = artifact.save()
    print(f"   ✓ Artifact saved to: {path}")

    # Load and analyze
    print("\n📊 Analyzing artifact...")
    loaded = load_artifact(test_dir)

    stats = loaded.get_statistics()
    print(f"   Total steps: {stats['total_steps']}")
    print(f"   Total rewards: {stats['total_rewards']}")
    print(f"   Avg step duration: {stats['avg_step_duration']:.4f}s")

    # Get specific step
    step_5 = loaded.get_step(5)
    if step_5:
        print(f"\n   Step 5 state: {step_5.state}")
        print(f"   Step 5 rewards: {step_5.rewards}")

    print("\n✅ Experiment Artifact Modal working!")

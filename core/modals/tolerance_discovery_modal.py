"""
TOLERANCE DISCOVERY MODAL - Auto-discover actuator tolerances from physics

MOP PRINCIPLE: MODAL-TO-MODAL COMMUNICATION
- Discovery Modal runs physics tests
- Discovers realistic error bounds (gravity droop, etc.)
- Passes tolerances to Actuator Modal
- Actuator Modal owns values
- Reward Modal reads from Actuator Modal

This is PURE AUTO-DISCOVERY from physics simulation!
"""

from dataclasses import dataclass, field
from typing import Dict
import numpy as np


@dataclass
class ToleranceDiscoveryModal:
    """Modal that discovers actuator tolerances from physics simulation

    MOP Principle #1 (AUTO-DISCOVERY): Let physics tell us realistic error bounds
    MOP Principle #5 (MODAL-TO-MODAL): Discovery Modal → Actuator Modal

    Usage:
        discovery = ToleranceDiscoveryModal()
        tolerances = discovery.discover_all_tolerances(ops)

        # Pass to actuator creation (MODAL-TO-MODAL!)
        actuators = create_all_actuators(tolerances=tolerances)
    """

    # Actuator test configurations
    actuator_configs: Dict[str, Dict] = field(default_factory=lambda: {
        "lift": {
            "test_positions": [0.3, 0.5, 0.7, 0.9],
            "unit": "meters"
        },
        "arm": {
            "test_positions": [0.1, 0.2, 0.3, 0.4],
            "unit": "meters"
        },
        "gripper": {
            "test_positions": [-0.01, 0.0, 0.02],
            "unit": "radians"
        },
        "wrist_yaw": {
            "test_positions": [-1.0, 0.0, 1.0],
            "unit": "radians"
        },
        "wrist_pitch": {
            "test_positions": [-1.0, 0.0, 0.5],
            "unit": "radians"
        },
        "wrist_roll": {
            "test_positions": [-1.0, 0.0, 1.0],
            "unit": "radians"
        },
        "head_pan": {
            "test_positions": [-1.0, 0.0, 1.0],
            "unit": "radians"
        },
        "head_tilt": {
            "test_positions": [-1.0, 0.0, 0.5],
            "unit": "radians"
        }
    })

    settle_steps: int = 500  # Steps to wait for physics to settle
    verbose: bool = True

    def discover_all_tolerances(self, ops) -> Dict[str, float]:
        """Discover tolerances for all actuators from physics

        MODAL-TO-MODAL COMMUNICATION: Returns tolerances to pass to Actuator Modal

        Args:
            ops: ExperimentOps instance (already compiled with robot)

        Returns:
            Dict mapping actuator names to tolerances (meters or radians)
            Example: {"lift": 0.0143, "arm": 0.0000, ...}
        """
        if self.verbose:
            print("=" * 80)
            print("TOLERANCE DISCOVERY MODAL - AUTO-DISCOVERING FROM PHYSICS")
            print("=" * 80)
            print()
            print("MOP Principle #5 (MODAL-TO-MODAL COMMUNICATION):")
            print("  - Discovery Modal runs physics tests")
            print("  - Discovers realistic error bounds")
            print("  - Passes to Actuator Modal (not hardcoded!)")
            print()

        tolerances = {}

        for actuator_name, config in self.actuator_configs.items():
            tolerance = self._discover_actuator_tolerance(
                ops, actuator_name, config
            )
            tolerances[actuator_name] = tolerance

        if self.verbose:
            print()
            print("=" * 80)
            print("TOLERANCE DISCOVERY COMPLETE")
            print("=" * 80)
            self._print_summary(tolerances)

        return tolerances

    def _discover_actuator_tolerance(self, ops, actuator_name: str, config: Dict) -> float:
        """Discover tolerance for single actuator

        Method:
        1. Command actuator to target position
        2. Run physics for settle_steps (let it stabilize)
        3. Measure actual position
        4. Calculate error = |target - actual|
        5. Return max error + 1 std dev (safety margin)
        """
        if self.verbose:
            print(f"┌{'─' * 78}┐")
            print(f"│ {actuator_name.upper():^76} │")
            print(f"└{'─' * 78}┘")
            print()

        errors = []

        for target_pos in config["test_positions"]:
            if self.verbose:
                print(f"  Testing {actuator_name} → {target_pos:.3f} {config['unit']}")

            # Command actuator to target position
            ops.engine.backend.set_controls({actuator_name: target_pos})

            # Run physics to settle
            for _ in range(self.settle_steps):
                ops.engine.step()

            # Sync actuator state from simulation
            ops.engine.backend.sync_actuators_from_backend(ops.engine.robot)

            # Get actual position
            actuator = ops.engine.robot.actuators[actuator_name]
            actual_pos = actuator.position

            # Calculate error
            error = abs(target_pos - actual_pos)
            errors.append(error)

            if self.verbose:
                status = "✓" if error < 0.01 else "⚠" if error < 0.05 else "✗"
                print(f"    {status} Commanded: {target_pos:6.3f}, "
                      f"Actual: {actual_pos:6.3f}, Error: {error*1000:5.1f}mm")

        # Calculate statistics
        max_error = max(errors)
        avg_error = np.mean(errors)
        std_error = np.std(errors)

        # Tolerance = max error + 1 std dev (safety margin)
        tolerance = max_error + std_error

        if self.verbose:
            print()
            print(f"  📊 Statistics:")
            print(f"     Max error:  {max_error*1000:5.1f}mm")
            print(f"     Avg error:  {avg_error*1000:5.1f}mm")
            print(f"     Std dev:    {std_error*1000:5.1f}mm")
            print(f"     TOLERANCE:  {tolerance*1000:5.1f}mm (max + 1σ)")
            print()

        return tolerance

    def _print_summary(self, tolerances: Dict[str, float]):
        """Print summary table of discovered tolerances"""
        print()
        print(f"{'Actuator':<15} {'Tolerance':<12} {'Physics Explanation':<40}")
        print("─" * 80)

        explanations = {
            "lift": "Gravity droop (vertical load)",
            "arm": "Perfect (horizontal, no gravity)",
            "gripper": "Spring compliance",
            "wrist_yaw": "Large error (needs investigation)",
            "wrist_pitch": "Gravity + inertia",
            "wrist_roll": "Minimal error",
            "head_pan": "Perfect (balanced)",
            "head_tilt": "Gravity droop"
        }

        for actuator_name, tolerance in tolerances.items():
            tol_mm = tolerance * 1000
            explanation = explanations.get(actuator_name, "Unknown")
            print(f"{actuator_name:<15} {tol_mm:>10.1f}mm {explanation:<40}")

        print()
        print("MODAL-TO-MODAL: Saving to JSON (modal creates JSON!)")
        print()

    def save_to_json(self, tolerances: Dict[str, float], output_path: str):
        """Save discovered tolerances to JSON - SELF-GENERATION!

        MOP Principle #2 (SELF-GENERATION): Modal WRITES JSON, doesn't read it

        Args:
            tolerances: Discovered tolerances from discover_all_tolerances()
            output_path: Where to save JSON file
        """
        import json
        from pathlib import Path

        # Modal creates JSON artifact
        json_data = {
            "_meta": {
                "description": "Auto-discovered actuator tolerances from physics simulation",
                "pattern": "MODAL-TO-MODAL",
                "source": "ToleranceDiscoveryModal",
                "principle": "MOP #1 (AUTO-DISCOVERY) + MOP #2 (SELF-GENERATION)"
            },
            "tolerances": tolerances
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        if self.verbose:
            print(f"✓ Tolerances saved to: {output_path}")
            print()



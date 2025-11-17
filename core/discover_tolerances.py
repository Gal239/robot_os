#!/usr/bin/env python3
"""
EMPIRICAL TOLERANCE DISCOVERY - PURE MOP!

Uses ToleranceDiscoveryModal to auto-discover tolerances from physics:
1. Position actuators (lift, arm, gripper, wrist, head) - measure settling error
2. Base movement (wheels) - measure odometry + IMU error

This is the MOP way - discover from physics, not hardcode!
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.tolerance_discovery_modal import ToleranceDiscoveryModal
import numpy as np


def discover_all_tolerances():
    """Discover tolerances for ALL actuators - PURE MOP!"""

    print("="*80)
    print("EMPIRICAL TOLERANCE DISCOVERY - PURE MOP!")
    print("="*80)
    print("\nUsing ToleranceDiscoveryModal for auto-discovery...")
    print("This will test both position actuators AND base movement!\n")

    # Create experiment
    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("tolerance_discovery", width=5, length=5, height=3)
    ops.add_robot("stretch")

    # Compile to create engine
    ops.compile()

    # Use ToleranceDiscoveryModal for PURE MOP auto-discovery!
    discovery = ToleranceDiscoveryModal(verbose=True)
    tolerances = discovery.discover_all_tolerances(ops)

    return tolerances


def save_tolerances(tolerances):
    """Save discovered tolerances to JSON file"""
    import json
    output_file = Path(__file__).parent / "modals" / "stretch" / "discovered_tolerances.json"

    with open(output_file, 'w') as f:
        json.dump(tolerances, f, indent=2)

    print(f"💾 Saved to: {output_file}")
    print("\nDiscovered tolerances:")
    for name, tol in sorted(tolerances.items()):
        if tol > 0:
            print(f"  {name:20s} = {tol:.6f}")


if __name__ == "__main__":
    tolerances = discover_all_tolerances()
    save_tolerances(tolerances)

    print("\n✅ Tolerance discovery complete!")
    print("Update actuator_modals.py to load from discovered_tolerances.json")
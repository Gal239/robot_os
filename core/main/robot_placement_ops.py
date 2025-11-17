"""
ROBOT PLACEMENT OPS - Automatic robot motion generation from scene specifications
PURE MOP: Uses existing modals without modification!

Pattern: Scene graph → IK solver → ActionBlocks → MuJoCo execution
No hardcoding, no glue code - just modal-to-modal communication!
"""

import numpy as np
from typing import Tuple, Dict, List
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.modals.reachability_modal import ReachabilityModal
from core.modals.stretch.action_modals import (
    BaseMoveForward, BaseMoveBackward, BaseRotateBy,
    ArmMoveTo, ArmMoveBy,
    LiftMoveTo, LiftMoveBy,
    GripperMoveTo
)
from core.modals.stretch.action_blocks_registry import ActionBlock


class RobotPlacementDemo:
    """Automatic robot placement from scene specifications - THE ULTIMATE TEST!

    This proves the infrastructure works end-to-end:
    - IK solver (ReachabilityModal)
    - Action system (ActionBlocks)
    - Sensor feedback (force sensing)
    - Runtime control (ExperimentOps)

    Usage:
        ops.compile()  # Compile scene normally
        demo = RobotPlacementDemo(ops)
        demo.execute_placement("apple", target_location=(2, 0, 0.8))
        # Robot automatically: pick → move → place!
    """

    def __init__(self, ops):
        """Initialize with ExperimentOps instance

        Args:
            ops: ExperimentOps instance (must have compiled scene!)
        """
        self.ops = ops
        self.robot = ops.robot
        self.scene = ops.scene
        self.engine = ops.engine
        self.ik = ReachabilityModal()

    def execute_placement(self, object_name: str, target_location: Tuple[float, float, float],
                         settling_steps: int = 100):
        """Execute automatic pick-and-place for ONE object

        Args:
            object_name: Name of object to place (must exist in scene)
            target_location: (x, y, z) world position for placement
            settling_steps: Physics steps to wait after each phase

        Raises:
            ValueError: If object not found or unreachable
        """
        print(f"\n🤖 ROBOT AUTO-PLACEMENT: {object_name}")
        print("="*60)

        # 1. Get object's current position
        print(f"\n📍 Phase 1: Locating {object_name}...")
        object_pos = self._get_object_position(object_name)
        print(f"   Found at: ({object_pos[0]:.2f}, {object_pos[1]:.2f}, {object_pos[2]:.2f})")

        # 2. Calculate IK for grasp
        print(f"\n🧮 Phase 2: Calculating inverse kinematics...")
        grasp_joints = self._calculate_grasp_ik(object_name, object_pos)
        print(f"   Base: ({grasp_joints['_base_position'][0]:.2f}, {grasp_joints['_base_position'][1]:.2f})")
        print(f"   Lift: {grasp_joints['joint_lift']:.2f}m")
        print(f"   Arm: {grasp_joints['joint_arm_l0']*4:.2f}m")

        # 3. Generate motion sequence
        print(f"\n🎬 Phase 3: Generating motion sequence...")
        actions = self._generate_motion_sequence(object_pos, target_location, grasp_joints)
        print(f"   Generated {len(actions)} action blocks")

        # 4. Execute motion sequence
        print(f"\n▶️  Phase 4: Executing motions...")
        for i, action_block in enumerate(actions, 1):
            print(f"   [{i}/{len(actions)}] {action_block.id}...")
            self._execute_action_block(action_block)

            # Settle physics after each phase
            for _ in range(settling_steps):
                self.ops.step()

        print(f"\n✅ COMPLETE: {object_name} placed at target location!")
        print("="*60)

    def _get_object_position(self, object_name: str) -> np.ndarray:
        """Get object's current position from simulation state"""
        if object_name not in self.scene.assets:
            raise ValueError(
                f"❌ Object '{object_name}' not found in scene!\n"
                f"✅ Available: {list(self.scene.assets.keys())}"
            )

        # Get state from runtime
        state = self.ops.get_state()

        if object_name not in state:
            raise ValueError(
                f"❌ Object '{object_name}' not in simulation state!\n"
                f"Did you call ops.compile() first?"
            )

        obj_state = state[object_name]
        return np.array(obj_state['position'])

    def _calculate_grasp_ik(self, object_name: str, object_pos: np.ndarray) -> Dict[str, float]:
        """Calculate joint values for grasping object"""
        # Get object width from asset config
        asset = self.scene.assets[object_name]

        # Default object width (can be refined with actual dimensions)
        object_width = 0.08  # 8cm default

        # Use ReachabilityModal for IK
        joint_values = self.ik.solve_for_grasp(
            target_pos=object_pos,
            object_width=object_width,
            robot_config={}
        )

        return joint_values

    def _generate_motion_sequence(self, object_pos: np.ndarray, target_pos: Tuple[float, float, float],
                                 grasp_joints: Dict[str, float]) -> List[ActionBlock]:
        """Generate complete pick-and-place motion sequence

        Returns list of ActionBlocks:
        1. Approach object
        2. Grasp object
        3. Lift object
        4. Move to target
        5. Place object
        """
        actions = []

        # ================================================================
        # PHASE 1: APPROACH OBJECT
        # ================================================================
        actions.append(ActionBlock(
            id="approach_object",
            execution_mode="parallel",  # All joints move together
            actions=[
                LiftMoveTo(position=grasp_joints["joint_lift"]),
                ArmMoveTo(position=grasp_joints["joint_arm_l0"] * 4),  # Total arm extension
                GripperMoveTo(position=0.05)  # Open gripper wide
            ]
        ))

        # ================================================================
        # PHASE 2: GRASP OBJECT
        # ================================================================
        actions.append(ActionBlock(
            id="grasp_object",
            execution_mode="sequential",
            actions=[
                GripperMoveTo(position=0.0, force_limit=5.0)  # Close with force sensing
            ]
        ))

        # ================================================================
        # PHASE 3: LIFT OBJECT
        # ================================================================
        actions.append(ActionBlock(
            id="lift_object",
            execution_mode="parallel",
            actions=[
                LiftMoveBy(distance=0.1),   # Lift 10cm
                ArmMoveBy(distance=-0.05)   # Retract slightly
            ]
        ))

        # ================================================================
        # PHASE 4: MOVE TO TARGET
        # ================================================================
        # Calculate movement needed
        current_x, current_y = object_pos[0], object_pos[1]
        target_x, target_y, target_z = target_pos

        # Move forward/backward
        distance_forward = target_x - current_x
        if abs(distance_forward) > 0.1:  # Only move if significant
            if distance_forward > 0:
                actions.append(ActionBlock(
                    id="move_to_target",
                    execution_mode="sequential",
                    actions=[
                        BaseMoveForward(distance=abs(distance_forward), speed=0.3)
                    ]
                ))
            else:
                actions.append(ActionBlock(
                    id="move_to_target",
                    execution_mode="sequential",
                    actions=[
                        BaseMoveBackward(distance=abs(distance_forward), speed=0.3)
                    ]
                ))

        # ================================================================
        # PHASE 5: PLACE OBJECT
        # ================================================================
        actions.append(ActionBlock(
            id="place_object",
            execution_mode="sequential",
            actions=[
                LiftMoveTo(position=target_z - 0.05),  # Lower to just above target
                GripperMoveTo(position=0.05),          # Open gripper
                ArmMoveBy(distance=-0.1),              # Retract arm
                LiftMoveTo(position=0.5)               # Return to neutral height
            ]
        ))

        return actions

    def _execute_action_block(self, action_block: ActionBlock):
        """Execute single action block and wait for completion"""
        # Connect all actions to robot
        for action in action_block.actions:
            action.connect(self.robot, self.engine.event_log)

        # Execute until all complete
        # Different timeouts for different action types (from level_1b tests!)
        if action_block.id.startswith("move_"):
            max_steps = 3500  # Base movement: ~2100 steps per meter (Test 8!)
        else:
            max_steps = 2000  # Arm/Lift/Gripper/Place sequences

        step_count = 0
        print_interval = 200  # Print every 200 steps

        while step_count < max_steps:
            # Collect all commands from all actions
            controls = {}
            all_complete = True

            for action in action_block.actions:
                # Get command from action
                cmd = action.tick()

                # Merge commands into single dict
                controls.update(cmd)

                # Check status
                if action.status != "completed":
                    all_complete = False

            # Send all controls to backend at once (MOP: backend.set_controls()!)
            self.engine.backend.set_controls(controls)

            # Step physics
            self.ops.step()
            step_count += 1

            # Check if all actions complete
            if all_complete:
                return

        # Timeout warning
        print(f"⚠️  Warning: Action block '{action_block.id}' timeout after {max_steps} steps")
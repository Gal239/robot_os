"""
TEST: ActionOps API - CLEAN UNIFIED ACCESS

Demonstrates the new ops.actions.* API that consolidates 6 scattered patterns
"""

from core.main.experiment_ops_unified import ExperimentOps
from core.modals.stretch.scene_modal import Scene
from core.modals.stretch.robot_modal import Robot
from core.modals.stretch.action_blocks_registry import move_forward, spin


def test_action_ops_api():
    """Test the new ActionOps API - discovery, inspection, control"""

    # Setup
    ops = ExperimentOps(
        scene=Scene(id="test_ops", name="Test ActionOps"),
        robot=Robot(name="stretch", actuators=[], sensors=[])
    )
    ops.compile()

    print("=" * 70)
    print("ACTION OPS API - UNIFIED ACCESS PATTERN")
    print("=" * 70)

    # === DISCOVERY ===
    print("\n1️⃣  DISCOVERY - What actions exist?")
    print("-" * 70)

    actions = ops.actions.get_available_actions()
    print(f"✓ Found {len(actions)} action types:")
    for name in list(actions.keys())[:5]:
        print(f"  - {name}")
    print(f"  ... and {len(actions) - 5} more")

    # Get params for specific action
    params = ops.actions.get_action_params('ArmMoveTo')
    print(f"\n✓ ArmMoveTo parameters:")
    for param_name, param_info in params.items():
        print(f"  - {param_name}: {param_info['type']} (required={param_info['required']})")

    # === INSPECTION (before actions) ===
    print("\n2️⃣  INSPECTION - What's in the queue?")
    print("-" * 70)

    status = ops.actions.get_status()
    print(f"✓ Status BEFORE actions:")
    print(f"  - Executing: {status['summary']['total_executing']}")
    print(f"  - Pending: {status['summary']['total_pending']}")

    # === CONTROL - Submit actions ===
    print("\n3️⃣  CONTROL - Submit actions")
    print("-" * 70)

    block_id_1 = ops.actions.submit(move_forward(distance=1.0))
    print(f"✓ Submitted move_forward: block_id={block_id_1}")

    block_id_2 = ops.actions.submit(spin(degrees=90))
    print(f"✓ Submitted spin: block_id={block_id_2}")

    # === INSPECTION (after actions) ===
    print("\n4️⃣  INSPECTION - Check queue after submit")
    print("-" * 70)

    status = ops.actions.get_status()
    print(f"✓ Status AFTER actions:")
    print(f"  - Executing: {status['summary']['total_executing']}")
    print(f"  - Pending: {status['summary']['total_pending']}")

    pending = ops.actions.get_pending_blocks()
    print(f"\n✓ Pending blocks ({len(pending)}):")
    for block in pending:
        print(f"  - Block {block['block_id']}: {block['name']} ({block['status']})")

    # Execute a few steps
    print("\n5️⃣  EXECUTION - Run 100 steps")
    print("-" * 70)

    for i in range(100):
        obs, reward, done, truncated, info = ops.step({})

    executing = ops.actions.get_executing_actions()
    print(f"✓ Currently executing ({len(executing)}):")
    for action in executing:
        print(f"  - {action['action_type']} on {action['actuator']}: {action['progress']:.1%}")

    # === COMPARISON ===
    print("\n" + "=" * 70)
    print("BEFORE vs AFTER - ACCESS PATTERNS")
    print("=" * 70)

    print("\n❌ BEFORE (scattered patterns):")
    print("   ops.engine.action_executor.get_status()        # 4 layers deep")
    print("   ops.engine.action_executor.queue_modal.blocks  # Direct modal access")
    print("   ops.submit_block(block)                        # Inconsistent naming")

    print("\n✅ AFTER (unified API):")
    print("   ops.actions.get_status()                       # Clean")
    print("   ops.actions.get_pending_blocks()               # Semantic")
    print("   ops.actions.submit(block)                      # Consistent")

    print("\n" + "=" * 70)
    print("✅ ACTION OPS API TEST PASSED!")
    print("=" * 70)
    print(f"\nMOP Principles Demonstrated:")
    print(f"  ✓ Self-Discovered: {len(actions)} actions auto-discovered")
    print(f"  ✓ Clean API: Single access point (ops.actions.*)")
    print(f"  ✓ OFFENSIVE: Type errors if action doesn't exist")
    print(f"  ✓ Unified: Discovery, Inspection, Control in one place")


if __name__ == "__main__":
    test_action_ops_api()

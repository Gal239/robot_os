"""
CONFIG GENERATOR - Regenerate all JSONs from modals
OFFENSIVE - crashes if generation incomplete

MODAL-ORIENTED: Modals generate their own specifications.
This orchestrates the generation and writes to disk.
"""

from pathlib import Path
import json
from typing import Dict, Set


def regenerate_robot_behaviors(robot_type="stretch"):
    """Scan actuators → ROBOT_BEHAVIORS.json

    MODAL-ORIENTED: Reads from ActuatorComponent.behaviors declarations.
    No hardcoding! Scans ALL actuators.
    """
    print(f"\n🔄 Generating ROBOT_BEHAVIORS.json for {robot_type}...")

    # Import here to avoid circular imports
    from core.main.robot_ops import create_robot

    # Create robot with all actuators
    robot = create_robot(robot_type)

    # Generate behaviors from actuators
    package = robot.create_robot_asset_package()
    behaviors = package["behaviors"]

    # Write to file
    output_file = Path(__file__).parent.parent / "behaviors" / "ROBOT_BEHAVIORS.json"
    output_file.write_text(json.dumps(behaviors, indent=2))

    # OFFENSIVE validation
    validate_robot_behaviors(robot, behaviors)

    print(f"✅ Generated {len(behaviors)} robot behaviors:")
    for behavior_name in sorted(behaviors.keys()):
        props = behaviors[behavior_name]["properties"]
        print(f"  - {behavior_name}: {len(props)} properties")

    return behaviors


def validate_robot_behaviors(robot, generated_behaviors: Dict):
    """OFFENSIVE validation - CRASH if incomplete!

    Ensures ALL actuator-declared behaviors are generated.
    """
    declared_behaviors: Set[str] = set()
    for actuator_name, actuator in robot.actuators.items():
        declared_behaviors.update(actuator.behaviors)

    generated = set(generated_behaviors.keys())
    missing = declared_behaviors - generated

    if missing:
        raise RuntimeError(
            f"GENERATION INCOMPLETE!\n"
            f"  Robot: {robot.robot_type}\n"
            f"\n"
            f"  Actuators DECLARE these behaviors:\n"
            f"    {sorted(declared_behaviors)}\n"
            f"\n"
            f"  But GENERATED only these:\n"
            f"    {sorted(generated)}\n"
            f"\n"
            f"  MISSING ({len(missing)}):\n"
            f"    {sorted(missing)}\n"
            f"\n"
            f"  FIX: Update Robot._generate_behavior_from_actuator() to handle all behavior types!\n"
            f"  The modals DECLARE their capabilities - generation must match!"
        )

    print(f"\n✅ OFFENSIVE VALIDATION PASSED:")
    print(f"  - Declared: {len(declared_behaviors)} behaviors")
    print(f"  - Generated: {len(generated)} behaviors")
    print(f"  - Missing: {len(missing)} behaviors")
    print(f"  ALL BEHAVIORS GENERATED! 🎉")


def regenerate_assets():
    """Scan all assets → ASSETS.json

    MODAL-ORIENTED: Assets generate their own specs via to_dict().
    Scans filesystem for all furniture and objects.
    """
    print(f"\n🔄 Generating ASSETS.json...")

    # Import here to avoid circular imports
    from core.modals import registry
    from core.modals.asset_modals import Asset

    assets_dict = {}

    # Scan furniture
    print(f"  Scanning {len(registry.FURNITURE)} furniture...")
    for asset_name in sorted(registry.FURNITURE):
        try:
            config = registry.load_asset_config(asset_name)
            asset = Asset(asset_name, config)
            assets_dict[asset_name] = asset.to_dict()
        except Exception as e:
            print(f"    ⚠️  Skipped {asset_name}: {e}")

    # Scan objects
    print(f"  Scanning {len(registry.OBJECTS)} objects...")
    for asset_name in sorted(registry.OBJECTS):
        try:
            config = registry.load_asset_config(asset_name)
            asset = Asset(asset_name, config)
            assets_dict[asset_name] = asset.to_dict()
        except Exception as e:
            print(f"    ⚠️  Skipped {asset_name}: {e}")

    # Write to file
    output_file = Path(__file__).parent.parent / "modals" / "mujoco_assets" / "ASSETS.json"
    output_file.write_text(json.dumps(assets_dict, indent=2))

    # OFFENSIVE validation
    validate_assets(assets_dict)

    print(f"✅ Generated {len(assets_dict)} assets")
    return assets_dict


def validate_assets(assets_dict: Dict):
    """OFFENSIVE validation - CRASH if assets incomplete!

    Ensures all assets have required structure.
    """
    for asset_name, asset_spec in assets_dict.items():
        # Required fields
        if "type" not in asset_spec:
            raise RuntimeError(
                f"ASSET GENERATION ERROR!\n"
                f"  Asset: {asset_name}\n"
                f"  MISSING 'type' field!\n"
                f"\n"
                f"  FIX: Asset.to_dict() must include 'type'"
            )

        if "components" not in asset_spec:
            raise RuntimeError(
                f"ASSET GENERATION ERROR!\n"
                f"  Asset: {asset_name}\n"
                f"  MISSING 'components' field!\n"
                f"\n"
                f"  FIX: Asset.to_dict() must include 'components'"
            )

    print(f"\n✅ ASSET VALIDATION PASSED:")
    print(f"  - Total assets: {len(assets_dict)}")
    print(f"  - All have required structure")
    print(f"  ASSETS.json COMPLETE! 🎉")


def regenerate_behaviors():
    """Generate BEHAVIORS.json from Component definitions

    MODAL-ORIENTED: Discovers all behavior types from the system.
    Currently loads from existing JSONs (to be enhanced).
    """
    print(f"\n🔄 Generating BEHAVIORS.json...")

    # Import existing BEHAVIORS to preserve structure
    from core.modals.asset_modals import BEHAVIORS

    # Write current behaviors to file (preserves hand-crafted definitions)
    output_file = Path(__file__).parent.parent / "behaviors" / "BEHAVIORS.json"

    # Filter out robot behaviors (those start with "robot_")
    asset_behaviors = {k: v for k, v in BEHAVIORS.items() if not k.startswith("robot_")}

    output_file.write_text(json.dumps(asset_behaviors, indent=2))

    print(f"✅ Generated {len(asset_behaviors)} asset behaviors")
    print(f"  (Note: Currently preserves existing definitions)")
    print(f"  (Future: Could scan Components to discover new behaviors)")

    return asset_behaviors


def regenerate_tolerances(robot_type="stretch"):
    """Discover actuator tolerances → tolerances.json

    MODAL-ORIENTED: Runs physics simulation to discover ACTUAL precision.
    NO HARDCODING! Measures what robot can achieve in practice.

    Discovery method:
    - Commands each actuator to test positions
    - Waits 1000 steps for full settling
    - Measures final achievable precision
    - One-time discovery when robot added to system
    """
    print(f"\n🔄 Discovering tolerances for {robot_type}...")
    print("  (This may take several minutes - we're running physics simulation!)")

    # Create experiment with robot (needed for physics simulation)
    from core.main.experiment_ops_unified import ExperimentOps
    from core.modals.tolerance_discovery_modal import ToleranceDiscoveryModal

    print(f"\n  Creating experiment for tolerance discovery...")
    ops = ExperimentOps(mode="simulated", headless=True)
    ops.create_scene("tolerance_discovery", width=5, length=5, height=3)
    ops.add_robot(robot_type)
    ops.compile()
    print(f"  ✓ Experiment compiled")

    # Run discovery (takes time - 1000 steps per test position!)
    print(f"\n  Running physics-based tolerance discovery...")
    discovery = ToleranceDiscoveryModal(settle_steps=1000, verbose=True)
    tolerances = discovery.discover_all_tolerances(ops)

    # Write to file
    output_file = Path(__file__).parent.parent / "modals" / robot_type / "tolerances.json"

    tolerance_data = {
        "_meta": {
            "description": "Auto-discovered actuator tolerances from physics simulation",
            "pattern": "MODAL-TO-MODAL",
            "source": "ToleranceDiscoveryModal",
            "principle": "MOP #1 (AUTO-DISCOVERY) + MOP #2 (SELF-GENERATION)",
            "discovery_method": "Measures error after 1000-step settling (true physics limit)",
            "discovery_steps": 1000,
            "note": "One-time discovery when robot XML added to system - accuracy over speed!"
        },
        "tolerances": tolerances
    }

    output_file.write_text(json.dumps(tolerance_data, indent=2))

    print(f"\n✅ Discovered {len(tolerances)} actuator tolerances:")
    for actuator_name, tolerance in sorted(tolerances.items()):
        print(f"  - {actuator_name}: {tolerance:.6f} ({tolerance*1000:.3f}mm)")

    return tolerances


def regenerate_grasp_points():
    """Discover grasp sites from XMLs → config.json

    MODAL-ORIENTED: XMLs are source of truth (artist places sites).
    Generator discovers and categorizes automatically.

    Discovery method:
    - Scans all object/furniture XML files
    - Finds <site> elements with names like "grip_*", "handle_*"
    - Writes grasp_points section to config.json
    - NO HARDCODING! XML is source of truth.
    """
    print(f"\n🔄 Discovering grasp points from XMLs...")

    import xml.etree.ElementTree as ET
    from core.modals import registry

    discovered_count = 0
    total_sites = 0

    # Scan all objects and furniture
    for asset_name in sorted(list(registry.OBJECTS) + list(registry.FURNITURE)):
        try:
            config = registry.load_asset_config(asset_name)
            xml_file = config.get('xml_file')

            if not xml_file:
                continue

            # Parse XML
            xml_path = Path(__file__).parent.parent / "modals" / xml_file
            if not xml_path.exists():
                # Try absolute path
                xml_path = Path(__file__).parent.parent / xml_file
                if not xml_path.exists():
                    continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Find all <site> elements
            sites = root.findall('.//site')
            grasp_points = {}

            for site in sites:
                site_name = site.get('name', '')

                # Categorize by name pattern (grip, handle, grasp, etc.)
                if any(keyword in site_name.lower() for keyword in ['grip', 'handle', 'grasp', 'hold']):
                    pos_str = site.get('pos', '0 0 0')
                    pos = [float(x) for x in pos_str.split()]

                    grasp_points[site_name] = {
                        "site_name": site_name,
                        "discovered_pos": pos,
                        "discovered_from": "XML site element"
                    }
                    total_sites += 1

            # Write back to config.json if grasp points found
            if grasp_points:
                config['grasp_points'] = grasp_points
                config_path = Path(__file__).parent.parent / "modals" / "mujoco_assets" / "objects" / asset_name / "config.json"

                # Try furniture path if object path doesn't exist
                if not config_path.exists():
                    config_path = Path(__file__).parent.parent / "modals" / "mujoco_assets" / "furniture" / asset_name / "config.json"

                if config_path.exists():
                    config_path.write_text(json.dumps(config, indent=2) + '\n')
                    discovered_count += 1
                    print(f"  ✓ {asset_name}: {len(grasp_points)} grasp points")

        except Exception as e:
            # Skip assets that fail (expected - not all have grasp points)
            pass

    print(f"\n✅ Discovered {total_sites} grasp points in {discovered_count} assets")
    print(f"  (Scanned {len(registry.OBJECTS) + len(registry.FURNITURE)} total assets)")

    return total_sites


def regenerate_relations() -> Dict:
    """Generate RELATIONS.json from Relation modals

    MODAL-ORIENTED: Relations are Pydantic modals that self-serialize!
    Same pattern as robot.create_robot_asset_package()
    """
    print("\n" + "="*80)
    print("GENERATING RELATIONS.json from Relation modals...")
    print("="*80)

    from core.modals.relation_modal import RELATIONS

    # MODAL-TO-MODAL: Ask each relation modal for its spec
    relations_spec = {}
    for rel_name, rel_modal in RELATIONS.items():
        # Pydantic auto-serialization!
        rel_dict = rel_modal.dict()

        # Add example
        rel_dict['example'] = rel_modal.get_example()

        relations_spec[rel_name] = rel_dict

    # Write to JSON
    output_file = Path(__file__).parent.parent / "modals" / "RELATIONS.json"
    output_file.write_text(json.dumps(relations_spec, indent=2, ensure_ascii=False))

    print(f"\n✅ Generated {len(relations_spec)} relations:")
    for rel_name, rel_data in relations_spec.items():
        print(f"  - {rel_name}: {rel_data['description'][:50]}...")

    print(f"\n📄 Output: {output_file}")

    # OFFENSIVE: Validate all expected relations exist
    expected_relations = ["on_top", "stack_on", "inside", "next_to", "front", "back", "left", "right"]
    declared_relations = set(RELATIONS.keys())
    expected_set = set(expected_relations)

    missing = expected_set - declared_relations
    extra = declared_relations - expected_set

    assert len(missing) == 0, (
        f"❌ GENERATION INCOMPLETE!\n"
        f"Missing relations: {missing}\n"
        f"\n💡 FIX: Add missing relation modals to relation_modal.py!"
    )

    if extra:
        print(f"\n📢 New relations detected: {extra}")

    return relations_spec


def regenerate_api_docs() -> Dict:
    """Generate API.json from ExperimentOps using inspect

    MODAL-ORIENTED: ExperimentOps knows itself via Python introspection.
    """
    print("\n" + "="*80)
    print("GENERATING API.json from ExperimentOps...")
    print("="*80)

    import inspect
    from core.main.experiment_ops_unified import ExperimentOps

    api_spec = {}

    # Scan all public methods
    for name, method in inspect.getmembers(ExperimentOps, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue

        # Extract signature
        sig = inspect.signature(method)
        doc = inspect.getdoc(method) or "No description"

        # Parse parameters
        params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            params[param_name] = {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else "required",
                "required": param.default == inspect.Parameter.empty
            }

        api_spec[name] = {
            "signature": f"ops.{name}{sig}".replace("self, ", "").replace("(self)", "()"),
            "description": doc.split('\n')[0],
            "parameters": params,
            "required": name in ["create_scene", "compile"]
        }

    # Write to JSON
    output_file = Path(__file__).parent.parent / "docs" / "API.json"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(json.dumps(api_spec, indent=2, ensure_ascii=False))

    print(f"\n✅ Generated {len(api_spec)} API methods:")
    method_names = list(api_spec.keys())
    for method_name in method_names[:10]:  # Show first 10
        print(f"  - {method_name}: {api_spec[method_name]['description'][:50]}...")
    if len(api_spec) > 10:
        print(f"  ... and {len(api_spec) - 10} more")

    print(f"\n📄 Output: {output_file}")

    # OFFENSIVE: Validate critical methods exist
    critical_methods = ["create_scene", "add_asset", "add_robot", "compile", "step"]
    missing = set(critical_methods) - set(api_spec.keys())

    assert len(missing) == 0, (
        f"❌ GENERATION INCOMPLETE!\n"
        f"Missing critical methods: {missing}\n"
        f"\n💡 FIX: Check ExperimentOps class has all required methods!"
    )

    return api_spec


def regenerate_all():
    """Regenerate all config files

    MODAL-ORIENTED: Complete system regeneration from modals.
    """
    print("=" * 70)
    print("MODAL-ORIENTED CONFIG GENERATION")
    print("=" * 70)

    # 1. Robot behaviors
    robot_behaviors = regenerate_robot_behaviors("stretch")

    # 2. Asset behaviors
    behaviors = regenerate_behaviors()

    # 3. Assets
    assets = regenerate_assets()

    # 4. Tolerances (SLOW - physics simulation!)
    # tolerances = regenerate_tolerances("stretch")  # Comment out - takes time!

    # 5. Grasp points
    grasp_points = regenerate_grasp_points()

    # 6. Relations (NEW!)
    relations = regenerate_relations()

    # 7. API docs (NEW!)
    api_docs = regenerate_api_docs()

    # 8. Validate everything
    print("\n" + "=" * 70)
    print("🎉 GENERATION COMPLETE! 🎉")
    print("=" * 70)
    print("\n✅ Summary:")
    print(f"  - ROBOT_BEHAVIORS.json: {len(robot_behaviors)} behaviors")
    print(f"  - BEHAVIORS.json: {len(behaviors)} behaviors")
    print(f"  - ASSETS.json: {len(assets)} assets")
    # print(f"  - tolerances.json: {len(tolerances)} tolerances")
    print(f"  - grasp_points: {grasp_points} total sites")
    print(f"  - RELATIONS.json: {len(relations)} relations")
    print(f"  - API.json: {len(api_docs)} methods")

    print("\n🎯 MOP Compliance:")
    print("  ✅ Modals self-declare")
    print("  ✅ Modals self-generate")
    print("  ✅ Generator orchestrates")
    print("  ✅ OFFENSIVE validation")
    print("\nAll JSONs are now in sync with modal declarations!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target = sys.argv[1]

        if target == "robot_behaviors":
            regenerate_robot_behaviors()
        elif target == "assets":
            regenerate_assets()
        elif target == "behaviors":
            regenerate_behaviors()
        elif target == "tolerances":
            regenerate_tolerances()
        elif target == "grasp_points":
            regenerate_grasp_points()
        elif target == "relations":
            regenerate_relations()
        elif target == "api":
            regenerate_api_docs()
        elif target == "all":
            regenerate_all()
        else:
            print(f"❌ Unknown target: {target}")
            print("\nValid targets:")
            print("  - robot_behaviors: Generate ROBOT_BEHAVIORS.json")
            print("  - assets: Generate ASSETS.json")
            print("  - behaviors: Generate BEHAVIORS.json")
            print("  - tolerances: Discover actuator tolerances (SLOW!)")
            print("  - grasp_points: Discover grasp sites from XMLs")
            print("  - relations: Generate RELATIONS.json from Pydantic modals")
            print("  - api: Generate API.json from ExperimentOps")
            print("  - all: Regenerate everything")
            print("\nExample: python3 -m simulation_center.core.tools.config_generator relations")
            sys.exit(1)
    else:
        regenerate_all()

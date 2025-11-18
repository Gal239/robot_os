#!/usr/bin/env python3
"""Demo all is_facing() classifications with example dot products"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Show all classification ranges
print("\n" + "="*80)
print("is_facing() DOT PRODUCT CLASSIFICATIONS")
print("="*80)

classifications = [
    (1.00, "directly_facing", "perfect alignment (0°)"),
    (0.85, "facing", "strong alignment (~30°)"),
    (0.50, "partially_facing", "angled ~45-60°"),
    (0.00, "perpendicular", "side-by-side, ~90°"),
    (-0.50, "partially_away", "angled away ~120-135°"),
    (-0.85, "facing_away", "strong opposite (~150°)"),
    (-1.00, "directly_opposite", "perfect opposite, 180°"),
]

print(f"\n{'Dot Product':<15} {'Classification':<25} {'Description':<40}")
print("-" * 80)
for dot, classification, description in classifications:
    facing = "✓ FACING" if dot > 0.7 else "✗ not facing"
    print(f"{dot:>+6.2f}          {classification:<25} {description:<40} {facing}")

print("\n" + "="*80)
print("USAGE EXAMPLE:")
print("="*80)
print("""
result = ops.is_facing("stretch.arm", "apple")
print(result["dot_explain"])
# Output: "stretch.arm is perpendicular to apple (side-by-side, ~90°)"

if result["facing"]:  # Uses threshold (default 0.7)
    print(f"Arm is facing apple! (dot={result['dot']:.2f})")
else:
    print(f"Arm NOT facing apple (dot={result['dot']:.2f})")

# Classification breakdown:
# result["dot"]         = -0.122 (numeric)
# result["dot_class"]   = "perpendicular" (category)
# result["dot_explain"] = "stretch.arm is perpendicular to apple..." (full explanation)
# result["facing"]      = False (threshold check)
# result["distance"]    = 2.12 (meters)
""")

print("="*80)

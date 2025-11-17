"""
Relation Modals - PURE MOP Implementation

Each spatial relation is a PYDANTIC MODAL that:
- Self-describes (description, parameters)
- Self-validates (requires_dimensions, dimension_aware)
- Self-calculates (calculate method)
- Self-serializes (Pydantic .dict())

NO if/elif chains! Relations are modals that know themselves!

MOP PRINCIPLES:
- AUTO-DISCOVERY: Relations declare their own properties
- SELF-VALIDATION: validate() checks if relation can be applied
- SELF-CALCULATION: calculate() knows how to compute position
- SELF-SERIALIZATION: Pydantic .dict() for JSON generation
"""

from typing import Dict, Tuple, List, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class RelationModal(BaseModel, ABC):
    """Base Relation Modal - All relations inherit from this

    MOP PRINCIPLES:
    - AUTO-DISCOVERY: Relations declare their own properties
    - SELF-VALIDATION: validate() checks if relation can be applied
    - SELF-CALCULATION: calculate() knows how to compute position
    - SELF-SERIALIZATION: Pydantic .dict() for JSON generation
    """

    name: str
    description: str
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = Field(default_factory=list)
    offset_direction: Optional[str] = None

    class Config:
        # Allow arbitrary types for abstract methods
        arbitrary_types_allowed = True

    @abstractmethod
    def calculate(
        self,
        base_pos: Tuple[float, float, float],
        base_dims: Dict[str, float],
        placed_dims: Dict[str, float],
        **kwargs
    ) -> Tuple[float, float, float]:
        """SELF-CALCULATION: Modal knows how to calculate position!

        Args:
            base_pos: Base object position (x, y, z)
            base_dims: Base object dimensions {width, depth, height}
            placed_dims: Placed object dimensions {width, depth, height}
            **kwargs: Additional parameters (surface_position, offset, etc.)

        Returns:
            Calculated position (x, y, z)
        """
        pass

    def validate(self, has_dimensions: bool) -> bool:
        """SELF-VALIDATION: Check if relation can be applied

        OFFENSIVE: If requires_dimensions and not has_dimensions → crash!
        """
        if self.requires_dimensions and not has_dimensions:
            raise ValueError(
                f"❌ Relation '{self.name}' REQUIRES dimensions!\n"
                f"Either provide dimension data OR use manual distance parameter."
            )
        return True

    def get_example(self) -> str:
        """Generate example usage"""
        return f'ops.add_asset("object", relative_to="base", relation="{self.name}")'


class OnTopRelation(RelationModal):
    """Place object on surface - MOP: Modal knows how to calculate!"""

    name: str = "on_top"
    description: str = "Place object on surface with optional surface_position"
    dimension_aware: bool = True
    requires_dimensions: bool = False  # Can use manual distance fallback
    parameters: List[str] = ["surface_position", "offset", "distance"]
    offset_direction: str = "+Z"

    def calculate(self, base_pos, base_dims, placed_dims, surface_position=None, offset=None, **kwargs):
        """MuJoCo uses CENTER positioning - calculate from half-heights

        PURE MOP: Furniture declares its surface position via 'surface_z'!
        - If base_dims has 'surface_z': Use absolute surface position (furniture)
        - Otherwise: Calculate from center position (objects)
        """
        # Check if base has explicit surface position (furniture with "surface" behavior)
        if 'surface_z' in base_dims:
            # Furniture with declared surface - use absolute surface position
            surface_z = base_dims['surface_z'] + placed_dims['height']/2
        else:
            # Objects - use center + half-heights
            surface_z = base_pos[2] + base_dims['height']/2 + placed_dims['height']/2

        # Apply surface positioning if provided
        x_offset = 0.0
        y_offset = 0.0
        if offset:
            x_offset = offset[0] if len(offset) > 0 else 0.0
            y_offset = offset[1] if len(offset) > 1 else 0.0

        return (base_pos[0] + x_offset, base_pos[1] + y_offset, surface_z)


class StackOnRelation(RelationModal):
    """Vertical stacking - MOP: MUST use dimensions!"""

    name: str = "stack_on"
    description: str = "Vertically stack objects (center-aligned, uses real dimensions)"
    dimension_aware: bool = True
    requires_dimensions: bool = True  # OFFENSIVE: MUST have dimensions!
    parameters: List[str] = ["distance"]
    offset_direction: str = "+Z"

    def calculate(self, base_pos, base_dims, placed_dims, **kwargs):
        """PURE MOP: Dimensions from modals determine exact stacking height"""
        stack_z = base_pos[2] + base_dims['height']/2 + placed_dims['height']/2
        return (base_pos[0], base_pos[1], stack_z)


class InsideRelation(RelationModal):
    """Place inside container"""

    name: str = "inside"
    description: str = "Place object inside container"
    dimension_aware: bool = True
    requires_dimensions: bool = False
    parameters: List[str] = []

    def calculate(self, base_pos, base_dims, placed_dims, **kwargs):
        """Place at container center (slightly above bottom)"""
        inside_z = base_pos[2] - base_dims['height']/4
        return (base_pos[0], base_pos[1], inside_z)


class NextToRelation(RelationModal):
    """Place adjacent on same surface"""

    name: str = "next_to"
    description: str = "Place object adjacent on same surface"
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = ["spacing", "distance"]
    offset_direction: str = "+X"

    def calculate(self, base_pos, base_dims, placed_dims, spacing=0.0, **kwargs):
        """Account for BOTH widths"""
        x_offset = base_dims['width']/2 + placed_dims['width']/2 + spacing
        return (base_pos[0] + x_offset, base_pos[1], base_pos[2])


class FrontRelation(RelationModal):
    """Place in front (+Y axis)"""

    name: str = "front"
    description: str = "In front of object (+Y axis)"
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = ["distance"]
    offset_direction: str = "+Y"

    def calculate(self, base_pos, base_dims, placed_dims, distance=None, **kwargs):
        """Account for BOTH depths"""
        y_offset = base_dims['depth']/2 + placed_dims['depth']/2
        if distance:
            y_offset = distance
        return (base_pos[0], base_pos[1] + y_offset, base_pos[2])


class BackRelation(RelationModal):
    """Place behind (-Y axis)"""

    name: str = "back"
    description: str = "Behind object (-Y axis)"
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = ["distance"]
    offset_direction: str = "-Y"

    def calculate(self, base_pos, base_dims, placed_dims, distance=None, **kwargs):
        """Account for BOTH depths"""
        y_offset = base_dims['depth']/2 + placed_dims['depth']/2
        if distance:
            y_offset = distance
        return (base_pos[0], base_pos[1] - y_offset, base_pos[2])


class LeftRelation(RelationModal):
    """Place left (-X axis)"""

    name: str = "left"
    description: str = "Left of object (-X axis)"
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = ["distance"]
    offset_direction: str = "-X"

    def calculate(self, base_pos, base_dims, placed_dims, distance=None, **kwargs):
        """Account for BOTH widths"""
        x_offset = base_dims['width']/2 + placed_dims['width']/2
        if distance:
            x_offset = distance
        return (base_pos[0] - x_offset, base_pos[1], base_pos[2])


class RightRelation(RelationModal):
    """Place right (+X axis)"""

    name: str = "right"
    description: str = "Right of object (+X axis)"
    dimension_aware: bool = True
    requires_dimensions: bool = True
    parameters: List[str] = ["distance"]
    offset_direction: str = "+X"

    def calculate(self, base_pos, base_dims, placed_dims, distance=None, **kwargs):
        """Account for BOTH widths"""
        x_offset = base_dims['width']/2 + placed_dims['width']/2
        if distance:
            x_offset = distance
        return (base_pos[0] + x_offset, base_pos[1], base_pos[2])


# REGISTRY: All relation modals available
# MOP: Relations self-register by being in this dict!
RELATIONS: Dict[str, RelationModal] = {
    "on_top": OnTopRelation(),
    "stack_on": StackOnRelation(),
    "inside": InsideRelation(),
    "next_to": NextToRelation(),
    "front": FrontRelation(),
    "back": BackRelation(),
    "left": LeftRelation(),
    "right": RightRelation()
}


def get_relation(name: str) -> RelationModal:
    """Get relation modal by name - OFFENSIVE if not found!

    MOP: Relations expose themselves to other modals via registry.
    """
    if name not in RELATIONS:
        available = list(RELATIONS.keys())
        raise ValueError(
            f"❌ Relation '{name}' not found!\n"
            f"Available relations: {available}\n"
            f"\n💡 FIX: Add {name}Relation class to relation_modal.py!"
        )
    return RELATIONS[name]

#!/usr/bin/env python3
"""
QUATERNION UTILITIES - Foundation for orientation system

Pure math utilities for quaternion conversions.
MuJoCo uses (w, x, y, z) format for quaternions.
"""

import math
from typing import Tuple


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """
    Convert Euler angles to quaternion (MuJoCo format: w, x, y, z)

    Args:
        roll: Rotation around X axis (radians)
        pitch: Rotation around Y axis (radians)
        yaw: Rotation around Z axis (radians)

    Returns:
        Quaternion tuple (w, x, y, z)

    Example:
        >>> euler_to_quaternion(0, 0, 0)  # No rotation
        (1.0, 0.0, 0.0, 0.0)

        >>> euler_to_quaternion(0, 0, math.pi/2)  # 90° yaw (Z rotation)
        (0.707..., 0.0, 0.0, 0.707...)
    """
    # Calculate half angles
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    # Calculate quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return (w, x, y, z)


def quaternion_to_euler(w: float, x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert quaternion to Euler angles (radians)

    Args:
        w, x, y, z: Quaternion components (MuJoCo format)

    Returns:
        Tuple of (roll, pitch, yaw) in radians

    Example:
        >>> quaternion_to_euler(1, 0, 0, 0)  # Identity quaternion
        (0.0, 0.0, 0.0)
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90° if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def calculate_facing_quaternion(from_pos: Tuple[float, float, float],
                                 to_pos: Tuple[float, float, float]) -> Tuple[float, float, float, float]:
    """
    Calculate quaternion to make object at from_pos face toward to_pos

    Calculates the rotation needed to align the forward direction (+X for Stretch robot)
    with the vector pointing from from_pos to to_pos.

    Args:
        from_pos: Starting position (x, y, z)
        to_pos: Target position to face toward (x, y, z)

    Returns:
        Quaternion (w, x, y, z) that rotates forward direction to face target

    Example:
        >>> calculate_facing_quaternion((0, 0, 0), (1, 0, 0))  # Face +X
        # Returns identity quaternion (no rotation needed - already facing +X)
    """
    # Calculate direction vector
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    # Ignore Z component for 2D facing (ground plane rotation only)

    # Calculate yaw angle to face target (in XY plane)
    # atan2(dy, dx) because Stretch robot default forward is +X, not +Y
    yaw = math.atan2(dy, dx)

    # Convert yaw to quaternion (no roll/pitch, only yaw rotation)
    return euler_to_quaternion(0, 0, yaw)


def quaternion_multiply(q1: Tuple[float, float, float, float],
                       q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Multiply two quaternions (combine rotations)

    Args:
        q1: First quaternion (w, x, y, z)
        q2: Second quaternion (w, x, y, z)

    Returns:
        Result quaternion (w, x, y, z)

    Example:
        >>> q1 = (1, 0, 0, 0)  # Identity
        >>> q2 = (0.707, 0, 0, 0.707)  # 90° Z rotation
        >>> quaternion_multiply(q1, q2)
        (0.707, 0, 0, 0.707)  # Same as q2
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return (w, x, y, z)


def quaternion_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Normalize quaternion to unit length

    Args:
        q: Quaternion (w, x, y, z)

    Returns:
        Normalized quaternion (w, x, y, z)
    """
    w, x, y, z = q
    magnitude = math.sqrt(w*w + x*x + y*y + z*z)

    if magnitude == 0:
        # Return identity quaternion if magnitude is zero
        return (1.0, 0.0, 0.0, 0.0)

    return (w / magnitude, x / magnitude, y / magnitude, z / magnitude)


# Preset quaternions for common orientations
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)  # No rotation
NORTH_QUAT = (1.0, 0.0, 0.0, 0.0)     # Face +Y (forward in MuJoCo)
SOUTH_QUAT = (0.0, 0.0, 0.0, 1.0)     # Face -Y (180° rotation)
EAST_QUAT = (0.707, 0.0, 0.0, 0.707)  # Face +X (90° CW around Z)
WEST_QUAT = (0.707, 0.0, 0.0, -0.707) # Face -X (90° CCW around Z)


if __name__ == "__main__":
    # Test conversions
    print("Testing quaternion utilities...")

    # Test 1: Identity quaternion
    quat = euler_to_quaternion(0, 0, 0)
    print(f"Identity: {quat}")
    assert abs(quat[0] - 1.0) < 0.001

    # Test 2: 90° yaw
    quat = euler_to_quaternion(0, 0, math.pi/2)
    print(f"90° yaw: {quat}")
    assert abs(quat[0] - 0.707) < 0.01
    assert abs(quat[3] - 0.707) < 0.01

    # Test 3: Facing quaternion
    quat = calculate_facing_quaternion((0, 0, 0), (1, 0, 0))
    print(f"Face +X: {quat}")

    quat = calculate_facing_quaternion((0, 0, 0), (0, 1, 0))
    print(f"Face +Y: {quat}")

    # Test 4: Round trip (euler -> quat -> euler)
    roll, pitch, yaw = 0.1, 0.2, 0.3
    quat = euler_to_quaternion(roll, pitch, yaw)
    roll2, pitch2, yaw2 = quaternion_to_euler(*quat)
    print(f"Round trip: ({roll}, {pitch}, {yaw}) -> {quat} -> ({roll2}, {pitch2}, {yaw2})")
    assert abs(roll - roll2) < 0.001
    assert abs(pitch - pitch2) < 0.001
    assert abs(yaw - yaw2) < 0.001

    print("\nAll tests passed!")

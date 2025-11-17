"""
DIFFERENTIAL DRIVE KINEMATICS - Like Hello Robot stretch_mujoco

MOP: Physics constants from Hello Robot official specs (MEASURED from real robot!)
OFFENSIVE: No magic numbers - these are EMPIRICAL constants from hardware specs

Constants source: hello-robot/stretch_mujoco/stretch_mujoco/config.py
"""

# Hello Robot Stretch 2/3 constants (measured from real robot!)
WHEEL_DIAMETER = 0.1016  # meters (4 inches)
WHEEL_SEPARATION = 0.3153  # meters (track width / wheelbase)


def diff_drive_inv_kinematics(v_linear: float, omega: float) -> tuple:
    """Convert (linear_vel, angular_vel) → (left_wheel_rad/s, right_wheel_rad/s)

    This is the INVERSE kinematics - from desired robot motion to wheel commands.
    Used for control: "I want to move at 0.1 m/s forward" → wheel commands.

    Math from Hello Robot:
        w_left = (V - (omega * L / 2)) / R
        w_right = (V + (omega * L / 2)) / R

    CRITICAL: MuJoCo velocity actuators with gear=3 divide commands by 3!
    stretch.xml: <velocity gear="3" kv="20" ctrlrange="-6 6"/>
    MuJoCo applies: actual_velocity = command / gear
    So we must multiply by gear to compensate!

    Args:
        v_linear: Linear velocity (m/s, positive = forward)
        omega: Angular velocity (rad/s, positive = counter-clockwise)

    Returns:
        (w_left, w_right) in rad/s, SCALED for MuJoCo gear=3 actuators

    Example:
        # Move forward at 0.1 m/s, no rotation
        w_left, w_right = diff_drive_inv_kinematics(0.1, 0.0)
        # Both wheels: ~5.9 rad/s (gear=3 compensated!)

        # Spin in place counter-clockwise at 1 rad/s
        w_left, w_right = diff_drive_inv_kinematics(0.0, 1.0)
        # w_left: -9.3 rad/s, w_right: +9.3 rad/s (gear=3 compensated!)
    """
    R = WHEEL_DIAMETER / 2
    L = WHEEL_SEPARATION
    GEAR = 3  # From stretch.xml: <velocity gear="3" ...>

    # Standard differential drive formula
    w_left = (v_linear - (omega * L / 2)) / R
    w_right = (v_linear + (omega * L / 2)) / R

    # Apply gear ratio compensation (MuJoCo divides by gear)
    w_left *= GEAR
    w_right *= GEAR

    return (w_left, w_right)


def diff_drive_fwd_kinematics(w_left: float, w_right: float) -> tuple:
    """Convert (left_wheel_rad/s, right_wheel_rad/s) → (linear_vel, angular_vel)

    This is the FORWARD kinematics - from wheel speeds to robot motion.
    Used for odometry: "Wheels are spinning at X rad/s" → robot velocity.

    Math from Hello Robot:
        V = R * (w_left + w_right) / 2.0
        omega = R * (w_right - w_left) / L

    Args:
        w_left: Left wheel angular velocity (rad/s)
        w_right: Right wheel angular velocity (rad/s)

    Returns:
        (v_linear, omega) in m/s and rad/s

    Example:
        # Both wheels at 1.969 rad/s
        v, omega = diff_drive_fwd_kinematics(1.969, 1.969)
        # v ≈ 0.1 m/s, omega ≈ 0 rad/s (straight forward)

        # Left wheel stopped, right wheel at 6 rad/s
        v, omega = diff_drive_fwd_kinematics(0.0, 6.0)
        # v > 0, omega > 0 (forward + turning left)
    """
    R = WHEEL_DIAMETER / 2
    L = WHEEL_SEPARATION

    v_linear = R * (w_left + w_right) / 2.0
    omega = R * (w_right - w_left) / L

    return (v_linear, omega)

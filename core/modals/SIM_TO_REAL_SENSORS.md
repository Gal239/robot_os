# Sim-to-Real Sensor Parity for Stretch Robot

**CRITICAL PRINCIPLE**: Only include sensors in simulation that exist on the real Stretch robot. This ensures policies trained in simulation can transfer to real hardware.

## Real Stretch Robot Sensors

### 1. IMU (Inertial Measurement Unit)
**Location**: Base (site `base_imu` in XML)
**Hardware**: Bosch BNO055 or similar
**Outputs**:
- 3-axis gyroscope (angular velocity)
- 3-axis accelerometer (linear acceleration)

**MuJoCo XML**:
```xml
<sensor>
  <gyro name="base_gyro" site="base_imu"/>
  <accelerometer name="base_accel" site="base_imu"/>
</sensor>
```

**Modal Class**: `IMUSensor` in `sensors_modals.py`
- Sim: `sync_from_mujoco()` reads `data.sensordata` for gyro/accel
- Real: `sync_from_hardware()` reads from I2C/serial IMU interface
- API: `{"gyro": [x,y,z], "accel": [x,y,z]}`

---

### 2. 2D Lidar (Rangefinder)
**Location**: Base (site `lidar` in XML)
**Hardware**: RPLidar A1 or A2 (360° 2D laser scanner)
**Outputs**:
- 360° distance measurements (typically 360-720 points)
- 5-10 Hz scan rate

**MuJoCo XML**:
```xml
<sensor>
  <rangefinder name="base_lidar" site="lidar" cutoff="10.0"/>
</sensor>
```

**Modal Class**: `LidarSensor` in `sensors_modals.py`
- Sim: `sync_from_mujoco()` raycasts in 360° pattern
- Real: `sync_from_hardware()` reads from RPLidar SDK
- API: `{"ranges": [float], "angles": [float]}`

**Note**: MuJoCo's rangefinder is single-ray. Full 360° scan requires multiple rays or custom implementation.

---

### 3. Cameras
**Real Hardware**:
- **D435i** (head): Intel RealSense RGB-D (640x480 RGB + depth, 90° FOV)
- **D405** (wrist): Intel RealSense depth camera (640x480 depth, 87° FOV)
- **Nav Camera** (head): Wide-angle navigation camera (fisheye, ~170° FOV)

**MuJoCo XML**:
```xml
<camera name="head_camera" pos="0 0 0.15" euler="0 0 0" fovy="90"/>
<camera name="wrist_camera" pos="0 0 0" euler="0 0 0" fovy="87"/>
<camera name="nav_camera" pos="0 0 0.15" euler="0 0 0" fovy="170"/>
```

**Modal Class**: `CameraSensor` in `sensors_modals.py`
- Sim: `sync_from_mujoco()` renders via `mujoco.MjrContext`
- Real: `sync_from_hardware()` reads from RealSense SDK
- API: `{"rgb": ndarray, "depth": ndarray, "timestamp": float}`

---

### 4. Joint Encoders
**Location**: All actuated joints (11 total on Stretch)
**Hardware**: Magnetic encoders on Dynamixel servos
**Outputs**:
- Joint position (radians or meters)
- Joint velocity (rad/s or m/s)

**MuJoCo XML**: Implicit (no explicit sensor tag needed)

**Modal Class**: `JointStateSensor` in `sensors_modals.py`
- Sim: `sync_from_mujoco()` reads `data.qpos`, `data.qvel`
- Real: `sync_from_hardware()` reads from Dynamixel SDK
- API: `{"positions": Dict[str, float], "velocities": Dict[str, float]}`

---

### 5. Motor Current Sensing (Force/Torque Estimation)
**Location**: All actuated joints
**Hardware**: Dynamixel servos report current draw
**Outputs**:
- Motor current (mA)
- Estimated torque/force (derived from current)

**MuJoCo XML**: No direct equivalent

**Modal Class**: `GripperForceSensor` in `sensors_modals.py:726-893`
- Sim: `sync_from_mujoco()` uses **contact forces** from `data.contact`
- Real: `sync_from_hardware()` uses **motor current** × force constant
- API: `{"force_left": float, "force_right": float, "contact_left": bool, "contact_right": bool}`

**Key Insight**: Different physics source (contacts vs. current), same API! This is the correct sim-to-real pattern.

---

## Sensors NOT on Real Robot (Do Not Add to Sim)

### ❌ Tactile/Touch Sensors on Fingertips
- Real Stretch does NOT have dedicated tactile sensors
- Uses motor current sensing instead (see above)
- MuJoCo 3.3+ touch sensors would break sim-to-real parity

### ❌ Force/Torque Sensors on Joints
- Real Stretch does NOT have dedicated F/T sensors
- Uses motor current estimation instead
- MuJoCo force sensors would provide unrealistic data

### ❌ Pressure Sensors
- Not present on real hardware

---

## Sim-to-Real Best Practices

1. **Sensor Parity**: Every simulation sensor must have real hardware equivalent
2. **API Consistency**: Sim and real sensors expose identical data structures
3. **Modal Pattern**:
   - `sync_from_mujoco(model, data, robot)` for simulation
   - `sync_from_hardware(robot_interface)` for real robot
   - Same `get_data()` method for both
4. **Behavior Declaration**: Sensors self-declare `behaviors` for automatic integration with rewards/actions
5. **Physics Differences OK**: Different underlying physics (e.g., contacts vs. current) is fine as long as API matches

## Example: Adding a New Sensor

If you want to add a new sensor, ask:
1. **Does it exist on the real Stretch robot?** If NO, stop here.
2. What is the hardware model/interface?
3. How does it connect (I2C, USB, serial)?
4. What is the data output format?
5. How will `sync_from_mujoco()` approximate it in simulation?

Then implement both sync methods with matching APIs.

---

## References

- Real Stretch documentation: https://docs.hello-robot.com/0.2/
- Stretch hardware specs: https://hello-robot.com/stretch-3-product
- MuJoCo sensors: https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor
- Modal pattern: See `core/modals/sensors_modals.py`

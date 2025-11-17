# Virtual Cameras (Bird's Eye View)

## Overview

Virtual cameras are **simulation-only sensors** that provide custom viewpoints. They are NOT real sensors - they only exist in MuJoCo simulation and cannot be used with real robot hardware.

**Key Features:**
- 🎥 Render from any position/angle you specify
- 🔄 Interactive - change angle during simulation
- 💾 Automatically save to timeline videos
- 🚫 Simulation-only (not for real hardware)
- ⚡ Works in headless mode (no viewer window needed)

## Quick Start

```python
from simulation_center.core.main.experiment_ops_unified import ExperimentOps

# Create experiment
ops = ExperimentOps(mode="simulated", headless=True)
ops.create_scene("room", width=10, length=10)
ops.add_robot("stretch")

# Add bird's eye camera
ops.add_free_camera('birds_eye',
                   lookat=(5, 5, 0.5),  # Look at center of room
                   distance=8.0,         # 8 meters away
                   elevation=-45)        # Looking down at 45°

ops.compile()
ops.step()  # Camera renders!
```

**Result:** Camera automatically saves to `timeline/cameras/birds_eye_rgb.mp4`

## API Reference

### add_free_camera()

```python
ops.add_free_camera(
    camera_id="birds_eye",      # Camera name
    lookat=(0, 0, 0.5),         # [x, y, z] point to look at
    distance=5.0,                # Distance from lookat (meters)
    azimuth=90.0,                # Horizontal angle (degrees)
    elevation=-30.0,             # Vertical angle (degrees)
    width=640,                   # Image width (pixels)
    height=480                   # Image height (pixels)
)
```

**Parameters:**

- **camera_id**: Unique name for the camera (e.g., 'birds_eye', 'side_view', 'top_down')
- **lookat**: `[x, y, z]` coordinates of the point the camera looks at
- **distance**: How far the camera is from the lookat point (in meters)
- **azimuth**: Horizontal rotation angle in degrees
  - `0` = looking forward (positive X direction)
  - `90` = looking right (positive Y direction)
  - `180` = looking backward (negative X direction)
  - `270` = looking left (negative Y direction)
- **elevation**: Vertical rotation angle in degrees
  - `-90` = pure top-down view
  - `0` = horizontal view
  - `90` = bottom-up view
- **width, height**: Resolution in pixels (default: 640x480)

### set_camera_angle()

```python
ops.set_camera_angle(
    camera_id="birds_eye",
    lookat=(2, 2, 0.5),  # Optional: new lookat point
    distance=10.0,       # Optional: new distance
    azimuth=180,         # Optional: new horizontal angle
    elevation=-60        # Optional: new vertical angle
)
```

**Use cases:**
- **Orbit** around the scene
- **Zoom** in/out
- **Follow** the robot
- **Switch** between top-down and angled views

## Usage Examples

### Example 1: Basic Bird's Eye View

```python
ops = ExperimentOps(headless=True)
ops.create_scene("room", width=10, length=10)
ops.add_robot("stretch")

# Simple bird's eye view (default settings)
ops.add_free_camera('birds_eye')

ops.compile()

# Simulate
for i in range(100):
    ops.step()

# Video saved to: timeline/cameras/birds_eye_rgb.mp4
```

### Example 2: Multiple Camera Angles

```python
# Add multiple cameras for different viewpoints
ops.add_free_camera('top_down',
                   lookat=(5, 5, 0.5),
                   distance=10.0,
                   elevation=-90)  # Pure top-down

ops.add_free_camera('side_view',
                   lookat=(0, 0, 0.5),
                   distance=4.0,
                   azimuth=0,
                   elevation=-15)  # Side view

ops.add_free_camera('corner_view',
                   lookat=(5, 5, 0),
                   distance=12.0,
                   azimuth=45,
                   elevation=-35)  # Corner angle

ops.compile()
ops.step()

# Result:
# - timeline/cameras/top_down_rgb.mp4
# - timeline/cameras/side_view_rgb.mp4
# - timeline/cameras/corner_view_rgb.mp4
```

### Example 3: Orbit Around Scene

```python
ops.add_free_camera('orbiting_cam', distance=7.0, elevation=-40)
ops.compile()

# Orbit 360 degrees around the scene
for angle in range(0, 360, 3):
    ops.set_camera_angle('orbiting_cam', azimuth=angle)
    ops.step()

# Result: Smooth rotating video
```

### Example 4: Zoom In/Out

```python
ops.add_free_camera('zoom_cam', distance=10.0)
ops.compile()

# Zoom in gradually
for i in range(100):
    distance = 10.0 - (i * 0.05)  # 10m → 5m
    ops.set_camera_angle('zoom_cam', distance=distance)
    ops.step()
```

### Example 5: Follow Robot

```python
ops.add_free_camera('follow_cam', elevation=-30)
ops.compile()

# Follow robot during movement
for i in range(200):
    # Get robot position
    robot_pos = ops.robot.actuators['base'].position  # Or from odometry

    # Update camera to look at robot
    ops.set_camera_angle('follow_cam',
                        lookat=(robot_pos[0], robot_pos[1], 0.5))
    ops.step()
```

### Example 6: Top-Down → Angled Transition

```python
ops.add_free_camera('transition_cam')
ops.compile()

# Start with top-down view
ops.set_camera_angle('transition_cam', elevation=-90)

for i in range(50):
    ops.step()

# Gradually transition to angled view
for i in range(100):
    elevation = -90 + (i * 0.6)  # -90° → -30°
    ops.set_camera_angle('transition_cam', elevation=elevation)
    ops.step()

# End with angled view
for i in range(50):
    ops.step()
```

## Common Camera Positions

### Bird's Eye View (Default)
```python
ops.add_free_camera('birds_eye',
                   lookat=(5, 5, 0.5),  # Center of 10x10 room
                   distance=8.0,
                   azimuth=90,
                   elevation=-30)
```
**Good for:** Overall scene observation, navigation tasks

### Pure Top-Down
```python
ops.add_free_camera('top_down',
                   lookat=(5, 5, 0),
                   distance=8.0,
                   elevation=-90)
```
**Good for:** 2D navigation, overhead maps, spatial planning

### Side View (Following Robot)
```python
ops.add_free_camera('side_view',
                   lookat=(0, 0, 0.5),
                   distance=3.0,
                   azimuth=180,
                   elevation=-10)
```
**Good for:** Manipulation tasks, arm movements, gripper observations

### Corner View (Isometric)
```python
ops.add_free_camera('corner',
                   lookat=(5, 5, 0.5),
                   distance=12.0,
                   azimuth=45,
                   elevation=-35)
```
**Good for:** Nice cinematic view, demonstrations, debugging

### Close-Up on Table
```python
ops.add_free_camera('table_close',
                   lookat=(2, 0, 0.8),  # Table position
                   distance=2.0,
                   azimuth=90,
                   elevation=-25)
```
**Good for:** Manipulation tasks, object interactions

## Integration with Streaming

For real-time streaming (WebSocket/REPL), you can change camera angles dynamically:

```python
# In your streaming server
def handle_camera_command(message):
    camera_id = message['camera_id']
    azimuth = message.get('azimuth')
    elevation = message.get('elevation')

    ops.set_camera_angle(camera_id, azimuth=azimuth, elevation=elevation)
    ops.step()

    # Get latest frame
    camera = ops.robot.sensors[camera_id]
    return camera.rgb_image  # np.ndarray → encode as JPEG/PNG for streaming
```

**WebSocket message format:**
```json
{
  "type": "set_camera_angle",
  "camera_id": "birds_eye",
  "azimuth": 180,
  "elevation": -45
}
```

## Technical Details

### Camera Coordinate System
- **Lookat**: The point in 3D space the camera looks at `[x, y, z]`
- **Distance**: Radius from lookat point
- **Azimuth**: Angle around vertical axis (Z-axis)
- **Elevation**: Angle above/below horizontal plane

### Performance Notes
- Camera rendering is **expensive** (~50-100ms per camera per frame on CPU)
- Uses MuJoCo's offscreen rendering (works in headless mode)
- Multiple cameras render sequentially (not parallel)
- Recommend: Use 2-3 cameras max for real-time performance

### Rendering Details
- Uses `mujoco.Renderer` for offscreen rendering
- Renderer is cached and reused (creating renderer is expensive)
- Each camera has its own renderer instance
- Renders at the FPS specified in ExperimentOps (default: 30 FPS)

### Timeline Saving
- Cameras automatically save to `timeline/cameras/{camera_id}_rgb.mp4`
- Uses cv2.VideoWriter with H.264 codec
- Same FPS as other timeline data (controlled by `save_fps` parameter)
- Videos are finalized when simulation ends (call `ops.close()`)

## Limitations

1. **Simulation Only**: Cannot be used with real hardware (`mode="real"`)
2. **No Depth Maps**: Currently only RGB rendering (depth support can be added)
3. **Sequential Rendering**: Multiple cameras render one-by-one (not parallel)
4. **Fixed Resolution**: Resolution set at creation, cannot change during simulation

## Troubleshooting

### Camera not rendering
```python
# Check if camera was added
assert 'birds_eye' in ops.robot.sensors

# Check if compile() was called
assert ops.engine is not None

# Check if step() was called
ops.step()
camera = ops.robot.sensors['birds_eye']
assert camera.rgb_image is not None
```

### Video file empty or corrupted
```python
# Make sure to close ops at end
ops.close()  # This finalizes video files
```

### EGL rendering errors (headless mode)
```bash
# Set EGL backend before running
export MUJOCO_GL=egl
python your_script.py
```

### Camera renders black screen
```python
# Check lookat point is valid (within scene bounds)
# Check distance is reasonable (not too close or too far)
# Try default settings first:
ops.add_free_camera('test')  # Uses defaults
```

## See Also

- **Level 1A Test**: `simulation_center/core/tests/levels/level_1a_basic_infrastructure_test_new.py` (test_10)
- **Level 1D Test**: `simulation_center/core/tests/levels/level_1d_sensor_system.py` (test_09)
- **Level 1G Test**: `simulation_center/core/tests/levels/level_1g_view_system.py` (CameraViewsTest)
- **Sensor Modals**: `simulation_center/core/modals/stretch/sensors_modals.py` (FreeCameraSensor class)
- **ExperimentOps API**: `simulation_center/core/main/experiment_ops_unified.py`

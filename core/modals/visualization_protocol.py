"""
Visualization Protocol - Convert matplotlib figures to numpy arrays
"""

import numpy as np
import io


def fig_to_numpy(fig):
    """Convert matplotlib figure to numpy RGB array

    Args:
        fig: matplotlib Figure object

    Returns:
        numpy array of shape (H, W, 3) with RGB values
    """
    # Draw the figure to render it
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    buf = fig.canvas.buffer_rgba()

    # Convert to numpy array
    img = np.frombuffer(buf, dtype=np.uint8)

    # Reshape to (H, W, 4)
    w, h = fig.canvas.get_width_height()
    img = img.reshape((h, w, 4))

    # Convert RGBA to RGB (drop alpha channel)
    img_rgb = img[:, :, :3]

    return img_rgb

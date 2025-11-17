"""
SIM TO REAL ADAPTER - Makes simulation match reality
"""

import time


class SimToRealAdapter:
    def __init__(self):
        """Track sim/real differences dynamically"""
        self.real_start = None
        self.sim_start = None
        self.ratios = {
            'time': 1.0,      # sim_time / real_time
            'velocity': 1.0   # actual / commanded
        }

    def update(self, sim_status):
        """Update ratios from simulation status

        Args:
            sim_status: Status from sim.pull_status()
        """
        current_real = time.time()
        current_sim = sim_status.time if hasattr(sim_status, 'time') else 0

        # Initialize on first call
        if not self.real_start:
            self.real_start = current_real
            self.sim_start = current_sim
            return

        # Calculate time ratio
        real_elapsed = current_real - self.real_start
        sim_elapsed = current_sim - self.sim_start

        if real_elapsed > 1.0:  # Update after 1+ seconds
            self.ratios['time'] = sim_elapsed / real_elapsed

    def adapt_velocity(self, desired_velocity):
        """Convert desired real velocity to sim command

        Args:
            desired_velocity: What we want in real world (m/s)
        Returns:
            What to send to sim
        """
        # Just pass through - velocity is what it is
        # The sim handles the physics
        return desired_velocity

    def adapt_time(self, real_seconds):
        """Convert real seconds to sim seconds

        Args:
            real_seconds: Real world time
        Returns:
            Simulation time
        """
        return real_seconds * self.ratios['time']

    def wait_sim_time(self, sim_seconds):
        """Wait for simulation seconds (not real seconds)

        Args:
            sim_seconds: How long in simulation time
        """
        if self.ratios['time'] > 0:
            real_seconds = sim_seconds / self.ratios['time']
            time.sleep(real_seconds)
        else:
            time.sleep(sim_seconds)

    def status(self):
        """Get current adapter state"""
        return {
            'time_ratio': self.ratios['time'],
            'sim_speed': f"{self.ratios['time']:.2f}x real-time",
            'velocity_scale': 1.0 / 0.18  # How much we scale commands
        }
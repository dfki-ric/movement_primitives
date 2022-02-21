"""Minimum jerk trajectory."""
import numpy as np
from .base import PointToPointMovement
from .data._minimum_jerk import generate_minimum_jerk


class MinimumJerkTrajectory(PointToPointMovement):
    """Precomputed point to point movement with minimum jerk.

    Parameters
    ----------
    n_dims : int
        State space dimensions.

    execution_time : float
        Execution time of the DMP.

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.
    """
    def __init__(self, n_dims, execution_time, dt=0.01):
        super(MinimumJerkTrajectory, self).__init__(n_dims, n_dims)
        self.X = None
        self.Xd = None
        self.execution_time = execution_time
        self.dt = dt
        self.step_idx = 0
        self.initialized = False

    def reset(self):
        """Reset initial state and time."""
        self.step_idx = 0
        self.initialized = False

    def step(self, last_y, last_yd):
        """Perform step.

        Parameters
        ----------
        last_y : array, shape (n_dims,)
            Last state.

        last_yd : array, shape (n_dims,)
            Last time derivative of state (e.g., velocity).

        Returns
        -------
        y : array, shape (n_dims,)
            Next state.

        yd : array, shape (n_dims,)
            Next time derivative of state (e.g., velocity).
        """
        if not self.initialized:
            self.X, self.Xd, _ = generate_minimum_jerk(
                self.start_y, self.goal_y, self.execution_time, self.dt)
            self.initialized = True

        self.current_y = self.X[self.step_idx]
        self.current_yd = self.Xd[self.step_idx]
        self.step_idx += 1

        return np.copy(self.current_y), np.copy(self.current_yd)

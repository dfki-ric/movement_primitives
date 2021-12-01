import numpy as np
from ..base import PointToPointMovement


class DMPBase(PointToPointMovement):
    """Base class of Dynamical Movement Primitives (DMPs)."""
    def __init__(self, n_pos_dims, n_vel_dims):
        super(DMPBase, self).__init__(n_pos_dims, n_vel_dims)

        self.initialized = False

    def reset(self):
        """Reset DMP to initial state and time."""
        self.t = 0.0
        self.last_t = None
        self.current_y = np.copy(self.start_y)
        self.current_yd = np.copy(self.start_yd)

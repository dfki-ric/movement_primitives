import numpy as np


class DMPBase:
    """Base class of Dynamical Movement Primitives (DMPs)."""
    def __init__(self, n_pos_dims, n_vel_dims):
        self.n_dims = n_pos_dims
        self.n_vel_dims = n_vel_dims

        self.t = 0.0
        self.last_t = None

        self.start_y = np.zeros(n_pos_dims)
        self.start_yd = np.zeros(n_vel_dims)
        self.start_ydd = np.zeros(n_vel_dims)

        self.goal_y = np.zeros(n_pos_dims)
        self.goal_yd = np.zeros(n_vel_dims)
        self.goal_ydd = np.zeros(n_vel_dims)

        self.initialized = False

        self.current_y = np.zeros(n_pos_dims)
        self.current_yd = np.zeros(n_vel_dims)

    def reset(self):
        """Reset DMP to initial state and time."""
        self.t = 0.0
        self.last_t = None
        self.current_y = np.copy(self.start_y)
        self.current_yd = np.copy(self.start_yd)

    def configure(self, t=None,
                  start_y=None, start_yd=None, start_ydd=None,
                  goal_y=None, goal_yd=None, goal_ydd=None):
        """Set meta parameters of DMP.

        Parameters
        ----------
        t : float, optional
            Time at current step

        start_y : array, shape (n_dims,)
            Initial state

        start_yd : array, shape (n_vel_dims,)
            Initial velocity

        start_ydd : array, shape (n_vel_dims,)
            Initial acceleration

        goal_y : array, shape (n_dims,)
            Goal state

        goal_yd : array, shape (n_vel_dims,)
            Goal velocity

        goal_ydd : array, shape (n_vel_dims,)
            Goal acceleration
        """
        if t is not None:
            self.t = t
        if start_y is not None:
            self.start_y = start_y
        if start_yd is not None:
            self.start_yd = start_yd
        if start_ydd is not None:
            self.start_ydd = start_ydd
        if goal_y is not None:
            self.goal_y = goal_y
        if goal_yd is not None:
            self.goal_yd = goal_yd
        if goal_ydd is not None:
            self.goal_ydd = goal_ydd

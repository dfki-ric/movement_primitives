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


class WeightParametersMixin:
    """Mixin class providing common access methods to forcing term weights."""
    def get_weights(self):
        """Get weight vector of DMP.

        Returns
        -------
        weights : array, shape (N * n_weights_per_dim,)
            Current weights of the DMP. N depends on the type of DMP
        """
        return self.forcing_term.weights.ravel()

    def set_weights(self, weights):
        """Set weight vector of DMP.

        Parameters
        ----------
        weights : array, shape (N * n_weights_per_dim,)
            New weights of the DMP. N depends on the type of DMP
        """
        self.forcing_term.weights[:, :] = weights.reshape(*self.forcing_term.shape)

    @property
    def n_weights(self):
        """Total number of weights configuring the forcing term."""
        return np.product(self.forcing_term.shape)

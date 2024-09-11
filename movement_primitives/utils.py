import numpy as np


def ensure_1d_array(value, n_dims, var_name):
    """Process scalar or array-like input to ensure it is a 1D numpy array of the correct shape."""
    value = np.atleast_1d(value).astype(float).flatten()
    if value.shape[0] == 1:
        value = value * np.ones(n_dims)
    elif value.shape != (n_dims,):
        raise ValueError(f"{var_name} has incorrect shape, expected ({n_dims},) got {value.shape}")
    return value

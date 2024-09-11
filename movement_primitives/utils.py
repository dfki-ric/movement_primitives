import numpy as np


def ensure_1d_array(value, dim, label):
    """Process scalar or array-like input to ensure it is a 1D numpy array of the correct shape."""
    value = np.atleast_1d(value).astype(float).flatten()
    if value.shape[0] == 1:
        value = value * np.ones(dim)
    elif value.shape != (dim,):
        raise ValueError(f"{label} has incorrect shape, expected ({dim},) got {value.shape}")
    return value

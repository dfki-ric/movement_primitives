"""Utility functions."""
import numpy as np


def pick_closest_quaternion(quaternion, target_quaternion):
    """Resolve quaternion ambiguity and pick the closest one to the target.

    .. warning::

        There are always two quaternions that represent the exact same
        orientation: q and -q. The problem is that a DMP that moves from q to q
        does not move at all, while a DMP that moves from q to -q moves a lot.
        We have to make sure that start_y always contains the quaternion
        representation that is closest to the previous start_y!

    Parameters
    ----------
    quaternion : array-like, shape (4,)
        Quaternion (w, x, y, z) of which we are unsure whether we want to
        select quaternion or -quaternion.

    target_quaternion : array-like, shape (4,)
        Target quaternion (w, x, y, z) to which we want to be close.

    Returns
    -------
    closest_quaternion : array, shape (4,)
        Quaternion that is closest (Euclidean norm) to the target quaternion.
    """
    quaternion = np.asarray(quaternion)
    target_quaternion = np.asarray(target_quaternion)
    if (np.linalg.norm(-quaternion - target_quaternion) <
            np.linalg.norm(quaternion - target_quaternion)):
        return -quaternion
    else:
        return quaternion

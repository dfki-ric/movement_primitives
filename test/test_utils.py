import numpy as np
from numpy.testing import assert_array_almost_equal
import pytransform3d.rotations as pr
from movement_primitives.utils import pick_closest_quaternion


def test_pick_closest_quaternion():
    random_state = np.random.RandomState(131)
    for _ in range(5):
        q = pr.random_quaternion(random_state)
        q2 = pick_closest_quaternion(q, q)
        assert_array_almost_equal(q, q2)
        q2 = pick_closest_quaternion(-q, q)
        assert_array_almost_equal(q, q2)

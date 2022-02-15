import numpy as np
from movement_primitives.minimum_jerk_trajectory import MinimumJerkTrajectory
from numpy.testing import assert_array_almost_equal


def test_step_through_minimum_jerk_trajectory():
    mjt = MinimumJerkTrajectory(3, 1.0, 0.01)
    mjt.configure(start_y=np.zeros(3), goal_y=np.ones(3))
    y = np.zeros(3)
    yd = np.zeros(3)

    y, yd = mjt.n_steps_open_loop(y, yd, 50)
    assert_array_almost_equal(y, 0.49 * np.ones(3), decimal=2)
    assert_array_almost_equal(yd, 1.9, decimal=2)

    y, yd = mjt.n_steps_open_loop(y, yd, 1)
    assert_array_almost_equal(y, 0.51 * np.ones(3), decimal=2)
    assert_array_almost_equal(yd, 1.9, decimal=2)

    y, yd = mjt.n_steps_open_loop(y, yd, 50)
    assert_array_almost_equal(y, np.ones(3))
    assert_array_almost_equal(yd, np.zeros(3))

    mjt.reset()

    y, yd = mjt.n_steps_open_loop(y, yd, 101)
    assert_array_almost_equal(y, np.ones(3))
    assert_array_almost_equal(yd, np.zeros(3))

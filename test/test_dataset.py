import numpy as np
from movement_primitives.data import (
    generate_minimum_jerk, load_lasa, generate_1d_trajectory_distribution)
from numpy.testing import assert_array_almost_equal
from nose.tools import (assert_raises_regexp, assert_greater_equal,
                        assert_less_equal, assert_equal)


def test_minimum_jerk_boundaries():
    random_state = np.random.RandomState(1)
    x0 = random_state.randn(2)
    g = x0 + random_state.rand(2)

    X, Xd, Xdd = generate_minimum_jerk(x0, g)
    assert_array_almost_equal(X[:, 0], x0)
    assert_array_almost_equal(Xd[:, 0], np.zeros(2))
    assert_array_almost_equal(Xdd[:, 0], np.zeros(2))
    assert_array_almost_equal(X[:, -1], g)
    assert_array_almost_equal(Xd[:, -1], np.zeros(2))
    assert_array_almost_equal(Xdd[:, -1], np.zeros(2))

    for d in range(X.shape[0]):
        for t in range(X.shape[1]):
            assert_greater_equal(X[d, t], x0[d])
            assert_less_equal(X[d, t], g[d])

    assert_raises_regexp(ValueError, "Shape .* must be equal",
                         generate_minimum_jerk, x0, np.zeros(1))


def test_lasa():
    T, X, Xd, Xdd, dt, shape_name = load_lasa(0)
    assert_equal(T.shape[0], X.shape[0])
    assert_equal(T.shape[1], X.shape[1])
    assert_equal(X.shape[2], 2)
    assert_equal(Xd.shape[0], X.shape[0])
    assert_equal(Xd.shape[1], X.shape[1])
    assert_equal(Xd.shape[2], X.shape[2])
    assert_equal(Xdd.shape[0], X.shape[0])
    assert_equal(Xdd.shape[1], X.shape[1])
    assert_equal(Xdd.shape[2], X.shape[2])
    assert_equal(shape_name, "Angle")


def test_toy1d():
    T1, X1 = generate_1d_trajectory_distribution(
        n_demos=1, n_steps=11, noise_per_step_range=0.0,
        initial_offset_range=0, final_offset_range=0)

    T2, X2 = generate_1d_trajectory_distribution(
        n_demos=1000, n_steps=11, noise_per_step_range=20.0,
        initial_offset_range=0, final_offset_range=0)

    assert_array_almost_equal(
        np.mean(X1, axis=0), np.mean(X2, axis=0), decimal=1)

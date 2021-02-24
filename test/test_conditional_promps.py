import numpy as np
from movement_primitives.data import generate_1d_trajectory_distribution
from movement_primitives.promp import ProMP
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_greater, assert_almost_equal


def test_certain_conditioning():
    n_demos = 100
    n_steps = 101
    T, Y = generate_1d_trajectory_distribution(n_demos, n_steps)
    y_conditional_cov = np.array([0.0])
    promp = ProMP(n_dims=1, n_weights_per_dim=50)
    promp.imitate([T] * n_demos, Y)
    Y_mean = promp.mean_trajectory(T)
    Y_var = promp.var_trajectory(T)
    assert_array_almost_equal(Y_mean, np.mean(Y, axis=0), decimal=3)
    assert_greater(Y_var[0], 0)

    for y_cond in np.linspace(-1, 2.5, 7):
        cpromp = promp.condition_position(np.array([y_cond]), y_cov=y_conditional_cov, t=0.0, t_max=1.0)
        Y_cmean = cpromp.mean_trajectory(T)
        assert_almost_equal(Y_cmean[0, 0], y_cond)
        Y_cvar = cpromp.var_trajectory(T)
        assert_almost_equal(Y_cvar[0, 0], 0)

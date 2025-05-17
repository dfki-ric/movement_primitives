import numpy as np
from movement_primitives.data import generate_1d_trajectory_distribution
from movement_primitives.promp import ProMP, via_points
from numpy.testing import assert_array_almost_equal
import pytest


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
    assert Y_var[0] > 0

    for y_cond in np.linspace(-1, 2.5, 7):
        cpromp = promp.condition_position(np.array([y_cond]), y_cov=y_conditional_cov, t=0.0, t_max=1.0)
        Y_cmean = cpromp.mean_trajectory(T)
        assert Y_cmean[0, 0] == pytest.approx(y_cond)
        Y_cvar = cpromp.var_trajectory(T)
        assert Y_cvar[0, 0] == pytest.approx(0)


def test_conditioning_with_default_value():
    n_demos = 100
    n_steps = 101
    T, Y = generate_1d_trajectory_distribution(n_demos, n_steps)
    promp = ProMP(n_dims=1, n_weights_per_dim=50)
    promp.imitate([T] * n_demos, Y)
    Y_mean = promp.mean_trajectory(T)
    Y_var = promp.var_trajectory(T)
    assert_array_almost_equal(Y_mean, np.mean(Y, axis=0), decimal=3)
    assert Y_var[0] > 0

    cpromp = promp.condition_position(np.array([0.0]), t=0.0, t_max=1.0)
    Y_cmean = cpromp.mean_trajectory(T)
    assert Y_cmean[0, 0] == pytest.approx(0.0)
    Y_cvar = cpromp.var_trajectory(T)
    assert Y_cvar[0, 0] == pytest.approx(0)


def test_multi_via_points():
    n_demos = 100
    n_steps = 101
    T, Y = generate_1d_trajectory_distribution(n_demos, n_steps)
    promp = ProMP(n_dims=1, n_weights_per_dim=50)
    promp.imitate([T] * n_demos, Y)
    Y_mean = promp.mean_trajectory(T)
    Y_var = promp.var_trajectory(T)
    assert_array_almost_equal(Y_mean, np.mean(Y, axis=0), decimal=3)
    assert Y_var[0] > 0

    y_cond = np.array([-1, 2.5, 4, 7])
    y_conditional_cov = np.zeros(4)
    ts = np.array([0.0, 0.3, 0.7, 1.0])
    idx = np.array([
        np.argwhere(T==ts[0]).squeeze(),
        np.argwhere(T==ts[1]).squeeze(),
        np.argwhere(np.logical_and(T>ts[2] - 0.001, T<ts[2]+0.001)).squeeze(),
        np.argwhere(T==ts[-1]).squeeze()
    ])
    cpromp = via_points(
        promp=promp,
        y_cond=y_cond,
        y_conditional_cov=y_conditional_cov,
        ts=ts
    )
    Y_cmean = cpromp.mean_trajectory(T)
    assert Y_cmean[idx].squeeze() == pytest.approx(y_cond)
    Y_cvar = cpromp.var_trajectory(T)
    assert Y_cvar[idx] == pytest.approx(0)

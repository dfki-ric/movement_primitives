import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from numpy.testing import assert_array_almost_equal
import pytest


def test_final_velocity():
    dt = 0.01
    execution_time = 1.0
    T = np.arange(0, execution_time + dt, dt)
    Y = np.column_stack((np.cos(np.pi * T), -np.cos(np.pi * T)))

    dmp = DMPWithFinalVelocity(n_dims=2, execution_time=execution_time)
    dmp.imitate(T, Y)

    for goal_yd in [0.0, 1.0, 2.0]:
        goal_yd_expected = np.array([goal_yd, goal_yd])
        dmp.configure(goal_yd=goal_yd_expected)
        _, Y = dmp.open_loop(run_t=execution_time)
        goal_yd_actual = (Y[-1] - Y[-2]) / dt
        assert_array_almost_equal(goal_yd_expected, goal_yd_actual, decimal=2)


def test_step_through_dmp_with_final_velocity():
    dt = 0.01
    execution_time = 1.0
    T = np.arange(0, execution_time + dt, dt)
    Y = np.column_stack((np.cos(np.pi * T), -np.cos(np.pi * T)))

    dmp = DMPWithFinalVelocity(n_dims=2, execution_time=execution_time)
    dmp.imitate(T, Y)
    dmp.configure(start_y=np.array([0, 0], dtype=float),
                  goal_y=np.array([1, 1], dtype=float))
    current_y = np.copy(dmp.start_y)
    current_yd = np.copy(dmp.start_yd)
    path = [np.copy(current_y)]
    while dmp.t <= dmp.execution_time_:
        current_y, current_yd = dmp.step(current_y, current_yd)
        path.append(np.copy(current_y))
    assert_array_almost_equal(np.vstack(path), dmp.open_loop()[1], decimal=5)


def test_temporal_scaling():
    execution_time = 2.0
    dt = 0.001

    dmp = DMPWithFinalVelocity(n_dims=2, execution_time=execution_time, dt=dt,
                               n_weights_per_dim=100)

    T = np.arange(0.0, execution_time + dt, dt)
    Y_demo = np.empty((len(T), 2))
    Y_demo[:, 0] = np.cos(np.pi * T)
    Y_demo[:, 1] = 0.5 + np.cos(0.5 * np.pi * T)
    dmp.imitate(T, Y_demo)
    goal_yd = np.array([0.5, -1.0])

    dmp.configure(goal_yd=goal_yd)
    T2, Y2 = dmp.open_loop()
    assert T2[-1] == pytest.approx(2.0)
    Yd2 = np.column_stack((np.gradient(Y2[:, 0]),
                           np.gradient(Y2[:, 1]))) / dmp.dt_
    assert_array_almost_equal(Yd2[-1], dmp.goal_yd, decimal=1)

    dmp.execution_time_ = 1.0
    T1, Y1 = dmp.open_loop()
    assert T1[-1] == pytest.approx(1.0)
    Yd1 = np.column_stack((np.gradient(Y1[:, 0]),
                           np.gradient(Y1[:, 1]))) / dmp.dt_
    assert_array_almost_equal(Yd1[-1], dmp.goal_yd, decimal=1)

    dmp.execution_time_ = 4.0
    T4, Y4 = dmp.open_loop()
    assert T4[-1] == pytest.approx(4.0, rel=1e-2)
    Yd4 = np.column_stack((np.gradient(Y4[:, 0]),
                           np.gradient(Y4[:, 1]))) / dmp.dt_
    assert_array_almost_equal(Yd4[-1], dmp.goal_yd, decimal=1)

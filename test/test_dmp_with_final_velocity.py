import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity
from numpy.testing import assert_array_almost_equal


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
    while dmp.t <= dmp.execution_time:
        current_y, current_yd = dmp.step(current_y, current_yd)
        path.append(np.copy(current_y))
    assert_array_almost_equal(np.vstack(path), dmp.open_loop()[1])

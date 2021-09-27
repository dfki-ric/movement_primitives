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

import numpy as np
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance2D
from movement_primitives.dmp_potential_field import potential_field_2d
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_potential_field_2d():
    start_y = np.array([0, 0], dtype=float)
    goal_y = np.array([1, 1], dtype=float)
    obstacle = np.array([0.85, 0.5])
    random_state = np.random.RandomState(1)

    dmp = DMP(n_dims=2, n_weights_per_dim=10, dt=0.01, execution_time=1.0)
    dmp.forcing_term.weights[:, :] = random_state.randn(
        *dmp.forcing_term.weights.shape) * 500.0
    dmp.configure(start_y=start_y, goal_y=goal_y)
    coupling_term = CouplingTermObstacleAvoidance2D(obstacle)

    x_range = -0.2, 1.2
    y_range = -0.2, 1.2
    n_ticks = 15

    position = np.copy(start_y)
    velocity = np.zeros_like(start_y)

    while dmp.t <= dmp.execution_time:
        xx, yy, ft, ts, ct, acc = potential_field_2d(
            dmp, x_range, y_range, n_ticks, obstacle)

        assert_array_equal(xx.shape, (n_ticks, n_ticks))
        assert_array_equal(yy.shape, (n_ticks, n_ticks))
        assert_array_equal(ft.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(ts.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(ct.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(acc.shape, (n_ticks, n_ticks, 2))

        assert_array_almost_equal(ft + ts + ct, acc)
        xx, yy, ft, ts, ct, acc = potential_field_2d(
            dmp, x_range, y_range, n_ticks)

        assert_array_equal(xx.shape, (n_ticks, n_ticks))
        assert_array_equal(yy.shape, (n_ticks, n_ticks))
        assert_array_equal(ft.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(ts.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(ct.shape, (n_ticks, n_ticks, 2))
        assert_array_equal(acc.shape, (n_ticks, n_ticks, 2))

        assert_array_almost_equal(ft + ts + ct, acc)

        position, velocity = dmp.step(
            position, velocity, coupling_term=coupling_term)

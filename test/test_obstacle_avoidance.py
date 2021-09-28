import numpy as np
from scipy.spatial.distance import cdist
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance2D, CouplingTermObstacleAvoidance3D
from nose.tools import assert_less


def test_obstacle_avoidance_2d():
    execution_time = 1.0
    start_y = np.zeros(2)
    goal_y = np.ones(2)
    obstacle = 0.5 * np.ones(2)
    random_state = np.random.RandomState(42)

    dmp = DMP(n_dims=len(start_y), execution_time=execution_time,
              n_weights_per_dim=10, dt=0.01)
    dmp.configure(start_y=start_y, goal_y=goal_y)

    T, Y = dmp.open_loop()
    min_dist = min(cdist(Y, obstacle[np.newaxis]))
    assert_less(min_dist, 0.03)
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(2)
        coupling_term = CouplingTermObstacleAvoidance2D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert_less(0.1, min_dist)


def test_obstacle_avoidance_3d():
    execution_time = 1.0
    start_y = np.zeros(3)
    goal_y = np.ones(3)
    obstacle = 0.5 * np.ones(3)
    random_state = np.random.RandomState(42)

    dmp = DMP(n_dims=len(start_y), execution_time=execution_time,
              n_weights_per_dim=10, dt=0.01)
    dmp.configure(start_y=start_y, goal_y=goal_y)

    T, Y = dmp.open_loop()
    min_dist = min(cdist(Y, obstacle[np.newaxis]))
    assert_less(min_dist, 0.03)
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(3)
        coupling_term = CouplingTermObstacleAvoidance3D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert_less(0.15, min_dist)

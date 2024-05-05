import numpy as np
from scipy.spatial.distance import cdist
from movement_primitives.dmp import (
    DMP, CouplingTermObstacleAvoidance2D, CouplingTermObstacleAvoidance3D,
    DMPWithFinalVelocity)
from numpy.testing import assert_array_almost_equal


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
    assert min_dist < 0.03
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(2)
        coupling_term = CouplingTermObstacleAvoidance2D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert 0.1 < min_dist


def test_obstacle_avoidance_2d_rk4_python():
    execution_time = 1.0
    start_y = np.zeros(2)
    goal_y = np.ones(2)
    obstacle = 0.5 * np.ones(2)
    random_state = np.random.RandomState(42)

    dmp = DMP(n_dims=len(start_y), execution_time=execution_time,
              n_weights_per_dim=10, dt=0.01)
    dmp.configure(start_y=start_y, goal_y=goal_y)

    T, Y = dmp.open_loop(step_function="rk4")
    min_dist = min(cdist(Y, obstacle[np.newaxis]))
    assert min_dist < 0.03
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(2)
        coupling_term = CouplingTermObstacleAvoidance2D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term, step_function="rk4")
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert 0.1 < min_dist


def test_obstacle_avoidance_2d_fast():
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
    assert min_dist < 0.03
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(2)
        coupling_term = CouplingTermObstacleAvoidance2D(obstacle, fast=True)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert 0.1 < min_dist


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
    assert min_dist < 0.03
    for _ in range(20):
        start_y_random = 0.2 + 0.2 * random_state.randn(3)
        coupling_term = CouplingTermObstacleAvoidance3D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert 0.13 < min_dist


def test_obstacle_avoidance_3d_with_final_velocity():
    execution_time = 1.0
    start_y = np.zeros(3)
    goal_y = np.ones(3)
    obstacle = 0.5 * np.ones(3)
    random_state = np.random.RandomState(42)

    dmp = DMPWithFinalVelocity(
        n_dims=len(start_y), execution_time=execution_time,
        n_weights_per_dim=10, dt=0.01)
    dmp.configure(start_y=start_y, goal_y=goal_y, goal_yd=np.ones(3))

    T, Y = dmp.open_loop()
    min_dist = min(cdist(Y, obstacle[np.newaxis]))
    assert min_dist < 0.03
    for _ in range(10):
        start_y_random = 0.2 + 0.2 * random_state.randn(3)
        coupling_term = CouplingTermObstacleAvoidance3D(obstacle)
        dmp.configure(start_y=start_y_random)
        T, Y = dmp.open_loop(coupling_term=coupling_term)
        min_dist = min(cdist(Y, obstacle[np.newaxis]))
        assert 0.1 < min_dist
        goal_yd_actual = (Y[-1] - Y[-2]) / 0.01
        assert_array_almost_equal(np.ones(3), goal_yd_actual, decimal=1)

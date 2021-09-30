import numpy as np
from movement_primitives.dmp import (DMP, CouplingTermPos1DToPos1D,
                                     CouplingTermPos3DToPos3D)
from nose.tools import assert_almost_equal


def test_coupling_1d_to_1d_pos():
    dt = 0.01
    execution_time = 2.0
    dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=200)
    ct = CouplingTermPos1DToPos1D(desired_distance=0.5, lf=(1.0, 0.0), k=1.0)

    T = np.linspace(0.0, execution_time, 101)
    Y = np.empty((len(T), 2))
    Y[:, 0] = np.cos(2.5 * np.pi * T)
    Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
    dmp.imitate(T, Y)

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T, Y = dmp.open_loop()

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T, Y = dmp.open_loop(coupling_term=ct)

    distances = Y[:, 1] - Y[:, 0]
    assert_almost_equal(np.mean(distances), ct.desired_distance, places=3)


def test_coupling_3d_to_3d_pos():
    dt = 0.01

    dmp = DMP(n_dims=6, execution_time=1.0, dt=dt, n_weights_per_dim=10,
              int_dt=0.0001)
    ct = CouplingTermPos3DToPos3D(desired_distance=np.array([0.1, 0.5, 1.0]),
                                  lf=(0.0, 1.0), k=1.0, c1=30.0, c2=100.0)

    T = np.linspace(0.0, 1.0, 101)
    Y = np.empty((len(T), 6))
    Y[:, 0] = np.cos(np.pi * T)
    Y[:, 1] = np.sin(np.pi * T)
    Y[:, 2] = np.sin(2 * np.pi * T)
    Y[:, 3] = np.cos(np.pi * T)
    Y[:, 4] = np.sin(np.pi * T)
    Y[:, 5] = 0.5 + np.sin(2 * np.pi * T)
    dmp.imitate(T, Y)

    T, Y = dmp.open_loop(coupling_term=ct)

    assert_almost_equal(
        np.median(Y[:, 3] - Y[:, 0]), ct.desired_distance[0], places=1)
    assert_almost_equal(
        np.median(Y[:, 4] - Y[:, 1]), ct.desired_distance[1], places=1)
    assert_almost_equal(
        np.median(Y[:, 5] - Y[:, 2]), ct.desired_distance[2], places=1)

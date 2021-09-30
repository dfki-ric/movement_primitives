import numpy as np
from movement_primitives.dmp import DMP, CouplingTermPos1DToPos1D
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

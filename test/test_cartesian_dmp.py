import numpy as np
from movement_primitives.dmp import CartesianDMP
from movement_primitives.testing.simulation import UR5Simulation
from pytransform3d import rotations as pr
from numpy.testing import assert_array_almost_equal


def test_imitate_cartesian_dmp():
    dt = 0.001
    execution_time = 1.0

    dmp = CartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=0.0001)
    Y = np.zeros((1001, 7))
    T = np.linspace(0, 1, len(Y))
    sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
    Y[:, 0] = 0.6
    Y[:, 1] = -0.2 + 0.4 * sigmoid
    Y[:, 2] = 0.45
    start_aa = np.array([0.0, 1.0, 0.0, 0.25 * np.pi])
    goal_aa = np.array([0.0, 0.0, 1.0, 0.25 * np.pi])
    for t in range(len(Y)):
        frac = sigmoid[t]
        aa_t = (1.0 - frac) * start_aa + frac * goal_aa
        aa_t[:3] /= np.linalg.norm(aa_t[:3])
        Y[t, 3:] = pr.quaternion_from_axis_angle(aa_t)
    dmp.imitate(T, Y, allow_final_velocity=True)
    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T2, Y2 = dmp.open_loop()
    assert_array_almost_equal(T, T2)
    assert_array_almost_equal(Y, Y2, decimal=2)

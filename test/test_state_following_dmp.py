import numpy as np
from movement_primitives.dmp import StateFollowingDMP
from nose.tools import assert_almost_equal, assert_less


def test_state_following_dmp2d():
    start_y = np.array([0.0, 1.0])
    dt = 0.001
    execution_time = 3.0
    n_viapoints = 8

    dmp = StateFollowingDMP(n_dims=2, execution_time=execution_time, dt=dt, n_viapoints=n_viapoints)
    dmp.forcing_term.viapoints[:, 0] = np.linspace(0, 1, n_viapoints)
    dmp.forcing_term.viapoints[:, 1] = np.linspace(1, 2, n_viapoints)
    dmp.configure(start_y=start_y)

    T, Y = dmp.open_loop(run_t=execution_time)

    assert_almost_equal(T[0], 0.0, places=3)
    assert_almost_equal(T[-1], execution_time, places=2)

    for i in range(n_viapoints):
        viapoint = dmp.forcing_term.viapoints[i]
        min_dist = np.min(np.linalg.norm(Y - viapoint[np.newaxis], axis=1))
        assert_less(min_dist, 0.02)

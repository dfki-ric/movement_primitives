import numpy as np
from movement_primitives.dmp import StateFollowingDMP
from movement_primitives.dmp._state_following_dmp import StateFollowingForcingTerm
import pytest


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

    assert T[0] == pytest.approx(0.0, rel=1e-3)
    assert T[-1] == pytest.approx(execution_time, rel=1e-2)

    for i in range(n_viapoints):
        viapoint = dmp.forcing_term.viapoints[i]
        min_dist = np.min(np.linalg.norm(Y - viapoint[np.newaxis], axis=1))
        assert min_dist < 0.02


def test_state_following_dmp2d_stepwise():
    start_y = np.array([0.0, 1.0])
    dt = 0.001
    execution_time = 3.0
    n_viapoints = 8

    dmp = StateFollowingDMP(n_dims=2, execution_time=execution_time, dt=dt, n_viapoints=n_viapoints)
    dmp.forcing_term.viapoints[:, 0] = np.linspace(0, 1, n_viapoints)
    dmp.forcing_term.viapoints[:, 1] = np.linspace(1, 2, n_viapoints)
    dmp.configure(start_y=start_y)

    current_y = np.copy(dmp.start_y)
    current_yd = np.copy(dmp.start_yd)
    Y = [np.copy(current_y)]
    for _ in np.arange(0.0, execution_time + dt, dt):
        current_y, current_yd = dmp.step(current_y, current_yd)
        Y.append(np.copy(current_y))
    Y = np.array(Y)

    for i in range(n_viapoints):
        viapoint = dmp.forcing_term.viapoints[i]
        min_dist = np.min(np.linalg.norm(Y - viapoint[np.newaxis], axis=1))
        assert min_dist < 0.02


def test_invalid_viapoints():
    with pytest.raises(ValueError, match="The number of viapoints"):
        StateFollowingForcingTerm(2, -1, 1.0, 0.0, 0.7, 6)


def test_invalid_times():
    with pytest.raises(ValueError, match="Goal must be chronologically after start"):
        StateFollowingForcingTerm(2, 5, 0.0, 1.0, 0.7, 6)

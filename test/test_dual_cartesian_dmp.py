import numpy as np
import pytest

from movement_primitives.dmp import DualCartesianDMP


def test_invalid_step_function():
    dmp = DualCartesianDMP(
        execution_time=1.0, dt=0.01, n_weights_per_dim=10, int_dt=0.001)
    with pytest.raises(ValueError, match="Step function"):
        dmp.open_loop(step_function="invalid")


def test_temporal_scaling():
    execution_time = 2.0
    dt = 0.01

    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt, n_weights_per_dim=100)

    T = np.arange(0.0, execution_time + dt, dt)
    Y_demo = np.zeros((len(T), 14))
    Y_demo[:, 0] = np.cos(2.5 * np.pi * T)
    Y_demo[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
    Y_demo[:, 3] = 1.0
    Y_demo[:, 10] = 1.0
    dmp.imitate(T, Y_demo)

    dmp.configure(start_y=Y_demo[0], goal_y=Y_demo[-1])
    _, Y2 = dmp.open_loop()

    dmp.execution_time_ = 1.0
    _, Y1 = dmp.open_loop()

    dmp.execution_time_ = 4.0
    _, Y4 = dmp.open_loop()

    assert np.linalg.norm(Y1 - Y2[::2]) / len(Y1) < 1e-3
    assert np.linalg.norm(Y2 - Y4[::2]) / len(Y2) < 1e-3

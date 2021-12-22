import numpy as np
from movement_primitives.dmp import CartesianDMP
from pytransform3d import rotations as pr
from numpy.testing import assert_array_almost_equal, assert_raises_regex


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


def test_step_through_cartesian_dmp():
    dt = 0.001
    execution_time = 1.0

    dmp = CartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=0.0001)
    dmp.configure(start_y=np.array([0, 0, 0, 1, 0, 0, 0], dtype=float),
                  goal_y=np.array([1, 1, 1, 0, 1, 0, 0], dtype=float))
    current_y = np.copy(dmp.start_y)
    current_yd = np.copy(dmp.start_yd)
    path = [np.copy(current_y)]
    while dmp.t <= dmp.execution_time:
        current_y, current_yd = dmp.step(current_y, current_yd)
        path.append(np.copy(current_y))
    assert_array_almost_equal(np.vstack(path), dmp.open_loop()[1])


def test_compare_python_cython():
    from copy import deepcopy
    from movement_primitives.dmp._cartesian_dmp import dmp_step_quaternion_python
    from movement_primitives.dmp_fast import dmp_step_quaternion as dmp_step_quaternion_cython
    kwargs = dict(
        last_t=1.999, t=2.0,
        current_y=np.array([1.0, 0.0, 0.0, 0.0]), current_yd=np.zeros([3]),
        goal_y=np.array([1.0, 0.0, 0.0, 0.0]), goal_yd=np.zeros([3]), goal_ydd=np.zeros([3]),
        start_y=np.array([1.0, 0.0, 0.0, 0.0]), start_yd=np.zeros([3]), start_ydd=np.zeros([3]),
        goal_t=2.0, start_t=0.0, alpha_y=25.0, beta_y=6.25,
        forcing_term=lambda x: 10000 * np.ones((3, 1)), coupling_term=None, int_dt=0.0001
    )
    kwargs_python = deepcopy(kwargs)
    dmp_step_quaternion_python(**kwargs_python)
    kwargs_cython = deepcopy(kwargs)
    dmp_step_quaternion_cython(**kwargs_cython)

    assert_array_almost_equal(kwargs_python["current_y"], kwargs_cython["current_y"])
    assert_array_almost_equal(kwargs_python["current_yd"], kwargs_cython["current_yd"])


def test_get_set_weights():
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

    weights = dmp.get_weights()
    dmp.set_weights(weights)

    T3, Y3 = dmp.open_loop()
    assert_array_almost_equal(T, T3)
    assert_array_almost_equal(Y, Y3, decimal=2)


def test_invalid_step_function():
    dmp = CartesianDMP(
        execution_time=1.0, dt=0.01,
        n_weights_per_dim=10, int_dt=0.001)
    assert_raises_regex(ValueError, "Step function", dmp.open_loop,
                        step_function="invalid")
    assert_raises_regex(ValueError, "Step function", dmp.open_loop,
                        quaternion_step_function="invalid")

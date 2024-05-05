import numpy as np
from pytransform3d.rotations import quaternion_gradient, assert_quaternion_equal
from movement_primitives.dmp import (
    DMP, DMPWithFinalVelocity, CartesianDMP, DualCartesianDMP)
from numpy.testing import assert_array_almost_equal


def _setup_circle():
    T = np.linspace(0, 1, 101)
    x = np.sin(T ** 2 * 1.99 * np.pi)
    y = np.cos(T ** 2 * 1.99 * np.pi)
    return T, x, y


def test_smooth_scaling_dmp():
    T, x, y = _setup_circle()
    Y = np.column_stack((x, y))
    new_start, new_goal = np.array([5.0, 0.5]), np.array([-5.0, 0.5])

    dmp = DMP(n_dims=2, execution_time=1.0, dt=0.01,
              n_weights_per_dim=20, smooth_scaling=False)
    dmp.imitate(T, Y)
    dmp.configure(start_y=new_start, goal_y=new_goal)
    for step_function in ["rk4", "euler", "euler-cython", "rk4-cython"]:
        dmp.smooth_scaling = False
        _, Y_dmp = dmp.open_loop(step_function=step_function)
        dmp.smooth_scaling = True
        _, Y_dmp_smooth = dmp.open_loop(step_function=step_function)
        Yd_dmp = np.gradient(Y_dmp, axis=0)
        Ydd_dmp = np.gradient(Yd_dmp, axis=0)
        Yd_dmp_smooth = np.gradient(Y_dmp_smooth, axis=0)
        Ydd_dmp_smooth = np.gradient(Yd_dmp_smooth, axis=0)
        # with smooth scaling we avoid large accelerations in the beginning
        assert np.max(np.abs(Ydd_dmp_smooth)) < 0.2 * np.max(np.abs(Ydd_dmp))
        assert_array_almost_equal(Y_dmp[-1], new_goal, decimal=1)


def test_smooth_scaling_cartesian_dmp():
    T, x, y = _setup_circle()
    z = qx = qy = qz = np.zeros_like(x)
    qw = np.ones_like(x)
    Y = np.column_stack((x, y, z, qw, qx, qy, qz))
    new_start = np.array([5.0, 0.5, 0, 0, 0, 1, 0])
    new_goal = np.array([-5.0, 0.5, 0, 0, 1, 0, 0])

    dmp = CartesianDMP(execution_time=1.0, dt=0.01, n_weights_per_dim=20, smooth_scaling=False)
    dmp.imitate(T, Y)
    dmp.configure(start_y=new_start, goal_y=new_goal)
    for step_function in ["cython", "python"]:
        dmp.smooth_scaling = False
        _, Y_dmp = dmp.open_loop(quaternion_step_function=step_function)
        dmp.smooth_scaling = True
        _, Y_dmp_smooth = dmp.open_loop(quaternion_step_function=step_function)
        Yd_pos_dmp = np.gradient(Y_dmp[:, :3], axis=0)
        Ydd_pos_dmp = np.gradient(Yd_pos_dmp[:, :3], axis=0)
        Yd_pos_dmp_smooth = np.gradient(Y_dmp_smooth, axis=0)
        Ydd_pos_dmp_smooth = np.gradient(Yd_pos_dmp_smooth, axis=0)
        Yd_orn_dmp = quaternion_gradient(Y_dmp[:, 3:])
        Ydd_orn_dmp = np.gradient(Yd_orn_dmp, axis=0)
        Yd_orn_dmp_smooth = quaternion_gradient(Y_dmp_smooth[:, 3:])
        Ydd_orn_dmp_smooth = np.gradient(Yd_orn_dmp_smooth, axis=0)
        # with smooth scaling we avoid large accelerations in the beginning
        assert np.max(np.abs(Ydd_pos_dmp_smooth)) < 0.2 * np.max(np.abs(Ydd_pos_dmp))
        assert np.max(np.abs(Ydd_orn_dmp_smooth)) < 0.2 * np.max(np.abs(Ydd_orn_dmp))
        assert_array_almost_equal(Y_dmp[-1, :3], new_goal[:3], decimal=1)
        assert_quaternion_equal(Y_dmp[-1, 3:], new_goal[3:], decimal=4)


def test_smooth_scaling_dual_cartesian_dmp():
    T, x, y = _setup_circle()
    z = qx = qy = qz = np.zeros_like(x)
    qw = np.ones_like(x)
    Y = np.column_stack((x, y, z, qw, qx, qy, qz, x, y, z, qw, qx, qy, qz))
    new_start = np.array([5.0, 0.5, 0, 0, 0, 1, 0, 5.0, 0.5, 0, 0, 0, 1, 0])
    new_goal = np.array([-5.0, 0.5, 0, 0, 1, 0, 0, -5.0, 0.5, 0, 0, 1, 0, 0])

    dmp = DualCartesianDMP(
        execution_time=1.0, dt=0.01, n_weights_per_dim=20,
        smooth_scaling=False)
    dmp.imitate(T, Y)
    dmp.configure(start_y=new_start, goal_y=new_goal)
    for step_function in ["cython", "python"]:
        dmp.smooth_scaling = False
        _, Y_dmp = dmp.open_loop(step_function=step_function)
        dmp.smooth_scaling = True
        _, Y_dmp_smooth = dmp.open_loop(step_function=step_function)
        Yd_pos_dmp = np.gradient(Y_dmp[:, :3], axis=0)
        Ydd_pos_dmp = np.gradient(Yd_pos_dmp[:, :3], axis=0)
        Yd_pos_dmp_smooth = np.gradient(Y_dmp_smooth, axis=0)
        Ydd_pos_dmp_smooth = np.gradient(Yd_pos_dmp_smooth, axis=0)
        Yd_orn_dmp = quaternion_gradient(Y_dmp[:, 3:7])
        Ydd_orn_dmp = np.gradient(Yd_orn_dmp, axis=0)
        Yd_orn_dmp_smooth = quaternion_gradient(Y_dmp_smooth[:, 3:7])
        Ydd_orn_dmp_smooth = np.gradient(Yd_orn_dmp_smooth, axis=0)
        # with smooth scaling we avoid large accelerations in the beginning
        assert np.max(np.abs(Ydd_pos_dmp_smooth)) < 0.2 * np.max(np.abs(Ydd_pos_dmp))
        assert np.max(np.abs(Ydd_orn_dmp_smooth)) < 0.2 * np.max(np.abs(Ydd_orn_dmp))
        assert_array_almost_equal(Y_dmp[-1, :3], new_goal[:3], decimal=1)
        assert_quaternion_equal(Y_dmp[-1, 3:7], new_goal[3:7], decimal=4)

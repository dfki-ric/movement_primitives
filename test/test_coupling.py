import numpy as np
from movement_primitives.dmp import (
    DMP, CouplingTermPos1DToPos1D, CouplingTermPos3DToPos3D, DualCartesianDMP,
    CouplingTermDualCartesianPose)
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from nose.tools import assert_almost_equal, assert_less


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


def test_coupling_term_dual_cartesian_pose():
    dt = 0.01
    int_dt = 0.001
    execution_time = 1.0

    desired_distance = np.array([  # right arm to left arm
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.2],
        [0.0, 0.0, 0.0, 1.0]
    ])
    desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle(
        [np.deg2rad(180), 0, 0])
    ct = CouplingTermDualCartesianPose(
        desired_distance=desired_distance, couple_position=True,
        couple_orientation=True, lf=(1.0, 0.0), k=1, c1=0.1, c2=1000)  # c2=10000 in simulation

    Y_demo = np.zeros((1001, 14))
    T_demo = np.linspace(0, 1, len(Y_demo))
    sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T_demo - 0.5)) + 1.0)
    radius = 0.5

    circle1 = radius * np.cos(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
    circle2 = radius * np.sin(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
    Y_demo[:, 0] = circle1
    Y_demo[:, 1] = 0.55
    Y_demo[:, 2] = circle2
    R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
    R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(0)])
    # introduces coupling error (default goal: -90; error at: -110)
    R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-110)])
    q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
    q_end = -pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
    for i, t in enumerate(T_demo):
        Y_demo[i, 3:7] = pr.quaternion_slerp(q_start, q_end, t)

    circle1 = radius * np.cos(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
    circle2 = radius * np.sin(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
    Y_demo[:, 7] = circle1
    Y_demo[:, 8] = 0.55
    Y_demo[:, 9] = circle2
    R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
    R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-180)])
    R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-270)])
    q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
    q_end = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
    for i, t in enumerate(T_demo):
        Y_demo[i, 10:] = pr.quaternion_slerp(q_start, q_end, t)

    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=int_dt, p_gain=0.0)
    dmp.imitate(T_demo, Y_demo)
    T, Y = dmp.open_loop(coupling_term=ct)
    for y in Y:
        left2origin = pt.transform_from_pq(y[:7])
        right2origin = pt.transform_from_pq(y[7:])
        right2left = pt.concat(right2origin, pt.invert_transform(left2origin))
        error = np.linalg.norm(right2left - desired_distance)
        assert_less(error, 0.25)

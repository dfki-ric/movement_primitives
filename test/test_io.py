import os
import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import batch_rotations as pbr
from movement_primitives.dmp import DMP, DualCartesianDMP
from movement_primitives.promp import ProMP
from movement_primitives.io import (
    write_pickle, read_pickle, write_yaml, read_yaml, write_json, read_json)
from numpy.testing import assert_array_almost_equal


def test_pickle_io():
    dmp = DMP(n_dims=6, execution_time=1.0, dt=0.01, n_weights_per_dim=10,
              int_dt=0.0001)

    T = np.linspace(0.0, 1.0, 101)
    Y = np.empty((len(T), 6))
    Y[:, 0] = np.cos(np.pi * T)
    Y[:, 1] = np.sin(np.pi * T)
    Y[:, 2] = np.sin(2 * np.pi * T)
    Y[:, 3] = np.cos(np.pi * T)
    Y[:, 4] = np.sin(np.pi * T)
    Y[:, 5] = 0.5 + np.sin(2 * np.pi * T)
    dmp.imitate(T, Y)

    try:
        write_pickle("dmp.pickle", dmp)
        dmp2 = read_pickle("dmp.pickle")
    finally:
        if os.path.exists("dmp.pickle"):
            os.remove("dmp.pickle")

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T1, Y1 = dmp.open_loop()

    dmp2.configure(start_y=Y[0], goal_y=Y[-1])
    T2, Y2 = dmp2.open_loop()

    assert_array_almost_equal(T1, T2)
    assert_array_almost_equal(Y1, Y2)


def test_yaml_io():
    dmp = DualCartesianDMP(
        execution_time=1.0, dt=0.01, n_weights_per_dim=10, int_dt=0.001)

    Y = np.zeros((1001, 14))
    T = np.linspace(0, 1, len(Y))
    sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
    radius = 0.5

    circle1 = radius * np.cos(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
    circle2 = radius * np.sin(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
    Y[:, 0] = circle1
    Y[:, 1] = 0.55
    Y[:, 2] = circle2
    R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
    R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(0)])
    R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-90)])
    q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
    q_end = -pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
    Y[:, 3:7] = pbr.quaternion_slerp_batch(q_start, q_end, T)

    circle1 = radius * np.cos(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
    circle2 = radius * np.sin(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
    Y[:, 7] = circle1
    Y[:, 8] = 0.55
    Y[:, 9] = circle2
    R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
    R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-180)])
    R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-270)])
    q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
    q_end = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
    Y[:, 10:] = pbr.quaternion_slerp_batch(q_start, q_end, T)

    dmp.imitate(T, Y)

    try:
        write_yaml("dmp.yaml", dmp)
        dmp2 = read_yaml("dmp.yaml")
    finally:
        if os.path.exists("dmp.yaml"):
            os.remove("dmp.yaml")

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T1, Y1 = dmp.open_loop()

    dmp2.configure(start_y=Y[0], goal_y=Y[-1])
    T2, Y2 = dmp2.open_loop()

    assert_array_almost_equal(T1, T2)
    assert_array_almost_equal(Y1, Y2)


def test_json_io():
    promp = ProMP(n_dims=1, n_weights_per_dim=100)

    random_state = np.random.RandomState(10)
    n_demos = 10
    n_steps = 101
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Y = np.empty((n_demos, n_steps, 1))
    for demo_idx in range(n_demos):
        Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
        Y[demo_idx, :, 0] += random_state.randn(n_steps) * 0.01

    promp.imitate(T, Y)

    try:
        write_json("promp.json", promp)
        promp2 = read_json("promp.json")
    finally:
        if os.path.exists("promp.json"):
            os.remove("promp.json")

    T = np.linspace(0.0, 1.0, n_steps)
    Y1 = promp.mean_trajectory(T)
    Y2 = promp2.mean_trajectory(T)

    assert_array_almost_equal(Y1, Y2)

import numpy as np
from movement_primitives.dmp import (
    DualCartesianDMP, CouplingTermDualCartesianDistance,
    CouplingTermDualCartesianPose)
import pytransform3d.rotations as pr
import pytransform3d.trajectories as ptr
from numpy.testing import assert_array_almost_equal
import pytest


def test_pose_coupling():
    dt = 0.01
    int_dt = 0.001
    execution_time = 1.0

    desired_distance = np.array([  # right arm to left arm
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.2],
        [0.0, 0.0, 0.0, 1.0]
    ])
    desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle([np.deg2rad(180), 0, 0])

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
    # introduces coupling error (default goal: -90; error at: -110)
    R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-110)])
    q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
    q_end = -pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
    for i, t in enumerate(T):
        Y[i, 3:7] = pr.quaternion_slerp(q_start, q_end, t)

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
    for i, t in enumerate(T):
        Y[i, 10:] = pr.quaternion_slerp(q_start, q_end, t)

    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=int_dt, p_gain=0.0)
    dmp.imitate(T, Y)

    _, Y_none = dmp.open_loop(coupling_term=None)

    right2left = _relative_poses(Y_none[80:90])
    right2left_mean = np.mean(right2left, axis=0)
    right2left_mean[:3, :3] = pr.norm_matrix(right2left_mean[:3, :3])
    pose_error = np.dot(right2left_mean, np.linalg.inv(desired_distance))
    assert np.linalg.norm(pose_error[:3, 3]) > 0.25
    assert pr.axis_angle_from_matrix(pose_error[:3, :3])[-1] > 0.25

    coupling_term_pose = CouplingTermDualCartesianPose(
        desired_distance=desired_distance, couple_position=True,
        couple_orientation=True, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000,
        verbose=0)
    _, Y_pose = dmp.open_loop(coupling_term=coupling_term_pose)

    right2left = _relative_poses(Y_pose[80:90])
    right2left_mean = np.mean(right2left, axis=0)
    assert_array_almost_equal(right2left_mean, desired_distance, decimal=1)

    coupling_term_pose = CouplingTermDualCartesianPose(
        desired_distance=desired_distance, couple_position=True,
        couple_orientation=False, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000,
        verbose=0)
    _, Y_position = dmp.open_loop(coupling_term=coupling_term_pose)

    right2left = _relative_poses(Y_position[80:90])
    right2left_mean = np.mean(right2left, axis=0)
    assert_array_almost_equal(
        right2left_mean[:3, 3], desired_distance[:3, 3], decimal=1)

    coupling_term_pose = CouplingTermDualCartesianPose(
        desired_distance=desired_distance, couple_position=False,
        couple_orientation=True, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000,
        verbose=0)
    _, Y_orientation = dmp.open_loop(coupling_term=coupling_term_pose)

    right2left = _relative_poses(Y_orientation[80:90])
    right2left_mean = np.mean(right2left, axis=0)
    assert_array_almost_equal(
        right2left_mean[:3, :3], desired_distance[:3, :3], decimal=1)

    coupling_term_dist = CouplingTermDualCartesianDistance(
        desired_distance=0.2, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000)
    _, Y_dist = dmp.open_loop(coupling_term=coupling_term_dist)

    right2left = _relative_poses(Y_dist[80:90])
    right2left_mean = np.mean(right2left, axis=0)
    assert np.linalg.norm(right2left_mean[:3, 3]) == pytest.approx(
        0.2, abs=0.02)


def _relative_poses(Y_pose):
    left2base = ptr.transforms_from_pqs(Y_pose[:, :7])
    right2base = ptr.transforms_from_pqs(Y_pose[:, 7:])
    base2left = ptr.invert_transforms(left2base)
    right2left = np.einsum("nij,njk->nik", base2left, right2base)
    return right2left


if __name__ == "__main__":
    test_pose_coupling()

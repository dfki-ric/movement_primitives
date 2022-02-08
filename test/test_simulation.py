import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr
from movement_primitives.testing.simulation import (
    KinematicsChain, UR5Simulation)
from movement_primitives.kinematics import Kinematics
from numpy.testing import assert_array_almost_equal


def test_inverse_kinematics():
    desired_ee2base = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw([0.5, 0, 0]),
        p=np.array([-0.3, 0, 0.5]))

    urdf_path = "examples/data/urdf/ur5.urdf"
    ee_frame = "ur5_tool0"
    base_frame = "ur5_base_link"
    joint_names = [
        "ur5_shoulder_pan_joint", "ur5_shoulder_lift_joint", "ur5_elbow_joint",
        "ur5_wrist_1_joint", "ur5_wrist_2_joint", "ur5_wrist_3_joint"]

    chain1 = KinematicsChain(ee_frame, joint_names, urdf_path, debug_gui=False)

    with open(urdf_path, "r") as f:
        kin = Kinematics(f.read())
    chain2 = kin.create_chain(joint_names, base_frame, ee_frame)

    q0 = np.zeros(len(joint_names))
    q1 = chain1.inverse(
        pt.pq_from_transform(desired_ee2base), q0, n_iter=500,
        threshold=0.0001)
    q2 = chain2.inverse_with_random_restarts(
        desired_ee2base, n_restarts=10, tolerance=0.0001,
        random_state=np.random.RandomState(42))

    ee_pose1 = chain2.forward(q1)
    ee_pose2 = chain2.forward(q2)

    assert_array_almost_equal(ee_pose1, desired_ee2base, decimal=2)
    assert_array_almost_equal(ee_pose2, desired_ee2base, decimal=2)


def test_ur5():
    desired_ee2base = pt.transform_from(
        R=pr.active_matrix_from_extrinsic_roll_pitch_yaw([0.5, 0, 0]),
        p=np.array([-0.3, 0, 0.5]))

    ur5 = UR5Simulation(dt=0.01, gui=False, real_time=False)
    for _ in range(4):
        ur5.goto_ee_state(pt.pq_from_transform(desired_ee2base), wait_time=1.0)
        ur5.stop()

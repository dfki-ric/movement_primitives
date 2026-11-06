import numpy as np
import pytest
from movement_primitives.kinematics import Kinematics
from numpy.testing import assert_array_almost_equal


COMPI_URDF = """
<?xml version="1.0"?>
  <robot name="compi">
    <link name="linkmount"/>
    <link name="link1"/>
    <link name="link2"/>
    <link name="link3"/>
    <link name="link4"/>
    <link name="link5"/>
    <link name="link6"/>
    <link name="tcp"/>

    <joint name="joint1" type="revolute">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <parent link="linkmount"/>
      <child link="link1"/>
      <axis xyz="0 0 1.0"/>
    </joint>

    <joint name="joint2" type="revolute">
      <origin xyz="0 0 0.158" rpy="1.570796 0 0"/>
      <parent link="link1"/>
      <child link="link2"/>
      <axis xyz="0 0 -1.0"/>
    </joint>

    <joint name="joint3" type="revolute">
      <origin xyz="0 0.28 0" rpy="0 0 0"/>
      <parent link="link2"/>
      <child link="link3"/>
      <axis xyz="0 0 -1.0"/>
    </joint>

    <joint name="joint4" type="revolute">
      <origin xyz="0 0 0" rpy="-1.570796 0 0"/>
      <parent link="link3"/>
      <child link="link4"/>
      <axis xyz="0 0 1.0"/>
    </joint>

    <joint name="joint5" type="revolute">
      <origin xyz="0 0 0.34" rpy="1.570796 0 0"/>
      <parent link="link4"/>
      <child link="link5"/>
      <axis xyz="0 0 -1.0"/>
    </joint>

    <joint name="joint6" type="revolute">
      <origin xyz="0 0.346 0" rpy="-1.570796 0 0"/>
      <parent link="link5"/>
      <child link="link6"/>
      <axis xyz="0 0 1.0"/>
    </joint>

    <joint name="jointtcp" type="fixed">
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <parent link="link6"/>
      <child link="tcp"/>
    </joint>
  </robot>
"""


def test_forward_inverse():
    with open("examples/data/urdf/ur5.urdf", "r") as f:
        kin = Kinematics(f.read())
    joint_names = ["ur5_shoulder_pan_joint",
                   "ur5_shoulder_lift_joint",
                   "ur5_elbow_joint",
                   "ur5_wrist_1_joint",
                   "ur5_wrist_2_joint",
                   "ur5_wrist_3_joint"]
    chain = kin.create_chain(joint_names, "ur5_base_link", "ur5_tool0")
    random_state = np.random.RandomState(1232)
    for _ in range(5):
        q = (random_state.rand(len(chain.joint_limits))
             * (chain.joint_limits[:, 1] - chain.joint_limits[:, 0])
             + chain.joint_limits[:, 0])
        ee2base = chain.forward(q)
        q2 = chain.inverse_with_random_restarts(
            ee2base, random_state=random_state, n_restarts=20)
        ee2base2 = chain.forward(q2)
        assert_array_almost_equal(ee2base, ee2base2, decimal=3)


def test_forward_inverse_precise():
    with open("examples/data/urdf/ur5.urdf", "r") as f:
        kin = Kinematics(f.read())
    joint_names = ["ur5_shoulder_pan_joint",
                   "ur5_shoulder_lift_joint",
                   "ur5_elbow_joint",
                   "ur5_wrist_1_joint",
                   "ur5_wrist_2_joint",
                   "ur5_wrist_3_joint"]
    chain = kin.create_chain(joint_names, "ur5_base_link", "ur5_tool0")
    random_state = np.random.RandomState(1232)
    for _ in range(5):
        q = (random_state.rand(len(chain.joint_limits))
             * (chain.joint_limits[:, 1] - chain.joint_limits[:, 0])
             + chain.joint_limits[:, 0])
        ee2base = chain.forward(q)
        q2 = chain.local_inverse_with_random_restarts(
            ee2base, joint_angles=q + 0.1, interval=0.2,
            random_state=random_state, n_restarts=3, tolerance=0.0)
        ee2base2 = chain.forward(q2)
        assert_array_almost_equal(ee2base, ee2base2, decimal=3)


def test_forward_inverse_trajectory():
    kin = Kinematics(COMPI_URDF)
    chain = kin.create_chain(
        ["joint%d" % i for i in range(1, 7)], "compi", "tcp", verbose=0)

    Q = np.zeros((100, chain.n_joints))
    Q[:, 0] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 1] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 2] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 3] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 4] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 5] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))

    H = chain.forward_trajectory(Q)
    random_state = np.random.RandomState(2)
    Q2 = chain.inverse_trajectory(H, Q[0], random_state=random_state)
    H2 = chain.forward_trajectory(Q2)

    assert_array_almost_equal(H, H2, decimal=3)


def test_forward_inverse_trajectory_without_initialization():
    kin = Kinematics(COMPI_URDF)
    chain = kin.create_chain(
        ["joint%d" % i for i in range(1, 7)], "compi", "tcp", verbose=0)

    Q = np.zeros((100, chain.n_joints))
    Q[:, 0] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 1] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 2] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 3] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 4] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 5] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))

    H = chain.forward_trajectory(Q)
    random_state = np.random.RandomState(2)
    Q2 = chain.inverse_trajectory(
        H, None, random_state=random_state)
    H2 = chain.forward_trajectory(Q2)

    assert_array_almost_equal(H, H2, decimal=3)


def test_forward_inverse_trajectory_without_restarts():
    kin = Kinematics(COMPI_URDF)
    chain = kin.create_chain(
        ["joint%d" % i for i in range(1, 7)], "compi", "tcp", verbose=0)

    Q = np.zeros((100, chain.n_joints))
    Q[:, 0] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 1] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 2] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 3] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 4] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))
    Q[:, 5] = np.linspace(-0.5 * np.pi, 0.5 * np.pi, len(Q))

    H = chain.forward_trajectory(Q)
    Q2 = chain.inverse_trajectory(H, Q[0], random_restarts=False)
    H2 = chain.forward_trajectory(Q2)

    assert_array_almost_equal(H, H2, decimal=3)


@pytest.mark.parametrize("target_x,center,interval,expected", [
    (-0.8, 0.0, 0.1, -0.1), (0.8, 0.0, 0.1, 0.1),
    (0.8, 0.9, 0.05, 0.85), (-0.8, -0.9, 0.05, -0.85),
    (1.5, 0.95, 0.2, 1.0), (-1.5, -0.95, 0.2, -1.0),
    (0.05, 0.0, 0.1, 0.05)])
@pytest.mark.parametrize("n_restarts", [1, 3])
def test_local_inverse_respects_interval(
        target_x, center, interval, expected, n_restarts):
    urdf = """<robot name="slider">
    <link name="base"/><link name="tip"/>
    <joint name="slide" type="prismatic">
      <parent link="base"/><child link="tip"/><axis xyz="1 0 0"/>
      <limit lower="-1" upper="1" effort="1" velocity="1"/>
    </joint></robot>"""
    chain = Kinematics(urdf).create_chain(["slide"], "base", "tip")
    target = np.eye(4)
    target[0, 3] = target_x
    q = chain.local_inverse_with_random_restarts(
        target, np.array([center]), interval=interval, n_restarts=n_restarts,
        random_state=np.random.RandomState(0))
    assert abs(q[0] - center) <= interval + 1e-8
    assert -1.0 <= q[0] <= 1.0
    np.testing.assert_allclose(q[0], expected, atol=1e-6)


@pytest.mark.parametrize("random_restarts", [False, True])
def test_inverse_trajectory_interval_with_restarts(random_restarts):
    urdf = """<robot name="slider">
    <link name="base"/><link name="tip"/>
    <joint name="slide" type="prismatic">
      <parent link="base"/><child link="tip"/><axis xyz="1 0 0"/>
      <limit lower="-1" upper="1" effort="1" velocity="1"/>
    </joint></robot>"""
    chain = Kinematics(urdf).create_chain(["slide"], "base", "tip")
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[1:, 0, 3] = [0.8, -0.8, 0.8]
    q = chain.inverse_trajectory(
        poses, initial_joint_angles=np.array([0.0]), interval=0.1,
        random_restarts=random_restarts, random_state=np.random.RandomState(0))
    assert np.max(np.abs(np.diff(q[:, 0]))) <= 0.1 + 1e-8

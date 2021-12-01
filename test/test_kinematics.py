import numpy as np
from movement_primitives.kinematics import Kinematics
from numpy.testing import assert_array_almost_equal


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

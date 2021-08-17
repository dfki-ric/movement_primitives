import numpy as np
import pybullet
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from testing.simulation import analyze_robot, draw_transform, _pytransform_pose, _pybullet_pose


class KinematicsChain:
    def __init__(self, ee_frame, joints, urdf_path, debug_gui=False):
        if debug_gui:
            self.client_id = pybullet.connect(pybullet.GUI)
        else:
            self.client_id = pybullet.connect(pybullet.DIRECT)
        pybullet.resetSimulation(physicsClientId=self.client_id)

        self.chain = pybullet.loadURDF(
            urdf_path, useFixedBase=1, physicsClientId=self.client_id)
        self.joint_indices, self.link_indices = analyze_robot(
            robot=self.chain, physicsClientId=self.client_id)

        self.chain_joint_indices = [self.joint_indices[jn] for jn in joints]
        self.n_chain_joints = len(self.chain_joint_indices)
        self.ee_idx = self.link_indices[ee_frame]

    def inverse(self, desired_ee_state, q_current=None):
        if q_current is not None:
            # not effective in this step yet
            pybullet.setJointMotorControlArray(
                self.chain, self.chain_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=q_current, physicsClientId=self.client_id)
        ee_pos, ee_rot = _pybullet_pose(desired_ee_state)
        q = pybullet.calculateInverseKinematics(
            self.chain, self.ee_idx, ee_pos, ee_rot,
            maxNumIterations=100, residualThreshold=0.001,
            jointDamping=[0.1] * self.n_chain_joints,
            physicsClientId=self.client_id)
        return q


ee_frame = "LTCP_Link"
joints = ("ALShoulder1", "ALShoulder2", "ALShoulder3", "ALElbow", "ALWristRoll", "ALWristYaw", "ALWristPitch")
urdf_path = "pybullet-only-arms-urdf/submodels/left_arm.urdf"

left_arm_ik = KinematicsChain(ee_frame, joints, urdf_path)

left_arm_gui = KinematicsChain(ee_frame, joints, urdf_path, debug_gui=True)

orientation = pr.quaternion_from_matrix(pr.active_matrix_from_extrinsic_roll_pitch_yaw([np.pi, 0, 0.5 * np.pi]))
desired_pose = np.array([0.6, 0.3, 0.5] + orientation.tolist())
q = left_arm_ik.inverse(desired_pose, np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8]))

pybullet.setJointMotorControlArray(
    left_arm_gui.chain, left_arm_gui.chain_joint_indices,
    pybullet.POSITION_CONTROL,
    targetPositions=q,
    physicsClientId=left_arm_gui.client_id)
print(f"Desired joint state: {np.round(q, 2)}")
for _ in range(100):
    pybullet.stepSimulation(physicsClientId=left_arm_gui.client_id)


def get_joint_state(self):
    joint_states = pybullet.getJointStates(self.chain, left_arm_gui.chain_joint_indices, physicsClientId=left_arm_gui.client_id)
    positions = []
    for joint_state in joint_states:
        pos, vel, forces, torque = joint_state
        positions.append(pos)
    return np.asarray(positions)


q = get_joint_state(left_arm_gui)

print(f"Actual joint state: {np.round(q, 2)}")

_, _, _, _, left_pos, left_rot, _, _ = pybullet.getLinkState(
    left_arm_gui.chain, left_arm_gui.ee_idx, computeLinkVelocity=1,
    computeForwardKinematics=1, physicsClientId=left_arm_gui.client_id)
pybullet.addUserDebugLine(left_pos, [0, 0, 0], [1, 0, 0], 2, physicsClientId=left_arm_gui.client_id)
left_pose = _pytransform_pose(left_pos, left_rot)
print(f"Actual pose: {np.round(left_pose, 3)}")

print(f"Desired pose: {np.round(desired_pose, 3)}")
desired_pos, desired_rot = _pybullet_pose(desired_pose)

draw_transform(pt.transform_from_pq(left_pose), 0.1, left_arm_gui.client_id)
draw_transform(pt.transform_from_pq(desired_pose), 0.1, left_arm_gui.client_id)

while True:
    pybullet.stepSimulation(physicsClientId=left_arm_gui.client_id)

import numpy as np
import pybullet
from pytransform3d import rotations as pr
from simulation import analyze_robot, _pytransform_pose, _pybullet_pose


class LeftArmKinematics:
    def __init__(self, ee_frame="LTCP_Link",
                 joints=("ALShoulder1", "ALShoulder2", "ALShoulder3",
                         "ALElbow", "ALWristRoll", "ALWristYaw", "ALWristPitch"),
                 debug_gui=True):
        if debug_gui:
            self.client_id = pybullet.connect(pybullet.GUI)
        else:
            self.client_id = pybullet.connect(pybullet.DIRECT)
        pybullet.resetSimulation(physicsClientId=self.client_id)
        self.left_arm = pybullet.loadURDF(
            "pybullet-only-arms-urdf/submodels/left_arm.urdf",
            useFixedBase=1, physicsClientId=self.client_id)
        self.joint_indices, self.link_indices = analyze_robot(robot=self.left_arm, physicsClientId=self.client_id)

        self.left_arm_joint_indices = [self.joint_indices[jn] for jn in joints]
        self.n_joints = len(self.left_arm_joint_indices)
        self.left_arm_ee_idx_ik = self.link_indices[ee_frame]

        com2w_p, com2w_q, _, _, _, _ = pybullet.getLinkState(
            self.left_arm, 0, computeForwardKinematics=True, physicsClientId=self.client_id)
        self.com2world = com2w_p, com2w_q
        self.world2com = pybullet.invertTransform(*self.com2world)

        #self.robot2world = (0, 0, 0), (0.0, 0.0, 0.0, 1.0)  # not pybullet.getBasePositionAndOrientation(self.left_arm)
        #self.world2robot = pybullet.invertTransform(*self.robot2world)

    def inverse(self, left_ee_state, q_current=None):
        left_pos, left_rot = _pybullet_pose(left_ee_state)
        # ee2world
        #left_pos, left_rot = pybullet.multiplyTransforms(
        #    left_pos, left_rot, *self.robot2world)
        if q_current is not None:  # not effective in this step yet
            pybullet.setJointMotorControlArray(
                self.left_arm, self.left_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=q_current, physicsClientId=self.client_id)
        q = pybullet.calculateInverseKinematics(
            self.left_arm, self.left_arm_ee_idx_ik, left_pos, left_rot,
            maxNumIterations=100, residualThreshold=0.001,
            jointDamping=[0.1] * self.n_joints, physicsClientId=self.client_id)
        return q


left_arm_kin = LeftArmKinematics(debug_gui=True)  # TODO breaks as soon as it is not at (0, 0, 0)
q = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8])
pybullet.setTimeStep(0.001, physicsClientId=left_arm_kin.client_id)
pybullet.setJointMotorControlArray(
    left_arm_kin.left_arm, left_arm_kin.left_arm_joint_indices,
    pybullet.POSITION_CONTROL,
    targetPositions=q,
    physicsClientId=left_arm_kin.client_id)
print(f"Desired joint state: {np.round(q, 2)}")
#print(left_arm_kin.left_arm_pose)
orientation = pr.quaternion_from_matrix(pr.active_matrix_from_extrinsic_roll_pitch_yaw([np.pi, 0, 0.5 * np.pi]))
desired_pose = np.array([0.6, 0.3, 0.5] + orientation.tolist())
#desired_pose = np.array([-0.19557431,  0.55368418,  0.92392093,  0.91187376,  0.15737931,  0.37514561, 0.05462424])
q = left_arm_kin.inverse(desired_pose)
#for i in range(20):
#    print(pybullet.getJointInfo(left_arm_kin.left_arm, i))
#for i in range(20):
#    print(pybullet.getLinkState(left_arm_kin.left_arm, i))
pybullet.setJointMotorControlArray(
    left_arm_kin.left_arm, left_arm_kin.left_arm_joint_indices,
    pybullet.POSITION_CONTROL,
    targetPositions=q,
    physicsClientId=left_arm_kin.client_id)
print(f"Desired joint state: {np.round(q, 2)}")
for _ in range(100):
    pybullet.stepSimulation(physicsClientId=left_arm_kin.client_id)


def get_joint_state(self):
    joint_states = pybullet.getJointStates(self.left_arm, left_arm_kin.left_arm_joint_indices, physicsClientId=left_arm_kin.client_id)
    positions = []
    for joint_state in joint_states:
        pos, vel, forces, torque = joint_state
        positions.append(pos)
    return np.asarray(positions)


q = get_joint_state(left_arm_kin)

print(f"Actual joint state: {np.round(q, 2)}")

_, _, _, _, left_pos, left_rot, _, _ = pybullet.getLinkState(
    left_arm_kin.left_arm, left_arm_kin.left_arm_ee_idx_ik, computeLinkVelocity=1,
    computeForwardKinematics=1, physicsClientId=left_arm_kin.client_id)
pybullet.addUserDebugLine(left_pos, [0, 0, 0], [1, 0, 0], 2, physicsClientId=left_arm_kin.client_id)
left_pose = _pytransform_pose(left_pos, left_rot)
print(f"Actual pose: {np.round(left_pose, 3)}")

print(f"Desired pose: {np.round(desired_pose, 3)}")
desired_pos, desired_rot = _pybullet_pose(desired_pose)
pybullet.addUserDebugLine(desired_pos, [0, 0, 0], [0, 1, 0], 2, physicsClientId=left_arm_kin.client_id)

while True:
    pybullet.stepSimulation(physicsClientId=left_arm_kin.client_id)

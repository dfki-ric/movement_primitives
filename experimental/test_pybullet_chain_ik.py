import numpy as np
import pybullet
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from simulation import LeftArmKinematics, _pytransform_pose, _pybullet_pose, _pytransform_pose


print(pybullet.connect(pybullet.GUI))
pybullet.resetSimulation()
pybullet.setTimeStep(0.001)
left_arm_kin = LeftArmKinematics(left_arm_pos=(0, 0, 0))
q = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8])
pybullet.setJointMotorControlArray(
    left_arm_kin.left_arm, left_arm_kin.left_arm_joint_indices,
    pybullet.POSITION_CONTROL,
    targetPositions=q)
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
    targetPositions=q)
print(f"Desired joint state: {np.round(q, 2)}")
for _ in range(100):
    pybullet.stepSimulation()


def get_joint_state(self):
    joint_states = pybullet.getJointStates(self.left_arm, left_arm_kin.left_arm_joint_indices)
    positions = []
    for joint_state in joint_states:
        pos, vel, forces, torque = joint_state
        positions.append(pos)
    return np.asarray(positions)


q = get_joint_state(left_arm_kin)

print(f"Actual joint state: {np.round(q, 2)}")

left_ee_state = pybullet.getLinkState(
    left_arm_kin.left_arm, left_arm_kin.left_arm_ee_idx_ik, computeLinkVelocity=1,
    computeForwardKinematics=1)
left_pos = left_ee_state[4]
left_rot = left_ee_state[5]
pybullet.addUserDebugLine(left_pos, [0, 0, 0], [1, 0, 0], 2)
left_pos, left_rot = pybullet.multiplyTransforms(
    left_pos, left_rot, *left_arm_kin.robot2world)
left_pose = _pytransform_pose(left_pos, left_rot)
print(f"Actual pose: {np.round(left_pose, 3)}")

print(f"Desired pose: {np.round(desired_pose, 3)}")
desired_pos, desired_rot = _pybullet_pose(desired_pose)
desired_pos, desired_rot = pybullet.multiplyTransforms(
    desired_pos, desired_rot, *left_arm_kin.world2robot)
pybullet.addUserDebugLine(desired_pos, [0, 0, 0], [0, 1, 0], 2)

while True:
    pybullet.stepSimulation()

import numpy as np
import pybullet
import pybullet_data


class UR5Simulation:  # Quaternion convention: x, y, z, w
    def __init__(self, dt, gui=True, real_time=False):
        self.dt = dt
        if gui:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.resetSimulation()
        pybullet.setTimeStep(dt)
        pybullet.setRealTimeSimulation(1 if real_time else 0)
        pybullet.setGravity(0, 0, -9.81)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pybullet.loadURDF("plane.urdf", [0, 0, -1], useFixedBase=1)
        self.robot = pybullet.loadURDF("ur5_fts300_2f-140/urdf/ur5.urdf", [0, 0, 0], useFixedBase=1)

        self.base_pose = pybullet.getBasePositionAndOrientation(self.robot)
        self.inv_base_pose = pybullet.invertTransform(*self.base_pose)

        self.n_ur5_joints = 6
        self.ee_link_index = pybullet.getJointInfo(self.robot, self.n_ur5_joints)[16] + 1

        self.n_joints = pybullet.getNumJoints(self.robot)
        self.joint_indices = [
            i for i in range(self.n_joints) if pybullet.getJointInfo(self.robot, i)[2] == 0]  # joint type 0: revolute
        self.joint_names = {i: pybullet.getJointInfo(self.robot, i)[1] for i in self.joint_indices}
        # we cannot actually use them so far:
        self.joint_max_velocities = [pybullet.getJointInfo(self.robot, i)[11] for i in self.joint_indices]

        #print([self.joint_names[i] for i in self.joint_indices[:6]])

    def get_joint_state(self):
        joint_states = pybullet.getJointStates(self.robot, self.joint_indices[:self.n_ur5_joints])
        positions = []
        velocities = []
        for joint_state in joint_states:
            pos, vel, forces, torque = joint_state
            positions.append(pos)
            velocities.append(vel)
        return np.asarray(positions), np.asarray(velocities)

    def stop(self):
        pybullet.setJointMotorControlArray(
            self.robot, self.joint_indices[:self.n_ur5_joints], pybullet.VELOCITY_CONTROL,
            targetVelocities=np.zeros(self.n_ur5_joints))

    def set_desired_joint_state(self, joint_state, position_control=False):
        if position_control:
            pybullet.setJointMotorControlArray(
                self.robot, self.joint_indices[:self.n_ur5_joints], pybullet.POSITION_CONTROL,
                targetPositions=joint_state)
        else:  # velocity control
            pybullet.setJointMotorControlArray(
                self.robot, self.joint_indices[:self.n_ur5_joints], pybullet.VELOCITY_CONTROL,
                targetVelocities=joint_state)

    def get_ee_state(self):
        ee_state = pybullet.getLinkState(
            self.robot, self.ee_link_index, computeForwardKinematics=1)
        pos = ee_state[4]
        rot = ee_state[5]
        pos, rot = pybullet.multiplyTransforms(pos, rot, *self.inv_base_pose)
        return np.hstack((pos, [rot[-1]], rot[:-1]))  # xyzw -> wxyz

    def inverse_kinematics(self, ee2robot):
        pos = ee2robot[:3]
        rot = ee2robot[3:]
        rot = np.hstack((rot[1:], [rot[0]]))  # wxyz -> xyzw

        # ee2world
        pos, rot = pybullet.multiplyTransforms(pos, rot, *self.base_pose)

        q = pybullet.calculateInverseKinematics(
            self.robot, self.ee_link_index, pos, rot, maxNumIterations=100, residualThreshold=0.001)
        q = q[:self.n_ur5_joints]
        if any(np.isnan(q)):
            raise Exception("IK solver found no solution.")
        return q

    def set_desired_ee_state(self, ee_state):
        q = self.inverse_kinematics(ee_state)
        last_q, last_qd = self.get_joint_state()
        self.set_desired_joint_state((q - last_q) / self.dt, position_control=False)

    def step(self):
        pybullet.stepSimulation()

    def sim_loop(self, n_steps=None):
        if n_steps is None:
            while True:
                pybullet.stepSimulation()
        else:
            for _ in range(n_steps):
                pybullet.stepSimulation()
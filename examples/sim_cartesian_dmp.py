import numpy as np
from dmp import DMP
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
        self.plane = pybullet.loadURDF("plane.urdf", useFixedBase=1)
        self.robot = pybullet.loadURDF("ur5_fts300_2f-140/urdf/ur5.urdf", [0, 0, 1], useFixedBase=1)

        self.base_pose = pybullet.getBasePositionAndOrientation(self.robot)

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
                targetPositions=joint_state,
                #positionGains=[0.1] * self.n_ur5_joints, velocityGains=[0.0] * self.n_ur5_joints,
                #targetVelocities=[0] * self.n_ur5_joints,
                #forces=[10] * self.n_ur5_joints
            )
        else:  # velocity control
            pybullet.setJointMotorControlArray(
                self.robot, self.joint_indices[:self.n_ur5_joints], pybullet.VELOCITY_CONTROL,
                targetVelocities=joint_state)

    def get_ee_state(self):
        ee_state = pybullet.getLinkState(
            self.robot, self.ee_link_index, computeForwardKinematics=1)
        pos = ee_state[2]
        rot = ee_state[3]
        return np.hstack((pos, [rot[-1]], rot[:-1]))  # xyzw -> wxyz

    def inverse_kinematics(self, ee_state):
        pos = ee_state[:3]
        rot = ee_state[3:]
        rot = np.hstack((rot[1:], [rot[0]]))  # wxyz -> xyzw

        pos, rot = pybullet.multiplyTransforms(pos, rot, *self.base_pose)

        q = pybullet.calculateInverseKinematics(
            self.robot, self.ee_link_index, pos, rot, maxNumIterations=100, residualThreshold=0.01)
        q = q[:self.n_ur5_joints]
        return q

    def set_desired_ee_state(self, ee_state):
        q = self.inverse_kinematics(ee_state)
        last_q, last_qd = self.get_joint_state()
        self.set_desired_joint_state((q - last_qd) / self.dt, position_control=False)

    def step(self):
        pybullet.stepSimulation()

    def sim_loop(self, n_steps=None):
        if n_steps is None:
            while True:
                pybullet.stepSimulation()
        else:
            for _ in range(n_steps):
                pybullet.stepSimulation()


dt = 0.01

dmp = DMP(n_dims=7, execution_time=1.0, dt=0.001, n_weights_per_dim=10)
T = np.linspace(0.0, 1.0, 101)
Y = np.empty((len(T), 7))
Y[:, 0] = 0.3 + 0.1 * np.cos(np.pi * T)
Y[:, 1] = 0.3 + 0.1 * np.sin(np.pi * T)
Y[:, 2] = 0.0 + 0.1 * np.sin(2 * np.pi * T)
Y[:, 3] = 1.0
Y[:, 4] = 0.0
Y[:, 5] = 0.0
Y[:, 6] = 0.0
dmp.imitate(T, Y)
dmp.configure(start_y=Y[0], goal_y=Y[-1])
ur5 = UR5Simulation(dt=0.001, real_time=True)
q = ur5.inverse_kinematics(Y[0])
ur5.set_desired_joint_state(q, position_control=True)
ur5.sim_loop(1000)
#ur5.sim_loop()
print(ur5.get_ee_state())
positions = []
desired_positions = []
velocities = []
desired_velocities = []
last_v = np.zeros(7)
for i in range(1001):
    #last_p, last_v = ur5.get_ee_state()
    last_p = ur5.get_ee_state()
    p, v = dmp.step(last_p, last_v)
    print(np.linalg.norm(p - last_p))
    ur5.set_desired_ee_state(p)
    ur5.step()

    positions.append(last_p)
    desired_positions.append(p)
    velocities.append(last_v)
    desired_velocities.append(v)
    print("====")
    print(dmp.t)
    print(np.round(p, 2))
    print(np.round(last_p, 2))
    last_v = v
ur5.stop()

import matplotlib.pyplot as plt
P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plt.plot(P[:, 1], label="Actual")
plt.plot(dP[:, 1], label="Desired")
T, Y = dmp.open_loop(run_t=1.0)
plt.plot(Y[:, 1], label="Open loop")
plt.legend()
plt.show()
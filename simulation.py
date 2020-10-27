import numpy as np
import pybullet
import pybullet_data


# Quaternion convention: x, y, z, w

class PybulletSimulation:
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

def _pybullet_pose(pose):
    pos = pose[:3]
    rot = pose[3:]
    rot = np.hstack((rot[1:], [rot[0]]))  # wxyz -> xyzw
    return pos, rot

def _pytransform_pose(pos, rot):
    return np.hstack((pos, [rot[-1]], rot[:-1]))  # xyzw -> wxyz


class UR5Simulation(PybulletSimulation):
    def __init__(self, dt, gui=True, real_time=False):
        super(UR5Simulation, self).__init__(dt, gui, real_time)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pybullet.loadURDF(
            "plane.urdf", [0, 0, -1], useFixedBase=1)
        self.robot = pybullet.loadURDF(
            "ur5_fts300_2f-140/urdf/ur5.urdf", [0, 0, 0], useFixedBase=1)

        self.base_pose = pybullet.getBasePositionAndOrientation(self.robot)
        self.inv_base_pose = pybullet.invertTransform(*self.base_pose)

        self.n_ur5_joints = 6
        # one link after the base link of the last joint
        self.ee_link_index = pybullet.getJointInfo(
            self.robot, self.n_ur5_joints)[16] + 1

        self.n_joints = pybullet.getNumJoints(self.robot)
        self.joint_indices = [
            i for i in range(self.n_joints)
            if pybullet.getJointInfo(self.robot, i)[2] == 0]  # joint type 0: revolute
        self.joint_names = {i: pybullet.getJointInfo(self.robot, i)[1]
                            for i in self.joint_indices}
        # we cannot actually use them so far:
        self.joint_max_velocities = [pybullet.getJointInfo(self.robot, i)[11]
                                     for i in self.joint_indices]

        #print([self.joint_names[i] for i in self.joint_indices[:6]])

    def inverse_kinematics(self, ee2robot):
        pos, rot = _pybullet_pose(ee2robot)
        # ee2world
        pos, rot = pybullet.multiplyTransforms(pos, rot, *self.base_pose)

        q = pybullet.calculateInverseKinematics(
            self.robot, self.ee_link_index, pos, rot, maxNumIterations=100,
            residualThreshold=0.001)
        q = q[:self.n_ur5_joints]
        if any(np.isnan(q)):
            raise Exception("IK solver found no solution.")
        return q

    def get_joint_state(self):
        joint_states = pybullet.getJointStates(self.robot, self.joint_indices[:self.n_ur5_joints])
        positions = []
        velocities = []
        for joint_state in joint_states:
            pos, vel, forces, torque = joint_state
            positions.append(pos)
            velocities.append(vel)
        return np.asarray(positions), np.asarray(velocities)

    def set_desired_joint_state(self, joint_state, position_control=False):
        if position_control:
            pybullet.setJointMotorControlArray(
                self.robot, self.joint_indices[:self.n_ur5_joints],
                pybullet.POSITION_CONTROL,
                targetPositions=joint_state)
        else:  # velocity control
            pybullet.setJointMotorControlArray(
                self.robot, self.joint_indices[:self.n_ur5_joints],
                pybullet.VELOCITY_CONTROL, targetVelocities=joint_state)

    def get_ee_state(self, return_velocity=False):
        ee_state = pybullet.getLinkState(
            self.robot, self.ee_link_index, computeLinkVelocity=1,
            computeForwardKinematics=1)
        pos = ee_state[4]
        rot = ee_state[5]
        pos, rot = pybullet.multiplyTransforms(pos, rot, *self.inv_base_pose)
        if return_velocity:
            vel = ee_state[6]
            #ang_vel = ee_state[7]
            #ang_speed = np.linalg.norm(ang_vel)
            #ang_axis = np.asarray(ang_vel) / ang_speed
            vel, _ = pybullet.multiplyTransforms(
                vel, [0, 0, 0, 1], *self.inv_base_pose)
            # TODO transform angular velocity?
            return _pytransform_pose(pos, rot), np.hstack((vel, np.zeros(3)))
        else:
            return _pytransform_pose(pos, rot)

    def set_desired_ee_state(self, ee_state):
        q = self.inverse_kinematics(ee_state)
        last_q, last_qd = self.get_joint_state()
        self.set_desired_joint_state(
            (q - last_q) / self.dt, position_control=False)

    def step(self):
        pybullet.stepSimulation()

    def sim_loop(self, n_steps=None):
        if n_steps is None:
            while True:
                pybullet.stepSimulation()
        else:
            for _ in range(n_steps):
                pybullet.stepSimulation()

    def stop(self):
        pybullet.setJointMotorControlArray(
            self.robot, self.joint_indices[:self.n_ur5_joints],
            pybullet.VELOCITY_CONTROL,
            targetVelocities=np.zeros(self.n_ur5_joints))
        self.step()

    def goto_ee_state(self, ee_state, wait_time=1.0, text=None):
        if text:
            pos, rot = _pybullet_pose(ee_state)
            self.write(pos, text)
        q = self.inverse_kinematics(ee_state)
        self.set_desired_joint_state(q, position_control=True)
        self.sim_loop(int(wait_time / self.dt))

    def step_through_cartesian(self, steppable, last_p, last_v, execution_time, closed_loop=False):
        p, v = self.get_ee_state(return_velocity=True)
        desired_positions = [last_p]
        positions = [p]
        desired_velocities = [last_v]
        velocities = [v]

        for i in range(int(execution_time / self.dt)):
            if closed_loop:
                last_p, _ = self.get_ee_state(return_velocity=True)  # TODO last_v

            p, v = steppable.step(last_p, last_v)
            self.set_desired_ee_state(p)
            self.step()

            desired_positions.append(p)
            desired_velocities.append(v)

            last_v = v
            last_p = p

            p, v = self.get_ee_state(return_velocity=True)
            positions.append(p)
            velocities.append(v)

        self.stop()

        return (np.asarray(desired_positions),
                np.asarray(positions),
                np.asarray(desired_velocities),
                np.asarray(velocities))

    def write(self, pos, text):
        pybullet.addUserDebugText(text, pos, [0, 0, 0])
        pybullet.addUserDebugLine(pos, [0, 0, 0], [0, 0, 0], 2)


class LeftArmKinematics:
    def __init__(self, left_arm_pos=(-1, 0, 5)):
        self.left_arm_pos = left_arm_pos
        self.left_arm = pybullet.loadURDF(
            "abstract-urdf-gripper/urdf/rh5_left_arm.urdf", self.left_arm_pos, useFixedBase=1)
        self.left_arm_pose = self.left_arm_pos, (0.0, 0.0, 0.0, 1.0)  # not pybullet.getBasePositionAndOrientation(self.left_arm)
        self.left_arm_joint_indices = [4, 5, 6, 7, 8, 9, 10]
        self.n_joints = len(self.left_arm_joint_indices)
        self.left_arm_ee_idx_ik = 10

    def inverse(self, left_ee_state, q_current=None):
        left_pos, left_rot = _pybullet_pose(left_ee_state)
        # ee2world
        left_pos, left_rot = pybullet.multiplyTransforms(left_pos, left_rot, *self.left_arm_pose)
        if q_current is not None:  # not effective in this step yet
            pybullet.setJointMotorControlArray(
                self.left_arm, self.left_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=q_current)
        q = pybullet.calculateInverseKinematics(
            self.left_arm, self.left_arm_ee_idx_ik, left_pos, left_rot, maxNumIterations=100,
            residualThreshold=0.001, jointDamping=[0.1] * self.n_joints)
        return q


class RightArmKinematics:
    def __init__(self, right_arm_pos=(1, 0, 5)):
        self.right_arm_pos = right_arm_pos
        self.right_arm = pybullet.loadURDF(
            "abstract-urdf-gripper/urdf/rh5_right_arm.urdf", self.right_arm_pos, useFixedBase=1)
        self.right_arm_pose = self.right_arm_pos, (0.0, 0.0, 0.0, 1.0)  # not pybullet.getBasePositionAndOrientation(self.right_arm)
        self.right_arm_joint_indices = [4, 5, 6, 7, 8, 9, 10]
        self.n_joints = len(self.right_arm_joint_indices)
        self.right_arm_ee_idx_ik = 10

    def inverse(self, right_ee_state, q_current=None):
        right_pos, right_rot = _pybullet_pose(right_ee_state)
        # ee2world
        right_pos, right_rot = pybullet.multiplyTransforms(right_pos, right_rot, *self.right_arm_pose)
        if q_current is not None:  # not effective in this step yet
            pybullet.setJointMotorControlArray(
                self.right_arm, self.right_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=q_current)
        q = pybullet.calculateInverseKinematics(
            self.right_arm, self.right_arm_ee_idx_ik, right_pos, right_rot, maxNumIterations=100,
            residualThreshold=0.001, jointDamping=[0.1] * self.n_joints)
        return q


class RH5Simulation(PybulletSimulation):  # https://git.hb.dfki.de/bolero-environments/graspbullet/-/blob/transfit_wp5300/Grasping/grasping_env_rh5.py
    def __init__(self, dt, gui=True, real_time=False):
        super(RH5Simulation, self).__init__(dt, gui, real_time)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.base_pos = [0, 0, 0]
        self.plane = pybullet.loadURDF(
            "plane.urdf", [0, 0, -1], useFixedBase=1)
        self.robot = pybullet.loadURDF(
            "abstract-urdf-gripper/urdf/rh5_fixed.urdf", self.base_pos, useFixedBase=1)
        self.left_arm_kin = LeftArmKinematics()
        self.right_arm_kin = RightArmKinematics()

        self.base_pose = self.base_pos, (0.0, 0.0, 0.0, 1.0)  # not pybullet.getBasePositionAndOrientation(self.robot)
        self.inv_base_pose = pybullet.invertTransform(*self.base_pose)

        self.left_arm_joint_indices = [4, 5, 6, 7, 8, 9, 10]
        self.right_arm_joint_indices = [21, 22, 23, 24, 25, 26, 27]
        # base link of the joint after the last joint
        self.left_ee_link_index = pybullet.getJointInfo(
            self.robot, max(self.left_arm_joint_indices) + 1)[16]
        self.right_ee_link_index = pybullet.getJointInfo(
            self.robot, max(self.right_arm_joint_indices) + 1)[16]

    def inverse_kinematics(self, ee2robot):
        q = np.empty(len(self.left_arm_joint_indices) + len(self.right_arm_joint_indices))
        left_ee_state, right_ee_state = np.split(ee2robot, (7,))
        q_current, _ = self.get_joint_state()
        q[:len(self.left_arm_joint_indices)] = self.left_arm_kin.inverse(left_ee_state, q_current[:len(self.left_arm_joint_indices)])
        q[len(self.left_arm_joint_indices):] = self.right_arm_kin.inverse(right_ee_state, q_current[len(self.left_arm_joint_indices):])
        if any(np.isnan(q)):
            raise Exception("IK solver found no solution.")
        return q

    def get_joint_state(self):
        joint_states = pybullet.getJointStates(self.robot, self.left_arm_joint_indices + self.right_arm_joint_indices)
        positions = []
        velocities = []
        for joint_state in joint_states:
            pos, vel, forces, torque = joint_state
            positions.append(pos)
            velocities.append(vel)
        return np.asarray(positions), np.asarray(velocities)

    def set_desired_joint_state(self, joint_state, position_control=False):
        left_joint_state, right_joint_state = np.split(joint_state, (len(self.left_arm_joint_indices),))
        if position_control:
            pybullet.setJointMotorControlArray(
                self.robot, self.left_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=left_joint_state)
            pybullet.setJointMotorControlArray(
                self.robot, self.right_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=right_joint_state)
        else:  # velocity control
            pybullet.setJointMotorControlArray(
                self.robot, self.left_arm_joint_indices,
                pybullet.VELOCITY_CONTROL, targetVelocities=left_joint_state)
            pybullet.setJointMotorControlArray(
                self.robot, self.right_arm_joint_indices,
                pybullet.VELOCITY_CONTROL, targetVelocities=right_joint_state)

    def get_ee_state(self, return_velocity=False):
        left_ee_state = pybullet.getLinkState(
            self.robot, self.left_ee_link_index, computeLinkVelocity=1,
            computeForwardKinematics=1)
        left_pos = left_ee_state[4]
        left_rot = left_ee_state[5]
        left_pos, left_rot = pybullet.multiplyTransforms(left_pos, left_rot, *self.inv_base_pose)
        left_pose = _pytransform_pose(left_pos, left_rot)

        right_ee_state = pybullet.getLinkState(
            self.robot, self.right_ee_link_index, computeLinkVelocity=1,
            computeForwardKinematics=1)
        right_pos = right_ee_state[4]
        right_rot = right_ee_state[5]
        right_pos, right_rot = pybullet.multiplyTransforms(right_pos, right_rot, *self.inv_base_pose)
        right_pose = _pytransform_pose(right_pos, right_rot)

        if return_velocity:
            raise NotImplementedError()
            """
            left_vel = left_ee_state[6]
            #ang_vel = ee_state[7]
            #ang_speed = np.linalg.norm(ang_vel)
            #ang_axis = np.asarray(ang_vel) / ang_speed
            left_vel, _ = pybullet.multiplyTransforms(
                left_vel, [0, 0, 0, 1], *self.inv_base_pose)
            # TODO transform angular velocity?
            return _pytransform_pose(pos, rot), np.hstack((vel, np.zeros(3)))
            """
        else:
            return np.hstack((left_pose, right_pose))

    def set_desired_ee_state(self, ee_state, position_control=False):
        q = self.inverse_kinematics(ee_state)
        if position_control:
            self.set_desired_joint_state(q, position_control=True)
        else:
            last_q, _ = self.get_joint_state()
            self.set_desired_joint_state(
                (q - last_q) / self.dt, position_control=False)

    def step(self):
        pybullet.stepSimulation()

    def sim_loop(self, n_steps=None):
        if n_steps is None:
            while True:
                pybullet.stepSimulation()
        else:
            for _ in range(n_steps):
                pybullet.stepSimulation()

    def stop(self):
        #pybullet.setJointMotorControlArray(
        #    self.robot, self.left_arm_joint_indices + self.right_arm_joint_indices,
        #    pybullet.VELOCITY_CONTROL,
        #    targetVelocities=np.zeros(len(self.left_arm_joint_indices) + len(self.right_arm_joint_indices)))
        x = self.get_ee_state()
        q = self.inverse_kinematics(x)
        self.set_desired_joint_state(q, position_control=True)
        self.step()

    def goto_ee_state(self, ee_state, wait_time=1.0, text=None):
        if text:
            pos, rot = _pybullet_pose(ee_state)
            self.write(pos, text)
        q = self.inverse_kinematics(ee_state)
        self.set_desired_joint_state(q, position_control=True)
        self.sim_loop(int(wait_time / self.dt))

    def step_through_cartesian(self, steppable, last_p, last_v, execution_time, closed_loop=False, coupling_term=None):
        p = self.get_ee_state(return_velocity=False)   # TODO v
        desired_positions = [last_p]
        positions = [p]
        desired_velocities = [last_v]
        velocities = [np.zeros(12)]

        for i in range(int(execution_time / self.dt)):
            if closed_loop:
                last_p = self.get_ee_state(return_velocity=False)  # TODO last_v

            p, v = steppable.step(last_p, last_v, coupling_term=coupling_term)
            self.set_desired_ee_state(p)
            self.step()

            desired_positions.append(p)
            desired_velocities.append(v)

            last_v = v
            last_p = p

            p = self.get_ee_state(return_velocity=False)  # TODO v
            positions.append(p)
            #velocities.append(v)
            velocities.append(last_v)

        self.stop()

        return (np.asarray(desired_positions),
                np.asarray(positions),
                np.asarray(desired_velocities),
                np.asarray(velocities))

    def write(self, pos, text):
        pybullet.addUserDebugText(text, pos, [0, 0, 0])
        pybullet.addUserDebugLine(pos, [0, 0, 0], [0, 0, 0], 2)

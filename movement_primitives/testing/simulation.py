"""Robot simulations based on PyBullet.

Note that PyBullet internally represents orientations as quaternions with
scalar-last convention: (qx, qy, qz, qw).
"""
import numpy as np
import os
try:
    import pybullet
    import pybullet_data
    pybullet_available = True
except ImportError:
    pybullet_available = False
import pytransform3d.transformations as pt


class PybulletSimulation:
    """PyBullet simulation of a robot.

    Parameters
    ----------
    dt : float
        Length of time step

    gui : bool, optional (default: True)
        Show PyBullet GUI

    real_time : bool, optional (default: False)
        Simulate in real time
    """
    def __init__(self, dt, gui=True, real_time=False):
        assert pybullet_available
        self.dt = dt
        if gui:
            self.client_id = pybullet.connect(pybullet.GUI)
        else:
            self.client_id = pybullet.connect(pybullet.DIRECT)

        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 0)
        pybullet.resetDebugVisualizerCamera(2, 75, -30, [0, 0, 0])

        pybullet.resetSimulation(physicsClientId=self.client_id)
        pybullet.setTimeStep(dt, physicsClientId=self.client_id)
        pybullet.setRealTimeSimulation(
            1 if real_time else 0, physicsClientId=self.client_id)
        pybullet.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

    def step(self):
        """Simulation step."""
        assert pybullet.isConnected(self.client_id)
        pybullet.stepSimulation(physicsClientId=self.client_id)

    def sim_loop(self, n_steps=None):
        """Run simulation loop.

        Parameters
        ----------
        n_steps : int, optional (default: infinite)
            Number of simulation steps.
        """
        if n_steps is None:
            while pybullet.isConnected(self.client_id):
                pybullet.stepSimulation(physicsClientId=self.client_id)
        else:
            for _ in range(n_steps):
                if not pybullet.isConnected(self.client_id):
                    break
                pybullet.stepSimulation(physicsClientId=self.client_id)


def _pybullet_pose(pose):
    """Convert pose from (x, y, z, qw, qx, qy, qz) to ((x, y, z), (qx, qy, qz, qw))."""
    pos = pose[:3]
    rot = pose[3:]
    rot = np.hstack((rot[1:], [rot[0]]))  # wxyz -> xyzw
    return pos, rot


def _pytransform_pose(pos, rot):
    """Convert pose from ((x, y, z), (qx, qy, qz, qw)) to (x, y, z, qw, qx, qy, qz)."""
    return np.hstack((pos, [rot[-1]], rot[:-1]))  # xyzw -> wxyz


def draw_transform(pose2origin, s, client_id, lw=1):
    """Draw pose represented by transformation matrix.

    Parameters
    ----------
    pose2origin : array-like, shape (4, 4)
        Homogeneous transformation matrix

    s : float
        Scale, length of the coordinate axes

    client_id : int
        Physics client ID

    lw : int, optional (default: 1)
        Line width
    """
    pose2origin = pt.check_transform(pose2origin)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 0],
        [1, 0, 0], lw, physicsClientId=client_id)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 1],
        [0, 1, 0], lw, physicsClientId=client_id)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 2],
        [0, 0, 1], lw, physicsClientId=client_id)


def draw_pose(pose2origin, s, client_id, lw=1):
    """Draw pose represented by position and quaternion.

    Parameters
    ----------
    pose2origin : array-like, shape (7,)
        Position and quaternion: (x, y, z, qw, qx, qy, qz)

    s : float
        Scale, length of the coordinate axes

    client_id : int
        Physics client ID

    lw : int, optional (default: 1)
        Line width
    """
    pose2origin = pt.transform_from_pq(pose2origin)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 0],
        [1, 0, 0], lw, physicsClientId=client_id)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 1],
        [0, 1, 0], lw, physicsClientId=client_id)
    pybullet.addUserDebugLine(
        pose2origin[:3, 3], pose2origin[:3, 3] + s * pose2origin[:3, 2],
        [0, 0, 1], lw, physicsClientId=client_id)


def draw_trajectory(A2Bs, client_id, n_key_frames=10, s=1.0, lw=1):
    """Draw trajectory.

    Parameters
    ----------
    A2Bs : array-like, shape (n_steps, 4, 4)
        Homogeneous transformation matrices

    client_id : int
        Physics client ID

    n_key_frames : int, optional (default: 10)
        Number of coordinate frames

    s : float, optional (default: 1)
        Scale, length of the coordinate axes

    lw : int, optional (default: 1)
        Line width
    """
    key_frames_indices = np.linspace(
        0, len(A2Bs) - 1, n_key_frames, dtype=np.int)
    for idx in key_frames_indices:
        draw_transform(A2Bs[idx], s=s, client_id=client_id)
    for idx in range(len(A2Bs) - 1):
        pybullet.addUserDebugLine(
            A2Bs[idx, :3, 3], A2Bs[idx + 1, :3, 3], [0, 0, 0], lw,
            physicsClientId=client_id)


def get_absolute_path(urdf_path, model_prefix_path):
    autoproj_dir = None
    if "AUTOPROJ_CURRENT_ROOT" in os.environ and os.path.exists(os.environ["AUTOPROJ_CURRENT_ROOT"]):
        autoproj_dir = os.environ["AUTOPROJ_CURRENT_ROOT"]
    if autoproj_dir is not None and os.path.exists(os.path.join(autoproj_dir, model_prefix_path)):
        return os.path.join(autoproj_dir, model_prefix_path, urdf_path)
    else:
        return urdf_path


class UR5Simulation(PybulletSimulation):
    """PyBullet simulation of UR5 robot arm.

    Parameters
    ----------
    dt : float
        Length of time step

    gui : bool, optional (default: True)
        Show PyBullet GUI

    real_time : bool, optional (default: False)
        Simulate in real time
    """
    def __init__(self, dt, gui=True, real_time=False):
        super(UR5Simulation, self).__init__(dt, gui, real_time)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pybullet.loadURDF(
            "plane.urdf", [0, 0, -1], useFixedBase=1)
        self.robot = pybullet.loadURDF(
            "examples/data/urdf/ur5.urdf", [0, 0, 0], useFixedBase=1)

        self.base_pose = pybullet.getBasePositionAndOrientation(self.robot)
        self.inv_base_pose = pybullet.invertTransform(*self.base_pose)

        self.n_ur5_joints = 6
        # one link after the base link of the last joint
        self.ee_link_index = pybullet.getJointInfo(
            self.robot, self.n_ur5_joints)[16] + 2

        self.n_joints = pybullet.getNumJoints(self.robot)
        self.joint_indices = [
            i for i in range(self.n_joints)
            if pybullet.getJointInfo(self.robot, i)[2] == 0]  # joint type 0: revolute
        self.joint_names = {i: pybullet.getJointInfo(self.robot, i)[1]
                            for i in self.joint_indices}
        # we cannot actually use them so far:
        self.joint_max_velocities = [pybullet.getJointInfo(self.robot, i)[11]
                                     for i in self.joint_indices]

    def inverse_kinematics(self, ee2robot):
        """Inverse kinematics of UR5.

        Parameters
        ----------
        ee2robot : array-like, shape (7,)
            End-effector pose: (x, y, z, qw, qx, qy, qz).

        Returns
        -------
        q : array, shape (n_joints,)
            Joint angles
        """
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
        """Get joint state.

        Returns
        -------
        positions : array, shape (n_joints,)
            Joint angles

        velocities : array, shape (n_joints,)
            Joint velocities
        """
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


class KinematicsChain:
    """Wrapper of PyBullet simulation of one kinematic chain.

    This can be used to compute inverse kinematics.

    Parameters
    ----------
    ee_frame : str
        Frame of the end-effector.

    joints : list
        Names of joints.

    urdf_path : str
        Path to URDF file.

    debug_gui : bool, optional (default: False)
        Start PyBullet with GUI for debugging purposes.

    Attributes
    ----------
    client_id_ : int
        ID of the PyBullet instance.
    """
    def __init__(self, ee_frame, joints, urdf_path, debug_gui=False):
        if debug_gui:
            self.client_id_ = pybullet.connect(pybullet.GUI)
        else:
            self.client_id_ = pybullet.connect(pybullet.DIRECT)
        pybullet.resetSimulation(physicsClientId=self.client_id_)
        pybullet.setTimeStep(1.0, physicsClientId=self.client_id_)

        self.chain = pybullet.loadURDF(
            urdf_path, useFixedBase=1, physicsClientId=self.client_id_)
        self.joint_indices, self.link_indices, \
        self.joint_name_to_joint_idx_ik = analyze_robot(
            robot=self.chain, physicsClientId=self.client_id_,
            return_joint_indices=True, verbose=0)

        self.chain_joint_indices = [self.joint_indices[jn] for jn in joints]
        self.ee_idx = self.link_indices[ee_frame]
        self.q_indices = [self.joint_name_to_joint_idx_ik[jn] for jn in joints]

        self.n_all_chain_joints = pybullet.getNumJoints(
            self.chain, self.client_id_)

    def inverse(self, desired_ee_state, q_current=None, n_iter=100,
                threshold=0.001):
        """Compute inverse kinematics.

        Parameters
        ----------
        desired_ee_state : array-like, shape (7,)
            Desired pose of end effector: (x, y, z, qw, qx, qy, qz)

        q_current : array-like, shape (n_joints,)
            Current joint angles.

        n_iter : int, optional (default: 100)
            Number of iterations for the numerical inverse kinematics solver.

        threshold : float, optional (default: 0.001)
            Threshold for residual.

        Returns
        -------
        q : array, shape (n_joints,)
            Joint angles to reach the desired pose of the end effector.
            If the desired pose cannot be reached, this function will return
            joint angles that come as close as possible.
        """
        if q_current is not None:
            # we have to actively go to the current joint state
            # before computing inverse kinematics
            self.goto_joint_state(q_current)

        ee_pos, ee_rot = _pybullet_pose(desired_ee_state)
        q = pybullet.calculateInverseKinematics(
            self.chain, self.ee_idx, ee_pos, ee_rot,
            maxNumIterations=n_iter, residualThreshold=threshold,
            jointDamping=[0.1] * self.n_all_chain_joints,
            physicsClientId=self.client_id_)
        return np.asarray(q)[self.q_indices]

    def goto_joint_state(self, q_current, max_steps_to_joint_state=50,
                         joint_state_eps=0.001):
        pybullet.setJointMotorControlArray(
            self.chain, self.chain_joint_indices,
            pybullet.POSITION_CONTROL,
            targetPositions=q_current, physicsClientId=self.client_id_)

        for _ in range(max_steps_to_joint_state):
            pybullet.stepSimulation(physicsClientId=self.client_id_)
            q_internal = np.array([js[0] for js in pybullet.getJointStates(
                self.chain, self.chain_joint_indices,
                physicsClientId=self.client_id_)])
            if np.linalg.norm(q_current - q_internal) < joint_state_eps:
                break


class RH5Simulation(PybulletSimulation):
    """PyBullet simulation of RH5 humanoid robot.

    Parameters
    ----------
    dt : float
        Length of time step

    gui : bool, optional (default: True)
        Show PyBullet GUI

    real_time : bool, optional (default: False)
        Simulate in real time

    left_joints : tuple, optional
        Joints of the left arm

    right_joints : tuple, optional
        Joints of the right arm

    urdf_path : str, optional
        Path to URDF file

    left_arm_path : str, optional
        Path to URDF of left arm

    right_arm_path : str, optional
        Path to URDF of right arm
    """
    def __init__(self, dt, gui=True, real_time=False,
                 left_ee_frame="LTCP_Link", right_ee_frame="RTCP_Link",
                 left_joints=("ALShoulder1", "ALShoulder2", "ALShoulder3", "ALElbow", "ALWristRoll", "ALWristYaw", "ALWristPitch"),
                 right_joints=("ARShoulder1", "ARShoulder2", "ARShoulder3", "ARElbow", "ARWristRoll", "ARWristYaw", "ARWristPitch"),
                 urdf_path=get_absolute_path("pybullet-only-arms-urdf/urdf/RH5.urdf", "models/robots/rh5_models"),
                 left_arm_path=get_absolute_path("pybullet-only-arms-urdf/submodels/left_arm.urdf", "models/robots/rh5_models"),
                 right_arm_path=get_absolute_path("pybullet-only-arms-urdf/submodels/right_arm.urdf", "models/robots/rh5_models")):
        super(RH5Simulation, self).__init__(dt, gui, real_time)

        self.base_pos = (0, 0, 0)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = pybullet.loadURDF(
            "plane.urdf", (0, 0, -1), useFixedBase=1,
            physicsClientId=self.client_id)
        self.robot = pybullet.loadURDF(
            urdf_path, self.base_pos, useFixedBase=1,
            physicsClientId=self.client_id)
        self.joint_indices, self.link_indices = analyze_robot(
            robot=self.robot, physicsClientId=self.client_id)

        self.base_pose = self.base_pos, (0.0, 0.0, 0.0, 1.0)  # not pybullet.getBasePositionAndOrientation(self.robot)
        self.inv_base_pose = pybullet.invertTransform(*self.base_pose)

        self.n_joints = len(left_joints) + len(right_joints)
        self.n_left_joints = len(left_joints)
        self.left_arm_joint_indices = [self.joint_indices[jn] for jn in left_joints]
        self.right_arm_joint_indices = [self.joint_indices[jn] for jn in right_joints]
        self.left_ee_link_index = self.link_indices[left_ee_frame]
        self.right_ee_link_index = self.link_indices[right_ee_frame]

        self.left_ik = KinematicsChain(
            left_ee_frame, left_joints, left_arm_path)
        self.right_ik = KinematicsChain(
            right_ee_frame, right_joints, right_arm_path)

    def inverse_kinematics(self, ee2robot):
        """Inverse kinematics of RH5.

        Parameters
        ----------
        ee2robot : array-like, shape (14,)
            End-effector poses: two 7-tuples that contain the pose of the
            left and right end effector in the order (x, y, z, qw, qx, qy, qz).

        Returns
        -------
        q : array, shape (n_joints,)
            Joint angles
        """
        q = np.empty(self.n_joints)

        left_q = np.array([js[0] for js in pybullet.getJointStates(
            self.robot, self.left_arm_joint_indices, physicsClientId=self.client_id)])
        q[:self.n_left_joints] = self.left_ik.inverse(ee2robot[:7], left_q)

        right_q = np.array([js[0] for js in pybullet.getJointStates(
            self.robot, self.right_arm_joint_indices, physicsClientId=self.client_id)])
        q[self.n_left_joints:] = self.right_ik.inverse(ee2robot[7:], right_q)

        return q

    def get_joint_state(self):
        """Get joint state.

        Returns
        -------
        positions : array, shape (n_joints,)
            Joint angles

        velocities : array, shape (n_joints,)
            Joint velocities
        """
        joint_states = pybullet.getJointStates(
            self.robot, self.left_arm_joint_indices + self.right_arm_joint_indices,
            physicsClientId=self.client_id)
        positions = np.empty(self.n_joints)
        velocities = np.empty(self.n_joints)
        for joint_idx, joint_state in enumerate(joint_states):
            positions[joint_idx], velocities[joint_idx], forces, torque = joint_state
        return positions, velocities

    def set_desired_joint_state(self, joint_state, position_control=False):
        left_joint_state, right_joint_state = np.split(joint_state, (len(self.left_arm_joint_indices),))
        if position_control:
            pybullet.setJointMotorControlArray(
                self.robot, self.left_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=left_joint_state,
                physicsClientId=self.client_id)
            pybullet.setJointMotorControlArray(
                self.robot, self.right_arm_joint_indices,
                pybullet.POSITION_CONTROL,
                targetPositions=right_joint_state,
                physicsClientId=self.client_id)
        else:  # velocity control
            pybullet.setJointMotorControlArray(
                self.robot, self.left_arm_joint_indices,
                pybullet.VELOCITY_CONTROL, targetVelocities=left_joint_state,
                physicsClientId=self.client_id)
            pybullet.setJointMotorControlArray(
                self.robot, self.right_arm_joint_indices,
                pybullet.VELOCITY_CONTROL, targetVelocities=right_joint_state,
                physicsClientId=self.client_id)

    def get_ee_state(self, return_velocity=False):
        left_ee_state = pybullet.getLinkState(
            self.robot, self.left_ee_link_index, computeLinkVelocity=1,
            computeForwardKinematics=1, physicsClientId=self.client_id)
        left_pos = left_ee_state[4]
        left_rot = left_ee_state[5]
        left_pos, left_rot = pybullet.multiplyTransforms(left_pos, left_rot, *self.inv_base_pose)
        left_pose = _pytransform_pose(left_pos, left_rot)

        right_ee_state = pybullet.getLinkState(
            self.robot, self.right_ee_link_index, computeLinkVelocity=1,
            computeForwardKinematics=1, physicsClientId=self.client_id)
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

    def stop(self):
        ee_state = self.get_ee_state(return_velocity=False)
        self.goto_ee_state(ee_state)
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
        pybullet.addUserDebugText(text, pos, [0, 0, 0], physicsClientId=self.client_id)
        pybullet.addUserDebugLine(pos, [0, 0, 0], [0, 0, 0], 2, physicsClientId=self.client_id)


class SimulationMockup:  # runs steppables open loop
    def __init__(self, dt):
        self.dt = dt
        self.ee_state = None

    def goto_ee_state(self, ee_state):
        self.ee_state = np.copy(ee_state)

    def step_through_cartesian(self, steppable, last_p, last_v, execution_time, coupling_term=None):
        desired_positions = [np.copy(last_p)]
        positions = [np.copy(last_p)]
        desired_velocities = [np.copy(last_v)]
        velocities = [np.copy(last_v)]

        for i in range(int(execution_time / self.dt)):
            p, v = steppable.step(last_p, last_v, coupling_term=coupling_term)

            desired_positions.append(p)
            desired_velocities.append(v)

            positions.append(p)
            velocities.append(v)

            last_v = v
            last_p = p

        return (np.asarray(desired_positions),
                np.asarray(positions),
                np.asarray(desired_velocities),
                np.asarray(velocities))


def analyze_robot(urdf_path=None, robot=None, physicsClientId=None,
                  return_joint_indices=False, verbose=0):
    """Compute mappings between joint and link names and their indices.

    Parameters
    ----------
    urdf_path : str, optional (default: None)
        Path to URDF file.

    robot : int, optional (default: None)
        PyBullet ID of robot.

    physicsClientId : int, optional (default: None)
        ID of a running bullet instance.

    return_joint_indices : bool, optional (default: False)
        Return joint indices used in inverse kinematics.

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    joint_name_to_joint_id : dict
        Mapping from joint names to PyBullet IDs

    link_name_to_link_id : dict
        Mapping from link names to PyBullet IDs

    joint_name_to_joint_idx_ik : dict
        Mapping from joint names to array indices in result of inverse
        kinematics
    """
    if urdf_path is not None:
        assert robot is None
        physicsClientId = pybullet.connect(pybullet.DIRECT)
        pybullet.resetSimulation(physicsClientId=physicsClientId)
        robot = pybullet.loadURDF(urdf_path, physicsClientId=physicsClientId)
    assert robot is not None

    base_link, robot_name = pybullet.getBodyInfo(robot, physicsClientId=physicsClientId)

    if verbose:
        print()
        print("=" * 80)
        print(f"Robot name: {robot_name}")
        print(f"Base link: {base_link}")

    n_joints = pybullet.getNumJoints(robot, physicsClientId=physicsClientId)

    last_link_idx = -1
    link_id_to_link_name = {last_link_idx: base_link}
    joint_name_to_joint_id = {}
    joint_name_to_joint_idx_ik = {}

    if verbose:
        print(f"Number of joints: {n_joints}")

    joint_idx_ik = 0

    for joint_idx in range(n_joints):
        _, joint_name, joint_type, q_index, u_index, _, jd, jf, lo, hi,\
            max_force, max_vel, child_link_name, ja, parent_pos,\
            parent_orient, parent_idx = pybullet.getJointInfo(
            robot, joint_idx, physicsClientId=physicsClientId)

        child_link_name = child_link_name.decode("utf-8")
        joint_name = joint_name.decode("utf-8")

        if child_link_name not in link_id_to_link_name.values():
            last_link_idx += 1
            link_id_to_link_name[last_link_idx] = child_link_name
        assert parent_idx in link_id_to_link_name

        joint_name_to_joint_id[joint_name] = joint_idx

        joint_type = _joint_type(joint_type)

        if joint_type != "fixed":
            joint_name_to_joint_idx_ik[joint_name] = joint_idx_ik
            joint_idx_ik += 1

        if verbose:
            print(f"Joint #{joint_idx}: {joint_name} ({joint_type}), "
                  f"child link: {child_link_name}, parent link index: {parent_idx}")
            if joint_type == "fixed":
                continue
            print("=" * 80)
            print(f"Index in positional state variables: {q_index}, "
                  f"Index in velocity state variables: {u_index}")
            print(f"Joint limits: [{lo}, {hi}], max. force: {max_force}, "
                  f"max. velocity: {max_vel}")
            print("=" * 80)

    if verbose:
        for link_idx in sorted(link_id_to_link_name.keys()):
            print(f"Link #{link_idx}: {link_id_to_link_name[link_idx]}")

    link_name_to_link_id = {v: k for k, v in link_id_to_link_name.items()}

    if return_joint_indices:
        return (joint_name_to_joint_id, link_name_to_link_id,
                joint_name_to_joint_idx_ik)
    else:
        return joint_name_to_joint_id, link_name_to_link_id


def _joint_type(id):
    try:
        return {pybullet.JOINT_REVOLUTE: "revolute",
                pybullet.JOINT_PRISMATIC: "prismatic",
                pybullet.JOINT_SPHERICAL: "spherical",
                pybullet.JOINT_PLANAR: "planar",
                pybullet.JOINT_FIXED: "fixed"}[id]
    except KeyError:
        raise ValueError(f"Unknown joint type id {id}")

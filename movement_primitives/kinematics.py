"""Forward kinematics and a simple implementation of inverse kinematics."""
import numpy as np
import math
import numba
from pytransform3d.urdf import UrdfTransformManager
from scipy.optimize import minimize


class FastUrdfTransformManager(UrdfTransformManager):
    """Fast transformation manager that can load URDF files.

    This version has efficient numba-accelerated code to update joints.
    """
    def __init__(self):
        super(FastUrdfTransformManager, self).__init__(check=False)

    def set_joint(self, joint_name, value):
        """Set joint position.

        Note that joint values are clipped to their limits.

        Parameters
        ----------
        joint_name : string
            Name of the joint

        value : float
            Joint angle in radians in case of revolute joints or position
            in case of prismatic joint.
        """
        from_frame, to_frame, child2parent, axis, limits, joint_type = \
            self._joints[joint_name]
        value = min(max(value, limits[0]), limits[1])
        if joint_type == "revolute":
            joint2A = _fast_matrix_from_axis_angle(axis, value)
        else:
            joint2A = np.eye(4)
            joint2A[:3, 3] = value * axis
        self.transforms[(from_frame, to_frame)] = child2parent.dot(joint2A)

    def get_ee2base(self, ee_index, base_index):
        """Request a transform.

        Parameters
        ----------
        ee_index : int
            Index of the end-effector node

        base_index : int
            Index of the base node

        Returns
        -------
        ee2base : array-like, shape (4, 4)
            Homogeneous matrix that represents the transform from ee to base
        """
        return self._path_transform(self._shortest_path(ee_index, base_index))


@numba.jit(nopython=True, cache=True)
def _fast_matrix_from_axis_angle(axis, angle):
    """Compute transformation matrix from axis-angle.

    Parameters
    ----------
    axis : array-like, shape (3,)
        Axis of rotation: (x, y, z).

    angle : float
        Rotation angle.

    Returns
    -------
    A2B : array-like, shape (4, 4)
        Transformation matrix
    """
    ux, uy, uz = axis
    c = math.cos(angle)
    s = math.sin(angle)
    ci = 1.0 - c
    ciux = ci * ux
    ciuy = ci * uy
    ciuz = ci * uz
    return np.array([
        [ciux * ux + c, ciux * uy - uz * s, ciux * uz + uy * s, 0.0],
        [ciuy * ux + uz * s, ciuy * uy + c, ciuy * uz - ux * s, 0.0],
        [ciuz * ux - uy * s, ciuz * uy + ux * s, ciuz * uz + c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


class Kinematics:
    """Robot kinematics.

    Parameters
    ----------
    urdf : str
        URDF description of a robot

    mesh_path : str, optional (default: None)
        Path in which we search for meshes that are defined in the URDF.
        Meshes will be ignored if it is set to None.
    """
    def __init__(self, urdf, mesh_path=None):
        self.tm = FastUrdfTransformManager()
        self.tm.load_urdf(urdf, mesh_path=mesh_path)

    def create_chain(self, joint_names, base_frame, ee_frame, verbose=0):
        """Create kinematic chain.

        Parameters
        ----------
        joint_names : list
            Names of joints that should be used

        base_frame : str
            Name of the base link

        ee_frame : str
            Name of the end-effector link

        verbose : int, optional (default: 0)
            Verbosity level

        Returns
        -------
        chain : Chain
            Kinematic chain
        """
        return Chain(self.tm, joint_names, base_frame, ee_frame, verbose)


class Chain:
    """Kinematic chain.

    Parameters
    ----------
    tm : FastUrdfTransformManager
        Transformation manager

    joint_names : list
        Names of joints that should be used

    base_frame : str
        Name of the base link

    ee_frame : str
        Name of the end-effector link

    verbose : int, optional (default: 0)
        Verbosity level
    """
    def __init__(self, tm, joint_names, base_frame, ee_frame, verbose=0):
        self.tm = tm
        self.joint_names = joint_names
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.verbose = verbose

        self.joint_limits = np.array(
            [self.tm._joints[jn][4] for jn in self.joint_names])
        for i in range(len(self.joint_limits)):
            if np.isinf(self.joint_limits[i, 0]):
                self.joint_limits[i, 0] = -math.pi
            if np.isinf(self.joint_limits[i, 1]):
                self.joint_limits[i, 1] = math.pi

        self.n_joints = len(self.joint_names)
        assert len(self.joint_limits) == self.n_joints

        self.ee_index = self.tm.nodes.index(ee_frame)
        self.base_index = self.tm.nodes.index(base_frame)

    def forward(self, joint_angles):
        """Forward kinematics.

        Parameters
        ----------
        joint_angles : array, shape (n_joints,)
            Joint angles

        Returns
        -------
        ee2base : array, shape (4, 4)
            Transformation from end-effector to base frame
        """
        for i in range(self.n_joints):
            self.tm.set_joint(self.joint_names[i], joint_angles[i])
        return self.tm.get_ee2base(self.ee_index, self.base_index)

    def ee_pose_error(self, joint_angles, desired_pose, orientation_weight=1.0,
                      position_weight=1.0):
        """Compute pose error.

        Parameters
        ----------
        joint_angles : array-like, shape (n_joints,)
            Actual joint angles for which we compute forward kinematics.

        desired_pose : array-like, shape (4, 4)
            Desired pose.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        pose_error : float
            Weighted error between actual pose and desired pose.
        """
        return pose_dist(desired_pose, self.forward(joint_angles),
                         orientation_weight, position_weight)

    def inverse(
            self, desired_pose, initial_joint_angles, return_error=False,
            bounds=None, solver="SLSQP", orientation_weight=1.0,
            position_weight=1.0):
        """Inverse kinematics.

        Parameters
        ----------
        desired_pose : array, shape (4, 4)
            Desired transformation from end-effector to base frame

        initial_joint_angles : array, shape (n_joints,)
            Initial guess for joint angles

        return_error : bool, optional (default: False)
            Return error in addition to joint angles

        bounds : array, shape (n_joints, 2), optional (default: joint limits)
            Bounds for joint angle optimization

        solver : str, optional (default: 'SLSQP')
            Optimizer to solve inverse kinematics problem. Possible options:
            'SLSQP', 'L-BFGS-B', and 'COBYLA'.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution

        error : float, optional
            Pose error
        """
        if bounds is None:
            bounds = self.joint_limits
        res = minimize(
            self.ee_pose_error, initial_joint_angles,
            (desired_pose, orientation_weight, position_weight),
            method=solver, bounds=bounds)

        if self.verbose >= 2:
            print("Error: %g" % res["fun"])
        if return_error:
            return res["x"], res["fun"]
        else:
            return res["x"]

    def inverse_with_random_restarts(
            self, desired_pose, n_restarts=10, tolerance=1e-3,
            random_state=np.random, solver="SLSQP",
            orientation_weight=1.0, position_weight=1.0):
        """Compute inverse kinematics with multiple random restarts.

        Parameters
        ----------
        desired_pose : array-like, shape (4, 4)
            Desired pose.

        n_restarts : int, optional (default: 10)
            Maximum number of allowed restarts.

        tolerance : float, optional (default: 1e-3)
            Required tolerance to abort.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        solver : str, optional (default: 'SLSQP')
            Optimizer to solve inverse kinematics problem. Possible options:
            'SLSQP', 'L-BFGS-B', and 'COBYLA'.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution
        """
        assert n_restarts >= 1
        Q = []
        errors = []
        for _ in range(n_restarts):
            q, error = self.inverse(
                desired_pose, self._sample_joints_uniform(random_state),
                return_error=True, solver=solver,
                orientation_weight=orientation_weight,
                position_weight=position_weight)
            Q.append(q)
            errors.append(error)
            if error <= tolerance:
                break
        if self.verbose:
            print(np.round(errors, 4))
        return Q[np.argmin(errors)]

    def local_inverse_with_random_restarts(
            self, desired_pose, joint_angles, interval, n_restarts=10,
            tolerance=1e-3, random_state=np.random, solver="SLSQP",
            orientation_weight=1.0, position_weight=1.0):
        """Compute inverse kinematics with multiple random restarts.

        Parameters
        ----------
        desired_pose : array-like, shape (4, 4)
            Desired pose.

        joint_angles : array-like, shape (n_joints,)
            Initial guess for joint angles.

        interval : float
            We will search for a solution within the range
            [joint_angles - interval, joint_angles + interval].

        n_restarts : int, optional (default: 10)
            Maximum number of allowed restarts.

        tolerance : float, optional (default: 1e-3)
            Required tolerance to abort.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        solver : str, optional (default: 'SLSQP')
            Optimizer to solve inverse kinematics problem. Possible options:
            'SLSQP', 'L-BFGS-B', and 'COBYLA'.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        joint_angles : array, shape (n_joints,)
            Solution
        """
        assert n_restarts >= 1
        Q = []
        errors = []
        bounds = np.empty((self.n_joints, 2))
        bounds[:, 0] = joint_angles - interval
        bounds[:, 1] = joint_angles + interval
        q = joint_angles  # start with previous state
        for _ in range(n_restarts):
            q, error = self.inverse(
                desired_pose, q, return_error=True, solver=solver,
                orientation_weight=orientation_weight,
                position_weight=position_weight)
            Q.append(q)
            errors.append(error)
            if error <= tolerance:
                break
            q = self._sample_joints_uniform(random_state, bounds=bounds)
        return Q[np.argmin(errors)]

    def _sample_joints_uniform(self, random_state, bounds=None):
        if bounds is None:
            bounds = self.joint_limits
        return (random_state.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
                + bounds[:, 0])

    def forward_trajectory(self, Q):
        """Compute forward kinematics for a trajectory.

        Parameters
        ----------
        Q : array-like, shape (n_steps, n_joints)
            Joint angles.

        Returns
        -------
        H : array, shape (n_steps, 4, 4)
            End-effector poses.
        """
        H = np.empty((len(Q), 4, 4))
        for t in range(len(Q)):
            H[t] = self.forward(Q[t])
        return H

    def inverse_trajectory(
            self, H, initial_joint_angles=None, interval=0.1 * math.pi,
            random_restarts=True, random_state=np.random, solver="SLSQP",
            orientation_weight=1.0, position_weight=1.0):
        """Compute inverse kinematics for a trajectory.

        Parameters
        ----------
        H : array-like, shape (n_steps, 4, 4)
            Desired end-effector poses.

        initial_joint_angles : array-like, shape (n_joints,), optional (default: None)
            Initial guess for joint angles.

        interval : float
            We will search for a solution within the range
            [joint_angles - interval, joint_angles + interval] in each step.

        random_restarts : bool, optional (default: True)
            Allow random restarts if no solution is found.

        random_state : np.random.RandomState, optional (default: np.random)
            Random state.

        solver : str, optional (default: 'SLSQP')
            Optimizer to solve inverse kinematics problem. Possible options:
            'SLSQP', 'L-BFGS-B', and 'COBYLA'.

        orientation_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the orientation error.

        position_weight : float, optional (default: 1.0)
            Should be between 0.0 and 1.0 and represent the weighting for
            minimizing the position error.

        Returns
        -------
        Q : array, shape (n_steps, n_joints)
            Solution
        """
        Q = np.empty((len(H), len(self.joint_names)), dtype=float)

        if initial_joint_angles is None:
            Q[0] = self.inverse_with_random_restarts(
                H[0], random_state=random_state,
                orientation_weight=orientation_weight,
                position_weight=position_weight)
        else:
            Q[0] = self.inverse(
                H[0], initial_joint_angles, solver=solver,
                orientation_weight=orientation_weight,
                position_weight=position_weight)

        for t in range(1, len(H)):
            if self.verbose >= 2:
                print("Step: %d" % (t + 1))
            if random_restarts:
                Q[t] = self.local_inverse_with_random_restarts(
                    H[t], Q[t - 1], interval, random_state=random_state,
                    orientation_weight=orientation_weight,
                    position_weight=position_weight)
            else:
                bounds = np.empty((self.n_joints, 2), dtype=float)
                bounds[:, 0] = Q[t - 1] - interval
                bounds[:, 1] = Q[t - 1] + interval
                Q[t] = self.inverse(
                    H[t], Q[t - 1], False, bounds, solver=solver,
                    orientation_weight=orientation_weight,
                    position_weight=position_weight)
        return Q


@numba.jit(nopython=True, cache=True)
def pose_dist(
        ee2base_desired, ee2base_actual, orientation_weight, position_weight):
    ee_actual2ee_desired = np.linalg.inv(ee2base_actual).dot(ee2base_desired)
    trace = (ee_actual2ee_desired[0, 0] + ee_actual2ee_desired[1, 1]
             + ee_actual2ee_desired[2, 2])
    angle = math.acos(min((trace - 1.0) / 2.0, 1.0))
    orientation_error = min(angle, 2.0 * math.pi - angle)
    position_error = math.sqrt(
        ee_actual2ee_desired[0, 3] ** 2 + ee_actual2ee_desired[1, 3] ** 2
        + ee_actual2ee_desired[2, 3] ** 2)
    return (orientation_weight * orientation_error
            + position_weight * position_error)

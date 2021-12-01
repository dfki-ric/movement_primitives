import math
import numpy as np
from scipy.interpolate import interp1d
import pytransform3d.rotations as pr
import pytransform3d.batch_rotations as pbr
import pytransform3d.transformations as pt


EPSILON = 1e-10


class CouplingTermObstacleAvoidance2D:  # for DMP
    """Coupling term for obstacle avoidance in 2D."""
    def __init__(self, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
        self.obstacle_position = obstacle_position
        self.gamma = gamma
        self.beta = beta

    def coupling(self, y, yd):
        cdd = obstacle_avoidance_acceleration_2d(
            y, yd, self.obstacle_position, self.gamma, self.beta)
        return np.zeros_like(cdd), cdd


def obstacle_avoidance_acceleration_2d(
        y, yd, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
    """Compute acceleration for obstacle avoidance in 2D.

    Parameters
    ----------
    y : array, shape (..., 2)
        Current position(s).

    yd : array, shape (..., 2)
        Current velocity / velocities.

    obstacle_position : array, shape (2,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 2)
        Accelerations.
    """
    obstacle_diff = obstacle_position - y
    pad_width = ([[0, 0]] * (y.ndim - 1)) + [[0, 1]]
    obstacle_diff_0 = np.pad(obstacle_diff, pad_width, mode="constant",
                             constant_values=0.0)
    yd_0 = np.pad(yd, pad_width, mode="constant", constant_values=0.0)
    r = 0.5 * np.pi * pbr.norm_vectors(np.cross(obstacle_diff_0, yd_0))
    R = pbr.matrices_from_compact_axis_angles(r)[..., :2, :2]
    theta_nom = np.einsum(
        "ni,ni->n", obstacle_diff.reshape(-1, 2), yd.reshape(-1, 2))
    shape = y.shape[:-1]
    if shape:
        theta_nom = theta_nom.reshape(*shape)
    theta_denom = (np.linalg.norm(obstacle_diff, axis=-1)
                   * np.linalg.norm(yd, axis=-1) + EPSILON)
    theta = np.arccos(theta_nom / theta_denom)
    rotated_velocity = np.einsum(
        "nij,nj->ni", R.reshape(-1, 2, 2), yd.reshape(-1, 2))
    if shape:
        rotated_velocity = rotated_velocity.reshape(*(shape + (2,)))
    cdd = (gamma * rotated_velocity
           * (theta * np.exp(-beta * theta))[..., np.newaxis])
    return np.squeeze(cdd)


class CouplingTermObstacleAvoidance3D:  # for DMP
    """Coupling term for obstacle avoidance in 3D."""
    def __init__(self, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
        self.obstacle_position = obstacle_position
        self.gamma = gamma
        self.beta = beta

    def coupling(self, y, yd):
        cdd = obstacle_avoidance_acceleration_3d(
            y, yd, self.obstacle_position, self.gamma, self.beta)
        return np.zeros_like(cdd), cdd


def obstacle_avoidance_acceleration_3d(
        y, yd, obstacle_position, gamma=1000.0, beta=20.0 / math.pi):
    """Compute acceleration for obstacle avoidance in 3D.

    Parameters
    ----------
    y : array, shape (..., 3)
        Current position(s).

    yd : array, shape (..., 3)
        Current velocity / velocities.

    obstacle_position : array, shape (3,)
        Position of the point obstacle.

    gamma : float, optional (default: 1000)
        Obstacle avoidance parameter.

    beta : float, optional (default: 20 / pi)
        Obstacle avoidance parameter.

    Returns
    -------
    cdd : array, shape (..., 3)
        Accelerations.
    """
    obstacle_diff = obstacle_position - y
    r = 0.5 * np.pi * pr.norm_vector(np.cross(obstacle_diff, yd))
    R = pr.matrix_from_compact_axis_angle(r)
    theta = np.arccos(
        np.dot(obstacle_diff, yd)
        / (np.linalg.norm(obstacle_diff) * np.linalg.norm(yd) + EPSILON))
    cdd = gamma * np.dot(R, yd) * theta * np.exp(-beta * theta)
    return cdd


class CouplingTermPos1DToPos1D:
    """Couples position components of a 2D DMP with a virtual spring.

    A. Gams, B. Nemec, L. Zlajpah, M. Wächter, T. Asfour, A. Ude:
    Modulation of Motor Primitives using Force Feedback: Interaction with
    the Environment and Bimanual Tasks (2013), IROS,
    https://h2t.anthropomatik.kit.edu/pdf/Gams2013.pdf

    Parameters
    ----------
    desired_distance : float
        Desired distance between components.

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.
    """
    def __init__(self, desired_distance, lf, k=1.0, c1=100.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        da = y[0] - y[1]
        F12 = self.k * (-self.desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = self.c2 * self.c1 * F12 * self.lf[0]
        C21dot = self.c2 * self.c1 * F21 * self.lf[1]
        return np.array([C12, C21]), np.array([C12dot, C21dot])


class CouplingTermPos3DToPos3D:  # for DMP
    """Couples position components of a 6D DMP with a virtual spring in 3D.

    A. Gams, B. Nemec, L. Zlajpah, M. Wächter, T. Asfour, A. Ude:
    Modulation of Motor Primitives using Force Feedback: Interaction with
    the Environment and Bimanual Tasks (2013), IROS,
    https://h2t.anthropomatik.kit.edu/pdf/Gams2013.pdf

    Parameters
    ----------
    desired_distance : array, shape (3,)
        Desired distance between components.

    lf : array-like, shape (2,)
        Binary values that indicate which DMP(s) will be adapted.
        The variable lf defines the relation leader-follower. If lf[0] = lf[1],
        then both robots will adapt their trajectories to follow average
        trajectories at the defined distance dd between them [..]. On the other
        hand, if lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory
        to match the trajectory of DMP0, again at the distance dd and again
        only after learning. Vice versa applies as well. Leader-follower
        relation can be determined by a higher-level planner [..].

    k : float, optional (default: 1)
        Virtual spring constant that couples the positions.

    c1 : float, optional (default: 100)
        Scaling factor for spring forces in the velocity component and
        acceleration component.

    c2 : float, optional (default: 30)
        Scaling factor for spring forces in the acceleration component.
    """
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        da = y[:3] - y[3:6]
        # Why do we take -self.desired_distance here? Because this allows us
        # to regard the desired distance as the displacement of DMP1 with
        # respect to DMP0.
        F12 = self.k * (-self.desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]
        return np.hstack([C12, C21]), np.hstack([C12dot, C21dot])


class CouplingTermDualCartesianDistance:  # for DualCartesianDMP
    """Couples distance between 3D positions of a dual Cartesian DMP."""
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        actual_distance = y[:3] - y[7:10]
        desired_distance = (np.abs(self.desired_distance) * actual_distance
                            / np.linalg.norm(actual_distance))
        F12 = self.k * (desired_distance - actual_distance)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]
        return (np.hstack([C12, np.zeros(3), C21, np.zeros(3)]),
                np.hstack([C12dot, np.zeros(3), C21dot, np.zeros(3)]))


class CouplingTermDualCartesianOrientation:  # for DualCartesianDMP
    """Couples orientations of dual Cartesian DMP."""
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y, yd=None):
        q1 = y[3:7]
        q2 = y[10:]
        actual_distance = pr.compact_axis_angle_from_quaternion(
            pr.concatenate_quaternions(q1, pr.q_conj(q2)))
        actual_distance_norm = np.linalg.norm(actual_distance)
        if actual_distance_norm < np.finfo("float").eps:
            desired_distance = (np.abs(self.desired_distance)
                                * np.array([0.0, 0.0, 1.0]))
        else:
            desired_distance = (np.abs(self.desired_distance) * actual_distance
                                / actual_distance_norm)
        F12 = self.k * (desired_distance - actual_distance)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]
        return (np.hstack([np.zeros(3), C12, np.zeros(3), C21]),
                np.hstack([np.zeros(3), C12dot, np.zeros(3), C21dot]))


class CouplingTermDualCartesianPose:  # for DualCartesianDMP
    """Couples relative poses of dual Cartesian DMP."""
    def __init__(self, desired_distance, lf, couple_position=True,
                 couple_orientation=True, k=1.0, c1=1.0, c2=30.0, verbose=0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.couple_position = couple_position
        self.couple_orientation = couple_orientation
        self.k = k
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose

    def coupling(self, y, yd=None):
        return self.couple_distance(
            y, yd, self.k, self.c1, self.c2, self.lf, self.desired_distance,
            self.couple_position, self.couple_orientation)

    def couple_distance(self, y, yd, k, c1, c2, lf, desired_distance,
                        couple_position, couple_orientation):
        damping = 2.0 * np.sqrt(k * c2)

        vel_left = yd[:6]
        vel_right = yd[6:]

        left2base = pt.transform_from_pq(y[:7])
        right2left_pq = self._right2left_pq(y)

        actual_distance_pos = right2left_pq[:3]
        actual_distance_rot = right2left_pq[3:]

        desired_distance = pt.pq_from_transform(desired_distance)
        desired_distance_pos = desired_distance[:3]
        desired_distance_rot = desired_distance[3:]

        if self.verbose:
            print("Desired vs. actual:")
            print(np.round(desired_distance, 2))
            print(np.round(right2left_pq, 2))

        error_pos = desired_distance_pos - actual_distance_pos
        F12_pos = -k * error_pos
        F21_pos = k * error_pos

        F12_pos = pt.transform(left2base, pt.vector_to_direction(F12_pos))[:3]
        F21_pos = pt.transform(left2base, pt.vector_to_direction(F21_pos))[:3]

        C12_pos = lf[0] * c1 * F12_pos
        C21_pos = lf[1] * c1 * F21_pos

        C12dot_pos = lf[0] * (c2 * F12_pos - damping * vel_left[:3])
        C21dot_pos = lf[1] * (c2 * F21_pos - damping * vel_right[:3])

        if not couple_position:
            C12_pos *= 0
            C21_pos *= 0
            C12dot_pos *= 0
            C21dot_pos *= 0

        error_rot = pr.compact_axis_angle_from_quaternion(
            pr.concatenate_quaternions(desired_distance_rot,
                                       pr.q_conj(actual_distance_rot)))
        F12_rot = -k * error_rot
        F21_rot = k * error_rot

        F12_rot = pt.transform(left2base, pt.vector_to_direction(F12_rot))[:3]
        F21_rot = pt.transform(left2base, pt.vector_to_direction(F21_rot))[:3]

        C12_rot = lf[0] * c1 * F12_rot
        C21_rot = lf[1] * c1 * F21_rot

        C12dot_rot = lf[0] * (c2 * F12_rot - damping * vel_left[3:])
        C21dot_rot = lf[1] * (c2 * F21_rot - damping * vel_right[3:])

        if not couple_orientation:
            C12_rot *= 0
            C21_rot *= 0
            C12dot_rot *= 0
            C21dot_rot *= 0

        return (np.hstack([C12_pos, C12_rot, C21_pos, C21_rot]),
                np.hstack([C12dot_pos, C12dot_rot, C21dot_pos, C21dot_rot]))

    def _right2left_pq(self, y):
        left2base = pt.transform_from_pq(y[:7])
        right2base = pt.transform_from_pq(y[7:])
        base2left = pt.invert_transform(left2base)
        right2left = pt.concat(right2base, base2left)
        right2left_pq = pt.pq_from_transform(right2left)
        return right2left_pq


class CouplingTermDualCartesianTrajectory(CouplingTermDualCartesianPose):  # for DualCartesianDMP
    """Couples relative pose in dual Cartesian DMP with a given trajectory."""
    def __init__(self, offset, lf, dt, couple_position=True,
                 couple_orientation=True, k=1.0, c1=1.0, c2=30.0, verbose=1):
        self.offset = offset
        self.lf = lf
        self.dt = dt
        self.couple_position = couple_position
        self.couple_orientation = couple_orientation
        self.k = k
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose

    def imitate(self, T, Y):
        distance = np.empty((len(Y), 7))
        for t in range(len(Y)):
            distance[t] = self._right2left_pq(Y[t])
        self.desired_distance_per_dimension = [
            interp1d(T, distance[:, d], bounds_error=False,
                     fill_value="extrapolate")
            for d in range(distance.shape[1])
        ]
        self.t = 0.0

    def coupling(self, y, yd=None):
        desired_distance = np.empty(len(self.desired_distance_per_dimension))
        for d in range(len(desired_distance)):
            desired_distance[d] = self.desired_distance_per_dimension[d](self.t)
        desired_distance += self.offset
        self.t += self.dt
        return self.couple_distance(
            y, yd, self.k, self.c1, self.c2, self.lf,
            pt.transform_from_pq(desired_distance), self.couple_position,
            self.couple_orientation)

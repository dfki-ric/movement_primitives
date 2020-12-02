import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from libc.math cimport sqrt, cos, sin, acos


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dmp_step(
        double last_t, double t, np.ndarray[double, ndim=1] current_y, np.ndarray[double, ndim=1] current_yd,
        np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd,
        np.ndarray[double, ndim=1] goal_ydd, np.ndarray[double, ndim=1] start_y,
        np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd,
        double goal_t, double start_t, double alpha_y, double beta_y,
        object forcing_term, object coupling_term=None,
        tuple coupling_term_precomputed=None,
        double int_dt=0.001, double k_tracking_error=0.0,
        np.ndarray tracking_error=None):

    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        current_y[:] = start_y
        current_yd[:] = start_yd

    cdef double execution_time = goal_t - start_t

    cdef int n_dims = current_y.shape[0]

    cdef np.ndarray[double, ndim=1] current_ydd = np.empty(n_dims, dtype=np.float64)

    cdef np.ndarray[double, ndim=1] cd = np.zeros(n_dims, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] cdd = np.zeros(n_dims, dtype=np.float64)

    cdef np.ndarray[double, ndim=1] f = np.empty(n_dims, dtype=np.float64)

    cdef int d
    cdef double current_t
    cdef double dt

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd[:], cdd[:] = coupling_term.coupling(current_y, current_yd)
        elif coupling_term_precomputed is not None:
            cd[:] = coupling_term_precomputed[0]
            cdd[:] = coupling_term_precomputed[1]
        if tracking_error is not None:
            cdd += k_tracking_error * tracking_error / dt

        f[:] = forcing_term(current_t).squeeze()

        for d in range(n_dims):
            current_ydd[d] = (alpha_y * (beta_y * (goal_y[d] - current_y[d]) + execution_time * goal_yd[d] - execution_time * current_yd[d]) + goal_ydd[d] * execution_time ** 2 + f[d] + cdd[d]) / execution_time ** 2
            current_yd[d] += dt * current_ydd[d] + cd[d] / execution_time
            current_y[d] += dt * current_yd[d]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dmp_step_quaternion(
        double last_t, double t,
        np.ndarray[double, ndim=1] current_y, np.ndarray[double, ndim=1] current_yd,
        np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd,
        np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd,
        double goal_t, double start_t, double alpha_y, double beta_y,
        forcing_term,
        coupling_term=None,
        coupling_term_precomputed=None,
        double int_dt=0.001):

    if t <= start_t:
        current_y[:] = start_y
        current_yd[:] = start_yd

    cdef double execution_time = goal_t - start_t

    cdef np.ndarray[double, ndim=1] current_ydd = np.empty(3, dtype=np.float64)

    cdef np.ndarray[double, ndim=1] cd = np.zeros(3, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] cdd = np.zeros(3, dtype=np.float64)

    cdef np.ndarray[double, ndim=1] f = np.empty(3, dtype=np.float64)

    cdef int d
    cdef double current_t
    cdef double dt

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd[:], cdd[:] = coupling_term.coupling(current_y, current_yd)
        elif coupling_term_precomputed is not None:
            cd[:] = coupling_term_precomputed[0]
            cdd[:] = coupling_term_precomputed[1]

        f[:] = forcing_term(current_t).squeeze()

        current_ydd[:] = (alpha_y * (beta_y * compact_axis_angle_from_quaternion(concatenate_quaternions(goal_y, q_conj(current_y))) - execution_time * current_yd) + f + cdd) / execution_time ** 2
        current_yd += dt * current_ydd + cd / execution_time
        current_y[:] = concatenate_quaternions(quaternion_from_compact_axis_angle(dt * current_yd), current_y)


PPS = [0, 1, 2, 7, 8, 9]
PVS = [0, 1, 2, 6, 7, 8]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dmp_dual_cartesian_step(
        double last_t, double t,
        np.ndarray[double, ndim=1] current_y, np.ndarray[double, ndim=1] current_yd,
        np.ndarray[double, ndim=1] goal_y, np.ndarray[double, ndim=1] goal_yd, np.ndarray[double, ndim=1] goal_ydd,
        np.ndarray[double, ndim=1] start_y, np.ndarray[double, ndim=1] start_yd, np.ndarray[double, ndim=1] start_ydd,
        double goal_t, double start_t, double alpha_y, double beta_y,
        forcing_term, coupling_term=None,
        double int_dt=0.001,
        double k_tracking_error=0.0, np.ndarray tracking_error=None):
    if t <= start_t:
        current_y[:] = start_y
        current_yd[:] = start_yd

    cdef double execution_time = goal_t - start_t

    cdef np.ndarray[double, ndim=1] current_ydd = np.empty_like(current_yd)

    cdef int n_vel_dims = current_yd.shape[0]

    cdef np.ndarray[double, ndim=1] cd = np.zeros(n_vel_dims, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] cdd = np.zeros(n_vel_dims, dtype=np.float64)

    cdef np.ndarray[double, ndim=1] f = np.empty(n_vel_dims, dtype=np.float64)

    cdef int pps
    cdef int pvs
    cdef np.ndarray[long, ndim=2] POS_INDICES = np.array([[0, 0], [1, 1], [2, 2], [7, 6], [8, 7], [9, 8]], dtype=long)

    cdef double dt
    cdef double current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd[:], cdd[:] = coupling_term.coupling(current_y, current_yd)

        f[:] = forcing_term(current_t).squeeze()
        # TODO handle tracking error of orientation correctly
        if tracking_error is not None:
            for pps, pvs in POS_INDICES:
                cdd[pvs] += k_tracking_error * tracking_error[pps] / dt

        # position components
        for pps, pvs in POS_INDICES:
            current_ydd[pvs] = (alpha_y * (beta_y * (goal_y[pps] - current_y[pps]) + execution_time * goal_yd[pvs] - execution_time * current_yd[pvs]) + goal_ydd[pvs] * execution_time ** 2 + f[pvs] + cdd[pvs]) / execution_time ** 2
            current_yd[pvs] += dt * current_ydd[pvs] + cd[pvs] / execution_time
            current_y[pps] += dt * current_yd[pvs]

        # TODO handle tracking error of orientation correctly
        # orientation components
        for ops, ovs in ((slice(3, 7), slice(3, 6)), (slice(10, 14), slice(9, 12))):
            current_ydd[ovs] = (alpha_y * (beta_y * compact_axis_angle_from_quaternion(concatenate_quaternions(goal_y[ops], q_conj(current_y[ops]))) - execution_time * current_yd[ovs]) + f[ovs] + cdd[ovs]) / execution_time ** 2
            current_yd[ovs] += dt * current_ydd[ovs] + cd[ovs] / execution_time
            current_y[ops] = concatenate_quaternions(quaternion_from_compact_axis_angle(dt * current_yd[ovs]), current_y[ops])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef concatenate_quaternions(np.ndarray[double, ndim=1] q1, np.ndarray[double, ndim=1] q2):
    """Concatenate two quaternions.
    We use Hamilton's quaternion multiplication.
    Parameters
    ----------
    q1 : array-like, shape (4,)
        First quaternion
    q2 : array-like, shape (4,)
        Second quaternion
    Returns
    -------
    q12 : array-like, shape (4,)
        Quaternion that represents the concatenated rotation q1 * q2
    """
    cdef np.ndarray[double, ndim=1] q12 = np.empty(4)
    q12[0] = q1[0] * q2[0]
    cdef int i
    for i in range(1, 4):
        q12[0] -= q1[i] * q2[i]
    q12[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return q12


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef quaternion_from_compact_axis_angle(np.ndarray[double, ndim=1] a):
    """Compute quaternion from compact axis-angle (exponential map).
    We usually assume active rotations.
    Parameters
    ----------
    a : array-like, shape (4,)
        Axis of rotation and rotation angle: angle * (x, y, z)
    Returns
    -------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    """
    cdef double angle = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    cdef np.ndarray[double, ndim=1] axis
    if angle == 0.0:
        axis = np.array([1.0, 0.0, 0.0])
    else:
        axis = a / angle

    cdef np.ndarray[double, ndim=1] q = np.empty(4)
    q[0] = cos(angle / 2.0)
    cdef double s = sin(angle / 2.0)
    q[1] = s * axis[0]
    q[2] = s * axis[1]
    q[3] = s * axis[2]
    return q


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef q_conj(np.ndarray[double, ndim=1] q):
    """Conjugate of quaternion.
    The conjugate of a unit quaternion inverts the rotation represented by
    this unit quaternion. The conjugate of a quaternion q is often denoted
    as q*.
    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    Returns
    -------
    q_c : array-like, shape (4,)
        Conjugate (w, -x, -y, -z)
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef compact_axis_angle_from_quaternion(np.ndarray[double, ndim=1] q):
    """Compute compact axis-angle from quaternion (logarithmic map).
    We usually assume active rotations.
    Parameters
    ----------
    q : array-like, shape (4,)
        Unit quaternion to represent rotation: (w, x, y, z)
    Returns
    -------
    a : array-like, shape (3,)
        Axis of rotation and rotation angle: angle * (x, y, z). The angle is
        constrained to [0, pi].
    """
    cdef double p_norm = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if p_norm < 1e-16:
        return np.zeros(3)
    return q[1:] / (p_norm / 2.0 / acos(q[0]))

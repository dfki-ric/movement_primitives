import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool
from pytransform3d import rotations as pr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef dmp_step(
        double last_t, double t, np.ndarray[double, ndim=1] last_y, np.ndarray[double, ndim=1] last_yd,
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
        return start_y.copy(), start_yd.copy(), start_ydd.copy()

    cdef double execution_time = goal_t - start_t

    cdef int n_dims = last_y.shape[0]

    cdef np.ndarray[double, ndim=1] y = last_y
    cdef np.ndarray[double, ndim=1] yd = last_yd
    cdef np.ndarray[double, ndim=1] ydd = np.empty(n_dims, dtype=np.float64)

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
            cd[:], cdd[:] = coupling_term.coupling(y, yd)
        elif coupling_term_precomputed is not None:
            cd[:] = coupling_term_precomputed[0]
            cdd[:] = coupling_term_precomputed[1]
        if tracking_error is not None:
            cdd += k_tracking_error * tracking_error / dt

        f[:] = forcing_term(current_t).squeeze()

        for d in range(n_dims):
            ydd[d] = (alpha_y * (beta_y * (goal_y[d] - y[d]) + execution_time * goal_yd[d] - execution_time * yd[d]) + goal_ydd[d] * execution_time ** 2 + f[d] + cdd[d]) / execution_time ** 2
            yd[d] += dt * ydd[d] + cd[d] / execution_time
            y[d] += dt * yd[d]
    return y, yd


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def dmp_step_quaternion(
        last_t, t,
        last_y, last_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term,
        coupling_term=None,
        coupling_term_precomputed=None,
        int_dt=0.001):

    if t <= start_t:
        return start_y.copy(), start_yd.copy(), start_ydd.copy()

    cdef double execution_time = goal_t - start_t

    cdef np.ndarray[double, ndim=1] y = last_y
    cdef np.ndarray[double, ndim=1] yd = last_yd
    cdef np.ndarray[double, ndim=1] ydd = np.empty(3, dtype=np.float64)

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
            cd[:], cdd[:] = coupling_term.coupling(y, yd)
        elif coupling_term_precomputed is not None:
            cd[:] = coupling_term_precomputed[0]
            cdd[:] = coupling_term_precomputed[1]

        f[:] = forcing_term(current_t).squeeze()

        ydd = (alpha_y * (beta_y * pr.compact_axis_angle_from_quaternion(pr.concatenate_quaternions(goal_y, pr.q_conj(y))) - execution_time * yd) + f + cdd) / execution_time ** 2
        yd += dt * ydd + cd / execution_time
        y = pr.concatenate_quaternions(pr.quaternion_from_compact_axis_angle(dt * yd), y)
    return y, yd
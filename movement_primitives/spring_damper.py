"""Spring-damper based attractors."""
import numpy as np
import pytransform3d.rotations as pr
from .base import PointToPointMovement


class SpringDamper(PointToPointMovement):
    """Spring-damper system.

    This is similar to a DMP without the forcing term.

    Parameters
    ----------
    n_dims : int
        State space dimensions.

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.

    k : float, optional (default: 1)
        Spring constant.

    c : float, optional (default: 2 * sqrt(k) (critical damping))
        Damping coefficient.

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.
    """
    def __init__(self, n_dims, dt=0.01, k=1.0, c=None, int_dt=0.001):
        super(SpringDamper, self).__init__(n_dims, n_dims)
        self.n_dims = n_dims
        self.dt = dt
        self.k = k
        self.c = c
        self.int_dt = int_dt

        self.initialized = False
        self.configure()

    def step(self, last_y, last_yd, coupling_term=None):
        self.last_t = self.t
        self.t += self.dt

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        self.current_y[:], self.current_yd[:] = last_y, last_yd
        spring_damper_step(
            self.last_t, self.t,
            self.current_y, self.current_yd,
            self.goal_y,
            self.k, self.c,
            coupling_term=coupling_term,
            int_dt=self.int_dt)
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=1.0, coupling_term=None):
        return spring_damper_open_loop(
            self.dt,
            self.start_y, self.goal_y,
            self.k, self.c,
            coupling_term,
            run_t, self.int_dt)


class SpringDamperOrientation(PointToPointMovement):
    """Spring-damper system for quaternions.

    This is similar to a Quaternion DMP without the forcing term.

    Parameters
    ----------
    dt : float, optional (default: 0.01)
        Time difference between DMP steps.

    k : float, optional (default: 1)
        Spring constant.

    c : float, optional (default: 2 * sqrt(k) (critical damping))
        Damping coefficient.

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.
    """
    def __init__(self, dt=0.01, k=1.0, c=None, int_dt=0.001):
        super(SpringDamperOrientation, self).__init__(4, 3)

        self.dt = dt
        self.k = k
        self.c = c
        self.int_dt = int_dt

        self.initialized = False
        self.configure()

    def step(self, last_y, last_yd, coupling_term=None):
        self.last_t = self.t
        self.t += self.dt

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        self.current_y[:], self.current_yd[:] = last_y, last_yd
        spring_damper_step_quaternion(
            self.last_t, self.t,
            self.current_y, self.current_yd,
            self.goal_y,
            self.k, self.c,
            coupling_term=coupling_term,
            int_dt=self.int_dt)
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=1.0, coupling_term=None):
        return spring_damper_open_loop_quaternion(
            self.dt,
            self.start_y, self.goal_y,
            self.k, self.c,
            coupling_term,
            run_t, self.int_dt)


def spring_damper_step(
        last_t, t, current_y, current_yd, goal_y, k=1.0, c=None,
        coupling_term=None, coupling_term_precomputed=None, int_dt=0.001):
    if c is None:  # set for critical damping
        c = 2.0 * np.sqrt(k)

    current_ydd = np.empty_like(current_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(current_y)
        else:
            cd, cdd = np.zeros_like(current_y), np.zeros_like(current_y)
        if coupling_term_precomputed is not None:
            cd += coupling_term_precomputed[0]
            cdd += coupling_term_precomputed[1]

        current_ydd[:] = k * (goal_y - current_y) - c * current_yd
        current_yd += dt * current_ydd + cd
        current_y += dt * current_yd


def spring_damper_step_quaternion(
        last_t, t, current_y, current_yd, goal_y, k=1.0, c=None,
        coupling_term=None, coupling_term_precomputed=None, int_dt=0.001):
    if c is None:  # set for critical damping
        c = 2.0 * np.sqrt(k)

    current_ydd = np.empty_like(current_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(current_y)
        else:
            cd, cdd = np.zeros(3), np.zeros(3)
        if coupling_term_precomputed is not None:
            cd += coupling_term_precomputed[0]
            cdd += coupling_term_precomputed[1]

        current_ydd[:] = (
            k * pr.compact_axis_angle_from_quaternion(
                pr.concatenate_quaternions(goal_y, pr.q_conj(current_y)))
            - c * current_yd)
        current_yd += dt * current_ydd + cd
        current_y[:] = pr.concatenate_quaternions(
            pr.quaternion_from_compact_axis_angle(dt * current_yd), current_y)


def spring_damper_open_loop(
        dt, start_y, goal_y, k=1.0, c=None, coupling_term=None, run_t=1.0,
        int_dt=0.001):
    t = 0.0
    y = np.copy(start_y)
    yd = np.zeros_like(y)
    T = [t]
    Y = [np.copy(y)]
    while t < run_t:
        last_t = t
        t += dt
        spring_damper_step(
            last_t, t, y, yd,
            goal_y=goal_y,
            k=k, c=c, coupling_term=coupling_term, int_dt=int_dt)
        T.append(t)
        Y.append(np.copy(y))
    return np.asarray(T), np.asarray(Y)


def spring_damper_open_loop_quaternion(
        dt, start_y, goal_y, k=1.0, c=None, coupling_term=None, run_t=1.0,
        int_dt=0.001):
    t = 0.0
    y = np.copy(start_y)
    yd = np.zeros(3)
    T = [t]
    Y = [np.copy(y)]
    while t < run_t:
        last_t = t
        t += dt
        spring_damper_step_quaternion(
            last_t, t, y, yd, goal_y=goal_y, k=k, c=c,
            coupling_term=coupling_term, int_dt=int_dt)
        T.append(t)
        Y.append(np.copy(y))
    return np.asarray(T), np.asarray(Y)

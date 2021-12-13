import math
import numpy as np
from ._base import DMPBase
from ._canonical_system import canonical_system_alpha, phase


class StateFollowingDMP(DMPBase):
    """State following DMP (highly experimental).

    The DMP variant that is implemented here is described in

    J. Vidakovic, B. Jerbic, B. Sekoranja, M. Svaco, F. Suligoj:
    Task Dependent Trajectory Learning from Multiple Demonstrations Using
    Movement Primitives (2019),
    International Conference on Robotics in Alpe-Adria Danube Region (RAAD),
    pp. 275-282, doi: 10.1007/978-3-030-19648-6_32,
    https://link.springer.com/chapter/10.1007/978-3-030-19648-6_32

    Attributes
    ----------
    dt_ : float
        Time difference between DMP steps. This value can be changed to adapt
        the frequency.
    """
    def __init__(self, n_dims, execution_time, dt=0.01, n_viapoints=10,
                 int_dt=0.001):
        super(StateFollowingDMP, self).__init__(n_dims, n_dims)
        self.execution_time = execution_time
        self.dt_ = dt
        self.n_viapoints = n_viapoints
        self.int_dt = int_dt

        alpha_z = canonical_system_alpha(
            0.01, self.execution_time, 0.0, self.int_dt)

        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0

        self.forcing_term = StateFollowingForcingTerm(
            self.n_dims, self.n_viapoints, self.execution_time, 0.0, 0.1,
            alpha_z)

    def step(self, last_y, last_yd, coupling_term=None):
        assert len(last_y) == self.n_dims
        assert len(last_yd) == self.n_dims

        self.last_t = self.t
        self.t += self.dt_

        self.current_y[:], self.current_yd[:] = last_y, last_yd
        state_following_dmp_step(
            self.last_t, self.t,
            self.current_y, self.current_yd,
            self.goal_y, self.goal_yd, self.goal_ydd,
            self.start_y, self.start_yd, self.start_ydd,
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            forcing_term=self.forcing_term,
            coupling_term=coupling_term,
            int_dt=self.int_dt)
        return self.current_y, self.current_yd

    def open_loop(self, run_t=None, coupling_term=None):
        return state_following_dmp_open_loop(
            self.execution_time, 0.0, self.dt_, self.start_y, self.goal_y,
            self.alpha_y, self.beta_y, self.forcing_term, coupling_term, run_t,
            self.int_dt)

    def imitate(self, T, Y, regularization_coefficient=0.0,
                allow_final_velocity=False):
        raise NotImplementedError("imitation is not yet implemented")


class StateFollowingForcingTerm:
    """Defines the shape of the motion."""
    def __init__(self, n_dims, n_viapoints, goal_t, start_t, overlap, alpha_z):
        if n_viapoints <= 0:
            raise ValueError("The number of viapoints must be > 0!")
        self.n_viapoints = n_viapoints
        if start_t >= goal_t:
            raise ValueError("Goal must be chronologically after start!")
        self.goal_t = goal_t
        self.start_t = start_t
        self.overlap = overlap
        self.alpha_z = alpha_z

        self.viapoints = np.zeros((n_viapoints, n_dims))

        self._init_rbfs(n_viapoints, start_t)

    def _init_rbfs(self, n_viapoints, start_t):
        self.log_overlap = float(-math.log(self.overlap))
        self.execution_time = self.goal_t - self.start_t
        self.centers = np.empty(n_viapoints)
        self.widths = np.empty(n_viapoints)
        step = self.execution_time / self.n_viapoints
        # do first iteration outside loop because we need access to i and i - 1
        # in loop
        t = start_t
        self.centers[0] = phase(t, self.alpha_z, self.goal_t, self.start_t)
        for i in range(1, self.n_viapoints):
            # normally lower_border + i * step but lower_border is 0
            t = i * step
            self.centers[i] = phase(t, self.alpha_z, self.goal_t, self.start_t)
            # Choose width of RBF basis functions automatically so that the
            # RBF centered at one center has value overlap at the next center
            diff = self.centers[i] - self.centers[i - 1]
            self.widths[i - 1] = self.log_overlap / diff ** 2
        # Width of last Gaussian cannot be calculated, just use the same width
        # as the one before
        self.widths[self.n_viapoints - 1] = self.widths[self.n_viapoints - 2]

    def _activations(self, z, normalized):
        z = np.atleast_2d(z)  # 1 x n_steps
        squared_dist = (z - self.centers[:, np.newaxis]) ** 2
        activations = np.exp(-self.widths[:, np.newaxis] * squared_dist)
        if normalized:
            activations /= activations.sum(axis=0)
        return activations

    def __call__(self, t, int_dt=0.001):
        z = phase(t, alpha=self.alpha_z, goal_t=self.goal_t,
                  start_t=self.start_t, int_dt=int_dt)
        z = np.atleast_1d(z)
        return self._activations(z, normalized=True).T


def state_following_dmp_step(
        last_t, t, current_y, current_yd, goal_y, goal_yd, goal_ydd, start_y,
        start_yd, start_ydd, goal_t, start_t, alpha_y, beta_y, forcing_term,
        coupling_term=None, coupling_term_precomputed=None, int_dt=0.001):
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        return np.copy(start_y), np.copy(start_yd), np.copy(start_ydd)

    execution_time = goal_t - start_t

    current_ydd = np.empty_like(current_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(current_y, current_yd)
        else:
            cd, cdd = np.zeros_like(current_y), np.zeros_like(current_y)
        if coupling_term_precomputed is not None:
            cd += coupling_term_precomputed[0]
            cdd += coupling_term_precomputed[1]

        h = forcing_term(current_t).squeeze(axis=0)

        current_ydd[:] = np.sum(h[:, np.newaxis] * alpha_y * (
            beta_y * (forcing_term.viapoints - current_y)
            - 0.5 * execution_time * current_yd[np.newaxis])
                / (0.5 * execution_time) ** 2, axis=0)
        current_ydd += cdd / (0.5 * execution_time) ** 2
        current_yd += dt * current_ydd + cd / (0.5 * execution_time)
        current_y += dt * current_yd


def state_following_dmp_open_loop(
        goal_t, start_t, dt, start_y, goal_y, alpha_y, beta_y, forcing_term,
        coupling_term=None, run_t=None, int_dt=0.001):
    t = start_t
    y = np.copy(start_y)
    yd = np.zeros_like(y)
    T = [start_t]
    Y = [np.copy(y)]
    if run_t is None:
        run_t = goal_t
    while t < run_t:
        last_t = t
        t += dt
        state_following_dmp_step(
            last_t, t, y, yd,
            goal_y=goal_y, goal_yd=np.zeros_like(goal_y),
            goal_ydd=np.zeros_like(goal_y),
            start_y=start_y, start_yd=np.zeros_like(start_y),
            start_ydd=np.zeros_like(start_y),
            goal_t=goal_t, start_t=start_t, alpha_y=alpha_y, beta_y=beta_y,
            forcing_term=forcing_term, coupling_term=coupling_term,
            int_dt=int_dt)
        T.append(t)
        Y.append(np.copy(y))
    return np.asarray(T), np.asarray(Y)

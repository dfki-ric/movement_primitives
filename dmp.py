import numpy as np
from pytransform3d import rotations as pr


def canonical_system_alpha(goal_z, goal_t, start_t, int_dt=0.001):
    if goal_z <= 0.0:
        raise ValueError("Final phase must be > 0!")
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    execution_time = goal_t - start_t
    n_phases = int(execution_time / int_dt) + 1
    # assert that the execution_time is approximately divisible by int_dt
    assert abs(((n_phases - 1) * int_dt) - execution_time) < 0.05
    return (1.0 - goal_z ** (1.0 / (n_phases - 1))) * (n_phases - 1)


def phase(t, alpha, goal_t, start_t, int_dt=0.001, eps=1e-10):
    execution_time = goal_t - start_t
    b = max(1.0 - alpha * int_dt / execution_time, eps)
    return b ** ((t - start_t) / int_dt)


class ForcingTerm:
    def __init__(self, n_dims, n_weights_per_dim, goal_t, start_t, overlap, alpha_z):
        if n_weights_per_dim <= 0:
            raise ValueError("The number of weights per dimension must be > 1!")
        self.n_weights_per_dim = n_weights_per_dim
        if start_t >= goal_t:
            raise ValueError("Goal must be chronologically after start!")
        self.goal_t = goal_t
        self.start_t = start_t
        self.overlap = overlap
        self.alpha_z = alpha_z

        self._init_rbfs(n_dims, n_weights_per_dim, start_t)

    def _init_rbfs(self, n_dims, n_weights_per_dim, start_t):
        self.log_overlap = -np.log(self.overlap)
        self.execution_time = self.goal_t - self.start_t
        self.weights = np.zeros((n_dims, n_weights_per_dim))
        self.centers = np.empty(n_weights_per_dim)
        self.widths = np.empty(n_weights_per_dim)
        step = self.execution_time / (
                    self.n_weights_per_dim - 1)  # -1 because we want the last entry to be execution_time
        # do first iteration outside loop because we need access to i and i - 1 in loop
        t = start_t
        self.centers[0] = phase(t, self.alpha_z, self.goal_t, self.start_t)
        for i in range(1, self.n_weights_per_dim):
            t = i * step  # normally lower_border + i * step but lower_border is 0
            self.centers[i] = phase(t, self.alpha_z, self.goal_t, self.start_t)
            # Choose width of RBF basis functions automatically so that the
            # RBF centered at one center has value overlap at the next center
            diff = self.centers[i] - self.centers[i - 1]
            self.widths[i - 1] = self.log_overlap / diff ** 2
        # Width of last Gaussian cannot be calculated, just use the same width as the one before
        self.widths[self.n_weights_per_dim - 1] = self.widths[self.n_weights_per_dim - 2]

    def _activations(self, z, normalized):
        z = np.atleast_2d(z)  # 1 x n_steps
        squared_dist = (z - self.centers[:, np.newaxis]) ** 2
        activations = np.exp(-self.widths[:, np.newaxis] * squared_dist)
        if normalized:
            activations /= activations.sum(axis=0)
        return activations

    def design_matrix(self, T, int_dt=0.001):  # returns: n_weights_per_dim x n_steps
        Z = phase(T, alpha=self.alpha_z, goal_t=T[-1], start_t=T[0], int_dt=int_dt)
        return Z[np.newaxis, :] * self._activations(Z, normalized=True)

    def __call__(self, t, int_dt=0.001):
        z = phase(t, alpha=self.alpha_z, goal_t=self.goal_t, start_t=self.start_t, int_dt=int_dt)
        z = np.atleast_1d(z)
        activations = self._activations(z, normalized=True)
        return z[np.newaxis, :] * self.weights.dot(activations)


class DMP:
    def __init__(self, n_dims, execution_time, dt=0.01, n_weights_per_dim=10, int_dt=0.001, k_tracking_error=0.0):
        self.n_dims = n_dims
        self.execution_time = execution_time
        self.dt = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt
        self.k_tracking_error = k_tracking_error

        alpha_z = canonical_system_alpha(0.01, self.execution_time, 0.0, self.int_dt)
        self.forcing_term = ForcingTerm(self.n_dims, self.n_weights_per_dim, self.execution_time, 0.0, 0.8, alpha_z)

        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0

        self.last_t = None
        self.t = 0.0
        self.start_y = np.zeros(self.n_dims)
        self.start_yd = np.zeros(self.n_dims)
        self.start_ydd = np.zeros(self.n_dims)
        self.goal_y = np.zeros(self.n_dims)
        self.goal_yd = np.zeros(self.n_dims)
        self.goal_ydd = np.zeros(self.n_dims)
        self.initialized = False
        self.current_y = np.zeros(self.n_dims)
        self.current_yd = np.zeros(self.n_dims)
        self.configure()

    def configure(self, last_t=None, t=None, start_y=None, start_yd=None, start_ydd=None, goal_y=None, goal_yd=None, goal_ydd=None):
        if last_t is not None:
            self.last_t = last_t
        if t is not None:
            self.t = t
        if start_y is not None:
            self.start_y = start_y
        if start_yd is not None:
            self.start_yd = start_yd
        if start_ydd is not None:
            self.start_ydd = start_ydd
        if goal_y is not None:
            self.goal_y = goal_y
        if goal_yd is not None:
            self.goal_yd = goal_yd
        if goal_ydd is not None:
            self.goal_ydd = goal_ydd

    def step(self, last_y, last_yd, coupling_term=None):
        self.last_t = self.t
        self.t += self.dt

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        # https://github.com/studywolf/pydmps/blob/master/pydmps/cs.py
        tracking_error = self.current_y - last_y

        self.current_y[:], self.current_yd[:] = dmp_step(
            self.last_t, self.t,
            last_y, last_yd,
            self.goal_y, self.goal_yd, self.goal_ydd,
            self.start_y, self.start_yd, self.start_ydd,
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term,
            coupling_term,
            self.int_dt,
            k_tracking_error=self.k_tracking_error,
            tracking_error=tracking_error)
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None):
        return dmp_open_loop(
            self.execution_time, 0.0, self.dt,
            self.start_y, self.goal_y,
            self.alpha_y, self.beta_y,
            self.forcing_term,
            coupling_term,
            run_t, self.int_dt)

    def imitate(self, T, Y, regularization_coefficient=0.0, allow_final_velocity=False):
        self.forcing_term.weights[:, :] = dmp_imitate(
            T, Y,
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y, overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z, allow_final_velocity=allow_final_velocity)

class CartesianDMP:
    def __init__(self, execution_time, dt=0.01,
                 n_weights_per_dim=10, int_dt=0.001):
        self.n_dims = 7
        self.execution_time = execution_time
        self.dt = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt
        alpha_z = canonical_system_alpha(
            0.01, self.execution_time, 0.0, self.int_dt)
        self.forcing_term_pos = ForcingTerm(
            3, self.n_weights_per_dim, self.execution_time, 0.0, 0.8,
            alpha_z)
        self.forcing_term_rot = ForcingTerm(
            3, self.n_weights_per_dim, self.execution_time, 0.0, 0.8,
            alpha_z)

        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0

        self.last_t = None
        self.t = 0.0
        self.start_y = np.zeros(self.n_dims)
        self.start_yd = np.zeros(6)
        self.start_ydd = np.zeros(6)
        self.goal_y = np.zeros(self.n_dims)
        self.goal_yd = np.zeros(6)
        self.goal_ydd = np.zeros(6)
        self.configure()

    def configure(self, last_t=None, t=None, start_y=None, start_yd=None, start_ydd=None, goal_y=None, goal_yd=None, goal_ydd=None):
        if last_t is not None:
            self.last_t = last_t
        if t is not None:
            self.t = t
        if start_y is not None:
            self.start_y = start_y
        if start_yd is not None:
            self.start_yd = start_yd
        if start_ydd is not None:
            self.start_ydd = start_ydd
        if goal_y is not None:
            self.goal_y = goal_y
        if goal_yd is not None:
            self.goal_yd = goal_yd
        if goal_ydd is not None:
            self.goal_ydd = goal_ydd

    def step(self, last_y, last_yd, coupling_term=None):
        assert len(last_y) == 7
        assert len(last_yd) == 6

        self.last_t = self.t
        self.t += self.dt

        current_y = np.empty(self.n_dims)
        current_yd = np.empty(6)

        current_y[:3], current_yd[:3] = dmp_step(
            self.last_t, self.t,
            last_y[:3], last_yd[:3],
            self.goal_y[:3], self.goal_yd[:3], self.goal_ydd[:3],
            self.start_y[:3], self.start_yd[:3], self.start_ydd[:3],
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term_pos,
            coupling_term,
            self.int_dt)
        current_y[3:], current_yd[3:] = dmp_step_quaternion(
            self.last_t, self.t,
            last_y[3:], last_yd[3:],
            self.goal_y[3:], self.goal_yd[3:], self.goal_ydd[3:],
            self.start_y[3:], self.start_yd[3:], self.start_ydd[3:],
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term_rot,
            coupling_term,
            self.int_dt)
        return current_y, current_yd

    def open_loop(self, run_t=None, coupling_term=None):
        T, Yp = dmp_open_loop(
                self.execution_time, 0.0, self.dt,
                self.start_y[:3], self.goal_y[:3],
                self.alpha_y, self.beta_y,
                self.forcing_term_pos,
                coupling_term,
                run_t, self.int_dt)
        _, Yr = dmp_open_loop_quaternion(
                self.execution_time, 0.0, self.dt,
                self.start_y[3:], self.goal_y[3:],
                self.alpha_y, self.beta_y,
                self.forcing_term_rot,
                coupling_term,
                run_t, self.int_dt)
        # TODO open loop orientation dmp
        return (T, np.hstack((Yp, Yr)))

    def imitate(self, T, Y, regularization_coefficient=0.0,
                allow_final_velocity=False):
        self.forcing_term_pos.weights[:, :] = dmp_imitate(
            T, Y[:, :3],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y, overlap=self.forcing_term_pos.overlap,
            alpha_z=self.forcing_term_pos.alpha_z, allow_final_velocity=allow_final_velocity)
        self.forcing_term_rot.weights[:, :] = dmp_quaternion_imitation(
            T, Y[:, 3:],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y, overlap=self.forcing_term_rot.overlap,
            alpha_z=self.forcing_term_rot.alpha_z, allow_final_velocity=allow_final_velocity)


class DualCartesianDMP:  # TODO implement
    pass


# lf - Binary values that indicate which DMP(s) will be adapted.
# The variable lf defines the relation leader-follower. If lf[0] = lf[1],
# then both robots will adapt their trajectories to follow average trajectories
# at the defined distance dd between them [..]. On the other hand, if
# lf[0] = 0 and lf[1] = 1, only DMP1 will change the trajectory to match the
# trajectory of DMP0, again at the distance dd and again only after learning.
# Vice versa applies as well. Leader-follower relation can be determined by a
# higher-level planner [..].

class CouplingTerm:
    def __init__(self, desired_distance, lf, k=1.0, c1=100.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y):
        da = y[1] - y[0]
        F12 = self.k * (self.desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = self.c2 * self.c1 * F12 * self.lf[0]
        C21dot = self.c2 * self.c1 * F21 * self.lf[1]
        return np.array([C12, C21]), np.array([C12dot, C21dot])


class CouplingTermCartesianPosition:
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y):
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


class CouplingTermCartesianDistance:
    def __init__(self, desired_distance, lf, k=1.0, c1=1.0, c2=30.0):
        self.desired_distance = desired_distance
        self.lf = lf
        self.k = k
        self.c1 = c1
        self.c2 = c2

    def coupling(self, y):
        da = y[:3] - y[3:6]
        desired_distance = np.abs(self.desired_distance) * da / np.linalg.norm(da)
        F12 = self.k * (desired_distance - da)
        F21 = -F12
        C12 = self.c1 * F12 * self.lf[0]
        C21 = self.c1 * F21 * self.lf[1]
        C12dot = F12 * self.c2 * self.lf[0]
        C21dot = F21 * self.c2 * self.lf[1]
        return np.hstack([C12, C21]), np.hstack([C12dot, C21dot])


class CouplingTermCartesianPose:  # TODO implement
    pass


def dmp_step(last_t, t, last_y, last_yd, goal_y, goal_yd, goal_ydd, start_y, start_yd, start_ydd, goal_t, start_t, alpha_y, beta_y, forcing_term, coupling_term=None, int_dt=0.001, k_tracking_error=0.0, tracking_error=0.0):
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        return np.copy(start_y), np.copy(start_yd), np.copy(start_ydd)

    execution_time = goal_t - start_t

    y = np.copy(last_y)
    yd = np.copy(last_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(y)
        else:
            cd, cdd = np.zeros_like(y), np.zeros_like(y)

        f = forcing_term(current_t).squeeze()

        """
        # Schaal
        ydd = (alpha_y * (beta_y * (goal_y - last_y) + execution_time * goal_yd - execution_time * last_yd) + goal_ydd * execution_time ** 2 + f + cdd) / execution_time ** 2
        # Pastor
        #K, D = 100, 20
        #z = phase(t, alpha=forcing_term.alpha_z, goal_t=forcing_term.goal_t, start_t=forcing_term.start_t, int_dt=int_dt)
        #ydd = (K * (goal_y - last_y) - D * execution_time * last_yd - K * (goal_y - start_y) * z + K * f + cdd) / execution_time ** 2
        y += dt * yd
        yd += dt * ydd + cd / execution_time
        """
        coupling_sum = cdd + k_tracking_error * tracking_error / dt
        ydd = (alpha_y * (beta_y * (goal_y - last_y) + execution_time * goal_yd - execution_time * last_yd) + goal_ydd * execution_time ** 2 + f + coupling_sum) / execution_time ** 2
        yd += dt * ydd + cd / execution_time
        y += dt * yd
    return y, yd


def dmp_step_quaternion(
        last_t, t,
        last_y, last_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term,
        coupling_term=None,
        int_dt=0.001):
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        return np.copy(start_y), np.copy(start_yd), np.copy(start_ydd)

    execution_time = goal_t - start_t

    y = np.copy(last_y)
    yd = np.copy(last_yd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd, cdd = coupling_term.coupling(y)
        else:
            cd, cdd = np.zeros(3), np.zeros(3)

        f = forcing_term(current_t).squeeze()

        # TODO why factor 2???
        ydd = (alpha_y * (beta_y * pr.quaternion_log(pr.concatenate_quaternions(goal_y, pr.q_conj(last_y))) - execution_time * last_yd) + f + cdd) / execution_time ** 2
        yd += dt * ydd + cd / execution_time
        y = pr.concatenate_quaternions(pr.angular_velocity_exp(dt * yd), y)
    return y, yd
    # https://github.com/rock-learning/bolero/blob/master/src/representation/dmp/implementation/src/Dmp.cpp#L754


def dmp_imitate(T, Y, n_weights_per_dim, regularization_coefficient, alpha_y, beta_y, overlap, alpha_z, allow_final_velocity):
    if regularization_coefficient < 0.0:
        raise ValueError("Regularization coefficient must be >= 0!")

    forcing_term = ForcingTerm(Y.shape[1], n_weights_per_dim, T[-1], T[0], overlap, alpha_z)
    F = determine_forces(T, Y, alpha_y, beta_y, allow_final_velocity)  # n_steps x n_dims

    X = forcing_term.design_matrix(T)  # n_weights_per_dim x n_steps

    return ridge_regression(X, F, regularization_coefficient)


def determine_forces(T, Y, alpha_y, beta_y, allow_final_velocity):  # returns: n_steps x n_dims
    n_dims = Y.shape[1]
    DT = np.gradient(T)
    Yd = np.empty_like(Y)
    for d in range(n_dims):
        Yd[:, d] = np.gradient(Y[:, d]) / DT
    if not allow_final_velocity:
        Yd[-1, :] = 0.0
    Ydd = np.empty_like(Y)
    for d in range(n_dims):
        Ydd[:, d] = np.gradient(Yd[:, d]) / DT
    Ydd[-1, :] = 0.0

    execution_time = T[-1] - T[0]
    goal_y = Y[-1]
    goal_yd = Yd[-1]
    goal_ydd = Ydd[-1]
    F = np.empty((len(T), n_dims))
    for t in range(len(T)):
        F[t, :] = execution_time ** 2 * Ydd[t] - alpha_y * (beta_y * (goal_y - Y[t]) + goal_yd * execution_time - Yd[t] * execution_time) - execution_time ** 2 * goal_ydd
    return F


def dmp_quaternion_imitation(T, Y, n_weights_per_dim, regularization_coefficient, alpha_y, beta_y, overlap, alpha_z, allow_final_velocity):
    # https://github.com/rock-learning/bolero/blob/master/src/representation/dmp/implementation/src/Dmp.cpp#L702
    if regularization_coefficient < 0.0:
        raise ValueError("Regularization coefficient must be >= 0!")

    forcing_term = ForcingTerm(3, n_weights_per_dim, T[-1], T[0], overlap, alpha_z)
    F = determine_forces_quaternion(T, Y, alpha_y, beta_y, allow_final_velocity)  # n_steps x n_dims

    X = forcing_term.design_matrix(T)  # n_weights_per_dim x n_steps

    return ridge_regression(X, F, regularization_coefficient)


def determine_forces_quaternion(T, Y, alpha_y, beta_y, allow_final_velocity):
    # https://github.com/rock-learning/bolero/blob/master/src/representation/dmp/implementation/src/Dmp.cpp#L670
    n_dims = 3
    DT = np.gradient(T)
    Yd = pr.quaternion_gradient(Y) / DT[:, np.newaxis]
    if not allow_final_velocity:
        Yd[-1, :] = 0.0
    Ydd = np.empty_like(Yd)
    for d in range(n_dims):
        Ydd[:, d] = np.gradient(Yd[:, d]) / DT
    Ydd[-1, :] = 0.0

    execution_time = T[-1] - T[0]
    goal_y = Y[-1]
    F = np.empty((len(T), n_dims))
    for t in range(len(T)):
        F[t, :] = execution_time ** 2 * Ydd[t] - alpha_y * (beta_y * pr.quaternion_log(pr.concatenate_quaternions(goal_y, pr.q_conj(Y[t]))) - Yd[t] * execution_time)
    return F


def ridge_regression(X, F, regularization_coefficient):  # returns: n_dims x n_weights_per_dim
    return np.linalg.pinv(X.dot(X.T) + regularization_coefficient * np.eye(X.shape[0])).dot(X).dot(F).T


def dmp_open_loop(goal_t, start_t, dt, start_y, goal_y, alpha_y, beta_y, forcing_term, coupling_term=None, run_t=None, int_dt=0.001):
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
        y, yd = dmp_step(
            last_t, t, y, yd,
            goal_y=goal_y, goal_yd=np.zeros_like(goal_y), goal_ydd=np.zeros_like(goal_y),
            start_y=start_y, start_yd=np.zeros_like(start_y), start_ydd=np.zeros_like(start_y),
            goal_t=goal_t, start_t=start_t,
            alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term, coupling_term=coupling_term, int_dt=int_dt)
        T.append(t)
        Y.append(np.copy(y))
    return np.asarray(T), np.asarray(Y)


def dmp_open_loop_quaternion(goal_t, start_t, dt, start_y, goal_y, alpha_y, beta_y, forcing_term, coupling_term=None, run_t=None, int_dt=0.001):
    t = start_t
    y = np.copy(start_y)
    yd = np.zeros(3)
    T = [start_t]
    Y = [np.copy(y)]
    if run_t is None:
        run_t = goal_t
    while t < run_t:
        last_t = t
        t += dt
        y, yd = dmp_step_quaternion(
            last_t, t, y, yd,
            goal_y=goal_y, goal_yd=np.zeros_like(yd), goal_ydd=np.zeros_like(yd),
            start_y=start_y, start_yd=np.zeros_like(yd), start_ydd=np.zeros_like(yd),
            goal_t=goal_t, start_t=start_t,
            alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term, coupling_term=coupling_term, int_dt=int_dt)
        T.append(t)
        Y.append(np.copy(y))
    return np.asarray(T), np.asarray(Y)

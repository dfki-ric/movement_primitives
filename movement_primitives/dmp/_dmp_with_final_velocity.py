import numpy as np
from ._base import DMPBase
from ._forcing_term import ForcingTerm
from ._canonical_system import canonical_system_alpha
from ._dmp import dmp_imitate, dmp_open_loop


class DMPWithFinalVelocity(DMPBase):
    """Dynamical movement primitive (DMP) with final velocity.

    Implementation according to

    K. Muelling, J. Kober, O. Kroemer, J. Peters:
    Learning to Select and Generalize Striking Movements in Robot Table Tennis
    (2013), International Journal of Robotics Research 32(3), pp. 263-279,
    https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf

    Parameters
    ----------
    n_dims : int
        State space dimensions.

    execution_time : float
        Execution time of the DMP.

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.

    p_gain : float, optional (default: 0)
        Gain for proportional controller of DMP tracking error.
        The domain is [0, execution_time**2/dt].

    Attributes
    ----------
    dt_ : float
        Time difference between DMP steps. This value can be changed to adapt
        the frequency.
    """
    def __init__(self, n_dims, execution_time, dt=0.01, n_weights_per_dim=10,
                 int_dt=0.001, p_gain=0.0):
        super(DMPWithFinalVelocity, self).__init__(n_dims, n_dims)
        self.execution_time = execution_time
        self.dt_ = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt
        self.p_gain = p_gain

        alpha_z = canonical_system_alpha(0.01, self.execution_time, 0.0,
                                         self.int_dt)
        self.forcing_term = ForcingTerm(self.n_dims, self.n_weights_per_dim,
                                        self.execution_time, 0.0, 0.8, alpha_z)

        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0

    def step(self, last_y, last_yd, coupling_term=None):
        """DMP step.

        Parameters
        ----------
        last_y : array, shape (n_dims,)
            Last state.

        last_yd : array, shape (n_dims,)
            Last time derivative of state (e.g., velocity).

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        Returns
        -------
        y : array, shape (n_dims,)
            Next state.

        yd : array, shape (n_dims,)
            Next time derivative of state (e.g., velocity).
        """
        self.last_t = self.t
        self.t += self.dt_

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        # https://github.com/studywolf/pydmps/blob/master/pydmps/cs.py
        tracking_error = self.current_y - last_y

        dmp_step_euler_with_constraints(
            self.last_t, self.t,
            self.current_y, self.current_yd,
            self.goal_y, self.goal_yd, self.goal_ydd,
            self.start_y, self.start_yd, self.start_ydd,
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term,
            coupling_term=coupling_term,
            int_dt=self.int_dt,
            p_gain=self.p_gain,
            tracking_error=tracking_error)
        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None):
        """Run DMP open loop.

        Parameters
        ----------
        run_t : float, optional (default: execution_time)
            Run time of DMP. Can be shorter or longer than execution_time.

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        Returns
        -------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, n_dims)
            State at each step.
        """
        return dmp_open_loop(
            self.execution_time, 0.0, self.dt_,
            self.start_y, self.goal_y,
            self.alpha_y, self.beta_y,
            self.forcing_term,
            coupling_term,
            run_t, self.int_dt,
            dmp_step_euler_with_constraints,
            start_yd=self.start_yd, start_ydd=self.start_ydd,
            goal_yd=self.goal_yd, goal_ydd=self.goal_ydd)

    def imitate(self, T, Y, regularization_coefficient=0.0):
        """Imitate demonstration.

        Parameters
        ----------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, n_dims)
            State at each step.

        regularization_coefficient : float, optional (default: 0)
            Regularization coefficient for regression.
        """
        self.forcing_term.weights[:, :], start_y, start_yd, start_ydd, goal_y, goal_yd, goal_ydd = dmp_imitate(
            T, Y,
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z, allow_final_velocity=True,
            determine_forces=determine_forces)
        self.configure(
            start_y=start_y, start_yd=start_yd, start_ydd=start_ydd,
            goal_y=goal_y, goal_yd=goal_yd, goal_ydd=goal_ydd)


def solve_constraints(t0, t1, y0, y0d, y0dd, y1, y1d, y1dd):
    t02 = t0 * t0
    t03 = t02 * t0
    t04 = t03 * t0
    t05 = t04 * t0
    t12 = t1 * t1
    t13 = t12 * t1
    t14 = t13 * t1
    t15 = t14 * t1

    M = np.array([[1, t0, t02, t03, t04, t05],
                  [0, 1, 2 * t0, 3 * t02, 4 * t03, 5 * t04],
                  [0, 0, 2, 6 * t0, 12 * t02, 20 * t03],
                  [1, t1, t12, t13, t14, t15],
                  [0, 1, 2 * t1, 3 * t12, 4 * t13, 5 * t14],
                  [0, 0, 2, 6 * t1, 12 * t12, 20 * t13]])
    Y = np.vstack((y0, y0d, y0dd, y1, y1d, y1dd))

    # Solve M*b = y for b in each DOF at once
    B = np.linalg.solve(M, Y)
    return B


def apply_constraints(t, goal_y, goal_t, coefficients):
    if t > goal_t + np.finfo(float).eps:
        # For t > goal_t the polynomial should always 'pull' to the goal
        # position, but velocity and acceleration should be zero.
        # This is done to avoid diverging from the goal if the DMP is executed
        # longer than expected.
        return goal_y, np.zeros_like(goal_y), np.zeros_like(goal_y)
    else:
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        pos = np.array([1, t, t2, t3, t4, t5])
        vel = np.array([0, 1, 2 * t, 3 * t2, 4 * t3, 5 * t4])
        acc = np.array([0, 0, 2, 6 * t, 12 * t2, 20 * t3])

        g = np.dot(pos, coefficients)
        gd = np.dot(vel, coefficients)
        gdd = np.dot(acc, coefficients)
        return g, gd, gdd


def determine_forces(T, Y, alpha_y, beta_y, allow_final_velocity):
    """Determine forces that the forcing term should generate.

    Parameters
    ----------
    T : array, shape (n_steps,)
        Time of each step.

    Y : array, shape (n_steps, n_dims)
        Position at each step.

    alpha_y : float
        Parameter of the transformation system.

    beta_y : float
        Parameter of the transformation system.

    allow_final_velocity : bool
        Whether a final velocity is allowed. This should always be True for
        this function.

    Returns
    -------
    F : array, shape (n_steps, n_dims)
        Forces.

    start_y : array, shape (n_dims,)
        Start position.

    start_yd : array, shape (n_dims,)
        Start velocity.

    start_ydd : array, shape (n_dims,)
        Start acceleration.

    goal_y : array, shape (n_dims,)
        Final position.

    goal_yd : array, shape (n_dims,)
        Final velocity.

    goal_ydd : array, shape (n_dims,)
        Final acceleration.
    """
    assert allow_final_velocity

    n_dims = Y.shape[1]
    DT = np.diff(T)

    Yd = np.empty_like(Y)
    Yd[0] = 0.0
    for d in range(n_dims):
        Yd[1:, d] = np.diff(Y[:, d]) / DT

    Ydd = np.empty_like(Y)
    Ydd[0] = 0.0
    for d in range(n_dims):
        Ydd[1:, d] = np.diff(Yd[:, d]) / DT

    coefficients = solve_constraints(
        T[0], T[-1], Y[0], Yd[0], Ydd[0], Y[-1], Yd[-1], Ydd[-1])

    execution_time = T[-1] - T[0]
    F = np.empty((len(T), n_dims))
    for i in range(len(T)):
        g, gd, gdd = apply_constraints(T[i], Y[-1], T[-1], coefficients)
        F[i, :] = execution_time ** 2 * Ydd[i] - alpha_y * (
            beta_y * (g - Y[i]) + gd * execution_time
            - Yd[i] * execution_time) - execution_time ** 2 * gdd
    return F, Y[0], Yd[0], Ydd[0], Y[-1], Yd[-1], Ydd[-1]


def dmp_step_euler_with_constraints(
        last_t, t, current_y, current_yd, goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd, goal_t, start_t, alpha_y, beta_y,
        forcing_term, coupling_term=None, coupling_term_precomputed=None,
        int_dt=0.001, p_gain=0.0, tracking_error=0.0):
    """Integrate regular DMP for one step with Euler integration.

    Parameters
    ----------
    last_t : float
        Time at last step.

    t : float
        Time at current step.

    current_y : array, shape (n_dims,)
        Current position. Will be modified.

    current_yd : array, shape (n_dims,)
        Current velocity. Will be modified.

    goal_y : array, shape (n_dims,)
        Goal position.

    goal_yd : array, shape (n_dims,)
        Goal velocity.

    goal_ydd : array, shape (n_dims,)
        Goal acceleration.

    start_y : array, shape (n_dims,)
        Start position.

    start_yd : array, shape (n_dims,)
        Start velocity.

    start_ydd : array, shape (n_dims,)
        Start acceleration.

    goal_t : float
        Time at the end.

    start_t : float
        Time at the start.

    alpha_y : float
        Constant in transformation system.

    beta_y : float
        Constant in transformation system.

    forcing_term : ForcingTerm
        Forcing term.

    coupling_term : CouplingTerm, optional (default: None)
        Coupling term. Must have a function coupling(y, yd) that returns
        additional velocity and acceleration.

    coupling_term_precomputed : tuple
        A precomputed coupling term, i.e., additional velocity and
        acceleration.

    int_dt : float, optional (default: 0.001)
        Time delta used internally for integration.

    p_gain : float, optional (default: 0)
        Proportional gain for tracking error.

    tracking_error : float, optional (default: 0)
        Tracking error from last step.
    """
    if start_t >= goal_t:
        raise ValueError("Goal must be chronologically after start!")

    if t <= start_t:
        return np.copy(start_y), np.copy(start_yd), np.copy(start_ydd)

    execution_time = goal_t - start_t

    coefficients = solve_constraints(
        start_t, goal_t, start_y, start_yd, start_ydd,
        goal_y, goal_yd, goal_ydd)

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

        f = forcing_term(current_t).squeeze()

        g, gd, gdd = apply_constraints(current_t, goal_y, goal_t, coefficients)

        coupling_sum = cdd + p_gain * tracking_error / dt
        ydd = (alpha_y * (beta_y * (g - current_y)
                          + execution_time * gd
                          - execution_time * current_yd)
               + gdd * execution_time ** 2
               + f + coupling_sum) / execution_time ** 2
        current_yd += dt * ydd + cd / execution_time
        current_y += dt * current_yd

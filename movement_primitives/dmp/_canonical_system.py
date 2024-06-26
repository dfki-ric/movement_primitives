def canonical_system_alpha(goal_z, goal_t, start_t, int_dt=0.001):
    """Compute parameter alpha of canonical system.

    Parameters
    ----------
    goal_z : float
        Value of phase variable at the end of the execution (> 0).

    goal_t : float
        Time at which the execution should be done. Make sure that
        goal_t > start_t.

    start_t : float
        Time at which the execution should start.

    int_dt : float, optional (default: 0.001)
        Time delta that is used internally for integration.

    Returns
    -------
    alpha : float
        Value of the alpha parameter of the canonical system.

    Raises
    ------
    ValueError
        If input values are invalid.
    """
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
    r"""Map time to phase.

    According to [1]_, the differential Equation

    .. math::

        \tau \dot{z} = -\alpha_z z

    describes the evolution of the phase variable z. Starting from the initial
    position :math:`z_0 = 1`, the phase value converges monotonically to 0.
    Instead of using an iterative procedure to calculate the current value of
    z, it is computed directly for a fixed :math:`\Delta t` through

    .. math::

        (1 - \alpha_z \frac{\Delta t}{\tau})^{\frac{t}{\Delta t}}

    Parameters
    ----------
    t : float
        Current time.

    alpha : float
        Value of the alpha parameter of the canonical system.

    goal_t : float
        Time at which the execution should be done.

    start_t : float
        Time at which the execution should start.

    int_dt : float, optional (default: 0.001)
        Time delta that is used internally for integration.

    eps : float, optional (default: 1e-10)
        Small number used to avoid numerical issues.

    Returns
    -------
    z : float
        Value of phase variable.

    References
    ----------
    .. [1] Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P., Schaal, S.
       (2013). Dynamical Movement Primitives: Learning Attractor Models for
       Motor Behaviors. Neural Computation 25 (2), 328-373. DOI:
       10.1162/NECO_a_00393,
       https://homes.cs.washington.edu/~todorov/courses/amath579/reading/DynamicPrimitives.pdf
    """
    execution_time = goal_t - start_t
    b = max(1.0 - alpha * int_dt / execution_time, eps)
    return b ** ((t - start_t) / int_dt)

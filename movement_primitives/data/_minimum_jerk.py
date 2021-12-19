import numpy as np


def generate_minimum_jerk(start, goal, execution_time=1.0, dt=0.01):
    """Create a minimum jerk trajectory.

    A minimum jerk trajectory from :math:`x_0` to :math:`g` minimizes
    the third time derivative of the positions:

    .. math::

        \\arg \min_{x_0, \ldots, x_T} \int_{t=0}^T \dddot{x}(t)^2 dt

    The trajectory will have

    .. code-block:: python

        n_steps = 1 + execution_time / dt

    steps because we start at 0 seconds and end at execution_time seconds.

    Parameters
    ----------
    start : array-like, shape (n_dims,)
        Initial state

    goal : array-like, shape (n_dims,)
        Goal state

    execution_time : float, optional (default: 1)
        Execution time in seconds

    dt : float, optional (default: 0.01)
        Time between successive steps in seconds

    Returns
    -------
    X : array, shape (n_dims, n_steps)
        The positions of the trajectory

    Xd : array, shape (n_dims, n_steps)
        The velocities of the trajectory

    Xdd : array, shape (n_task_dims, n_steps)
        The accelerations of the trajectory

    Raises
    ------
    ValueError
        If the shapes of the initial and goal state do not match.
    """
    x0 = np.asarray(start)
    g = np.asarray(goal)
    if x0.shape != g.shape:
        raise ValueError("Shape of initial state %s and goal %s must be equal"
                         % (x0.shape, g.shape))

    n_task_dims = x0.shape[0]
    n_steps = 1 + int(execution_time / dt)

    X = np.zeros((n_task_dims, n_steps))
    Xd = np.zeros((n_task_dims, n_steps))
    Xdd = np.zeros((n_task_dims, n_steps))

    x = x0.copy()
    xd = np.zeros(n_task_dims)
    xdd = np.zeros(n_task_dims)

    X[:, 0] = x
    for t in range(1, n_steps):
        tau = execution_time - t * dt

        if tau >= dt:
            dist = g - x

            a1 = 0
            a0 = xdd * tau ** 2
            v1 = 0
            v0 = xd * tau

            t1 = dt
            t2 = dt ** 2
            t3 = dt ** 3
            t4 = dt ** 4
            t5 = dt ** 5

            c1 = (6. * dist + (a1 - a0) / 2. - 3. * (v0 + v1)) / tau ** 5
            c2 = (-15. * dist + (3. * a0 - 2. * a1) / 2. + 8. * v0 +
                  7. * v1) / tau ** 4
            c3 = (10. * dist + (a1 - 3. * a0) / 2. - 6. * v0 -
                  4. * v1) / tau ** 3
            c4 = xdd / 2.
            c5 = xd
            c6 = x

            x = c1 * t5 + c2 * t4 + c3 * t3 + c4 * t2 + c5 * t1 + c6
            xd = (5. * c1 * t4 + 4 * c2 * t3 + 3 * c3 * t2 + 2 * c4 * t1 + c5)
            xdd = (20. * c1 * t3 + 12. * c2 * t2 + 6. * c3 * t1 + 2. * c4)

        X[:, t] = x
        Xd[:, t] = xd
        Xdd[:, t] = xdd

    return X, Xd, Xdd

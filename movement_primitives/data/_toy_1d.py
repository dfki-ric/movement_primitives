import numpy as np


def generate_1d_trajectory_distribution(
        n_demos, n_steps, initial_offset_range=3.0, final_offset_range=0.1,
        noise_per_step_range=20.0, random_state=np.random.RandomState(0)):
    """Generates toy data for testing and demonstration.

    Parameters
    ----------
    n_demos : int
        Number of demonstrations

    n_steps : int
        Number of steps

    initial_offset_range : float, optional (default: 3)
        Range of initial offset from cosine

    final_offset_range : float, optional (default: 0.1)
        Range of final offset from cosine

    noise_per_step_range : float, optional (default: 20)
        Factor for noise in each step

    random_state : RandomState, optional (default: seed 0)
        Random state

    Returns
    -------
    T : array, shape (n_steps,)
        Times

    Y : array, shape (n_demos, n_steps, 1)
        Demonstrations (positions)
    """
    T = np.linspace(0, 1, n_steps)
    Y = np.empty((n_demos, n_steps, 1))

    A = create_finite_differences_matrix_1d(n_steps, dt=1.0 / (n_steps - 1))
    cov = np.linalg.inv(A.T.dot(A))
    L = np.linalg.cholesky(cov)

    for demo_idx in range(n_demos):
        Y[demo_idx, :, 0] = np.cos(2 * np.pi * T)
        if initial_offset_range or final_offset_range:
            initial_offset = initial_offset_range * (random_state.rand() - 0.5)
            final_offset = final_offset_range * (random_state.rand() - 0.5)
            Y[demo_idx, :, 0] += np.linspace(
                initial_offset, final_offset, n_steps)
        if noise_per_step_range:
            noise_per_step = (noise_per_step_range
                              * L.dot(random_state.randn(n_steps)))
            Y[demo_idx, :, 0] += noise_per_step
    return T, Y


def create_finite_differences_matrix_1d(n_steps, dt):
    """Finite difference matrix to compute accelerations from positions.

    Parameters
    ----------
    n_steps : int
        Number of steps of the resulting trajectory.

    dt : float
        Time between steps.

    Returns
    -------
    A : array, shape (n_steps + 2, n_steps)
        Finite difference matrix.
    """
    A = np.zeros((n_steps + 2, n_steps), dtype=np.float)
    super_diagonal = (np.arange(n_steps), np.arange(n_steps))
    sub_diagonal = (np.arange(2, n_steps + 2), np.arange(n_steps))
    A[super_diagonal] = np.ones(n_steps)
    A[sub_diagonal] = np.ones(n_steps)
    main_diagonal = (np.arange(1, n_steps + 1), np.arange(n_steps))
    A[main_diagonal] = -2 * np.ones(n_steps)
    return A / (dt ** 2)

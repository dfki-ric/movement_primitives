import os

import numpy as np
from gmr import MVN
from tqdm import tqdm

from movement_primitives.dmp import DualCartesianDMP


def propagate_weight_distribution_to_state_space(dataset, n_weights_per_dim, cache_filename=None, alpha=1e-3, kappa=10.0, verbose=0):
    """Learn DMPs from dataset and propagate MVN of weights to state space.

    Parameters
    ----------
    dataset : list of tuples
        Dataset for imitation. Each tuple contains the time steps and the
        trajectory of a demonstration.

    n_weights_per_dim : int
        Number of DMP weights that will be used for each state dimension.

    cache_filename : str, optional (default: None)
        It is quite costly to propagate sigma points to state space. The
        trajectories can be cached in a file with this option.

    alpha : float, optional (default: 1e-3)
        Parameter for sigma-point propagation.

    kappa : float, optional (default: 10.0)
        Parameter for sigma-point propagation.

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    mvn : MVN
        Distribution over trajectories in state space. Note that trajectories
        will be represented by a 1d array and have to be reshaped to
        (n_steps, n_dims).
    """
    if cache_filename is not None and os.path.exists(cache_filename):
        trajectories = np.loadtxt(cache_filename)
    else:
        mvn, mean_execution_time = estimate_dmp_parameter_distribution(
            dataset=dataset, n_weights_per_dim=n_weights_per_dim,
            verbose=verbose)
        trajectories = propagate_to_state_space(mvn=mvn, n_weights_per_dim=n_weights_per_dim, execution_time=mean_execution_time, alpha=alpha, kappa=kappa, verbose=verbose)

        if cache_filename is not None:
            np.savetxt(cache_filename, trajectories)

    return estimate_state_distribution(trajectories, alpha=alpha, kappa=kappa, n_weights_per_dim=n_weights_per_dim)


def estimate_dmp_parameter_distribution(dataset, n_weights_per_dim, verbose=0):
    if verbose:
        print("Estimate DMP parameter distribution...")

    all_weights = []
    all_starts = []
    all_goals = []
    all_execution_times = []
    for T, P in tqdm(dataset):
        execution_time = T[-1]
        dt = np.mean(np.diff(T))
        if dt < 0.005:  # HACK
            continue

        dmp = DualCartesianDMP(
            execution_time=execution_time, dt=dt,
            n_weights_per_dim=n_weights_per_dim, int_dt=0.01)
        dmp.imitate(T, P)
        weights = dmp.get_weights()

        all_weights.append(weights)
        all_starts.append(P[0])
        all_goals.append(P[-1])
        all_execution_times.append(execution_time)
    all_parameters = np.vstack([
        np.hstack((w, s, g, e)) for w, s, g, e in zip(
            all_weights, all_starts, all_goals, all_execution_times)])

    mvn = MVN()
    mvn.from_samples(all_parameters)
    return mvn, np.mean(all_execution_times)


def propagate_to_state_space(mvn, n_weights_per_dim, execution_time, alpha, kappa, verbose=0):
    if verbose:
        print("Propagating to state space...")

    n_weights = 2 * 6 * n_weights_per_dim
    n_dims = 2 * 7
    weight_indices = np.arange(n_weights)
    start_indices = np.arange(n_weights, n_weights + n_dims)
    goal_indices = np.arange(n_weights + n_dims, n_weights + 2 * n_dims)

    points = mvn.sigma_points(alpha=alpha, kappa=kappa)
    trajectories = []
    for i, parameters in tqdm(list(enumerate(points))):
        weights = parameters[weight_indices]
        start = parameters[start_indices]
        goal = parameters[goal_indices]
        dmp = DualCartesianDMP(
            execution_time=execution_time, dt=0.1,
            n_weights_per_dim=n_weights_per_dim, int_dt=0.01)
        dmp.configure(start_y=start, goal_y=goal)
        dmp.set_weights(weights)
        T, P = dmp.open_loop(run_t=execution_time)
        trajectories.append(P.ravel())

    return np.vstack(trajectories)


def estimate_state_distribution(trajectories, alpha, kappa, n_weights_per_dim):
    print("Estimate distribution in state space...")
    n_weights = 2 * 6 * n_weights_per_dim
    n_dims = 2 * 7
    n_features = n_weights + 2 * n_dims + 1
    initial_mean = np.zeros(n_features)
    initial_cov = np.eye(n_features)
    return MVN(initial_mean, initial_cov, random_state=42).estimate_from_sigma_points(
        trajectories, alpha=alpha, kappa=kappa)

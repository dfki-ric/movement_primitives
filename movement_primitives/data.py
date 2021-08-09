import glob

import numpy as np
import pandas as pd
from mocap import array_from_dataframe
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.cleaning import smooth_quaternion_trajectory, median_filter
from tqdm import tqdm


def smooth_dual_arm_trajectories_pq(Ps, median_filter_window=5):
    """Make orientation representation smooth.

    Note that the argument Ps will be manipulated and the function does not
    return anything.

    Parameters
    ----------
    Ps : list
        List of dual arm trajectories represented by position of the left arm,
        orientation quaternion of the left arm, position of the right arm,
        and orientation quaternion of the right arm.

    median_filter_window : int, optional (default: 5)
        Window size of the median filter
    """
    for P in Ps:
        P[:, 3:7] = smooth_quaternion_trajectory(P[:, 3:7])
        P[:, 10:] = smooth_quaternion_trajectory(P[:, 10:])
        P[:, :] = median_filter(P, window_size=median_filter_window)


def smooth_single_arm_trajectories_pq(Ps, median_filter_window=5):
    """Make orientation representation smooth.

    Note that the argument Ps will be manipulated and the function does not
    return anything.

    Parameters
    ----------
    Ps : list
        List of single arm trajectories represented by position of the arm and
        orientation quaternion of the arm.

    median_filter_window : int, optional (default: 5)
        Window size of the median filter
    """
    for P in Ps:
        P[:, 3:7] = smooth_quaternion_trajectory(P[:, 3:7])
        P[:, :] = median_filter(P, window_size=median_filter_window)


def transpose_dataset(dataset):
    """Converts list of demo data to multiple lists of demo properties.

    For example, one entry might contain time steps and poses so that we
    generate one list for all time steps and one list for all poses
    (trajectories).

    Parameters
    ----------
    dataset : list of tuples
        There are n_demos entries. Each entry describes one demonstration
        completely. An entry might, for example, contain an array of time
        (T, shape (n_steps,)) or the dual arm trajectories
        (P, shape (n_steps, 14)).

    Returns
    -------
    dataset : tuple of lists
        Each entry contains a list of length n_demos, where the i-th entry
        corresponds to an attribute of the i-th demo. For example, the first
        list might contain to all time arrays of the demonstrations and the
        second entry might correspond to all trajectories.
    """
    n_samples = len(dataset)
    if n_samples == 0:
        raise ValueError("Empty dataset")

    n_arrays = len(dataset[0])
    arrays = [[] for _ in range(n_arrays)]
    for demo in dataset:
        for i in range(n_arrays):
            arrays[i].append(demo[i])
    return arrays


def load_kuka_dataset(pattern, context_names=None, verbose=0):
    """Load dataset obtained from kinesthetic teaching of dual arm Kuka system.

    Parameters
    ----------
    pattern : str
        Pattern that defines csv files that should be loaded, e.g., *.csv

    context_names : list, optional (default: None)
        Contexts that should be loaded

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    dataset : list of tuples
        Each entry contains either an array of time (T, shape (n_steps,)),
        the dual arm trajectories (P, shape (n_steps, 14)) represented by
        positions of the left arm, orientation quaternion of the left arm,
        positions of the right arm, and the orientation quaternion of the
        right arm, or T, P, and the context (shape, (n_context_dims,)).
    """
    filenames = list(glob.glob(pattern))
    if verbose:
        print("Loading dataset...")
        filenames = tqdm(filenames)
    return [load_kuka_demo(f, context_names, verbose=verbose) for f in filenames]


def load_kuka_demo(filename, context_names=None, verbose=0):
    """Load a single demonstration from the dual arm Kuka system from csv.

    Parameters
    ----------
    filename : str
        Name of the csv file.

    context_names : list, optional (default: None)
        Name of context variables that should be loaded.

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    T : array, shape (n_steps,)
        Time steps

    P : array, shape (n_steps, 14)
        Dual arm trajectories represented by position of the left arm,
        orientation quaternion of the left arm, position of the right arm,
        and orientation quaternion of the right arm.

    context : array, shape (len(context_names),), optional
        Values of context variables. These will only be returned if
        context_names are given.
    """
    if verbose:
        tqdm.write("Loading '%s'" % filename)
    trajectory = pd.read_csv(filename, sep=" ")

    if context_names is not None:
        context = trajectory[list(context_names)].iloc[0].to_numpy()
        if verbose:
            tqdm.write("Context: %s" % (context,))

    patterns = ["time\.microseconds",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.position\.data.*",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation\.re.*",
                "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation\.im.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.position\.data.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation\.re.*",
                "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation\.im.*"]
    columns = match_columns(trajectory, patterns)
    trajectory = trajectory[columns]

    group_rename = {
        "(time\.microseconds)": "Time",
        "(kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.position\.data).*": "left_pose",
        "(kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\.orientation).*": "left_pose",
        "(kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.position\.data).*": "right_pose",
        "(kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\.orientation).*": "right_pose"
    }
    trajectory = rename_stream_groups(trajectory, group_rename)

    trajectory["Time"] = trajectory["Time"] / 1e6
    trajectory["Time"] -= trajectory["Time"].iloc[0]
    T = trajectory["Time"].to_numpy()

    P = array_from_dataframe(
        trajectory,
        ["left_pose[0]", "left_pose[1]", "left_pose[2]",
         "left_pose.re", "left_pose.im[0]", "left_pose.im[1]", "left_pose.im[2]",
         "right_pose[0]", "right_pose[1]", "right_pose[2]",
         "right_pose.re", "right_pose.im[0]", "right_pose.im[1]", "right_pose.im[2]"])

    if context_names is None:
        return T, P
    else:
        return T, P, context


def load_rh5_demo(filename, verbose=0):
    """Load a single demonstration from the RH5 robot from csv.

    Parameters
    ----------
    filename : str
        Name of the csv file.

    verbose : int, optional (default: 0)
        Verbosity level

    Returns
    -------
    T : array, shape (n_steps,)
        Time steps

    P : array, shape (n_steps, 14)
        Dual arm trajectories represented by position of the left arm,
        orientation quaternion of the left arm, position of the right arm,
        and orientation quaternion of the right arm.
    """
    if verbose:
        tqdm.write("Loading '%s'" % filename)
    trajectory = pd.read_csv(filename, sep=" ")
    patterns = ["time\.microseconds",
                "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.position\.data.*",
                "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation\.re.*",
                "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation\.im.*",
                "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.position\.data.*",
                "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation\.re.*",
                "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation\.im.*"]
    columns = match_columns(trajectory, patterns)
    trajectory = trajectory[columns]

    group_rename = {
        "(time\.microseconds)": "Time",
        "(rh5_left_arm_posture_ctrl\.current_feedback\.pose\.position\.data).*": "left_pose",
        "(rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation).*": "left_pose",
        "(rh5_right_arm_posture_ctrl\.current_feedback\.pose\.position\.data).*": "right_pose",
        "(rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation).*": "right_pose"
    }
    trajectory = rename_stream_groups(trajectory, group_rename)

    trajectory["Time"] = trajectory["Time"] / 1e6
    trajectory["Time"] -= trajectory["Time"].iloc[0]
    T = trajectory["Time"].to_numpy()

    P = array_from_dataframe(
        trajectory,
        ["left_pose[0]", "left_pose[1]", "left_pose[2]", "left_pose.re", "left_pose.im[0]", "left_pose.im[1]",
         "left_pose.im[2]",
         "right_pose[0]", "right_pose[1]", "right_pose[2]", "right_pose.re", "right_pose.im[0]", "right_pose.im[1]",
         "right_pose.im[2]"])

    return T, P


def load_mia_demo(filename, dt=0.01, verbose=0):
    """Load a single demonstration for the Mia hand from csv.

    Parameters
    ----------
    filename : str
        Name of the csv file.

    dt : float, optional (default: 0.01)
        Time between steps.

    verbose : int, optional (default: 0)
        Verbosity level.

    Returns
    -------
    T : array, shape (n_steps,)
        Time steps

    P : array, shape (n_steps, 11)
        Position of the palm frame, orientation of the palm frame as
        quaternion, and joint angles. Pose of the palm frame is relative to
        the manipulated object. Order of joint angles is "j_index_fle",
        "j_mrl_fle", "j_thumb_fle", "j_thumb_opp".
    """
    trajectory = pd.read_csv(filename)
    T = np.arange(0.0, dt * len(trajectory), dt)
    if len(T) != len(trajectory):
        T = T[:len(trajectory)]
    P = trajectory[
        ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy",
         "base_qz", "j_index_fle", "j_mrl_fle", "j_thumb_fle", "j_thumb_opp"]
        ]
    return T, P.to_numpy()


def generate_1d_trajectory_distribution(
        n_demos, n_steps,
        initial_offset_range=3.0, final_offset_range=0.1, noise_per_step_range=20.0,
        random_state=np.random.RandomState(0)):
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
            Y[demo_idx, :, 0] += np.linspace(initial_offset, final_offset, n_steps)
        if noise_per_step_range:
            noise_per_step = noise_per_step_range * L.dot(random_state.randn(n_steps))
            Y[demo_idx, :, 0] += noise_per_step
    return T, Y


def create_finite_differences_matrix_1d(n_steps, dt):
    """Finite difference matrix to compute accelerations from positions."""
    A = np.zeros((n_steps + 2, n_steps), dtype=np.float)
    super_diagonal = (np.arange(n_steps), np.arange(n_steps))
    sub_diagonal = (np.arange(2, n_steps + 2), np.arange(n_steps))
    A[super_diagonal] = np.ones(n_steps)
    A[sub_diagonal] = np.ones(n_steps)
    main_diagonal = (np.arange(1, n_steps + 1), np.arange(n_steps))
    A[main_diagonal] = -2 * np.ones(n_steps)
    return A / (dt ** 2)

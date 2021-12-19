import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import mocap
import mocap.cleaning
import mocap.pandas_utils


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
        P[:, 3:7] = mocap.cleaning.smooth_quaternion_trajectory(P[:, 3:7])
        P[:, 10:] = mocap.cleaning.smooth_quaternion_trajectory(P[:, 10:])
        P[:, :] = mocap.cleaning.median_filter(
            P, window_size=median_filter_window)


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
        P[:, 3:7] = mocap.cleaning.smooth_quaternion_trajectory(P[:, 3:7])
        P[:, :] = mocap.cleaning.median_filter(
            P, window_size=median_filter_window)


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

    Raises
    ------
    ValueError
        If the dataset is empty.
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
    return [load_kuka_demo(f, context_names, verbose=verbose)
            for f in filenames]


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

    PREFIX_LEFT = "kuka_lbr_cart_pos_ctrl_left\.current_feedback\.pose\."
    PREFIX_RIGHT = "kuka_lbr_cart_pos_ctrl_right\.current_feedback\.pose\."
    patterns = [
        "time\.microseconds",
        f"{PREFIX_LEFT}position\.data.*",
        f"{PREFIX_LEFT}orientation\.re.*",
        f"{PREFIX_LEFT}orientation\.im.*",
        f"{PREFIX_RIGHT}position\.data.*",
        f"{PREFIX_RIGHT}orientation\.re.*",
        f"{PREFIX_RIGHT}orientation\.im.*"]
    columns = mocap.pandas_utils.match_columns(trajectory, patterns)
    trajectory = trajectory[columns]

    group_rename = {
        "(time\.microseconds)": "Time",
        f"({PREFIX_LEFT}position\.data).*": "left_pose",
        f"({PREFIX_LEFT}orientation).*": "left_pose",
        f"({PREFIX_RIGHT}position\.data).*": "right_pose",
        f"({PREFIX_RIGHT}orientation).*": "right_pose"
    }
    trajectory = mocap.pandas_utils.rename_stream_groups(trajectory, group_rename)

    trajectory["Time"] = trajectory["Time"] / 1e6
    trajectory["Time"] -= trajectory["Time"].iloc[0]
    T = trajectory["Time"].to_numpy()

    P = mocap.array_from_dataframe(
        trajectory,
        ["left_pose[0]", "left_pose[1]", "left_pose[2]", "left_pose.re",
         "left_pose.im[0]", "left_pose.im[1]", "left_pose.im[2]",
         "right_pose[0]", "right_pose[1]", "right_pose[2]", "right_pose.re",
         "right_pose.im[0]", "right_pose.im[1]", "right_pose.im[2]"])

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
    PREFIX_LEFT = "rh5_left_arm_posture_ctrl\.current_feedback\.pose\."
    PREFIX_RIGHT = "rh5_right_arm_posture_ctrl\.current_feedback\.pose\."
    patterns = [
        "time\.microseconds",
        f"{PREFIX_LEFT}position\.data.*",
        f"{PREFIX_LEFT}orientation\.re.*",
        f"{PREFIX_LEFT}orientation\.im.*",
        f"{PREFIX_RIGHT}position\.data.*",
        f"{PREFIX_RIGHT}orientation\.re.*",
        f"{PREFIX_RIGHT}orientation\.im.*"]
    columns = mocap.pandas_utils.match_columns(trajectory, patterns)
    trajectory = trajectory[columns]

    group_rename = {
        "(time\.microseconds)": "Time",
        f"({PREFIX_LEFT}position\.data).*": "left_pose",
        f"({PREFIX_LEFT}orientation).*": "left_pose",
        f"({PREFIX_RIGHT}position\.data).*": "right_pose",
        f"({PREFIX_RIGHT}orientation).*": "right_pose"
    }
    trajectory = mocap.pandas_utils.rename_stream_groups(trajectory, group_rename)

    trajectory["Time"] = trajectory["Time"] / 1e6
    trajectory["Time"] -= trajectory["Time"].iloc[0]
    T = trajectory["Time"].to_numpy()

    P = mocap.array_from_dataframe(
        trajectory,
        ["left_pose[0]", "left_pose[1]", "left_pose[2]", "left_pose.re",
         "left_pose.im[0]", "left_pose.im[1]", "left_pose.im[2]",
         "right_pose[0]", "right_pose[1]", "right_pose[2]", "right_pose.re",
         "right_pose.im[0]", "right_pose.im[1]", "right_pose.im[2]"])

    return T, P


def load_mia_demo(filename, dt=0.01, ignore_columns=(), verbose=0):
    """Load a single demonstration for the Mia hand from csv.

    Parameters
    ----------
    filename : str
        Name of the csv file.

    dt : float, optional (default: 0.01)
        Time between steps.

    ignore_columns : list or tuple, optional (default: ())
        Columns that should not be loaded.

    verbose : int, optional (default: 0)
        Verbosity level.

    Returns
    -------
    T : array, shape (n_steps,)
        Time steps

    P : array, shape (n_steps, 11 - len(ignore_columns))
        Position of the palm frame, orientation of the palm frame as
        quaternion, and joint angles. Pose of the palm frame is relative to
        the manipulated object. Order of joint angles is "j_index_fle",
        "j_mrl_fle", "j_thumb_fle", "j_thumb_opp".
    """
    trajectory = pd.read_csv(filename)
    T = np.arange(0.0, dt * len(trajectory), dt)
    if len(T) != len(trajectory):
        T = T[:len(trajectory)]

    ALL_COLUMNS = [
        "base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy",
        "base_qz", "j_index_fle", "j_mrl_fle", "j_thumb_fle", "j_thumb_opp"]
    # quadratic complexity; since order is relevant, we cannot use a set
    # difference
    columns = [c for c in ALL_COLUMNS if c not in ignore_columns]
    if verbose >= 2:
        tqdm.write("Loading columns: [%s]" % ", ".join(columns))

    P = trajectory[columns].to_numpy()
    return T, P

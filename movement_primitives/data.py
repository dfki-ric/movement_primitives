"""Tools for loading datasets."""
import glob
import os
import scipy.io
import zipfile
import io
try:
    from urllib2 import urlopen
except:
    from urllib.request import urlopen
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    from mocap import array_from_dataframe
    from mocap.pandas_utils import match_columns, rename_stream_groups
    from mocap.cleaning import smooth_quaternion_trajectory, median_filter
    mocap_available = True
except ImportError:
    mocap_available = False
    import warnings
    warnings.warn("mocap library is not available, data module will not work")


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
        Goal

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


def load_lasa(shape_idx):
    """Load demonstrations from LASA dataset.

    The LASA dataset contains 2D handwriting motions recorded from a
    Tablet-PC. It can be found `here
    <https://bitbucket.org/khansari/lasahandwritingdataset>`_
    Take a look at the `detailed explanation
    <http://cs.stanford.edu/people/khansari/DSMotions#SEDS_Benchmark_Dataset>`_
    for more information.

    The following plot shows multiple demonstrations for the same shape.

    .. plot::

        import matplotlib.pyplot as plt
        from movement_primitives.data import load_lasa
        X, Xd, Xdd, dt, shape_name = load_lasa(0)
        plt.figure()
        plt.title(shape_name)
        plt.plot(X[:, :, 0].T, X[:, :, 1].T)
        plt.show()

    Parameters
    ----------
    shape_idx : int
        Choose demonstrated shape, must be within range(30).

    Returns
    -------
    T : array, shape (n_demos, n_steps)
        Times

    X : array, shape (n_demos, n_steps, n_dims)
        Positions

    Xd : array, shape (n_demos, n_steps, n_dims)
        Velocities

    Xdd : array, shape (n_demos, n_steps, n_dims)
        Accelerations

    dt : float
        Time between steps

    shape_name : string
        Name of the Matlab file from which we load the demonstrations
        (without suffix)
    """
    dataset_path = get_common_dataset_path()
    if not os.path.isdir(dataset_path + "lasa_data"):
        url = urlopen("http://bitbucket.org/khansari/lasahandwritingdataset/get/38304f7c0ac4.zip")
        z = zipfile.ZipFile(io.BytesIO(url.read()))
        z.extractall(dataset_path)
        os.rename(dataset_path + z.namelist()[0],
                  dataset_path + "lasa_data" + os.sep)

    dataset_path += "lasa_data" + os.sep + "DataSet" + os.sep
    demos, shape_name = _load_from_matlab_file(dataset_path, shape_idx)
    X, Xd, Xdd, dt = _convert_demonstrations(demos)
    t = np.linspace(0, X.shape[1], X.shape[1])
    T = np.tile(t, (X.shape[0], 1))
    return T, X, Xd, Xdd, dt, shape_name


def _load_from_matlab_file(dataset_path, shape_idx):
    """Load demonstrations from Matlab files."""
    file_name = sorted(os.listdir(dataset_path))[shape_idx]
    return (scipy.io.loadmat(dataset_path + file_name)["demos"][0],
            file_name[:-4])


def _convert_demonstrations(demos):
    """Convert Matlab struct to numpy arrays."""
    tmp = []
    for demo_idx in range(demos.shape[0]):
        # The Matlab format is strange...
        demo = demos[demo_idx][0, 0]
        # Positions, velocities and accelerations
        tmp.append((demo[0], demo[2], demo[3]))

    X = np.transpose([P for P, _, _ in tmp], [0, 2, 1])
    Xd = np.transpose([V for _, V, _ in tmp], [0, 2, 1])
    Xdd = np.transpose([A for _, _, A in tmp], [0, 2, 1])

    dt = float(demos[0][0, 0][4])

    return X, Xd, Xdd, dt


def get_common_dataset_path():
    """Returns the path where all external datasets are stored."""
    dataset_path = os.path.expanduser("~")
    dataset_path += os.sep + ".movement_primitive_data" + os.sep
    return dataset_path

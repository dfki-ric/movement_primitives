import glob

import pandas as pd
from mocap import array_from_dataframe
from mocap.pandas_utils import match_columns, rename_stream_groups
from tqdm import tqdm


def transpose_dataset(dataset):
    """Converts list of demo data to multiple lists of demo properties.

    For example, one entry might contain time steps and poses so that we
    generate one list for all time steps and one list for all poses
    (trajectories).
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
    """Load dataset obtained from kinesthetic teaching of dual arm Kuka system."""
    filenames = list(glob.glob(pattern))
    if verbose:
        print("Loading dataset...")
        filenames = tqdm(filenames)
    return [load_kuka_demo(f, context_names, verbose=verbose) for f in filenames]


def load_kuka_demo(filename, context_names=None, verbose=0):
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


def load_rh5_demo(path):
    trajectory = pd.read_csv(path, sep=" ")
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

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from dmp import DualCartesianDMP
from gmr import MVN


def sqrt_sym_mat(cov):
    """Compute square root of a covariance matrix."""
    cov = np.triu(cov) + np.triu(cov, 1).T
    D, B = np.linalg.eigh(cov)
    # HACK: avoid numerical problems
    D = np.maximum(D, np.finfo(np.float).eps)
    return B.dot(np.diag(np.sqrt(D))).dot(B.T)


def sigma_points(mvn, alpha=1e-3, kappa=0.0):
    """TODO"""
    n_dims = len(mvn.mean)
    lmbda = alpha ** 2 * (n_dims + kappa) - n_dims
    offset = sqrt_sym_mat((n_dims + lmbda) * mvn.covariance)

    points = np.empty(((2 * n_dims + 1), n_dims))
    points[0, :] = mvn.mean
    for i in range(n_dims):
        points[1 + i, :] = mvn.mean + offset[i]
        points[1 + n_dims + i:, :] = mvn.mean - offset[i]
    return points


def estimate_from_sigma_points(transformed_sigma_points, n_dims, alpha=1e-3, beta=2.0, kappa=0.0, random_state=None):
    """TODO

    https://www.seas.harvard.edu/courses/cs281/papers/unscented.pdf

    Parameters
    ----------
    alpha : float, optional (default: 1e-3)
        Determines the spread of the sigma points around the mean and is
        usually set to a small positive value.

    beta : float, optional (default: 2)
        Encodes information about the distribution. For Gaussian distributions,
        beta=2 is the optimal choice.

    kappa : float, optional (default: 0)
        A secondary scaling parameter which is usually set to 0.
    """
    lmbda = alpha ** 2 * (n_dims + kappa) - n_dims

    mean_weight_0 = lmbda / (n_dims + lmbda)
    cov_weight_0 = lmbda / (n_dims + lmbda) + (1 - alpha ** 2 + beta)
    weights_i = 1.0 / (2.0 * (n_dims + lmbda))
    mean_weights = np.empty(len(transformed_sigma_points))
    mean_weights[0] = mean_weight_0
    mean_weights[1:] = weights_i
    cov_weights = np.empty(len(transformed_sigma_points))
    cov_weights[0] = cov_weight_0
    cov_weights[1:] = weights_i

    mean = np.sum(mean_weights[:, np.newaxis] * transformed_sigma_points, axis=0)
    sigma_points_minus_mean = transformed_sigma_points - mean
    covariance = sigma_points_minus_mean.T.dot(np.diag(cov_weights)).dot(sigma_points_minus_mean)
    return MVN(mean=mean, covariance=covariance, random_state=random_state)


# TODO unit test
"""
mvn = MVN(mean=np.zeros(2), covariance=np.eye(2))
points = sigma_points(mvn, alpha=10, kappa=1000000000)
points[:, 1] *= 10
points += np.array([0.5, -3.0])
mvn = estimate_from_sigma_points(points, 2, alpha=10, kappa=1000000000)
print(mvn.mean, mvn.covariance)
exit()
#"""


def load_data(path):
    trajectory = pd.read_csv(path, sep=" ")
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
    return T, P

n_weights_per_dim = 5
n_weights = 2 * 6 * n_weights_per_dim
n_dims = 14

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"

alpha = 1e-3
kappa = 10.0

if os.path.exists("trajectories.txt"):
    trajectories = np.loadtxt("trajectories.txt")
else:
    print("Loading...")
    all_weights = []
    all_starts = []
    all_goals = []
    all_execution_times = []
    for idx, path in tqdm(list(enumerate(glob.glob(pattern)))):
        T, P = load_data(path)

        execution_time = T[-1]
        dt = np.mean(np.diff(T))
        if dt < 0.005:  # HACK
            continue
        dmp = DualCartesianDMP(
            execution_time=execution_time, dt=dt,
            n_weights_per_dim=n_weights_per_dim, int_dt=0.01, k_tracking_error=0.0)
        dmp.imitate(T, P)
        weights = dmp.get_weights()
        all_weights.append(weights)
        all_starts.append(P[0])
        all_goals.append(P[-1])
        all_execution_times.append(execution_time)

    all_parameters = np.vstack([
        np.hstack((w, s, g, e)) for w, s, g, e in zip(
            all_weights, all_starts, all_goals, all_execution_times)])
    random_state = np.random.RandomState(0)
    mvn = MVN(random_state=random_state)
    mvn.from_samples(all_parameters)

    weight_indices = np.arange(n_weights)
    start_indices = np.arange(n_weights, n_weights + n_dims)
    goal_indices = np.arange(n_weights + n_dims, n_weights + 2 * n_dims)
    execution_time_indices = np.arange(n_weights + 2 * n_dims, n_weights + 2 * n_dims + 1)
    execution_time = np.mean(all_execution_times)

    points = sigma_points(mvn, alpha=alpha, kappa=kappa)

    trajectories = []
    print("Propagating...")
    for i, parameters in tqdm(list(enumerate(points))):
        weights = parameters[weight_indices]
        start = parameters[start_indices]
        goal = parameters[goal_indices]
        dmp = DualCartesianDMP(
            execution_time=execution_time, dt=0.01,
            n_weights_per_dim=n_weights_per_dim, int_dt=0.01)
        dmp.configure(start_y=start, goal_y=goal)
        dmp.set_weights(weights)
        T, P = dmp.open_loop(run_t=execution_time)
        trajectories.append(P.ravel())

    trajectories = np.vstack(trajectories)
    np.savetxt("trajectories.txt", trajectories)

print(trajectories)
mvn = estimate_from_sigma_points(
    trajectories, n_weights + 2 * n_dims + 1, alpha=alpha, kappa=kappa)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

for idx, path in tqdm(list(enumerate(glob.glob(pattern)))):
    T, P = load_data(path)
P = mvn.mean.reshape(-1, 2 * 7)
sigma = np.sqrt(np.diag(mvn.covariance).reshape(-1, 2 * 7))
for d in range(3):
    ax = plt.subplot(3, 1, 1 + d)
    ax.fill_between(np.arange(len(P)), P[:, d] - sigma[:, d], P[:, d] + sigma[:, d], alpha=0.2)
plt.show()

exit()


fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()

class SelectDemo:
    def __init__(self, fig, left, right):
        self.fig = fig
        self.left = left
        self.right = right
        self.show = True

    def __call__(self, vis, key, modifier):  # sometimes we don't receive the correct keys, why?
        self.show = not self.show
        if self.show and modifier:
            for g in self.left.geometries + self.right.geometries:
                vis.remove_geometry(g)
        elif not self.show and not modifier:
            for g in self.left.geometries + self.right.geometries:
                vis.add_geometry(g)
        fig.view_init(azim=0, elev=25)
        return True

with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)

P = mvn.mean.reshape(-1, 2 * 7)
left = fig.plot_trajectory(P=P[:, :7], s=0.02)
right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
#fig.visualizer.register_key_action_callback(str("1"), SelectDemo(fig, left, right))

fig.view_init(azim=0, elev=25)
fig.show()
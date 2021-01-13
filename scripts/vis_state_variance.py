import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
from movement_primitives.dmp import DualCartesianDMP
from movement_primitives.plot import plot_trajectory_in_rows, plot_distribution_in_rows
from gmr import MVN
import matplotlib.pyplot as plt


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


def weight_space_to_state_space(pattern, n_weights_per_dim, random_state=np.random.RandomState(0), cache_filename=None, alpha=1e-3, kappa=10.0, verbose=0):
    if cache_filename is not None and os.path.exists(cache_filename):
        trajectories = np.loadtxt(cache_filename)
    else:
        mvn, mean_execution_time = estimate_parameter_distribution(pattern=pattern, n_weights_per_dim=n_weights_per_dim, random_state=random_state, verbose=verbose)
        trajectories = propagate_to_state_space(mvn=mvn, n_weights_per_dim=n_weights_per_dim, execution_time=mean_execution_time, alpha=alpha, kappa=kappa, verbose=verbose)

        if cache_filename is not None:
            np.savetxt(cache_filename, trajectories)

    #axes = plot_trajectory_in_rows(trajectories[0].reshape(-1, 14), subplot_shape=(7, 2))
    #trajectories2 = np.loadtxt("trajectories_backup.txt")
    #plot_trajectory_in_rows(trajectories2[0].reshape(-1, 14), axes=axes)
    #plt.show()

    return estimate_state_distribution(trajectories, alpha=alpha, kappa=kappa, n_weights_per_dim=n_weights_per_dim)


def estimate_parameter_distribution(pattern, n_weights_per_dim, random_state, verbose=0):
    if verbose:
        print("Load data and estimate DMP parameter distribution from dataset...")

    all_weights = []
    all_starts = []
    all_goals = []
    all_execution_times = []
    for idx, path in tqdm(list(enumerate(glob.glob(pattern)))):
        T, P = load_data(path)
        #plot_trajectory_in_rows(P, T, subplot_shape=(7, 2))
        #plt.show()

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

    mvn = MVN(random_state=random_state)
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
        #plot_trajectory_in_rows(P, T, subplot_shape=(7, 2))
        #plt.show()
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



pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"
n_weights_per_dim = 5
cache_filename = "trajectories.txt"
mvn = weight_space_to_state_space(
    pattern, n_weights_per_dim, cache_filename=cache_filename,
    alpha=1e-3, kappa=10.0, verbose=1)


def sample_trajectories(mvn, n_samples, n_dims):
    print("Sampling...")
    sampled_trajectories = mvn.sample(n_samples)
    n_steps = sampled_trajectories.shape[1] // n_dims
    return sampled_trajectories.reshape(n_samples, n_steps, n_dims)

#"""
from scipy.interpolate import interp1d


mean_trajectory = mvn.mean.reshape(-1, 2 * 7)
sigma = np.sqrt(np.diag(mvn.covariance).reshape(-1, 2 * 7))

n_samples = 100
n_dims = 2 * 7
sampled_trajectories = sample_trajectories(mvn, n_samples, n_dims)
n_steps = sampled_trajectories.shape[1]

normalized_length_trajectories = []
for idx, path in tqdm(list(enumerate(glob.glob(pattern)))):
    trajectory = load_data(path)[1]
    new_trajectory = np.empty((n_steps, trajectory.shape[1]))
    for d in range(trajectory.shape[1]):
        fun = interp1d(np.arange(len(trajectory)), trajectory[:, d])
        x = np.linspace(0, 1, n_steps) * (len(trajectory) - 1)
        new_trajectory[:, d] = fun(x)
    normalized_length_trajectories.append(new_trajectory)

axes = None
for i, trajectory in enumerate(sampled_trajectories):
    axes = plot_trajectory_in_rows(trajectory, label="sampled" if i == 0 else None, color="r", alpha=0.1, axes=axes, subplot_shape=(7, 2))
for i, trajectory in enumerate(normalized_length_trajectories):
    plot_trajectory_in_rows(trajectory, label="demos" if i == 0 else None, color="orange", alpha=1, axes=axes)
plot_distribution_in_rows(mean_trajectory, sigma, label="distribution", color="b", alpha=0.2, axes=axes)
axes[0].legend()
plt.show()
#"""

"""
import pytransform3d.visualizer as pv
from kinematics import Kinematics

with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="kuka_lbr/urdf/")
right_chain = kin.create_chain(
    ["kuka_lbr_r_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_r_tcp", verbose=0)
left_chain = kin.create_chain(
    ["kuka_lbr_l_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_l_tcp", verbose=0)


fig = pv.figure()
fig.plot_basis(s=0.1)

graph = fig.plot_graph(kin.tm, "kuka_lbr", show_visuals=True)


def animation_callback(
        step, left_chain, right_chain, graph, joint_trajectories,
        n_samples, n_steps):
    k = step // n_steps
    t = step % n_steps
    print("Demonstration %d/%d, step %d/%d" % (k + 1, n_samples, t + 1, n_steps))
    q_left = joint_trajectories[k][0][t]
    q_right = joint_trajectories[k][1][t]
    left_chain.forward(q_left)
    right_chain.forward(q_right)
    graph.set_data()
    return graph


joint_trajectories = np.empty((n_samples, 2, n_steps, 7))
random_state = np.random.RandomState(2)
print("Inverse kinematics...")
for k in tqdm(range(len(sampled_trajectories))):
    P = sampled_trajectories[k]
    #plot_trajectory_in_rows(P, subplot_shape=(7, 2))
    #plt.show()
    H_left = np.empty((len(P), 4, 4))
    H_right = np.empty((len(P), 4, 4))
    for t in range(len(P)):
        H_left[t] = pt.transform_from_pq(P[t, :7])
        H_right[t] = pt.transform_from_pq(P[t, 7:])
    Q_left = left_chain.inverse_trajectory(H_left, np.zeros(7), random_state=random_state)
    Q_right = right_chain.inverse_trajectory(H_right, np.zeros(7), random_state=random_state)
    joint_trajectories[k, 0] = Q_left
    joint_trajectories[k, 1] = Q_right

for P in sampled_trajectories:
    fig.plot_trajectory(P[:, :7], s=0.05, n_frames=3)
    fig.plot_trajectory(P[:, 7:], s=0.05, n_frames=3)

fig.view_init()
fig.animate(animation_callback, len(joint_trajectories) * len(joint_trajectories[0][0]), loop=True,
            fargs=(left_chain, right_chain, graph, joint_trajectories, n_samples, n_steps))
fig.show()
#"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
from pytransform3d import transformations as pt
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from dmp import DualCartesianDMP
from gmr import MVN


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

pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
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

    points = mvn.sigma_points(alpha=alpha, kappa=kappa)

    trajectories = []
    print("Propagating...")
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

    trajectories = np.vstack(trajectories)
    np.savetxt("trajectories.txt", trajectories)

n_features = n_weights + 2 * n_dims + 1
initial_mean = np.zeros(n_features)
initial_cov = np.eye(n_features)
mvn = MVN(initial_mean, initial_cov, random_state=42).estimate_from_sigma_points(
    trajectories, alpha=alpha, kappa=kappa)

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

mean_trajectory = mvn.mean.reshape(-1, 2 * 7)
sigma = np.sqrt(np.diag(mvn.covariance).reshape(-1, 2 * 7))
n_steps = len(mean_trajectory)

"""
all_trajectories = []
for idx, path in tqdm(list(enumerate(glob.glob(pattern)))):
    trajectory = load_data(path)[1]
    new_trajectory = np.empty((n_steps, trajectory.shape[1]))
    for d in range(trajectory.shape[1]):
        fun = interp1d(np.arange(len(trajectory)), trajectory[:, d])
        x = np.linspace(0, 1, n_steps) * (len(trajectory) - 1)
        new_trajectory[:, d] = fun(x)
    all_trajectories.append(new_trajectory)

sampled_trajectories = mvn.sample(100)
for d in range(3):
    ax = plt.subplot(3, 1, 1 + d)
    for f in [1, 2, 3]:
        ax.fill_between(
            np.arange(len(mean_trajectory)),
            mean_trajectory[:, d] - f * sigma[:, d],
            mean_trajectory[:, d] + f * sigma[:, d],
            color="b", alpha=0.1)
    for new_trajectory in all_trajectories:
        ax.plot(new_trajectory[:, d], color="orange")
    for trajectory in sampled_trajectories:
        P = trajectory.reshape(-1, 14)
        ax.plot(P[:, d], color="green", alpha=0.2)
    for t in range(0, n_steps, 10):
        mmvn = mvn.marginalize(np.array([2 * 7 * t + d]))
        std = np.sqrt(mmvn.covariance[0, 0])
        plt.scatter([t, t, t], [mmvn.mean[0] - 2 * std, mmvn.mean[0], mmvn.mean[0] + 2 * std], s=5, color="red")
plt.show()
#"""

#"""
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


def animation_callback(step, left_chain, right_chain, graph, joint_trajectories):
    k = step // len(joint_trajectories[0][0])
    t = step % len(joint_trajectories[0][0])
    print(k, t)
    q_left = joint_trajectories[k][0][t]
    q_right = joint_trajectories[k][1][t]
    left_chain.forward(q_left)
    right_chain.forward(q_right)
    graph.set_data()
    return graph


fig = pv.figure()
fig.plot_basis(s=0.1)

graph = fig.plot_graph(kin.tm, "kuka_lbr", show_visuals=True)

sampled_trajectories = mvn.sample(20)
sampled_trajectories = [trajectory.reshape(-1, 2 * 7) for trajectory in sampled_trajectories]
joint_trajectories = []
random_state = np.random.RandomState(2)
for k in range(len(sampled_trajectories)):
    print(k)
    P = sampled_trajectories[k]
    H_left = np.empty((len(P), 4, 4))
    H_right = np.empty((len(P), 4, 4))
    for t in range(len(P)):
        H_left[t] = pt.transform_from_pq(P[t, :7])
        H_right[t] = pt.transform_from_pq(P[t, 7:])
    Q_left = left_chain.inverse_trajectory(H_left, np.zeros(7), random_state=random_state)
    Q_right = right_chain.inverse_trajectory(H_right, np.zeros(7), random_state=random_state)
    joint_trajectories.append((Q_left, Q_right))


for P in sampled_trajectories:
    fig.plot_trajectory(P[:, :7], s=0.05, n_frames=3)
    fig.plot_trajectory(P[:, 7:], s=0.05, n_frames=3)

fig.view_init()
fig.animate(animation_callback, len(joint_trajectories) * len(joint_trajectories[0][0]), loop=True, fargs=(left_chain, right_chain, graph, joint_trajectories))
fig.show()
#"""

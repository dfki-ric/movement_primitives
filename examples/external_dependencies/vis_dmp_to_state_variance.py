import numpy as np
from tqdm import tqdm
from mocap.dataset_loader import load_kuka_dataset

from movement_primitives.dmp_to_state_space_distribution import propagate_weight_distribution_to_state_space
from movement_primitives.plot import plot_trajectory_in_rows, plot_distribution_in_rows
from pytransform3d import transformations as pt
import pytransform3d.visualizer as pv
import matplotlib.pyplot as plt
from movement_primitives.kinematics import Kinematics


pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"

n_weights_per_dim = 5
cache_filename = "trajectories.txt"
dataset = load_kuka_dataset(pattern, verbose=1)

mvn = propagate_weight_distribution_to_state_space(
    dataset, n_weights_per_dim, cache_filename=cache_filename,
    alpha=1e-3, kappa=10.0, verbose=1)


def sample_trajectories(mvn, n_samples, n_dims):
    print("Sampling...")
    sampled_trajectories = mvn.sample(n_samples)
    n_steps = sampled_trajectories.shape[1] // n_dims
    return sampled_trajectories.reshape(n_samples, n_steps, n_dims)


########################################################
# Compare demonstrations with state space distribution #
########################################################

mean_trajectory = mvn.mean.reshape(-1, 2 * 7)
sigma = np.sqrt(np.diag(mvn.covariance).reshape(-1, 2 * 7))

n_samples = 100
n_dims = 2 * 7
sampled_trajectories = sample_trajectories(mvn, n_samples, n_dims)
n_steps = sampled_trajectories.shape[1]

# normalize training data
normalized_length_trajectories = []
from scipy.interpolate import interp1d
for _, trajectory in tqdm(dataset):
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


################################################################
# 3D animation of sampled trajectories from state distribution #
################################################################

# IK is slow, reduce number of animated trajectories
n_animated_trajectories = 2

with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="kuka_lbr/urdf/")
right_chain = kin.create_chain(
    ["kuka_lbr_r_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_r_tcp", verbose=0)
left_chain = kin.create_chain(
    ["kuka_lbr_l_joint_%d" % i for i in range(1, 8)],
    "kuka_lbr", "kuka_lbr_l_tcp", verbose=0)

joint_trajectories = np.empty((n_animated_trajectories, 2, n_steps, 7))
random_state = np.random.RandomState(2)
print("Inverse kinematics...")
for k in tqdm(range(n_animated_trajectories)):
    P = sampled_trajectories[k]
    H_left = np.empty((len(P), 4, 4))
    H_right = np.empty((len(P), 4, 4))
    for t in range(len(P)):
        H_left[t] = pt.transform_from_pq(P[t, :7])
        H_right[t] = pt.transform_from_pq(P[t, 7:])
    Q_left = left_chain.inverse_trajectory(H_left, np.zeros(7), random_state=random_state)
    Q_right = right_chain.inverse_trajectory(H_right, np.zeros(7), random_state=random_state)
    joint_trajectories[k, 0] = Q_left
    joint_trajectories[k, 1] = Q_right


fig = pv.figure()
fig.plot_basis(s=0.1)
graph = fig.plot_graph(kin.tm, "kuka_lbr", show_visuals=True)
for P in sampled_trajectories:
    fig.plot_trajectory(P[:, :7], s=0.05, n_frames=3)
    fig.plot_trajectory(P[:, 7:], s=0.05, n_frames=3)
fig.view_init()


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


fig.animate(animation_callback, len(joint_trajectories) * len(joint_trajectories[0][0]), loop=True,
            fargs=(left_chain, right_chain, graph, joint_trajectories, n_samples, n_steps))
fig.show()

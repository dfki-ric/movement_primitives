import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pytransform3d import trajectories as ptr
from pytransform3d import transformations as pt
from pytransform3d.batch_rotations import smooth_quaternion_trajectory
from mocap.dataset_loader import load_kuka_dataset, transpose_dataset, smooth_dual_arm_trajectories_pq
from mocap.cleaning import median_filter
from gmr import GMM

from movement_primitives.promp import ProMP
from movement_primitives.plot import plot_distribution_in_rows, PALETTE


def generate_training_data(
        pattern, n_weights_per_dim, context_names, smooth_quaterions, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    if smooth_quaterions:
        smooth_dual_arm_trajectories_pq(Ps)

    Ps_left = []
    Ps_right = []
    Ps_diff = []
    print("Computing relative poses...")
    for P in tqdm.tqdm(Ps):
        P_left = P[:, :7]
        P_right = P[:, 7:]

        left2base = ptr.transforms_from_pqs(P_left)
        right2base = ptr.transforms_from_pqs(P_right)

        left2right = np.empty_like(left2base)
        for t in range(len(left2right)):
            left2right[t] = pt.concat(left2base[t], pt.invert_transform(right2base[t], check=False), check=False)
        P_diff = ptr.pqs_from_transforms(left2right)

        if smooth_quaterions:
            P_diff[:, 3:] = smooth_quaternion_trajectory(P_diff[:, 3:])
            P_diff[:, :] = median_filter(P_diff, window_size=5)

        Ps_left.append(P_left)
        Ps_right.append(P_right)
        Ps_diff.append(P_diff)

    n_demos = len(Ts)
    n_dims = Ps_left[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights_left = np.empty((n_demos, n_dims * n_weights_per_dim))
    weights_right = np.empty((n_demos, n_dims * n_weights_per_dim))
    weights_diff = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights_left[demo_idx] = promp.weights(Ts[demo_idx], Ps_left[demo_idx])
        weights_right[demo_idx] = promp.weights(Ts[demo_idx], Ps_right[demo_idx])
        weights_diff[demo_idx] = promp.weights(Ts[demo_idx], Ps_diff[demo_idx])

    return weights_left, weights_right, weights_diff, Ts, Ps_left, Ps_right, Ps_diff, contexts


plot_training_data = False
n_dims = 7
n_weights_per_dim = 10
random_state = np.random.RandomState(0)
# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"

weights_left, weights_right, weights_diff, Ts, Ps_left, Ps_right, Ps_diff, contexts = generate_training_data(
    pattern, n_weights_per_dim, context_names=context_names,
    smooth_quaterions=True, verbose=2)

gmm_left = GMM(n_components=3, random_state=random_state)
gmm_left.from_samples(np.hstack((contexts, weights_left)))
gmm_right = GMM(n_components=3, random_state=random_state)
gmm_right.from_samples(np.hstack((contexts, weights_right)))
gmm_diff = GMM(n_components=3, random_state=random_state)
gmm_diff.from_samples(np.hstack((contexts, weights_diff)))

n_validation_samples = 100
n_steps = 100
T_query = np.linspace(0, 1, n_steps)
context_indices = np.arange(len(context_names))

plt.figure()
axes = None
for panel_width, color, idx in zip([0.3, 0.4, 0.5], PALETTE[:3], range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))
    context = np.array([panel_width, 0.0, 1.0])

    cgmm_left = gmm_left.condition(context_indices, context).to_mvn()
    cpromp_left = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    cpromp_left.from_weight_distribution(cgmm_left.mean, cgmm_left.covariance)

    cgmm_right = gmm_right.condition(context_indices, context).to_mvn()
    cpromp_right = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    cpromp_right.from_weight_distribution(cgmm_right.mean, cgmm_right.covariance)

    cgmm_diff = gmm_diff.condition(context_indices, context).to_mvn()
    cpromp_diff = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    cpromp_diff.from_weight_distribution(cgmm_diff.mean, cgmm_diff.covariance)

    # mean and standard deviation in state space
    mean_left = cpromp_left.mean_trajectory(T_query)
    std_left = np.sqrt(cpromp_left.var_trajectory(T_query))

    mean_right = cpromp_right.mean_trajectory(T_query)
    std_right = np.sqrt(cpromp_right.var_trajectory(T_query))

    mean_diff = cpromp_diff.mean_trajectory(T_query)
    std_diff = np.sqrt(cpromp_diff.var_trajectory(T_query))

    mean = np.hstack((mean_left, mean_right, mean_diff))
    std = np.hstack((std_left, std_right, std_diff))

    axes = plot_distribution_in_rows(
        mean, std, T_query, axes=axes, subplot_shape=(7, 3),
        color=color, transpose=True, alpha=0.3, fill_between=True,
        std_factors=[2], label="panel width = %.2f" % panel_width)

    for i in range(mean.shape[1]):
        m = np.mean(mean[:, i])
        axes[i].set_ylim((m - 0.25, m + 0.25))

    axes[-1].legend(loc="upper left")

plt.show()

import numpy as np
import tqdm
from mocap.dataset_loader import load_kuka_dataset, transpose_dataset
from movement_primitives.dmp import DualCartesianDMP
from movement_primitives.dmp_to_state_space_distribution import propagate_to_state_space, estimate_state_distribution
from movement_primitives.visualization import plot_pointcloud, ToggleGeometry
from mocap.cleaning import median_filter
from gmr import GMM
from pytransform3d.batch_rotations import smooth_quaternion_trajectory
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager


def generate_training_data(
        pattern, n_weights_per_dim, context_names, smooth_quaterions, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    if smooth_quaterions:
        for P in Ps:
            P[:, 3:7] = smooth_quaternion_trajectory(P[:, 3:7])
            P[:, 10:] = smooth_quaternion_trajectory(P[:, 10:])
            P[:, :] = median_filter(P, window_size=5)

    n_demos = len(Ts)

    print("Computing params...")
    params = np.empty((n_demos, 2 * 6 * n_weights_per_dim + 2 * 7 + 2 * 7 + 1))
    exclude_indices = []
    for demo_idx in tqdm.tqdm(range(n_demos)):
        execution_time = Ts[demo_idx][-1]
        dt = np.mean(np.diff(Ts[demo_idx]))
        if dt < 0.005:  # HACK
            exclude_indices.append(demo_idx)
            if verbose:
                tqdm.tqdm.write("Excluding %d, dt = %g" % (demo_idx, dt))
        dmp = DualCartesianDMP(
            execution_time=execution_time, dt=dt,
            n_weights_per_dim=n_weights_per_dim, int_dt=0.01)
        dmp.imitate(Ts[demo_idx], Ps[demo_idx])
        params[demo_idx] = np.hstack((
            dmp.get_weights(), Ps[demo_idx][0], Ps[demo_idx][-1], execution_time))
        if np.any(np.isnan(params[demo_idx])):
            exclude_indices.append(demo_idx)
            if verbose:
                tqdm.tqdm.write("Excluding %d, NaN" % demo_idx)

    params = np.vstack([params[i] for i in range(n_demos) if i not in exclude_indices])
    Ts = [Ts[i] for i in range(n_demos) if i not in exclude_indices]
    Ps = [Ps[i] for i in range(n_demos) if i not in exclude_indices]
    contexts = np.vstack([contexts[i] for i in range(n_demos) if i not in exclude_indices])

    if verbose:
        print("Excluded samples: %s" % (exclude_indices,))

    return params, Ts, Ps, contexts


n_dims = 14
n_weights_per_dim = 10
alpha = 1e-3
kappa = 10.0
# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"
weights, Ts, Ps, contexts = generate_training_data(pattern, n_weights_per_dim, context_names, smooth_quaterions=True, verbose=1)
X = np.hstack((contexts, weights))

random_state = np.random.RandomState(0)
gmm = GMM(n_components=5, random_state=random_state)
gmm.from_samples(X)

n_validation_samples = 100

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(
    tm, "kuka_lbr", show_frames=True, show_visuals=True,
    whitelist=["kuka_lbr_l_tcp", "kuka_lbr_r_tcp"], s=0.2)


def sample_trajectories(mvn, n_samples, n_dims):
    print("Sampling...")
    sampled_trajectories = mvn.sample(n_samples)
    n_steps = sampled_trajectories.shape[1] // n_dims
    return sampled_trajectories.reshape(n_samples, n_steps, n_dims)


for panel_width, color, idx in zip([0.3, 0.4, 0.5], ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))
    context = np.array([panel_width, 0.0, 1.0])
    conditional_weight_distribution = gmm.condition(np.arange(len(context)), context).to_mvn()
    average_execution_time = np.mean([T[-1] for T in Ts])
    trajectories = propagate_to_state_space(conditional_weight_distribution, n_weights_per_dim, average_execution_time, alpha=alpha, kappa=kappa, verbose=1)
    conditional_state_distribution = estimate_state_distribution(trajectories, alpha=alpha, kappa=kappa, n_weights_per_dim=n_weights_per_dim)

    mean = conditional_state_distribution.mean.reshape(-1, 14)
    fig.plot_trajectory(mean[:, :7], s=0.05, c=color)
    fig.plot_trajectory(mean[:, 7:], s=0.05, c=color)

    n_samples = 100
    n_dims = 2 * 7
    sampled_trajectories = sample_trajectories(conditional_state_distribution, n_samples, n_dims)

    pcl_points = []
    distances = []
    stds = []
    for P in sampled_trajectories:
        # uncomment to check orientations
        #left = fig.plot_trajectory(P=P[:, :7], s=0.02)
        #right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
        pcl_points.extend(P[:, :3])
        pcl_points.extend(P[:, 7:10])

        ee_distances = np.linalg.norm(P[:, :3] - P[:, 7:10], axis=1)
        distances.append(np.mean(ee_distances))
        stds.append(np.std(ee_distances))
    print("Mean average distance of end-effectors = %.2f, mean std. dev. = %.3f"
          % (np.mean(distances), np.mean(stds)))

    pcl = plot_pointcloud(fig, pcl_points, color)
    key = ord(str((idx + 1) % 10))
    fig.visualizer.register_key_action_callback(key, ToggleGeometry(fig, pcl))

fig.show()

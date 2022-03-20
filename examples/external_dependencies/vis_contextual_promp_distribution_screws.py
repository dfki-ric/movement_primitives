import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.trajectories as ptr
from pytransform3d.urdf import UrdfTransformManager
from mocap.cleaning import smooth_exponential_coordinates, median_filter
from mocap.dataset_loader import load_kuka_dataset, transpose_dataset

from movement_primitives.visualization import plot_pointcloud, ToggleGeometry
from movement_primitives.promp import ProMP
from gmr import GMM


def generate_training_data(
        pattern, n_weights_per_dim, context_names, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    Es = []
    for P in Ps:
        E = np.empty((len(P), 2 * 6))
        E[:, :6] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, :7]))
        E[:, 6:] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, 7:]))
        E[:, :6] = smooth_exponential_coordinates(E[:, :6])
        E[:, 6:] = smooth_exponential_coordinates(E[:, 6:])
        E[:, :] = median_filter(E, 5)
        Es.append(E)

        #import matplotlib.pyplot as plt
        #from movement_primitives.plot import plot_trajectory_in_rows
        #plot_trajectory_in_rows(E, subplot_shape=(6, 2))
        #plt.show()

    n_demos = len(Ts)
    n_dims = Es[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights[demo_idx] = promp.weights(Ts[demo_idx], Es[demo_idx])

    return weights, Ts, Es, contexts


plot_training_data = False
n_dims = 12
n_weights_per_dim = 10
# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"

weights, Ts, Es, contexts = generate_training_data(
    pattern, n_weights_per_dim, context_names=context_names, verbose=2)
X = np.hstack((contexts, weights))

random_state = np.random.RandomState(0)
gmm = GMM(n_components=5, random_state=random_state)
gmm.from_samples(X)

n_steps = 100
T_query = np.linspace(0, 1, n_steps)

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)

for panel_width, color, idx in zip([0.3, 0.4, 0.5], ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))

    context = np.array([panel_width, 0.0, 1.0])
    conditional_weight_distribution = gmm.condition(np.arange(len(context)), context).to_mvn()
    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    promp.from_weight_distribution(
        conditional_weight_distribution.mean,
        conditional_weight_distribution.covariance)
    mean = promp.mean_trajectory(T_query)
    var = promp.var_trajectory(T_query)
    samples = promp.sample_trajectories(T_query, 100, random_state)

    c = [0, 0, 0]
    c[idx] = 1
    fig.plot_trajectory(ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(mean[:, :6])), s=0.05, c=tuple(c))
    fig.plot_trajectory(ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(mean[:, 6:])), s=0.05, c=tuple(c))

    pcl_points = []
    distances = []
    stds = []
    for E in samples:
        P_left = ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(E[:, :6]))
        P_right = ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(E[:, 6:]))
        left2base_ee_pos = P_left[:, :3]
        right2base_ee_pos = P_right[:, :3]
        pcl_points.extend(left2base_ee_pos)
        pcl_points.extend(right2base_ee_pos)

        ee_distances = np.linalg.norm(left2base_ee_pos - right2base_ee_pos, axis=1)
        distances.append(np.mean(ee_distances))
        stds.append(np.std(ee_distances))
    print("Mean average distance of end-effectors = %.2f, mean std. dev. = %.3f"
          % (np.mean(distances), np.mean(stds)))

    pcl = plot_pointcloud(fig, pcl_points, color)

    key = ord(str((idx + 1) % 10))
    fig.visualizer.register_key_action_callback(key, ToggleGeometry(fig, pcl))

if plot_training_data:
    for E in Es:
        left2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, :6])
        right2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, 6:])
        pv.Trajectory(left2base_trajectory, s=0.02).add_artist(fig)
        pv.Trajectory(right2base_trajectory, s=0.02).add_artist(fig)

fig.view_init(azim=0, elev=25)
fig.show()

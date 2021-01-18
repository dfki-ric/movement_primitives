import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from pytransform3d.urdf import UrdfTransformManager
from mocap.cleaning import smooth_quaternion_trajectory, median_filter

from movement_primitives.visualization import plot_pointcloud, ToggleGeometry
from movement_primitives.data import load_kuka_dataset, transpose_dataset
from movement_primitives.promp import ProMP
from gmr import GMM


def generate_training_data(
        pattern, n_weights_per_dim, context_names, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    Es = []
    for P in Ps:
        P[:, 3:7] = smooth_quaternion_trajectory(P[:, 3:7])
        P[:, 10:] = smooth_quaternion_trajectory(P[:, 10:])
        P[:, :] = median_filter(P, 5)

        E = np.empty((len(P), 2 * 6))
        #for t in range(len(P)):
        #    E[t, :6] = pt.exponential_coordinates_from_transform(pt.transform_from_pq(P[t, :7]))
        #    E[t, 6:] = pt.exponential_coordinates_from_transform(pt.transform_from_pq(P[t, 7:]))
        E[:, :6] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, :7]))
        E[:, 6:] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, 7:]))
        for t in range(len(E)):
            E[t, :6] = pt.norm_exponential_coordinates(E[t, :6])
            E[t, 6:] = pt.norm_exponential_coordinates(E[t, 6:])
        E[:, :] = median_filter(E, 5)
        Es.append(E)

        # TODO still a lot of discontinuities
        for i, j in zip(range(len(E) - 1), range(1, len(E))):
            d = np.linalg.norm(E[i, :6] - E[j, :6])
            if d > 0.5:
                print(E[i, :6])
                print(np.linalg.norm(E[i, :3]))
                print(E[j, :6])
                print(np.linalg.norm(E[j, :3]))
                print(pt.transform_from_exponential_coordinates(E[i, :6]))
                print(pt.transform_from_exponential_coordinates(E[j, :6]))
            d = np.linalg.norm(E[i, 6:] - E[j, 6:])
            if d > 0.5:
                print(E[i, 6:])
                print(np.linalg.norm(E[i, 6:9]))
                print(E[j, 6:])
                print(np.linalg.norm(E[j, 6:9]))
                print(pt.transform_from_exponential_coordinates(E[i, 6:]))
                print(pt.transform_from_exponential_coordinates(E[j, 6:]))

        #import matplotlib.pyplot as plt
        #from movement_primitives.plot import plot_trajectory_in_rows
        #plot_trajectory_in_rows(P, subplot_shape=(7, 2))
        #plt.suptitle("PQ")
        #plt.show()
        #plot_trajectory_in_rows(E, subplot_shape=(6, 2))
        #plt.suptitle("St")
        #plt.show()

    n_demos = len(Ts)
    n_dims = Es[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights[demo_idx] = promp.weights(Ts[demo_idx], Es[demo_idx])

    return weights, Ts, Es, contexts


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
    samples = promp.sample_trajectories(T_query, 100, random_state)

    pcl_points = []
    distances = []
    stds = []
    for E in samples:
        left2base_ee_pos = E[:, 3:6]
        right2base_ee_pos = E[:, 9:]
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

# plot training data
for E in Es:
    left2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, :6])
    right2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, 6:])
    pv.Trajectory(left2base_trajectory, s=0.02).add_artist(fig)
    pv.Trajectory(right2base_trajectory, s=0.02).add_artist(fig)

fig.view_init(azim=0, elev=25)
fig.show()

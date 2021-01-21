import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from gmr import GMM

from movement_primitives.visualization import plot_pointcloud, ToggleGeometry
from movement_primitives.data import load_kuka_dataset, transpose_dataset, smooth_dual_arm_trajectories_pq
from movement_primitives.promp import ProMP


def generate_training_data(
        pattern, n_weights_per_dim, context_names, smooth_quaterions, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    if smooth_quaterions:
        smooth_dual_arm_trajectories_pq(Ps)

    n_demos = len(Ts)
    n_dims = Ps[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights[demo_idx] = promp.weights(Ts[demo_idx], Ps[demo_idx])

    return weights, Ts, Ps, contexts


plot_training_data = False
n_dims = 14
n_weights_per_dim = 10
# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"

weights, Ts, Ps, contexts = generate_training_data(
    pattern, n_weights_per_dim, context_names=context_names,
    smooth_quaterions=True, verbose=2)
X = np.hstack((contexts, weights))

random_state = np.random.RandomState(0)
gmm = GMM(n_components=5, random_state=random_state)
gmm.from_samples(X)

n_validation_samples = 100
n_steps = 100
T_query = np.linspace(0, 1, n_steps)

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(
    tm, "kuka_lbr", show_frames=True, show_visuals=True,
    whitelist=["kuka_lbr_l_tcp", "kuka_lbr_r_tcp"], s=0.2)

for panel_width, color, idx in zip([0.3, 0.4, 0.5], ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))

    context = np.array([panel_width, 0.0, 1.0])
    conditional_weight_distribution = gmm.condition(np.arange(len(context)), context).to_mvn()
    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    promp.from_weight_distribution(
        conditional_weight_distribution.mean,
        conditional_weight_distribution.covariance)
    samples = promp.sample_trajectories(T_query, n_validation_samples, random_state)

    # mean and standard deviation in state space
    mean = promp.mean_trajectory(T_query)
    std = np.sqrt(promp.var_trajectory(T_query))

    c = [0, 0, 0]
    c[idx] = 1
    fig.plot_trajectory(mean[:, :7], s=0.05, c=tuple(c))
    fig.plot_trajectory(mean[:, 7:], s=0.05, c=tuple(c))
    #fig.plot_trajectory(mean[:, :7] + std[:, :7], s=0.02, c=tuple(c))
    #fig.plot_trajectory(mean[:, 7:] + std[:, :7], s=0.02, c=tuple(c))
    #fig.plot_trajectory(mean[:, :7] - std[:, :7], s=0.02, c=tuple(c))
    #fig.plot_trajectory(mean[:, 7:] - std[:, :7], s=0.02, c=tuple(c))

    pcl_points = []
    distances = []
    stds = []
    for P in samples:
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

if plot_training_data:
    for P in Ps:
        left = fig.plot_trajectory(P=P[:, :7], s=0.02)
        right = fig.plot_trajectory(P=P[:, 7:], s=0.02)

fig.view_init(azim=0, elev=25)
fig.show()

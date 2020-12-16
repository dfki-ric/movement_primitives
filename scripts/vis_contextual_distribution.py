import glob
import numpy as np
import pandas as pd
import open3d as o3d
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from promp import ProMP
from gmr import GMM


def load_data(path):
    trajectory = pd.read_csv(path, sep=" ")

    # omitted: "left_arm", "right_arm"
    context = trajectory[["panel_width", "clockwise", "counterclockwise"]].iloc[0].to_numpy()

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

    return T, P, context


n_dims = 14
n_weights_per_dim = 10

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"


Ts = []
Ps = []
contexts = []
for idx, path in enumerate(list(glob.glob(pattern))):
    print("Loading %s" % path)
    T, P, context = load_data(path)
    Ts.append(T)
    Ps.append(P)
    contexts.append(context)

print("All contexts:")
print(np.array(contexts).tolist())

promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
weights = np.empty((len(Ps), n_dims * n_weights_per_dim))
for demo_idx in range(len(Ps)):
    weights[demo_idx] = promp.weights(Ts[demo_idx], Ps[demo_idx])

X = np.hstack((contexts, weights))

random_state = np.random.RandomState(0)
gmm = GMM(n_components=3, random_state=random_state)
gmm.from_samples(X)

mean_n_steps = int(np.mean([len(P) for P in Ps]))
mean_T = np.linspace(0, 1, mean_n_steps)

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)


class SelectContext:
    def __init__(self, fig, pcl):
        self.fig = fig
        self.pcl = pcl
        self.show = True

    def __call__(self, vis, key, modifier):  # sometimes we don't receive the correct keys, why?
        self.show = not self.show
        if self.show and modifier:
            vis.remove_geometry(self.pcl)
        elif not self.show and not modifier:
            vis.add_geometry(self.pcl)
        fig.view_init(azim=0, elev=25)
        return True


for panel_width, color, idx in zip([0.3, 0.4, 0.5], ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))

    context = np.array([panel_width, 1.0, 0.0])
    conditional_weight_distribution = gmm.condition(np.arange(3), context).to_mvn()
    promp.from_weight_distribution(
        conditional_weight_distribution.mean,
        conditional_weight_distribution.covariance)
    samples = promp.sample_trajectories(mean_T, 1000, random_state)

    pcl_points = []
    for P in samples:
        pcl_points.extend(P[:, :3])
        pcl_points.extend(P[:, 7:10])

        ee_distances = np.linalg.norm(P[:, :3] - P[:, 7:10], axis=1)
        average_ee_distance = np.mean(ee_distances)
        print("Average distance = %.2f" % average_ee_distance)
        #left = fig.plot_trajectory(P=P[:, :7], s=0.02, c=color)
        #right = fig.plot_trajectory(P=P[:, 7:], s=0.02, c=color)

    pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcl_points)))
    pcl = pcl.uniform_down_sample(10)
    colors = o3d.utility.Vector3dVector([color for _ in range(len(pcl.points))])
    pcl.colors = colors
    fig.add_geometry(pcl)

    key = ord(str((idx + 1) % 10))
    fig.visualizer.register_key_action_callback(key, SelectContext(fig, pcl))

"""
# plot training data
for P in Ps:
    left = fig.plot_trajectory(P=P[:, :7], s=0.02)
    right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
"""

fig.view_init(azim=0, elev=25)
fig.show()
import glob
import numpy as np
import pandas as pd
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from promp import ProMP


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


fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)

n_weights_per_dim = 10

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"

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


# first 5 trajectories are from training set
Ts = []
Ps = []
for idx, path in enumerate(list(glob.glob(pattern))[:5]):
    print("Loading %s" % path)
    T, P = load_data(path)
    Ts.append(T)
    Ps.append(P)

    if idx < 5:
        left = fig.plot_trajectory(P=P[:, :7], s=0.02)
        right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
        key = ord(str((idx + 1) % 10))
        fig.visualizer.register_key_action_callback(key, SelectDemo(fig, left, right))

mean_n_steps = int(np.mean([len(P) for P in Ps]))
mean_T = np.linspace(0, 1, mean_n_steps)
promp = ProMP(n_dims=14, n_weights_per_dim=n_weights_per_dim)
promp.imitate(Ts, Ps)

random_state = np.random.RandomState(0)

# next 5 trajectories are sampled from ProMP
for idx in range(5):
    print("Reproduce #%d" % (idx + 1))

    P = promp.sample_trajectories(mean_T, 1, random_state)[0]

    left = fig.plot_trajectory(P=P[:, :7], s=0.02)
    right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
    key = ord(str((idx + 6) % 10))
    fig.visualizer.register_key_action_callback(key, SelectDemo(fig, left, right))

fig.view_init(azim=0, elev=25)
fig.show()
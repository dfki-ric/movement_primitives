import numpy as np
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from mocap.dataset_loader import load_kuka_dataset
from movement_primitives.promp import ProMP


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
                vis.remove_geometry(g, False)
        elif not self.show and not modifier:
            for g in self.left.geometries + self.right.geometries:
                vis.add_geometry(g, False)
        return True


n_weights_per_dim = 10

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"

dataset = load_kuka_dataset(pattern, verbose=1)
Ts = [entry[0] for entry in dataset[:5]]
Ps = [entry[1] for entry in dataset[:5]]

mean_n_steps = int(np.mean([len(P) for P in Ps]))
mean_T = np.linspace(0, 1, mean_n_steps)
promp = ProMP(n_dims=14, n_weights_per_dim=n_weights_per_dim)
promp.imitate(Ts, Ps)

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)

# sample trajectories from ProMP
random_state = np.random.RandomState(0)
for idx in range(10):
    print("Reproduce #%d" % (idx + 1))

    P = promp.sample_trajectories(mean_T, 1, random_state)[0]

    left = fig.plot_trajectory(P=P[:, :7], s=0.02)
    right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
    key = ord(str((idx + 1) % 10))
    fig.visualizer.register_key_action_callback(key, SelectDemo(fig, left, right))

fig.view_init(azim=0, elev=25)
fig.show()

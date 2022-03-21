import numpy as np
from mocap.dataset_loader import load_kuka_dataset
import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
from movement_primitives.dmp import DualCartesianDMP


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

dataset = load_kuka_dataset(pattern, verbose=1)

all_weights = []
all_starts = []
all_goals = []
all_execution_times = []

# first N trajectories are from training set
for idx in range(5):
    T, P = dataset[idx]

    execution_time = T[-1]
    dt = np.mean(np.diff(T))
    if dt < 0.005:  # HACK
        continue
    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=n_weights_per_dim, int_dt=0.01, p_gain=0.0)
    dmp.imitate(T, P)
    weights = dmp.get_weights()
    all_weights.append(weights)
    all_starts.append(P[0])
    all_goals.append(P[-1])
    all_execution_times.append(execution_time)

    if idx < 5:
        left = fig.plot_trajectory(P=P[:, :7], s=0.02)
        right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
        key = ord(str((idx + 1) % 10))
        fig.visualizer.register_key_action_callback(key, SelectDemo(fig, left, right))

all_parameters = np.vstack([
    np.hstack((w, s, g, e)) for w, s, g, e in zip(
        all_weights, all_starts, all_goals, all_execution_times)])
mean_parameters = np.mean(all_parameters, axis=0)
cov_parameters = np.cov(all_parameters, rowvar=False)
random_state = np.random.RandomState(0)

# next 5 trajectories are sampled from Gaussian DMP
for idx in range(5):
    print("Reproduce #%d" % (idx + 1))

    params = random_state.multivariate_normal(mean_parameters, cov_parameters)
    n_weights = 2 * 6 * n_weights_per_dim
    weights = params[:n_weights]
    start = params[n_weights:n_weights + 14]
    goal = params[n_weights + 14:n_weights + 2 * 14]
    execution_time = params[-1]

    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=0.01,
        n_weights_per_dim=n_weights_per_dim, int_dt=0.01)
    dmp.set_weights(weights)
    dmp.configure(start_y=start, goal_y=goal)
    T, P = dmp.open_loop(run_t=execution_time)

    left = fig.plot_trajectory(P=P[:, :7], s=0.02)
    right = fig.plot_trajectory(P=P[:, 7:], s=0.02)
    key = ord(str((idx + 6) % 10))
    fig.visualizer.register_key_action_callback(key, SelectDemo(fig, left, right))

fig.view_init(azim=0, elev=25)
fig.show()
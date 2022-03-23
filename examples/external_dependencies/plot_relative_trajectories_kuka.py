import numpy as np
import matplotlib.pyplot as plt
from mocap.dataset_loader import load_kuka_dataset
from movement_primitives.plot import plot_trajectory_in_rows
from mocap.cleaning import median_filter
from pytransform3d.batch_rotations import smooth_quaternion_trajectory
from pytransform3d import trajectories as ptr


n_weights_per_dim = 10

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/01_peg_in_hole_both_arms/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/01_heavy_load_no_tilt_0cm_dual_arm/*.csv"
#pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/panel_450mm_counterclockwise/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*counterclockwise/*.csv"

dataset = load_kuka_dataset(pattern, verbose=1)

plt.figure()
axes = None
for T, P in dataset:
    P_left = P[:, :7]
    P_right = P[:, 7:]

    P_left[:, 3:] = smooth_quaternion_trajectory(P_left[:, 3:])
    P_right[:, 3:] = smooth_quaternion_trajectory(P_right[:, 3:])
    P_left[:, :] = median_filter(P_left, window_size=5)
    P_right[:, :] = median_filter(P_right, window_size=5)

    left2base = ptr.transforms_from_pqs(P_left)
    right2base = ptr.transforms_from_pqs(P_right)

    left2right = np.empty_like(left2base)
    for t in range(len(left2right)):
        left2right[t] = left2base[t].dot(np.linalg.inv(right2base[t]))
    P_diff = ptr.pqs_from_transforms(left2right)

    P_diff[:, 3:] = smooth_quaternion_trajectory(P_diff[:, 3:])
    P_diff[:, :] = median_filter(P_diff, window_size=5)

    P = np.hstack((P_left, P_right, P_diff))

    axes = plot_trajectory_in_rows(P, T, axes=axes, subplot_shape=(7, 3), transpose=True)
plt.tight_layout()
plt.show()

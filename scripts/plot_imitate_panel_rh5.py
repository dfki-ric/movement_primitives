import numpy as np
import pandas as pd
from mocap.pandas_utils import match_columns, rename_stream_groups
from mocap.conversion import array_from_dataframe
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
import pytransform3d.trajectories as ptr
import pytransform3d.rotations as pr
from movement_primitives.dmp import DualCartesianDMP, CouplingTermDualCartesianTrajectory

path = "data/rh5/20200831-1541/csv_processed/dual_arm_anticlockwise/20200831-1541_2.csv"
trajectory = pd.read_csv(path, sep=" ")
patterns = ["time\.microseconds",
            "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.position\.data.*",
            "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation\.re.*",
            "rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation\.im.*",
            "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.position\.data.*",
            "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation\.re.*",
            "rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation\.im.*"]
columns = match_columns(trajectory, patterns)
trajectory = trajectory[columns]
group_rename = {
    "(time\.microseconds)": "Time",
    "(rh5_left_arm_posture_ctrl\.current_feedback\.pose\.position\.data).*": "left_pose",
    "(rh5_left_arm_posture_ctrl\.current_feedback\.pose\.orientation).*": "left_pose",
    "(rh5_right_arm_posture_ctrl\.current_feedback\.pose\.position\.data).*": "right_pose",
    "(rh5_right_arm_posture_ctrl\.current_feedback\.pose\.orientation).*": "right_pose"
}
trajectory = rename_stream_groups(trajectory, group_rename)
trajectory["Time"] = trajectory["Time"] / 1e6
trajectory["Time"] -= trajectory["Time"].iloc[0]
print(trajectory.head())

T = trajectory["Time"].to_numpy()
P = array_from_dataframe(
    trajectory,
    ["left_pose[0]", "left_pose[1]", "left_pose[2]", "left_pose.re", "left_pose.im[0]", "left_pose.im[1]", "left_pose.im[2]",
     "right_pose[0]", "right_pose[1]", "right_pose[2]", "right_pose.re", "right_pose.im[0]", "right_pose.im[1]", "right_pose.im[2]"])
execution_time = T[-1]
dt = np.mean(np.diff(T))

dmp = DualCartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=0.001, k_tracking_error=0.0)
dmp.imitate(T, P)
dmp.configure(start_y=P[0], goal_y=P[-1])
offset = np.array([-0.05, 0.05, 0, 1, 0, 0, 0])
ct = CouplingTermDualCartesianTrajectory(
    offset=offset, dt=dt, couple_position=True, couple_orientation=False,
    lf=(1.0, 0.0), k=1, c1=0.1, c2=100)  # c2=10000 in simulation
ct.imitate(T, P)
T2, P2 = dmp.open_loop(run_t=execution_time, coupling_term=ct)

ax = make_3d_axis(ax_s=1)
ax.set_xlim((0.4, 1))
ax.set_ylim((-0.3, 0.3))
ax.set_zlim((0, 0.6))
ptr.plot_trajectory(ax=ax, P=P[:, :7], s=0.02, alpha=0.2)
ptr.plot_trajectory(ax=ax, P=P[:, 7:], s=0.02, alpha=0.2)
ptr.plot_trajectory(ax=ax, P=P2[:, :7], s=0.01)
ptr.plot_trajectory(ax=ax, P=P2[:, 7:], s=0.01)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from pytransform3d.plot_utils import make_3d_axis
import pytransform3d.trajectories as ptr
from mocap.dataset_loader import load_rh5_demo
from movement_primitives.dmp import DualCartesianDMP, CouplingTermDualCartesianTrajectory


path = "data/rh5/20200831-1541/csv_processed/dual_arm_anticlockwise/20200831-1541_2.csv"
T, P = load_rh5_demo(path)
execution_time = T[-1]
dt = np.mean(np.diff(T))

dmp = DualCartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=0.001, p_gain=0.0)
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

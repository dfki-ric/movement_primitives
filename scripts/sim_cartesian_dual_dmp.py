import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DualCartesianDMP, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from movement_primitives.testing.simulation import RH5Simulation, draw_pose

dt = 0.0005
execution_time = 1.0
plot = False

#ct = CouplingTermDualCartesianDistance(desired_distance=0.2, lf=(1.0, 0.0), k=1, c1=0.1, c2=1000)
#ct = CouplingTermDualCartesianOrientation(desired_distance=np.deg2rad(25), lf=(1.0, 0.0), k=0.1)
desired_distance = pt.transform_from(  # right arm to left arm
    R=pr.active_matrix_from_intrinsic_euler_zyx([np.pi, 0, np.deg2rad(15)]),
    p=np.array([0.0, -0.3, 0.0])
)
ct = CouplingTermDualCartesianPose(desired_distance=desired_distance, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000)

rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)
q0 = np.array([-1.57, 0.76, 0, -1.3, 0, 0, -0.55, 1.57, -0.76, 0, 1.3, 0, 0, -0.55])
rh5.set_desired_joint_state(q0, position_control=True)
rh5.sim_loop(int(1.0 / dt))

ee_state = rh5.get_ee_state()
left2base_start = ee_state[:7]
right2base_start = ee_state[7:]
draw_pose(left2base_start, s=0.1, client_id=rh5.client_id)
draw_pose(right2base_start, s=0.1, client_id=rh5.client_id)

Y = np.empty((101, 14))
Y[:, :] = ee_state[np.newaxis]
T = np.linspace(0, 1, len(Y))
Y[:, 7] += 0.1 * np.sin(T * np.pi) - 0.1
Y[:, 8] -= 0.35 * np.sin(T * 2 * np.pi)
Y[:, 9] += 0.35 * np.cos(T * 2 * np.pi)


for coupling_term in [ct, None]:
    # TODO reset DMP properly
    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=dt, p_gain=0.0)
    dmp.imitate(T, Y)
    dmp.configure(start_y=Y[0], goal_y=Y[-1])

    #import time
    #if coupling_term is not None:
    #    time.sleep(10)
    rh5.goto_ee_state(Y[0])
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time, coupling_term=coupling_term)
    P = np.asarray(positions)
    dP = np.asarray(desired_positions)
    V = np.asarray(velocities)
    dV = np.asarray(desired_velocities)

    if plot:
        for subplot_idx, plot_dim in enumerate(range(14)):
            plt.subplot(3, 7, 1 + subplot_idx)
            plt.plot(T, Y[:, plot_dim], label="Demo %d" % plot_dim, c="k")
            plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]], c="k")
            plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim], label="Actual %d" % plot_dim, c="r")
            plt.scatter([[0, execution_time]], [[P[0, plot_dim], P[-1, plot_dim]]], c="r")
            plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
            plt.scatter([[0, execution_time]], [[dP[0, plot_dim], dP[-1, plot_dim]]], c="g")
            plt.ylim((min(P[:, plot_dim] - 0.05), max(P[:, plot_dim]) + 0.05))
        for subplot_idx, plot_dim in enumerate(range(7)):
            plt.subplot(3, 7, 15 + subplot_idx)
            plt.plot(T, Y[:, plot_dim] - Y[:, 7 + plot_dim], label="Demo %d" % plot_dim, c="k")
            plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim] - P[:, 7 + plot_dim], label="Actual %d" % plot_dim, c="r")
            plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim] - dP[:, 7 + plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
            D = P[:, plot_dim] - P[:, 7 + plot_dim]
            plt.ylim((min(D - 0.05), max(D) + 0.05))
        plt.legend()
        plt.show()

rh5.sim_loop()

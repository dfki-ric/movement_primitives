import numpy as np
import matplotlib.pyplot as plt
from dmp import DualCartesianDMP, CouplingTermDualCartesianDistance, CouplingTermDualCartesianOrientation, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
from simulation import RH5Simulation

dt = 0.0001
execution_time = 1.0

#ct = CouplingTermDualCartesianDistance(desired_distance=0.9, lf=(1.0, 0.0), k=1.0, c1=0.0, c2=1000)
#ct = CouplingTermDualCartesianOrientation(desired_distance=np.deg2rad(25), lf=(1.0, 0.0), k=0.1)
desired_distance = np.array([  # right arm to left arm
    [1.0, 0.0, 0.0, 0.8],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle([np.deg2rad(-1), 0, 0])
ct = CouplingTermDualCartesianPose(desired_distance=desired_distance, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000)

rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)

Y = np.zeros((1001, 14))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
Y[:, 0] = -0.5 + 0.15 * (sigmoid - 0.5)
Y[:, 1] = 0.55# + 0.15 * (sigmoid - 0.5) - 0.05
Y[:, 2] = 0.25
Y[:, 3] = 1.0
Y[:, 7] = 0.5 + 0.15 * (sigmoid - 0.5)
Y[:, 8] = 0.55# + 0.15 * (sigmoid - 0.5) - 0.05
Y[:, 9] = 0.25 - 0.05 - 0.1 * (sigmoid - 0.5)
Y[:, 10] = 1.0


for coupling_term in [ct, None]:
    # TODO reset DMP properly
    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=dt, k_tracking_error=0.0)
    dmp.imitate(T, Y)
    dmp.configure(start_y=Y[0], goal_y=Y[-1])

    #import time
    #if coupling_term is not None:
    #    time.sleep(20)
    rh5.goto_ee_state(Y[0])
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time, coupling_term=coupling_term)
    P = np.asarray(positions)
    dP = np.asarray(desired_positions)
    V = np.asarray(velocities)
    dV = np.asarray(desired_velocities)

    for subplot_idx, plot_dim in enumerate(range(14)):
        plt.subplot(3, 7, 1 + subplot_idx)
        plt.plot(T, Y[:, plot_dim], label="Demo %d" % plot_dim, c="k")
        plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]], c="k")
        plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim], label="Actual %d" % plot_dim, c="r")
        plt.scatter([[0, execution_time]], [[P[0, plot_dim], P[-1, plot_dim]]], c="r")
        plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
        plt.scatter([[0, execution_time]], [[dP[0, plot_dim], dP[-1, plot_dim]]], c="g")
        plt.ylim((min(P[:, plot_dim] - 0.05), max(P[:, plot_dim]) + 0.05))
        #T, Y = dmp.open_loop(run_t=2.0)
        #plt.plot(T, Y[:, plot_dim], label="Open loop %d" % plot_dim)
        #plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
    for subplot_idx, plot_dim in enumerate(range(7)):
        plt.subplot(3, 7, 15 + subplot_idx)
        plt.plot(T, Y[:, plot_dim] - Y[:, 7 + plot_dim], label="Demo %d" % plot_dim, c="k")
        plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim] - P[:, 7 + plot_dim], label="Actual %d" % plot_dim, c="r")
        plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim] - dP[:, 7 + plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
        D = P[:, plot_dim] - P[:, 7 + plot_dim]
        plt.ylim((min(D - 0.05), max(D) + 0.05))
        #T, Y = dmp.open_loop(run_t=2.0)
        #plt.plot(T, Y[:, plot_dim], label="Open loop %d" % plot_dim)
        #plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
    plt.legend()
    plt.show()

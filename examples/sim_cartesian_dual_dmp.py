import numpy as np
from dmp import DualCartesianDMP, CouplingTermDualCartesianDistance
from simulation import RH5Simulation

dt = 0.001
execution_time = 1.0

dmp = DualCartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=0.001, k_tracking_error=0.0)
coupling_term = CouplingTermDualCartesianDistance(desired_distance=0.5, lf=(1.0, 1.0), k=0.1)
rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)

Y = np.zeros((1001, 14))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
Y[:, 0] = -0.5 + 0.15 * (sigmoid - 0.5)
Y[:, 1] = 0.55
Y[:, 2] = 0.4
Y[:, 3] = 1.0
Y[:, 7] = 0.5 + 0.15 * (sigmoid - 0.5)
Y[:, 8] = 0.55
Y[:, 9] = 0.4
Y[:, 10] = 1.0
dmp.imitate(T, Y)
dmp.configure(start_y=Y[0], goal_y=Y[-1])

while True:
    rh5.set_desired_ee_state(Y[0], position_control=True)
    rh5.sim_loop(1000)
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time, coupling_term=coupling_term)
    import matplotlib.pyplot as plt
    P = np.asarray(positions)
    dP = np.asarray(desired_positions)
    V = np.asarray(velocities)
    dV = np.asarray(desired_velocities)

    plot_dims = [0, 7]
    for plot_dim in plot_dims:
        plt.plot(T, Y[:, plot_dim], label="Demo %d" % plot_dim)
        plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
        plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim], label="Actual %d" % plot_dim)
        plt.scatter([[0, execution_time]], [[P[0, plot_dim], P[-1, plot_dim]]])
        plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim], label="Desired %d" % plot_dim)
        plt.scatter([[0, execution_time]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
        T, Y = dmp.open_loop(run_t=2.0)
        plt.plot(T, Y[:, plot_dim], label="Open loop %d" % plot_dim)
        plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
    plt.legend()
    plt.show()

"""
=================================
Simulate a DMP in Cartesian Space
=================================

A normal DMP is used to represent a Cartesian trajectory given by positions
and quaternions.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP
from movement_primitives.testing.simulation import UR5Simulation


dt = 0.01
execution_time = 1.0

dmp = DMP(n_dims=7, execution_time=execution_time, dt=dt,
          n_weights_per_dim=10, int_dt=0.01, p_gain=2.5)
Y = np.zeros((1001, 7))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
Y[:, 0] = 0.6
Y[:, 1] = -0.2 + 0.4 * sigmoid + np.linspace(-0.4, 0.4, len(Y)) ** 2
Y[:, 2] = 0.45
Y[:, 4] = 1.0
dmp.imitate(T, Y)
dmp.configure(start_y=Y[0], goal_y=Y[-1])

ur5 = UR5Simulation(dt=dt, real_time=False)
for _ in range(4):
    ur5.goto_ee_state(Y[0], wait_time=1.0)
    ur5.stop()

desired_positions, positions, desired_velocities, velocities = \
    ur5.step_through_cartesian(dmp, Y[0], np.zeros(7), 4 * execution_time, closed_loop=True)

P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plot_dim = 1
plt.plot(Y[:, plot_dim], label="Demo")
plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.plot(P[:, plot_dim], label="Actual")
plt.scatter([[0, len(P)]], [[P[0, plot_dim], P[-1, plot_dim]]])
plt.plot(dP[:, plot_dim], label="Desired")
plt.scatter([[0, len(dP)]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
T, Y = dmp.open_loop(run_t=1.0)
plt.plot(Y[:, plot_dim], label="Open loop")
plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.legend()
plt.show()

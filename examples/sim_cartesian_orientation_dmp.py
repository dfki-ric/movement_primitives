"""
========================
Simulate a Cartesian DMP
========================

A Cartesian DMP is used to represent a Cartesian trajectory given by positions
and quaternions.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import CartesianDMP
from movement_primitives.testing.simulation import UR5Simulation
from pytransform3d import rotations as pr


dt = 0.001
execution_time = 1.0

dmp = CartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=0.0001)
Y = np.zeros((1001, 7))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
Y[:, 0] = 0.6
Y[:, 1] = -0.2 + 0.4 * sigmoid
Y[:, 2] = 0.45
start_aa = np.array([0.0, 1.0, 0.0, 0.25 * np.pi])
goal_aa = np.array([0.0, 0.0, 1.0, 0.25 * np.pi])
for t in range(len(Y)):
    frac = sigmoid[t]
    aa_t = (1.0 - frac) * start_aa + frac * goal_aa
    aa_t[:3] /= np.linalg.norm(aa_t[:3])
    Y[t, 3:] = pr.quaternion_from_axis_angle(aa_t)
dmp.imitate(T, Y, allow_final_velocity=True)
dmp.configure(start_y=Y[0], goal_y=Y[-1])

ur5 = UR5Simulation(dt=dt, real_time=False)
for _ in range(4):
    ur5.goto_ee_state(Y[0], wait_time=1.0)
    ur5.stop()

desired_positions, positions, desired_velocities, velocities = \
    ur5.step_through_cartesian(dmp, Y[0], np.zeros(6), execution_time)

P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plot_dim = 5
plt.plot(T, Y[:, plot_dim], label="Demo")
plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim], label="Actual")
plt.scatter([[0, execution_time]], [[P[0, plot_dim], P[-1, plot_dim]]])
plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim], label="Desired")
plt.scatter([[0, execution_time]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
T, Y = dmp.open_loop(run_t=2.0)
plt.plot(T, Y[:, plot_dim], label="Open loop")
plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.legend()
plt.show()

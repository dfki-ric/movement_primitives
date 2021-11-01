"""
=============================
Simulate Spring-Damper System
=============================

A spring-damper system is used to control the Cartesian pose of a robot arm.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.spring_damper import SpringDamper
from movement_primitives.testing.simulation import UR5Simulation


dt = 0.01

sd = SpringDamper(n_dims=7, k=2.0, c=None, dt=dt, int_dt=0.001)
start = np.array([0.6, 0.2, 0.45, 1.0, 0.0, 0.0, 0.0])
goal = np.array([0.6, 0.0, 0.45, 1.0, 0.0, 0.0, 0.0])
sd.configure(start_y=start, goal_y=goal)

ur5 = UR5Simulation(dt=dt, real_time=False)
for _ in range(4):
    ur5.goto_ee_state(start, wait_time=1.0)
    ur5.stop()

desired_positions, positions, desired_velocities, velocities = \
    ur5.step_through_cartesian(sd, start, np.zeros(7), 10.0, closed_loop=True)

P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plot_dim = 1
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.title("Position")
T, Y = sd.open_loop(run_t=10.0)
plt.plot(P[:, plot_dim], label="Actual")
plt.scatter([[0, len(P)]], [[P[0, plot_dim], P[-1, plot_dim]]])
plt.plot(dP[:, plot_dim], label="Desired")
plt.scatter([[0, len(dP)]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
plt.plot(Y[:, plot_dim], label="Open loop")
plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.legend()
plt.subplot(132)
plt.title("Velocity")
plt.plot(np.gradient(P[:, plot_dim]) / dt, label="Actual")
plt.plot(np.gradient(dP[:, plot_dim]) / dt, label="Desired")
plt.plot(np.gradient(Y[:, plot_dim]) / dt, label="Open loop")
plt.subplot(133)
plt.title("Acceleration")
plt.plot(np.gradient(np.gradient(P[:, plot_dim])) / dt ** 2, label="Actual")
plt.plot(np.gradient(np.gradient(dP[:, plot_dim])) / dt ** 2, label="Desired")
plt.plot(np.gradient(np.gradient(Y[:, plot_dim])) / dt ** 2, label="Open loop")
plt.tight_layout()
plt.show()

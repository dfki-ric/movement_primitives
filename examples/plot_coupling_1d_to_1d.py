"""
===================
Two Coupled 1D DMPs
===================

Two 1D DMPs are spatially coupled with a virtual spring that forces them to
keep a distance. One of them is the leader DMP and the other one is the
follower DMP.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermPos1DToPos1D


dt = 0.01
execution_time = 2.0
desired_distance = 0.5
dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=200)
coupling_term = CouplingTermPos1DToPos1D(
    desired_distance=desired_distance, lf=(1.0, 0.0), k=1.0)

T = np.linspace(0.0, execution_time, 101)
Y = np.empty((len(T), 2))
Y[:, 0] = np.cos(2.5 * np.pi * T)
Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
dmp.imitate(T, Y)

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(131)
ax1.set_title("Dimension 1")
ax1.set_ylim((-3, 3))
ax2 = fig.add_subplot(132)
ax2.set_title("Dimension 2")
ax2.set_ylim((-3, 3))
ax1.plot(T, Y[:, 0], label="Demo")
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.plot(T, Y[:, 1], label="Demo")
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop()
ax1.plot(T, Y[:, 0], label="Reproduction")
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.plot(T, Y[:, 1], label="Reproduction")
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop(coupling_term=coupling_term)
ax1.plot(T, Y[:, 0], label="Coupled 1")
ax2.plot(T, Y[:, 1], label="Coupled 2")
ax1.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
ax2.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

ax1.legend(loc="best")

ax = fig.add_subplot(133)
ax.set_title("Distance")
ax.set_ylim((-3, 3))
ax.plot(T, Y[:, 1] - Y[:, 0], label="Actual distance")
ax.plot([T[0], T[-1]], [desired_distance, desired_distance],
        label="Desired distance")

ax.legend(loc="best")

plt.show()

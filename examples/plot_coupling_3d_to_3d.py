"""
===================
Two Coupled 3D DMPs
===================

Two 3D DMPs are spatially coupled with a virtual spring that forces them to
keep a distance. One of them is the leader DMP and the other one is the
follower DMP.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermPos3DToPos3D


dt = 0.01

dmp = DMP(n_dims=6, execution_time=1.0, dt=dt, n_weights_per_dim=10,
          int_dt=0.0001)
ct = CouplingTermPos3DToPos3D(desired_distance=np.array([0.1, 0.5, 1.0]),
                              lf=(0.0, 1.0), k=1.0, c1=30.0, c2=100.0)

T = np.linspace(0.0, 1.0, 101)
Y = np.empty((len(T), 6))
Y[:, 0] = np.cos(np.pi * T)
Y[:, 1] = np.sin(np.pi * T)
Y[:, 2] = np.sin(2 * np.pi * T)
Y[:, 3] = np.cos(np.pi * T)
Y[:, 4] = np.sin(np.pi * T)
Y[:, 5] = 0.5 + np.sin(2 * np.pi * T)
dmp.imitate(T, Y)

fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(3, 2)
ax = fig.add_subplot(gs[:, 0], projection="3d")

ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Demo 1")
ax.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Demo 2")

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop()
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Reproduction 1")
ax.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Reproduction 2")

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop(coupling_term=ct)
ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], label="Coupled 1")
ax.plot(Y[:, 3], Y[:, 4], Y[:, 5], label="Coupled 2")

ax.legend(loc="best")

ax = fig.add_subplot(gs[0, 1])
ax.plot(T, Y[:, 3] - Y[:, 0], label="Actual distance (X)")
ax.plot([T[0], T[-1]], [ct.desired_distance[0], ct.desired_distance[0]],
        label="Desired distance (X)")

ax = fig.add_subplot(gs[1, 1])
ax.plot(T, Y[:, 4] - Y[:, 1], label="Actual distance (Y)")
ax.plot([T[0], T[-1]], [ct.desired_distance[1], ct.desired_distance[1]],
        label="Desired distance (Y)")

ax = fig.add_subplot(gs[2, 1])
ax.plot(T, Y[:, 5] - Y[:, 2], label="Actual distance (Z)")
ax.plot([T[0], T[-1]], [ct.desired_distance[2], ct.desired_distance[2]],
        label="Desired distance (Z)")

plt.show()

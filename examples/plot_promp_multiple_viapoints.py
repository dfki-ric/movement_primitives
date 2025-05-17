"""
===================================
Plot ProMP with Multiple Via Points
===================================

This example shows how to use ProMP with multiple via points.
"""
print(__doc__)

import numpy as np
from movement_primitives.promp import ProMP, via_points
from movement_primitives.data import generate_1d_trajectory_distribution
import matplotlib.pyplot as plt


n_demos = 100
n_steps = 101
T, Y = generate_1d_trajectory_distribution(n_demos, n_steps)
promp = ProMP(n_dims=1, n_weights_per_dim=50)
promp.imitate([T] * n_demos, Y)
Y_mean = promp.mean_trajectory(T)
Y_conf = 1.96 * np.sqrt(promp.var_trajectory(T))

y_cond = np.array([0.5, -0.5, 0.0, 1.0])
y_conditional_cov = np.zeros(4)
ts = np.array([0.2, 0.5, 0.7, 1.0])
cpromp = via_points(
    promp=promp,
    y_cond=y_cond,
    y_conditional_cov=y_conditional_cov,
    ts=ts,
)
Y_cmean = cpromp.mean_trajectory(T)
Y_cconf = 1.96 * np.sqrt(cpromp.var_trajectory(T))

plt.figure(figsize=(10, 5))

ax1 = plt.subplot(121)
ax1.set_title("Training set and ProMP")
ax1.fill_between(T, (Y_mean - Y_conf).ravel(), (Y_mean + Y_conf).ravel(), color="r", alpha=0.3)
ax1.plot(T, Y_mean, c="r", lw=2, label="ProMP")
ax1.plot(T, Y[:, :, 0].T, c="k", alpha=0.1)
ax1.set_xlim((-0.05, 1.05))
ax1.set_ylim((-2.5, 3))
ax1.legend(loc="best")

ax2 = plt.subplot(122)
ax2.set_title("Conditioned ProMP")
ax2.scatter(ts, y_cond, marker="*", s=100, c="b", label="Viapoints")
ax2.fill_between(T, (Y_cmean - Y_cconf).ravel(), (Y_cmean + Y_cconf).ravel(), color="b", alpha=0.3)
ax2.plot(T, Y_cmean, c="b", lw=2, label="Conditioned ProMP")
ax2.set_xlim((-0.05, 1.05))
ax2.set_ylim((-2.5, 3))
ax2.legend(loc="best")

ax1.set_xlabel("Time $t$ [s]")
ax1.set_ylabel("Position $y$ [m]")
ax2.set_xlabel("Time $t$ [s]")
plt.tight_layout()
plt.show()

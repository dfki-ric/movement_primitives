"""
=================
Conditional ProMP
=================

Probabilistic Movement Primitives (ProMPs) define distributions over
trajectories that can be conditioned on viapoints. In this example, we
plot the resulting posterior distribution after conditioning on varying
start positions.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.data import generate_1d_trajectory_distribution
from movement_primitives.promp import ProMP


n_demos = 100
n_steps = 101
T, Y = generate_1d_trajectory_distribution(n_demos, n_steps)
y_conditional_cov = np.array([0.025])
promp = ProMP(n_dims=1, n_weights_per_dim=10)
promp.imitate([T] * n_demos, Y)
Y_mean = promp.mean_trajectory(T)
Y_conf = 1.96 * np.sqrt(promp.var_trajectory(T))

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
ax2.set_title("Conditional ProMPs")

for color, y_cond in zip("rgbcmyk", np.linspace(-1, 2.5, 7)):
    cpromp = promp.condition_position(np.array([y_cond]), y_cov=y_conditional_cov, t=0.0, t_max=1.0)
    Y_cmean = cpromp.mean_trajectory(T)
    Y_cconf = 1.96 * np.sqrt(cpromp.var_trajectory(T))

    ax2.scatter([0], [y_cond], marker="*", s=100, c=color, label="$y_0 = %.2f$" % y_cond)
    ax2.fill_between(T, (Y_cmean - Y_cconf).ravel(), (Y_cmean + Y_cconf).ravel(), color=color, alpha=0.3)
    ax2.plot(T, Y_cmean, c=color, lw=2)
    ax2.set_xlim((-0.05, 1.05))
    ax2.set_ylim((-2.5, 3))
    ax2.legend(loc="best")

ax1.set_xlabel("Time $t$ [s]")
ax1.set_ylabel("Position $y$ [m]")
ax2.set_xlabel("Time $t$ [s]")
plt.tight_layout()
plt.show()

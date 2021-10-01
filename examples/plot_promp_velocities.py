"""
=====================
Velocities of a ProMP
=====================

ProMPs can generate positions and velocities of a trajectory.
"""
print(__doc__)


import numpy as np
from movement_primitives.promp import ProMP
import matplotlib.pyplot as plt


n_weights_per_dim = 100

promp = ProMP(n_dims=1, n_weights_per_dim=n_weights_per_dim)

random_state = np.random.RandomState(10)
n_demos = 10
n_steps = 101
T = np.empty((n_demos, n_steps))
T[:, :] = np.linspace(0.0, 1.0, n_steps)
Y = np.empty((n_demos, n_steps, 1))
for demo_idx in range(n_demos):
    Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
    Y[demo_idx, :, 0] += random_state.randn(n_steps) * 0.01
promp.imitate(T, Y, verbose=1)

random_state = np.random.RandomState(0)
samples = promp.sample_trajectories(T[0], 10, random_state)

mean_trajectory = promp.mean_trajectory(T[0])
std_trajectory = np.sqrt(promp.var_trajectory(T[0]))

mean_velocities = promp.mean_velocities(T[0])
std_velocities = np.sqrt(promp.var_velocities(T[0]))

dt = T[0, 1] - T[0, 0]
int_velocities = mean_trajectory[0] + np.cumsum(mean_velocities * dt)

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(121)
for demo_idx in range(n_demos):
    label = "Demo" if demo_idx == 0 else None
    ax.plot(T[demo_idx, :], Y[demo_idx, :], c="k", alpha=0.1, label=label)
ax.plot(T[0], mean_trajectory, label="Reproduction", c="r", lw=3)
ax.plot(T[0], int_velocities, label="Integrated velocities", c="b")
factor = 2
ax.fill_between(
    T[0],
    mean_trajectory[:, 0] - factor * std_trajectory[:, 0],
    mean_trajectory[:, 0] + factor * std_trajectory[:, 0],
    alpha=0.3, label="1-$\sigma$ interval")
for sample in samples:
    ax.plot(T[0], sample, c="g", alpha=0.3)
ax.legend(loc="best")

ax = fig.add_subplot(122)
ax.plot(T[0], mean_velocities, label="Velocities", c="r", lw=3)
factor = 2
ax.fill_between(
    T[0],
    mean_velocities[:, 0] - factor * std_velocities[:, 0],
    mean_velocities[:, 0] + factor * std_velocities[:, 0],
    alpha=0.3, label="1-$\sigma$ interval")
for i, sample in enumerate(samples):
    label = "Samples" if i == 0 else None
    ax.plot(T[0], np.gradient(sample, axis=0) * n_steps, c="g", alpha=0.3,
            label=label)
ax.legend(loc="best")

plt.show()

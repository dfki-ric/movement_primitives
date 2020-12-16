import matplotlib.pyplot as plt
import numpy as np
from promp import ProMP


n_weights_per_dim = 50

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

for demo_idx in range(n_demos):
    plt.plot(T[demo_idx, :], Y[demo_idx, :], c="k", alpha=0.1)

random_state = np.random.RandomState(0)
samples = promp.sample_trajectories(T[0], 10, random_state)

mean_trajectory = promp.mean_trajectory(T[0])
plt.plot(T[0], mean_trajectory, label="Reproduction", c="r", lw=3)
var_trajectory = np.sqrt(promp.var_trajectory(T[0]))
factor = 2
plt.fill_between(
    T[0],
    mean_trajectory[:, 0] - factor * var_trajectory[:, 0],
    mean_trajectory[:, 0] + factor * var_trajectory[:, 0],
    alpha=0.3)
for sample in samples:
    plt.plot(T[0], sample, c="g", alpha=0.3)

plt.show()

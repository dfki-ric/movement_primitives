import matplotlib.pyplot as plt
import numpy as np
from promp import ProMP


n_weights_per_dim = 15

promp = ProMP(n_dims=2, n_weights_per_dim=n_weights_per_dim)

random_state = np.random.RandomState(10)
n_demos = 20
n_steps = 51
T = np.empty((n_demos, n_steps))
T[:, :] = np.linspace(0.0, 1.0, n_steps)
Ys = np.empty((n_demos, n_steps, 2))
for demo_idx in range(n_demos):
    Ys[demo_idx, :, 0] = np.sin(2 * np.pi * T[demo_idx] + random_state.randn() * 0.3)
    Ys[demo_idx, :, 1] = 0.5 + np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.5)
promp.imitate(T, Ys, verbose=1)

plt.subplot(121)
for demo_idx in range(n_demos):
    plt.plot(Ys[demo_idx, :, 0], Ys[demo_idx, :, 1], c="k", alpha=0.3)

mean_trajectory = promp.mean_trajectory(T[0])
plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], label="Reproduction", c="r", lw=3)
plt.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1], c="r")

random_state = np.random.RandomState(0)
samples = promp.sample_trajectories(T[0], 20, random_state)

for sample in samples:
    plt.plot(sample[:, 0], sample[:, 1], c="g", alpha=0.3)

plt.legend(loc="best")

plt.subplot(122)
var_trajectory = np.sqrt(promp.var_trajectory(T[0]))
factor = 2
plt.fill_between(
    T[0],
    mean_trajectory[:, 0] - factor * var_trajectory[:, 0],
    mean_trajectory[:, 0] + factor * var_trajectory[:, 0],
    color="orange", alpha=0.5)
plt.plot(T[0], mean_trajectory[:, 0], color="orange")
for Y in Ys:
    plt.plot(T[0], Y[:, 0], c="k", alpha=0.3, ls="--")
for sample in samples:
    plt.plot(T[0], sample[:, 0], c="blue", alpha=0.3)
plt.fill_between(
    T[0],
    mean_trajectory[:, 1] - factor * var_trajectory[:, 1],
    mean_trajectory[:, 1] + factor * var_trajectory[:, 1],
    color="blue", alpha=0.5)
plt.plot(T[0], mean_trajectory[:, 1], color="blue")
for Y in Ys:
    plt.plot(T[0], Y[:, 1], c="k", alpha=0.3, ls="--")
for sample in samples:
    plt.plot(T[0], sample[:, 1], c="orange", alpha=0.3)

plt.show()

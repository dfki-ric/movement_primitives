import matplotlib.pyplot as plt
import numpy as np
from promp import ProMP


start_t = 0.0
goal_t = 1.0
dt = 0.01

promp = ProMP(n_dims=2, execution_time=1.0, dt=0.01, n_weights_per_dim=10)

random_state = np.random.RandomState(10)
n_demos = 10
n_steps = 101
T = np.empty((n_demos, n_steps))
T[:, :] = np.linspace(0.0, 1.0, n_steps)
Y = np.empty((n_demos, n_steps, 2))
for demo_idx in range(n_demos):
    Y[demo_idx, :, 0] = np.cos(2.5 * np.pi * T[demo_idx]) + random_state.randn(n_steps) * 0.05
    Y[demo_idx, :, 1] = 0.5 + np.cos(1.5 * np.pi * T[demo_idx]) + random_state.randn(n_steps) * 0.05
promp.imitate_scmtl(T, Y, verbose=1)

for demo_idx in range(n_demos):
    plt.plot(Y[demo_idx, :, 0], Y[demo_idx, :, 1], c="k", alpha=0.3)

mean_trajectory = promp.mean_trajectory(T[0])
plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], label="Reproduction", c="r", lw=3)
plt.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1], label="Reproduction")

random_state = np.random.RandomState(0)
samples = promp.sample_trajectories(T[0], 100, random_state)

for sample in samples:
    plt.plot(sample[:, 0], sample[:, 1], c="g", alpha=0.05)

plt.legend(loc="best")
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from promp import ProMP


start_t = 0.0
goal_t = 1.0
dt = 0.01

dmp = ProMP(n_dims=1, execution_time=1.0, dt=0.01, n_weights_per_dim=5)

random_state = np.random.RandomState(10)
n_demos = 10
n_steps = 11
T = np.empty((n_demos, n_steps))
T[:, :] = np.linspace(0.0, 1.0, n_steps)
Y = np.empty((n_demos, n_steps, 1))
for demo_idx in range(n_demos):
    Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
dmp.imitate(T, Y)

activations = dmp._rbfs(T[0])
mean = dmp.weight_mean.dot(activations)

for demo_idx in range(n_demos):
    plt.plot(T[demo_idx, :], Y[demo_idx, :])

plt.plot(T[0], mean, label="Reproduction")

plt.show()

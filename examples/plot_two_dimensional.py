import matplotlib.pyplot as plt
import numpy as np
from dmp import DMP


start_t = 0.0
goal_t = 1.0
dt = 0.01

dmp = DMP(n_dims=2, execution_time=1.0, dt=0.01, n_weights_per_dim=10)

T = np.linspace(0.0, 1.0, 101)
Y = np.empty((len(T), 2))
Y[:, 0] = np.cos(2.5 * np.pi * T)
Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
dmp.imitate(T, Y)

plt.scatter(Y[:, 0], Y[:, 1], label="Demo")

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop()
plt.scatter(Y[:, 0], Y[:, 1], label="Reproduction")

dmp.configure(start_y=np.array([1.0, 1.5]), goal_y=np.array([0.2, 0.3]))
T, Y = dmp.open_loop(run_t=1.0)
plt.scatter(Y[:, 0], Y[:, 1], label="Adaptation")

plt.legend(loc="best")
plt.show()
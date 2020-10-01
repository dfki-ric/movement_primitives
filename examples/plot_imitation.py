import matplotlib.pyplot as plt
import numpy as np
from dmp import DMP


start_t = 0.0
goal_t = 1.0
dt = 0.01

dmp = DMP(n_dims=1, execution_time=1.0, dt=0.01, n_weights_per_dim=10)

T = np.linspace(0.0, 1.0, 101)
Y = np.cos(2 * np.pi * T)[:, np.newaxis]
dmp.imitate(T, Y)

plt.plot(T, Y, label="Demo")
plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop()
plt.plot(T, Y, label="Reproduction")
plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

dmp.configure(start_y=np.array([1.0]), goal_y=np.array([0.5]))
T, Y = dmp.open_loop(run_t=2.0)
plt.plot(T, Y, label="Adaptation")
plt.scatter([start_t, goal_t], [1.0, 0.5])

plt.legend(loc="best")
plt.show()
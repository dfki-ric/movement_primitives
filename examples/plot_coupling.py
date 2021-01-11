import matplotlib.pyplot as plt
import numpy as np
from movement_primitives import DMP, CouplingTerm

dt = 0.01

execution_time = 2.0
dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=200)

T = np.linspace(0.0, execution_time, 101)
Y = np.empty((len(T), 2))
Y[:, 0] = np.cos(2.5 * np.pi * T)
Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
dmp.imitate(T, Y)

plt.plot(T, Y, label="Demo")
plt.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
plt.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop()
plt.plot(T, Y, label="Reproduction")
plt.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
plt.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

dmp.configure(start_y=Y[0], goal_y=Y[-1])
T, Y = dmp.open_loop(coupling_term=CouplingTerm(desired_distance=0.5, lf=(1.0, 0.0), k=0.01))
plt.plot(T, Y[:, 0], label="Coupled 1")
plt.plot(T, Y[:, 1], label="Coupled 2")
plt.scatter([T[0], T[-1]], [Y[0, 0], Y[-1, 0]])
plt.scatter([T[0], T[-1]], [Y[0, 1], Y[-1, 1]])

plt.legend(loc="best")
plt.show()

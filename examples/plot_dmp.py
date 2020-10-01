import matplotlib.pyplot as plt
import numpy as np
from dmp import DMP


start_y = np.array([0.0])
goal_y = np.array([1.0])
dt = 0.001

dmp = DMP(n_dims=1, execution_time=1.0, dt=0.01, n_weights_per_dim=6)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.forcing_term.weights = 1000 * np.random.randn(*dmp.forcing_term.weights.shape)

T, Y = dmp.open_loop()

plt.plot(T, Y)
plt.show()

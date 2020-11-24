import matplotlib.pyplot as plt
import numpy as np
from dmp import StateFollowingDMP


start_y = np.array([0.0])
goal_y = np.array([1.0])
dt = 0.001

dmp = StateFollowingDMP(n_dims=1, execution_time=1.0, dt=0.01, n_viapoints=10)
dmp.configure(start_y=start_y, goal_y=goal_y)

T, Y = dmp.open_loop()

plt.plot(T, Y)
plt.show()

import numpy as np
from movement_primitives import DualCartesianDMP
import timeit


start_y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
goal_y = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
dt = 0.001
int_dt = 0.0001

dmp = DualCartesianDMP(execution_time=1.0, dt=dt, n_weights_per_dim=6, int_dt=int_dt)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.forcing_term.weights = 1000 * np.random.randn(*dmp.forcing_term.weights.shape)

times = timeit.repeat(dmp.open_loop, repeat=10, number=1)
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))
# Pure python
# Mean: 0.58188; Std. dev.: 0.00225
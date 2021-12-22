from functools import partial
import numpy as np
from movement_primitives.dmp import DualCartesianDMP
import timeit


start_y = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
goal_y = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
dt = 0.01
int_dt = 0.001

dmp = DualCartesianDMP(execution_time=1.0, dt=dt, n_weights_per_dim=6, int_dt=int_dt)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.set_weights(1000 * np.random.randn(*dmp.get_weights().shape))

times = timeit.repeat(partial(dmp.open_loop, step_function="cython"), repeat=10, number=1)
print("Cython")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

times = timeit.repeat(partial(dmp.open_loop, step_function="python"), repeat=10, number=1)
print("Python")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

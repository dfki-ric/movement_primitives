import numpy as np
from movement_primitives.dmp import DMP
from functools import partial
import timeit


n_dims = 10
start_y = np.zeros(n_dims)
goal_y = np.ones(n_dims)
dt = 0.001
int_dt = 0.0001

dmp = DMP(n_dims=n_dims, execution_time=1.0, dt=dt, n_weights_per_dim=6, int_dt=int_dt)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.set_weights(1000 * np.random.randn(*dmp.get_weights().shape))

times = timeit.repeat(partial(dmp.open_loop, step_function="rk4"), repeat=10, number=1)
print("RK4")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

times = timeit.repeat(partial(dmp.open_loop, step_function="euler"), repeat=10, number=1)
print("Euler (Python)")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

times = timeit.repeat(partial(dmp.open_loop, step_function="euler-cython"), repeat=10, number=1)
print("Euler (Cython)")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

times = timeit.repeat(partial(dmp.open_loop, step_function="rk4-cython"), repeat=10, number=1)
print("RK4 (Cython)")
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

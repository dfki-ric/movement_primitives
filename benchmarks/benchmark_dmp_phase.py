from functools import partial
import numpy as np
from movement_primitives.dmp._canonical_system import canonical_system_alpha
from movement_primitives.dmp._canonical_system import phase as phase_python
from movement_primitives.dmp_fast import phase as phase_cython
import timeit


goal_t = 1.0
start_t = 0.0
int_dt = 0.001
alpha = canonical_system_alpha(0.01, goal_t, start_t, int_dt)
times = timeit.repeat(partial(phase_python, 0.5, alpha, goal_t, start_t, int_dt), repeat=1000, number=1000)
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))
times = timeit.repeat(partial(phase_cython, 0.5, alpha, goal_t, start_t, int_dt), repeat=1000, number=1000)
print("Mean: %.5f; Std. dev.: %.5f" % (np.mean(times), np.std(times)))

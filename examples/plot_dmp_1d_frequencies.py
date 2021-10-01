"""
======================
Change DMP frequencies
======================

In this example, we modify the size of a time step in a DMP and compare its
effect.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP

start_y = np.array([0.0])
goal_y = np.array([1.0])
execution_time = 1.0

dmp = DMP(n_dims=1, execution_time=execution_time)
dmp.configure(start_y=start_y, goal_y=goal_y)

plt.figure(figsize=(5, 8))
ax = plt.subplot(111)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
for dt in [0.1, 0.075, 0.01]:
    # modify dt after DMP has been created
    dmp.dt_ = dt
    T, Y = dmp.open_loop(run_t=0.3)
    ax.scatter(T, Y, s=5)
    ax.plot(T, Y, label="dt = %g" % dt)
ax.legend()
plt.show()

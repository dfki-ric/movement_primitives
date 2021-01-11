import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.spring_damper import SpringDamper


start_y = np.array([0.0])
goal_y = np.array([1.0])
dt = 0.01

for k in [5.0, 10.0, 50.0]:
    for c_factor in [0.5, 1.0, 2.0]:
        c = c_factor * 2.0 * np.sqrt(k)
        sd = SpringDamper(n_dims=1, dt=0.01, k=k, c=c, int_dt=0.001)
        sd.configure(start_y=start_y, goal_y=goal_y)
        T, Y = sd.open_loop(run_t=10.0)
        plt.plot(T, Y, label="k = %d, c = %.1f" % (k, c))
plt.plot([0.0, 10.0], [goal_y, goal_y])
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity


dt = 0.01
execution_time = 1.0
T = np.arange(0, execution_time + dt, dt)
Y = np.cos(np.pi * T)[:, np.newaxis]

dmp = DMPWithFinalVelocity(n_dims=1, execution_time=execution_time)
dmp.imitate(T, Y)

plt.figure(figsize=(8, 5))
ax1 = plt.subplot(121)
ax1.set_xlabel("Time")
ax1.set_ylabel("Position")
ax2 = plt.subplot(122)
ax2.set_xlabel("Time")
ax2.set_ylabel("Velocity")
ax1.plot(T, Y, label="Demo")
ax2.plot(T, np.gradient(Y, axis=0) / dmp.dt_)
ax2.scatter([T[-1]], (Y[-1] - Y[-2]) / dmp.dt_)
for goal_yd in [0.0, 1.0, 2.0]:
    dmp.configure(goal_yd=goal_yd)
    T, Y = dmp.open_loop(run_t=1)
    ax1.plot(T, Y, label="goal_yd = %g" % goal_yd)
    ax2.plot(T, np.gradient(Y, axis=0) / dmp.dt_)
    ax2.scatter([T[-1]], [goal_yd])
ax1.legend()
plt.show()

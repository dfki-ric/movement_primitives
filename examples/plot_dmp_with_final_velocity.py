"""
=======================
DMP with Final Velocity
=======================

Not all DMPs allow a final velocity > 0. In this case we analyze the effect
of changing final velocities in an appropriate variation of the DMP
formulation that allows to set the final velocity.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMPWithFinalVelocity


dt = 0.01
execution_time = 1.0
T = np.arange(0, execution_time + dt, dt)
Y = np.column_stack((np.cos(np.pi * T), -np.cos(np.pi * T)))

dmp = DMPWithFinalVelocity(n_dims=2, execution_time=execution_time)
dmp.imitate(T, Y)

plt.figure(figsize=(10, 8))
ax1 = plt.subplot(221)
ax1.set_title("Dimension 1")
ax1.set_xlabel("Time")
ax1.set_ylabel("Position")
ax2 = plt.subplot(222)
ax2.set_title("Dimension 2")
ax2.set_xlabel("Time")
ax2.set_ylabel("Position")
ax3 = plt.subplot(223)
ax3.set_xlabel("Time")
ax3.set_ylabel("Velocity")
ax4 = plt.subplot(224)
ax4.set_xlabel("Time")
ax4.set_ylabel("Velocity")
ax1.plot(T, Y[:, 0], label="Demo")
ax2.plot(T, Y[:, 1], label="Demo")
ax3.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
ax4.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
ax3.scatter([T[-1]], (Y[-1, 0] - Y[-2, 0]) / dmp.dt_)
ax4.scatter([T[-1]], (Y[-1, 1] - Y[-2, 1]) / dmp.dt_)
for goal_yd in [0.0, 1.0, 2.0]:
    dmp.configure(goal_yd=np.array([goal_yd, goal_yd]))
    T, Y = dmp.open_loop(run_t=execution_time)
    ax1.plot(T, Y[:, 0], label="goal_yd = %g" % goal_yd)
    ax2.plot(T, Y[:, 1], label="goal_yd = %g" % goal_yd)
    ax3.plot(T, np.gradient(Y[:, 0]) / dmp.dt_)
    ax4.plot(T, np.gradient(Y[:, 1]) / dmp.dt_)
    ax3.scatter([T[-1]], [goal_yd])
    ax4.scatter([T[-1]], [goal_yd])
ax1.legend()
plt.tight_layout()
plt.show()

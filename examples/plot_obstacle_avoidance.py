import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance


dt = 0.01
execution_time = 1.0
T_demo = np.arange(0, execution_time + dt, dt)
Y_demo = np.column_stack((np.cos(np.pi * T_demo), np.sin(1.5 * np.pi * T_demo)))

dmp = DMP(n_dims=2, execution_time=execution_time)
dmp.imitate(T_demo, Y_demo)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
for obstacle_x in [-0.05, 0.05, 0.15]:
    obstacle_position = np.array([obstacle_x, 0.5])
    coupling_term = CouplingTermObstacleAvoidance(obstacle_position)
    T, Y = dmp.open_loop(run_t=execution_time, coupling_term=coupling_term)
    ax.plot(Y[:, 0], Y[:, 1], label="obstacle_x = %g" % obstacle_x)
    ax.scatter(obstacle_position[0], obstacle_position[1])
ax.plot(Y_demo[:, 0], Y_demo[:, 1], label="Demo")
ax.legend()
plt.tight_layout()
plt.show()

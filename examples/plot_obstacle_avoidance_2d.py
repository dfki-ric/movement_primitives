"""
========================
Obstacle Avoidance in 2D
========================

Plots a 2D DMP that goes through a point obstacle when there is no coupling
term for obstacle avoidance and a 2D DMP that avoids the point obstacle with
a coupling term.
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance2D


execution_time = 1.0
start_y = np.zeros(2)
goal_y = np.ones(2)

dmp = DMP(n_dims=2, execution_time=execution_time, n_weights_per_dim=3)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.set_weights(np.array([-50.0, 100.0, 300.0, -200.0, -200.0, -200.0]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
obstacle_position = np.array([0.92, 0.5])
T, Y = dmp.open_loop(run_t=execution_time)
ax.plot(Y[:, 0], Y[:, 1], label="Original")
coupling_term = CouplingTermObstacleAvoidance2D(obstacle_position)
T, Y = dmp.open_loop(run_t=execution_time, coupling_term=coupling_term)
ax.plot(Y[:, 0], Y[:, 1], label="Obstacle avoidance")
ax.scatter(start_y[0], start_y[1], c="r", label="Start")
ax.scatter(goal_y[0], goal_y[1], c="g", label="Goal")
ax.scatter(obstacle_position[0], obstacle_position[1], c="y", label="Obstacle")
ax.legend()
plt.tight_layout()
plt.show()

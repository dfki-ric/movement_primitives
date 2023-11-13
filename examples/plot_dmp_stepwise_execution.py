"""
=========================
Stepwise Execution of DMP
=========================

Since DMPs are executed open loop in most examples, in this case we will
demonstrate how to perform stepwise execution of a DMP, which is necessary to
incorporate sensor data in the execution.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP, CouplingTermObstacleAvoidance2D


execution_time = 1.0
dt = 0.001
start_y = np.zeros(2)
goal_y = np.ones(2)

dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=3)
dmp.configure(start_y=start_y, goal_y=goal_y)
dmp.set_weights(np.array([-50.0, 100.0, 300.0, -200.0, -200.0, -200.0]))
obstacle_position = np.array([0.92, 0.5])
coupling_term = CouplingTermObstacleAvoidance2D(obstacle_position, fast=True)

# on a real system, these would be measured values
y = np.copy(dmp.start_y)
yd = np.copy(dmp.start_yd)

T = []
Y = []

total_time = 0.0
for t in np.arange(0.0, 2.0, dt):  # on real system, t would be measured time
    T.append(t)
    Y.append(y)
    start_time = time.time()
    y, yd = dmp.step(y, yd, coupling_term, step_function="rk4-cython")
    total_time += time.time() - start_time
T.append(2.0)
Y.append(y)

T = np.array(T)
Y = np.array(Y)

print("Total runtime of DMP: %g s" % total_time)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.plot(Y[:, 0], Y[:, 1], label="Obstacle avoidance")
ax.scatter(start_y[0], start_y[1], c="r", label="Start")
ax.scatter(goal_y[0], goal_y[1], c="g", label="Goal")
ax.scatter(obstacle_position[0], obstacle_position[1], c="y", label="Obstacle")
ax.legend()
plt.tight_layout()
plt.show()

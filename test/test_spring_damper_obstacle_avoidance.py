import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.spring_damper import SpringDamper
from movement_primitives.dmp import CouplingTermObstacleAvoidance2D


def test_spring_damper_obstacle_avoidance(returns=False):
    start_y = np.zeros(2)
    goal_y = np.ones(2)

    sd = SpringDamper(n_dims=2, dt=0.01)
    sd.configure(start_y=start_y, goal_y=goal_y)
    obstacle_position = np.array([0.45, 0.5])
    ct = CouplingTermObstacleAvoidance2D(obstacle_position=obstacle_position)
    T, Y = sd.open_loop(run_t=10.0, coupling_term=ct)
    min_dist = min(np.linalg.norm(Y - obstacle_position, axis=1))
    assert min_dist > 0.3
    if returns:
        return T, Y, sd, obstacle_position


if __name__ == "__main__":
    T, Y, sd, obstacle_position = test_spring_damper_obstacle_avoidance(True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.plot(Y[:, 0], Y[:, 1], label="Obstacle avoidance")
    T, Y = sd.open_loop(run_t=10.0)
    ax.plot(Y[:, 0], Y[:, 1], label="Original")
    ax.scatter(sd.start_y[0], sd.start_y[1], c="r", label="Start")
    ax.scatter(sd.goal_y[0], sd.goal_y[1], c="g", label="Goal")
    ax.scatter(obstacle_position[0], obstacle_position[1], c="y", label="Obstacle")
    ax.legend()
    plt.tight_layout()
    plt.show()

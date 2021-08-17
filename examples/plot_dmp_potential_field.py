import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.dmp import DMP
from movement_primitives.dmp_potential_field import plot_potential_field_2d


dmp = DMP(n_dims=2, n_weights_per_dim=10, dt=0.01, execution_time=1.0)
random_state = np.random.RandomState(1)
dmp.forcing_term.weights[:, :] = random_state.randn(
    *dmp.forcing_term.weights.shape) * 200.0
start_y = np.array([0, 0], dtype=float)
goal_y = np.array([1, 1], dtype=float)
dmp.configure(start_y=start_y, goal_y=goal_y)

n_rows, n_cols = 2, 3
n_subplots = n_rows * n_cols
x_range = -0.2, 1.2
y_range = -0.2, 1.2

position = np.copy(start_y)
velocity = np.zeros_like(start_y)

plt.figure(figsize=(9, 6))
positions = [position]
for i in range(n_subplots):
    ax = plt.subplot(n_rows, n_cols, i + 1, aspect="equal")
    ax.set_title(f"t = {dmp.t:.02f}", backgroundcolor="#ffffffff", y=0.05)

    plot_potential_field_2d(
        ax, dmp, x_range=x_range, y_range=y_range, n_ticks=15)
    plt.plot(start_y[0], start_y[1], "o", color="b", markersize=10)
    plt.plot(goal_y[0], goal_y[1], "o", color="g", markersize=10)

    path = np.array(positions)
    plt.plot(path[:, 0], path[:, 1], lw=5, color="k")

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    plt.setp(ax, xticks=(), yticks=())

    if i == n_subplots - 1:
        break

    while dmp.t <= dmp.execution_time * (1 + i) / (n_subplots - 1):
        position, velocity = dmp.step(position, velocity)
        positions.append(position)
plt.subplots_adjust(
    left=0.0, bottom=0.0, right=1.0, top=1.0, wspace=0.01, hspace=0.01)
plt.show()

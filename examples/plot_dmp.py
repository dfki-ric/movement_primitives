import matplotlib.pyplot as plt
import numpy as np
from dmp import canonical_system_alpha, ForcingTerm, dmp_open_loop


start_t = 0.0
goal_t = 1.0
start_y = np.array([0.0])
goal_y = np.array([1.0])
dt = 0.001

alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=goal_t, start_t=start_t)
alpha_y = 25.0
beta_y = alpha_y / 4.0
forcing_term = ForcingTerm(
    n_dims=start_y.shape[0], n_weights_per_dim=6, goal_t=goal_t, start_t=start_t, overlap=0.8, alpha_z=alpha_z)
forcing_term.weights = 1000 * np.random.randn(*forcing_term.weights.shape)

T, Y = dmp_open_loop(goal_t=goal_t, start_t=start_t, dt=dt, start_y=start_y, goal_y=goal_y, alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term)

plt.plot(T, Y)
plt.scatter([start_t], [start_y])
plt.scatter([goal_t], [goal_y])
plt.show()
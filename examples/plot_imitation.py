import matplotlib.pyplot as plt
import numpy as np
from dmp import canonical_system_alpha, ForcingTerm, dmp_imitate, dmp_open_loop


start_t = 0.0
goal_t = 1.0
start_y = np.array([1.0])
goal_y = np.array([0.5])
dt = 0.01

alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=goal_t, start_t=start_t)
alpha_y = 25.0
beta_y = alpha_y / 4.0
n_weights_per_dim = 10
forcing_term = ForcingTerm(
    n_dims=start_y.shape[0], n_weights_per_dim=n_weights_per_dim, goal_t=goal_t, start_t=start_t, overlap=0.8, alpha_z=alpha_z)

T = np.linspace(0.0, 1.0, 101)
Y = np.cos(2 * np.pi * T)[:, np.newaxis]
forcing_term.weights[:, :] = dmp_imitate(T, Y, n_weights_per_dim=n_weights_per_dim, regularization_coefficient=0.0, alpha_y=alpha_y, beta_y=beta_y, overlap=0.8, alpha_z=alpha_z, allow_final_velocity=False)


plt.plot(T, Y, label="Demo")
plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

T, Y = dmp_open_loop(goal_t=goal_t, start_t=start_t, dt=dt, start_y=Y[0], goal_y=Y[-1], alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term)
plt.plot(T, Y, label="Reproduction")
plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

T, Y = dmp_open_loop(goal_t=goal_t, start_t=start_t, dt=dt, start_y=start_y, goal_y=goal_y, alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term, run_t=2.0)
plt.plot(T, Y, label="Adaptation")
plt.scatter([start_t, goal_t], [start_y, goal_y])

plt.legend(loc="best")
plt.show()
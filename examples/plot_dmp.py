import matplotlib.pyplot as plt
import numpy as np
from dmp import canonical_system_alpha, ForcingTerm, dmp_step


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

last_t = start_t - dt
t = start_t
y = np.copy(start_y)
yd = np.zeros_like(y)
T = []
Y = []
while t < goal_t:
    last_t = t
    t += dt
    y, yd = dmp_step(
        last_t, t, y, yd,
        goal_y=goal_y, goal_yd=np.zeros_like(goal_y), goal_ydd=np.zeros_like(goal_y),
        start_y=start_y, start_yd=np.zeros_like(start_y), start_ydd=np.zeros_like(start_y),
        goal_t=goal_t, start_t=start_t,
        alpha_y=alpha_y, beta_y=beta_y, forcing_term=forcing_term)
    T.append(t)
    Y.append(np.copy(y))

plt.plot(T, np.ravel(Y))
plt.scatter([start_t], [start_y])
plt.scatter([goal_t], [goal_y])
plt.show()
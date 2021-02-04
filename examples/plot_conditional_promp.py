import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.promp import ProMP


def acceleration_L(n_steps, dt):
    A = create_fd_matrix_1d(n_steps, dt)
    cov = np.linalg.inv(A.T.dot(A))
    return np.linalg.cholesky(cov)


def create_fd_matrix_1d(n_steps, dt):
    A = np.zeros((n_steps + 2, n_steps), dtype=np.float)
    super_diagonal = (np.arange(n_steps), np.arange(n_steps))
    sub_diagonal = (np.arange(2, n_steps + 2), np.arange(n_steps))
    A[super_diagonal] = np.ones(n_steps)
    A[sub_diagonal] = np.ones(n_steps)
    main_diagonal = (np.arange(1, n_steps + 1), np.arange(n_steps))
    A[main_diagonal] = -2 * np.ones(n_steps)
    return A / (dt ** 2)


n_demos = 100
n_steps = 101
random_state = np.random.RandomState(0)
y_conditional_cov = np.array([0.0])#np.array([0.01])

T = np.linspace(0, 1, 101)
Y = np.empty((n_demos, n_steps, 1))
L = acceleration_L(n_steps, dt=1.0 / (n_steps - 1))
for demo_idx in range(n_demos):
    initial_offset = 3.0 * (random_state.rand() - 0.5)
    final_offset = 0.1 * (random_state.rand() - 0.5)
    noise_per_step = 20.0 * L.dot(random_state.randn(n_steps))
    Y[demo_idx, :, 0] = np.linspace(initial_offset, final_offset, n_steps) + np.cos(2 * np.pi * T) + noise_per_step

promp = ProMP(n_dims=1, n_weights_per_dim=10)
promp.imitate([T] * n_demos, Y)
Y_mean = promp.mean_trajectory(T)
Y_conf = 1.96 * np.sqrt(promp.var_trajectory(T))

plt.figure(figsize=(10, 5))

ax1 = plt.subplot(121)
ax1.set_title("Training set and ProMP")
ax1.fill_between(T, (Y_mean - Y_conf).ravel(), (Y_mean + Y_conf).ravel(), color="r", alpha=0.3)
ax1.plot(T, Y_mean, c="r", lw=2, label="ProMP")
ax1.plot(T, Y[:, :, 0].T, c="k", alpha=0.1)
ax1.set_xlim((-0.05, 1.05))
ax1.set_ylim((-2.5, 3))
ax1.legend(loc="best")

ax2 = plt.subplot(122)
ax2.set_title("Conditional ProMPs")

for color, y_cond in zip("rgbcmyk", np.linspace(-1, 2.5, 7)):
    cpromp = promp.condition_position(np.array([y_cond]), y_cov=y_conditional_cov, t=0.0, t_max=1.0)
    Y_cmean = cpromp.mean_trajectory(T)
    Y_cconf = 1.96 * np.sqrt(cpromp.var_trajectory(T))

    ax2.scatter([0], [y_cond], marker="*", s=100, c=color, label="$y_0 = %.2f$" % y_cond)
    ax2.fill_between(T, (Y_cmean - Y_cconf).ravel(), (Y_cmean + Y_cconf).ravel(), color=color, alpha=0.3)
    ax2.plot(T, Y_cmean, c=color, lw=2)
    ax2.set_xlim((-0.05, 1.05))
    ax2.set_ylim((-2.5, 3))
    ax2.legend(loc="best")

plt.tight_layout()
plt.show()

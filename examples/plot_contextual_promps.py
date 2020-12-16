import matplotlib.pyplot as plt
import numpy as np
from promp import ProMP
from gmr import GMM


n_dims = 2
n_weights_per_dim = 10

promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)

random_state = np.random.RandomState(10)
n_demos = 30
n_steps = 51
T = np.linspace(0.0, 1.0, n_steps)
contexts = np.linspace(0, 1, n_demos)
Ys = np.empty((n_demos, n_steps, 2))
for demo_idx in range(n_demos):
    Ys[demo_idx, :, 0] = np.sin((0.5 + contexts[demo_idx]) * np.pi * T + random_state.randn() * 0.1)
    Ys[demo_idx, :, 1] = 0.5 + np.sin(2 * np.pi * T + random_state.randn() * 0.1)

weights = np.empty((n_demos, n_dims * n_weights_per_dim))
for demo_idx in range(n_demos):
    weights[demo_idx] = promp.weights(T, Ys[demo_idx]).ravel()

gmm = GMM(n_components=2, random_state=random_state)
X = np.hstack((contexts[:, np.newaxis], weights))
gmm.from_samples(X)
cmvn = gmm.condition([0], contexts[-1]).to_mvn()
promp.from_weight_distribution(cmvn.mean, cmvn.covariance)

for demo_idx in range(n_demos):
    plt.plot(Ys[demo_idx, :, 0], Ys[demo_idx, :, 1], c="k", alpha=0.3)
plt.plot(Ys[-1, :, 0], Ys[-1, :, 1], c="r", alpha=0.3)

samples = promp.sample_trajectories(T, 10, random_state)
for sample in samples:
    plt.plot(sample[:, 0], sample[:, 1], c="g", alpha=0.5)

plt.show()

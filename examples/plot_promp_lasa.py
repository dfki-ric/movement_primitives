"""
============================
LASA Handwriting with ProMPs
============================

The LASA Handwriting dataset learned with ProMPs. The dataset consists of
2D handwriting motions. The first and third column of the plot represent
demonstrations and the second and fourth column show the imitated ProMPs
with 1-sigma interval.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.data import load_lasa
from movement_primitives.promp import ProMP


def draw(T, X, promp, idx, axs):
    h = int(idx / width)
    w = int(idx % width) * 2
    axs[h, w].plot(X[:, :, 0].T, X[:, :, 1].T)

    mean = promp.mean_trajectory(T[0])
    std = np.sqrt(promp.var_trajectory(T[0]))
    axs[h, w + 1].plot(mean[:, 0], mean[:, 1], c="r")
    axs[h, w + 1].plot(mean[:, 0] - std[:, 0], mean[:, 1] - std[:, 1], c="g")
    axs[h, w + 1].plot(mean[:, 0] + std[:, 0], mean[:, 1] + std[:, 1], c="g")

    axs[h, w + 1].set_xlim(axs[h, w].get_xlim())
    axs[h, w + 1].set_ylim(axs[h, w].get_ylim())
    axs[h, w].get_yaxis().set_visible(False)
    axs[h, w].get_xaxis().set_visible(False)
    axs[h, w + 1].get_yaxis().set_visible(False)
    axs[h, w + 1].get_xaxis().set_visible(False)


n_shapes = 10
width = 2
height = 5

fig, axes = plt.subplots(int(height), int(width * 2))

for i in range(n_shapes):
    T, X, Xd, Xdd, dt, shape_name = load_lasa(i)
    promp = ProMP(n_weights_per_dim=30, n_dims=2)
    promp.imitate(T, X)
    draw(T, X, promp, i, axes)
plt.show()

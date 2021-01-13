import matplotlib.pyplot as plt
from itertools import cycle
from seaborn.palettes import SEABORN_PALETTES


def plot_trajectory_in_rows(trajectory, t=None, axes=None, subplot_shape=None):
    """TODO doc"""
    n_steps, n_dims = trajectory.shape

    if subplot_shape is None:
        subplot_shape = (n_dims, 1)

    newaxes = axes is None
    if newaxes:
        axes = [plt.subplot(subplot_shape[0], subplot_shape[1], 1 + i)
                for i in range(n_dims)]

    if t is not None:
        xlabel = "Time [s]"
    else:
        t = range(n_steps)
        xlabel = "Step"

    colors = cycle(SEABORN_PALETTES["deep"])

    for i in range(n_dims):
        color = next(colors)
        axes[i].plot(t, trajectory[:, i], color=color, label="Dimension #%d" % i)

        if newaxes:
            axes[i].legend()
            if subplot_shape[0] * subplot_shape[1] - i in range(1, subplot_shape[1] + 1):
                axes[i].set_xlabel(xlabel)
            else:
                axes[i].set_xticks(())

    if newaxes:
        plt.tight_layout(h_pad=0)

    return axes
import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory_in_rows(trajectory, t=None, label=None, axes=None, subplot_shape=None, **kwargs):
    """TODO doc

    Note that you have to manually activate the legend for one plot if you
    need it.
    """
    n_steps, n_dims = trajectory.shape

    if subplot_shape is None:
        subplot_shape = (n_dims, 1)

    newaxes = axes is None
    if newaxes:
        axes = create_axes(n_dims, subplot_shape)

    if t is not None:
        xlabel = "Time [s]"
    else:
        t = range(n_steps)
        xlabel = "Step"

    for i in range(n_dims):
        axes[i].plot(t, trajectory[:, i], label=label, **kwargs)

    if newaxes:
        layout_axes(axes, n_dims, subplot_shape, xlabel, (t[0], t[-1]))

    return axes


def plot_distribution_in_rows(mean, std_dev, t=None, label=None, axes=None, subplot_shape=None, **kwargs):
    """TODO doc

    Note that you have to manually activate the legend for one plot if you
    need it.
    """
    n_steps, n_dims = mean.shape

    if subplot_shape is None:
        subplot_shape = (n_dims, 1)

    newaxes = axes is None
    if newaxes:
        axes = create_axes(n_dims, subplot_shape)

    if t is not None:
        xlabel = "Time [s]"
    else:
        t = range(n_steps)
        xlabel = "Step"

    if "color" in kwargs:
        color = kwargs["color"]
    else:
        color = None

    for i in range(n_dims):
        axes[i].plot(t, mean[:, i], **kwargs)
        for f in [1, 2, 3]:
            axes[i].fill_between(
                np.arange(n_steps),
                mean[:, i] - f * std_dev[:, i],
                mean[:, i] + f * std_dev[:, i],
                color=color, alpha=0.1, label=label if f == 1 else None)

    if newaxes:
        layout_axes(axes, n_dims, subplot_shape, xlabel, (t[0], t[-1]))

    return axes


def create_axes(n_dims, subplot_shape):
    return [plt.subplot(subplot_shape[0], subplot_shape[1], 1 + i)
            for i in range(n_dims)]


def layout_axes(axes, n_dims, subplot_shape, xlabel, xlim):
    for i in range(n_dims):
        axes[i].set_title("Dimension #%d" % i, loc="left", y=0)
        if subplot_shape[0] * subplot_shape[1] - i in range(1, subplot_shape[1] + 1):
            axes[i].set_xlabel(xlabel)
        else:
            axes[i].set_xticks(())
        axes[i].set_xlim(xlim)
    plt.tight_layout(h_pad=0)

import numpy as np
from .dmp import dmp_transformation_system


def potential_field_2d(dmp, x_range, y_range, n_ticks):
    """Discretize potential field of DMP.

    Parameters
    ----------
    dmp : DMP
        Dynamical movement primitive.

    x_range : tuple
        Range in x dimension.

    y_range : tuple
        Range in y dimension.

    n_ticks : int
        Number of ticks per dimension.

    Returns
    -------
    xx : array, shape (n_ticks, n_ticks)
        x coordinates

    yy : array, shape (n_ticks, n_ticks)
        y coordinates

    ft : array, shape (n_ticks, n_ticks, 2)
        Acceleration from forcing term for each position.

    ts: array, shape (n_ticks, n_ticks, 2)
        Acceleration from transformation system for each position.
    """
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], n_ticks),
                         np.linspace(y_range[0], y_range[1], n_ticks))
    Y = np.array((xx, yy)).transpose((1, 2, 0))
    Yd = np.empty_like(Y)
    try:
        Yd[:, :] = dmp.last_yd
    except AttributeError:
        Yd[:, :, :] = 0.0

    ts = dmp_transformation_system(
        Y, Yd, dmp.alpha_y, dmp.beta_y, dmp.goal_y, dmp.goal_yd, dmp.goal_ydd,
        dmp.execution_time)
    ft = np.empty_like(ts)
    ft[:, :] = (dmp.alpha_y * dmp.forcing_term(dmp.t) / dmp.execution_time ** 2).ravel()

    acc = ft + ts
    return xx, yy, ft, ts, acc


def plot_potential_field_2d(ax, dmp, x_range, y_range, n_ticks):
    xx, yy, ft, ts, acc = potential_field_2d(
        dmp, x_range, y_range, n_ticks)

    quiver_scale = np.abs(acc).max() * n_ticks
    ax.quiver(xx, yy, ts[:, :, 0], ts[:, :, 1], scale=quiver_scale, color="g")
    ax.quiver(xx, yy, ft[:, :, 0], ft[:, :, 1], scale=quiver_scale, color="r")
    ax.quiver(xx, yy, acc[:, :, 0], acc[:, :, 1], scale=quiver_scale, color="k")

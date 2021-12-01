"""Visualization of DMP as potential field."""
import numpy as np
from .dmp import dmp_transformation_system, obstacle_avoidance_acceleration_2d


def potential_field_2d(dmp, x_range, y_range, n_ticks, obstacle=None):
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

    obstacle : array, shape (2,), optional (default: None)
        Obstacle position in the plane.

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

    ct : array, shape (n_ticks, n_ticks, 2)
        Acceleration from coupling term for obstacle avoidance for each
        position.

    acc : array, shape (n_ticks, n_ticks, 2)
        Accumulated acceleration from all terms.
    """
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], n_ticks),
                         np.linspace(y_range[0], y_range[1], n_ticks))
    Y = np.array((xx, yy)).transpose((1, 2, 0))
    Yd = np.empty_like(Y)
    Yd[:, :] = dmp.current_yd

    ts = dmp_transformation_system(
        Y, Yd, dmp.alpha_y, dmp.beta_y, dmp.goal_y, dmp.goal_yd, dmp.goal_ydd,
        dmp.execution_time)
    ft = np.empty_like(ts)
    ft[:, :] = (dmp.forcing_term(dmp.t) / dmp.execution_time ** 2).ravel()

    if obstacle is None:
        ct = np.zeros_like(ts)
    else:
        ct = obstacle_avoidance_acceleration_2d(Y, Yd, obstacle)

    acc = ft + ts + ct
    return xx, yy, ft, ts, ct, acc


def plot_potential_field_2d(ax, dmp, x_range, y_range, n_ticks, obstacle=None,
                            exaggerate_arrows=25.0):  # pragma: no cover
    """Plot 2D potential field of a DMP.

    We will indicate the influence of the transformation system at each
    position with green arrows, the influence of the forcing term with
    red arrows, the influence of obstacle avoidance with yellow arrows,
    and the combined acceleration with a black arrow.

    Parameters
    ----------
    ax : Matplotlib axis
        Axis on which we draw the potential field.

    dmp : DMP
        DMP object.

    x_range : tuple
        Range of x-axis.

    y_range : tuple
        Range of y-axis.

    n_ticks : int
        Number of ticks per dimension.

    obstacle : array, shape (2,), optional (default: None)
        Obstacle position in the plane.

    exaggerate_arrows : float, optional (default: 25)
        Multiply arrow sizes by this factor.
    """
    xx, yy, ft, ts, ct, acc = potential_field_2d(
        dmp, x_range, y_range, n_ticks, obstacle)

    ft *= exaggerate_arrows
    ts *= exaggerate_arrows
    ct *= exaggerate_arrows
    acc *= exaggerate_arrows

    quiver_scale = np.abs(acc).max() * n_ticks
    ax.quiver(xx, yy, ts[:, :, 0], ts[:, :, 1], scale=quiver_scale, color="g")
    ax.quiver(xx, yy, ft[:, :, 0], ft[:, :, 1], scale=quiver_scale, color="r")
    ax.quiver(xx, yy, ct[:, :, 0], ct[:, :, 1], scale=quiver_scale, color="y")
    ax.quiver(xx, yy, acc[:, :, 0], acc[:, :, 1], scale=quiver_scale,
              color="k")

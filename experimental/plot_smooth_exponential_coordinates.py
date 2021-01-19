import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.plot import plot_trajectory_in_rows
from pytransform3d import transformations as pt


Sthetas = np.loadtxt("experimental/screw_trajectory.txt")


def smooth_exponential_coordinates(Sthetas):
    """Smooth trajectories of exponential coordinates."""
    Sthetas = np.copy(Sthetas)
    diffs = np.linalg.norm(Sthetas[:-1, :3] - Sthetas[1:, :3], axis=1)
    sums = np.linalg.norm(Sthetas[:-1, :3] + Sthetas[1:, :3], axis=1)
    before_jump_indices = np.where(diffs > sums)[0]

    """
    # workaround for interpolation artifacts:
    before_smooth_jump_indices = np.isclose(diffs, sums)
    before_smooth_jump_indices = np.where(
        np.logical_and(before_smooth_jump_indices[:-1],
                       before_smooth_jump_indices[1:]))[0]
    before_jump_indices = np.unique(
        np.hstack((before_jump_indices, before_smooth_jump_indices)))
    """

    before_jump_indices = before_jump_indices.tolist()
    before_jump_indices.append(len(Sthetas))

    slices_to_correct = np.array(
        list(zip(before_jump_indices[:-1], before_jump_indices[1:])))[::2]
    for i, j in slices_to_correct:
        Sthetas[i + 1:j] = mirror_screw_axis_direction(Sthetas[i + 1:j])
    return Sthetas


def mirror_screw_axis_direction(Sthetas):
    Sthetas_new = []
    for Stheta in Sthetas:
        S, theta = pt.screw_axis_from_exponential_coordinates(Stheta)
        q, s, h = pt.screw_parameters_from_screw_axis(S)
        s_new = -s
        theta_new = 2.0 * np.pi - theta
        h_new = -h * theta_new / theta
        Stheta_new = pt.screw_axis_from_screw_parameters(q, s_new, h_new) * theta_new
        Sthetas_new.append(Stheta_new)
    return np.vstack(Sthetas_new)


Sthetas_smooth = smooth_exponential_coordinates(Sthetas)

axes = plot_trajectory_in_rows(Sthetas, subplot_shape=(2, 3))
plot_trajectory_in_rows(Sthetas_smooth, axes=axes)
plt.show()

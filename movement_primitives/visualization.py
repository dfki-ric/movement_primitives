"""3D visualization tools for movement primitives."""
import numpy as np
import open3d as o3d
import scipy as sp
import pytransform3d.transformations as pt


def plot_pointcloud(fig, pcl_points, color, uniform_down_sample=1):
    """Plot point cloud.

    Parameters
    ----------
    fig : pytransform3d.visualizer.Figure
        Figure.

    pcl_points : array-like, shape (n_points, 3)
        Points of the point cloud.

    color : array-like, shape (3,)
        Color as RGB values between 0 and 1.

    uniform_down_sample : int, optional (default: 1)
        Apply downsampling with this factor.

    Returns
    -------
    pcl : open3d.geometry.PointCloud
        Point cloud.
    """
    pcl = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(np.array(pcl_points)))
    colors = o3d.utility.Vector3dVector(
        [color for _ in range(len(pcl.points))])
    pcl.colors = colors
    if uniform_down_sample > 1:
        pcl = pcl.uniform_down_sample(uniform_down_sample)
    fig.add_geometry(pcl)
    return pcl


class ToggleGeometry:
    """Key action callback to toggle between showing and hiding a geometry.

    Parameters
    ----------
    fig : pytransform3d.visualizer.Figure
        Figure.

    geometry : open3d.geometry.Geometry
        Geometry.
    """
    def __init__(self, fig, geometry):
        self.fig = fig
        self.geometry = geometry
        self.show = True

    def __call__(self, vis, key, modifier):
        self.show = not self.show
        if self.show and modifier:
            vis.remove_geometry(self.geometry, False)
        elif not self.show and not modifier:
            vis.add_geometry(self.geometry, False)
        return True


def to_ellipsoid(mean, cov):
    """Compute error ellipsoid.

    An error ellipsoid shows equiprobable points.

    Parameters
    ----------
    mean : array-like, shape (3,)
        Mean of distribution

    cov : array-like, shape (3, 3)
        Covariance of distribution

    Returns
    -------
    ellipsoid2origin : array, shape (4, 4)
        Ellipsoid frame in world frame

    radii : array, shape (3,)
        Radii of ellipsoid
    """
    radii, R = sp.linalg.eigh(cov)
    if np.linalg.det(R) < 0:  # undo reflection (exploit symmetry)
        R *= -1
    ellipsoid2origin = pt.transform_from(R=R, p=mean)
    return ellipsoid2origin, np.sqrt(np.abs(radii))

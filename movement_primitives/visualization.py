import numpy as np
import open3d as o3d


def plot_pointcloud(fig, pcl_points, color, uniform_down_sample=1):
    pcl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(pcl_points)))
    colors = o3d.utility.Vector3dVector(
        [color for _ in range(len(pcl.points))])
    pcl.colors = colors
    if uniform_down_sample > 1:
        pcl = pcl.uniform_down_sample(uniform_down_sample)
    fig.add_geometry(pcl)
    return pcl


class ToggleGeometry:
    """Open3D key action callback that toggles between displaying and hiding a geometry."""
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

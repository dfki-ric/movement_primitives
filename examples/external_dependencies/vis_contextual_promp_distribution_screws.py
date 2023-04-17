import numpy as np
import open3d as o3d
from matplotlib import cbook
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.uncertainty as pu
from pytransform3d.urdf import UrdfTransformManager
from mocap.cleaning import smooth_exponential_coordinates, median_filter
from mocap.dataset_loader import load_kuka_dataset, transpose_dataset

from movement_primitives.promp import ProMP
from gmr import GMM


class Surface(pv.Artist):
    """Surface.

    Parameters
    ----------
    x : array, shape (n_steps, n_steps)
        Coordinates on x-axis of grid on surface.
    y : array, shape (n_steps, n_steps)
        Coordinates on y-axis of grid on surface.
    z : array, shape (n_steps, n_steps)
        Coordinates on z-axis of grid on surface.
    c : array-like, shape (3,), optional (default: None)
        Color
    """
    def __init__(self, x, y, z, c=None):
        self.c = c
        self.mesh = o3d.geometry.TriangleMesh()
        self.set_data(x, y, z)

    def set_data(self, x, y, z):
        """Update data.

        Parameters
        ----------
        x : array, shape (n_steps, n_steps)
            Coordinates on x-axis of grid on surface.
        y : array, shape (n_steps, n_steps)
            Coordinates on y-axis of grid on surface.
        z : array, shape (n_steps, n_steps)
            Coordinates on z-axis of grid on surface.
        """
        polys = np.stack([cbook._array_patch_perimeters(a, 1, 1)
                          for a in (x, y, z)], axis=-1)
        vertices = polys.reshape(-1, 3)
        triangles = (
            [[4 * i + 0, 4 * i + 1, 4 * i + 2] for i in range(len(polys))] +
            [[4 * i + 2, 4 * i + 3, 4 * i + 0] for i in range(len(polys))] +
            [[4 * i + 0, 4 * i + 3, 4 * i + 2] for i in range(len(polys))] +
            [[4 * i + 2, 4 * i + 1, 4 * i + 0] for i in range(len(polys))]
        )
        self.mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mesh.triangles = o3d.utility.Vector3iVector(triangles)
        if self.c is not None:
            self.mesh.paint_uniform_color(self.c)
        self.mesh.compute_vertex_normals()

    @property
    def geometries(self):
        """Expose geometries.
        Returns
        -------
        geometries : list
            List of geometries that can be added to the visualizer.
        """
        return [self.mesh]


def generate_training_data(
        pattern, n_weights_per_dim, context_names, verbose=0):
    Ts, Ps, contexts = transpose_dataset(
        load_kuka_dataset(pattern, context_names, verbose=verbose))

    Es = []
    for P in Ps:
        E = np.empty((len(P), 2 * 6))
        E[:, :6] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, :7]))
        E[:, 6:] = ptr.exponential_coordinates_from_transforms(ptr.transforms_from_pqs(P[:, 7:]))
        E[:, :6] = smooth_exponential_coordinates(E[:, :6])
        E[:, 6:] = smooth_exponential_coordinates(E[:, 6:])
        E[:, :] = median_filter(E, 5)
        Es.append(E)

        #import matplotlib.pyplot as plt
        #from movement_primitives.plot import plot_trajectory_in_rows
        #plot_trajectory_in_rows(E, subplot_shape=(6, 2))
        #plt.show()

    n_demos = len(Ts)
    n_dims = Es[0].shape[1]

    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    weights = np.empty((n_demos, n_dims * n_weights_per_dim))
    for demo_idx in range(n_demos):
        weights[demo_idx] = promp.weights(Ts[demo_idx], Es[demo_idx])

    return weights, Ts, Es, contexts


plot_training_data = False
n_dims = 12
n_weights_per_dim = 10
# available contexts: "panel_width", "clockwise", "counterclockwise", "left_arm", "right_arm"
context_names = ["panel_width", "clockwise", "counterclockwise"]

#pattern = "data/kuka/20200129_peg_in_hole/csv_processed/*/*.csv"
#pattern = "data/kuka/20191213_carry_heavy_load/csv_processed/*/*.csv"
pattern = "data/kuka/20191023_rotate_panel_varying_size/csv_processed/*/*.csv"

weights, Ts, Es, contexts = generate_training_data(
    pattern, n_weights_per_dim, context_names=context_names, verbose=2)
X = np.hstack((contexts, weights))

random_state = np.random.RandomState(0)
gmm = GMM(n_components=5, random_state=random_state)
gmm.from_samples(X)

n_steps = 100
T_query = np.linspace(0, 1, n_steps)

fig = pv.figure(with_key_callbacks=True)
fig.plot_transform(s=0.1)
tm = UrdfTransformManager()
with open("kuka_lbr/urdf/kuka_lbr.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="kuka_lbr/urdf/")
fig.plot_graph(tm, "kuka_lbr", show_visuals=True)

for panel_width, color, idx in zip([0.3, 0.4, 0.5], ([1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]), range(3)):
    print("panel_width = %.2f, color = %s" % (panel_width, color))

    context = np.array([panel_width, 0.0, 1.0])
    conditional_weight_distribution = gmm.condition(np.arange(len(context)), context).to_mvn()
    promp = ProMP(n_dims=n_dims, n_weights_per_dim=n_weights_per_dim)
    promp.from_weight_distribution(
        conditional_weight_distribution.mean,
        conditional_weight_distribution.covariance)

    # mean and covariance in state space
    mean = promp.mean_trajectory(T_query)
    cov = promp.cov_trajectory(T_query).reshape(
        mean.shape[0], mean.shape[1], mean.shape[1], mean.shape[0])

    for t in range(0, len(mean), 1):
        mean_left = pt.transform_from_exponential_coordinates(mean[t, :6])
        mean_right = pt.transform_from_exponential_coordinates(mean[t, 6:12])
        cov_left = cov[t, :6, :6, t]
        cov_right = cov[t, 6:, 6:, t]
        x, y, z = pu.to_projected_ellipsoid(mean_left, cov_left, 0.5, 20)
        surface = Surface(x, y, z, c=color)
        surface.add_artist(fig)
        x, y, z = pu.to_projected_ellipsoid(mean_right, cov_right, 0.5, 20)
        surface = Surface(x, y, z, c=color)

if plot_training_data:
    for E in Es:
        left2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, :6])
        right2base_trajectory = ptr.transforms_from_exponential_coordinates(E[:, 6:])
        pv.Trajectory(left2base_trajectory, s=0.02).add_artist(fig)
        pv.Trajectory(right2base_trajectory, s=0.02).add_artist(fig)

fig.view_init(azim=0, elev=25)
fig.show()

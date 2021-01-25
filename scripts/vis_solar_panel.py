import pytransform3d.visualizer as pv
from pytransform3d.urdf import UrdfTransformManager
import open3d as o3d


def load_solar_panels():
    solar_panels = [o3d.io.read_triangle_mesh(f) for f in
                    ["solar_panels/solar_panel_02/meshes/stl/base link.stl",
                     "solar_panels/solar_panel_03/meshes/stl/base link.stl"]]
    for sp in solar_panels:
        sp.compute_vertex_normals()
    return solar_panels


solar_panel_idx = 0
solar_panel = load_solar_panels()[solar_panel_idx]

tm = UrdfTransformManager(check=False)
with open("abstract-urdf-gripper/urdf/rh5_fixed.urdf", "r") as f:
    tm.load_urdf(f.read(), mesh_path="abstract-urdf-gripper/urdf/")

fig = pv.figure()
fig.plot_transform(s=0.2)
fig.add_geometry(solar_panel)
fig.plot_graph(tm, "RH5", show_visuals=True, show_frames=True, whitelist=["ALWristPitch_Link", "ARWristPitch_Link"])
fig.view_init()
fig.show()

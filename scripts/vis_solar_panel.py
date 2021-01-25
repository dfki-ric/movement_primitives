import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
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
tm.set_joint("ALShoulder1", -1.57)
tm.set_joint("ARShoulder1", 1.57)
tm.set_joint("ALShoulder2", 1.25)
tm.set_joint("ARShoulder2", -1.25)
tm.set_joint("ALShoulder3", 0)
tm.set_joint("ARShoulder3", 0)
tm.set_joint("ALElbow", -1.75)
tm.set_joint("ARElbow", 1.75)
tm.set_joint("ALWristRoll", 0)
tm.set_joint("ARWristRoll", 0)
tm.set_joint("ALWristPitch", 0.8)
tm.set_joint("ARWristPitch", 0.8)

tcp_left = tm.get_transform("LTCP_Link", "BodyBase_Link")
tcp_right = tm.get_transform("RTCP_Link", "BodyBase_Link")
tcp_left_pos = tcp_left[:3, 3]
tcp_right_pos = tcp_right[:3, 3]
tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
y_axis = pr.norm_vector(0.5 * (tcp_left[:3, 1] + tcp_right[:3, 1]))
R_panel = pr.matrix_from_two_vectors(x_axis, y_axis)
panel_pose = pt.transform_from(R_panel, tcp_middle)

solar_panel.transform(panel_pose)

fig = pv.figure()
fig.plot_transform(s=0.3)
fig.add_geometry(solar_panel)
fig.plot_graph(tm, "BodyBase_Link", show_visuals=True, show_frames=True,
               whitelist=["ALWristPitch_Link", "ARWristPitch_Link", "LTCP_Link", "RTCP_Link"],
               s=0.1)
fig.plot_transform(panel_pose, s=0.2)
fig.view_init()
fig.show()

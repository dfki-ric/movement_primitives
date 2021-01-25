import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from kinematics import Kinematics
import open3d as o3d


def load_solar_panels():
    solar_panels = [o3d.io.read_triangle_mesh(f) for f in
                    ["solar_panels/solar_panel_02/meshes/stl/base link.stl",
                     "solar_panels/solar_panel_03/meshes/stl/base link.stl"]]
    for sp in solar_panels:
        sp.compute_vertex_normals()
    return solar_panels


def panel_pose(tm):
    tcp_left = tm.get_transform("LTCP_Link", "BodyBase_Link")
    tcp_right = tm.get_transform("RTCP_Link", "BodyBase_Link")
    tcp_left_pos = tcp_left[:3, 3]
    tcp_right_pos = tcp_right[:3, 3]
    tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
    x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
    y_axis = pr.norm_vector(0.5 * (tcp_left[:3, 1] + tcp_right[:3, 1]))
    R_panel = pr.matrix_from_two_vectors(x_axis, y_axis)
    return pt.transform_from(R_panel, tcp_middle)


solar_panel_idx = 0
panel_rotation_angle = np.deg2rad(60)
n_steps = 101

solar_panel = load_solar_panels()[solar_panel_idx]

with open("abstract-urdf-gripper/urdf/rh5_fixed.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="abstract-urdf-gripper/urdf/")
left_arm = kin.create_chain(
    ["ALShoulder1", "ALShoulder2", "ALShoulder3",
     "ALElbow", "ALWristRoll", "ALWristPitch"],
    "BodyBase_Link", "LTCP_Link")
right_arm = kin.create_chain(
    ["ARShoulder1", "ARShoulder2", "ARShoulder3",
     "ARElbow", "ARWristRoll", "ARWristPitch"],
    "BodyBase_Link", "RTCP_Link")
q0_left = np.array([-1.57, 1.25, 0, -1.75, 0, 0.8])
q0_right = np.array([1.57, -1.25, 0, 1.75, 0, 0.8])
left_arm.forward(q0_left)
right_arm.forward(q0_right)

left2base_start = kin.tm.get_transform("LTCP_Link", "BodyBase_Link")
right2base_start = kin.tm.get_transform("RTCP_Link", "BodyBase_Link")
tcp_left_pos = left2base_start[:3, 3]
tcp_right_pos = right2base_start[:3, 3]

panel2base_start = panel_pose(kin.tm)
left2panel_start = pt.concat(left2base_start, pt.invert_transform(panel2base_start))
right2panel_start = pt.concat(right2base_start, pt.invert_transform(panel2base_start))


rotation_axis = pr.unity
start2end = pt.rotate_transform(np.eye(4), pr.matrix_from_compact_axis_angle(rotation_axis * panel_rotation_angle))
left2panel_end = pt.concat(left2panel_start, start2end)
right2panel_end = pt.concat(right2panel_start, start2end)

left2base_end = pt.concat(left2panel_end, panel2base_start)
right2base_end = pt.concat(right2panel_end, panel2base_start)

start_left = pt.exponential_coordinates_from_transform(left2base_start)
end_left = pt.exponential_coordinates_from_transform(left2base_end)
start_right = pt.exponential_coordinates_from_transform(right2base_start)
end_right = pt.exponential_coordinates_from_transform(right2base_end)

t = np.linspace(0, 1, n_steps)
left_trajectory = start_left[np.newaxis] + t[:, np.newaxis] * (end_left[np.newaxis] - start_left[np.newaxis])
right_trajectory = start_right[np.newaxis] + t[:, np.newaxis] * (end_right[np.newaxis] - start_right[np.newaxis])
left_trajectory = ptr.transforms_from_exponential_coordinates(left_trajectory)
right_trajectory = ptr.transforms_from_exponential_coordinates(right_trajectory)

fig = pv.figure()
fig.plot_transform(s=0.3)
fig.add_geometry(solar_panel)
fig.plot_graph(kin.tm, "BodyBase_Link", show_visuals=True, show_frames=True,
               whitelist=["ALWristPitch_Link", "ARWristPitch_Link", "LTCP_Link", "RTCP_Link"],
               s=0.1)

solar_panel.transform(panel2base_start)
fig.plot_transform(panel2base_start, s=0.2)

fig.plot_transform(left2base_start, s=0.15)
fig.plot_transform(right2base_start, s=0.15)
fig.plot_transform(left2base_end, s=0.15)
fig.plot_transform(right2base_end, s=0.15)

pv.Trajectory(left_trajectory, s=0.05).add_artist(fig)
pv.Trajectory(right_trajectory, s=0.05).add_artist(fig)

fig.view_init()
fig.show()

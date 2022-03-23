import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from pytransform3d.batch_rotations import smooth_quaternion_trajectory
from mocap.cleaning import smooth_exponential_coordinates
from movement_primitives.kinematics import Kinematics
from movement_primitives.dmp import DualCartesianDMP
from movement_primitives.io import (write_json, write_yaml, write_pickle)
from movement_primitives.testing.simulation import get_absolute_path


def panel_pose(tm):
    tcp_left = tm.get_transform("LTCP_Link", "RH5_Root_Link")
    tcp_right = tm.get_transform("RTCP_Link", "RH5_Root_Link")
    tcp_left_pos = tcp_left[:3, 3]
    tcp_right_pos = tcp_right[:3, 3]
    tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
    x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
    y_axis = -pr.norm_vector(0.5 * (tcp_left[:3, 1] + tcp_right[:3, 1]))
    R_panel = pr.matrix_from_two_vectors(x_axis, y_axis)
    return pt.transform_from(R_panel, tcp_middle)


def animation_callback(step, graph, left_arm, right_arm, left_joint_trajectory, right_joint_trajectory, panel_mesh):
    left_arm.forward(left_joint_trajectory[step])
    right_arm.forward(right_joint_trajectory[step])
    graph.set_data()
    panel_mesh.set_data(panel_pose(graph.tm))
    return graph, panel_mesh


solar_panel_idx = 0
panel_rotation_angle = np.deg2rad(90)
n_steps = 201

with open(get_absolute_path("pybullet-only-arms-urdf/urdf/RH5.urdf", "models/robots/rh5_models"), "r") as f:
    kin = Kinematics(f.read(), mesh_path=get_absolute_path("pybullet-only-arms-urdf/urdf/", "models/robots/rh5_models"))
#kin.tm.write_png("graph.png", "twopi")
left_arm = kin.create_chain(
    ["ALShoulder1", "ALShoulder2", "ALShoulder3",
     "ALElbow", "ALWristRoll", "ALWristYaw", "ALWristPitch"],
    "RH5_Root_Link", "LTCP_Link")
right_arm = kin.create_chain(
    ["ARShoulder1", "ARShoulder2", "ARShoulder3",
     "ARElbow", "ARWristRoll", "ARWristYaw", "ARWristPitch"],
    "RH5_Root_Link", "RTCP_Link")

#q0_left = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8])
#q0_right = np.array([1.57, -1.25, 0, 1.75, 0, 0, 0.8])
q0_left = np.array([-1.57, 0.9, 0, -1.3, 0, 0, -0.55])
q0_right = np.array([1.57, -0.9, 0, 1.3, 0, 0, -0.55])

left_arm.forward(q0_left)
right_arm.forward(q0_right)

left2base_start = kin.tm.get_transform("LTCP_Link", "RH5_Root_Link")
right2base_start = kin.tm.get_transform("RTCP_Link", "RH5_Root_Link")
tcp_left_pos = left2base_start[:3, 3]
tcp_right_pos = right2base_start[:3, 3]

panel2base_start = panel_pose(kin.tm)
left2panel_start = pt.concat(left2base_start, pt.invert_transform(panel2base_start))
right2panel_start = pt.concat(right2base_start, pt.invert_transform(panel2base_start))


rotation_axis = -pr.unity

start2end = pt.rotate_transform(np.eye(4), pr.matrix_from_compact_axis_angle(rotation_axis * panel_rotation_angle))
left2panel_end = pt.concat(left2panel_start, start2end)
right2panel_end = pt.concat(right2panel_start, start2end)

left2base_end = pt.concat(left2panel_end, panel2base_start)
right2base_end = pt.concat(right2panel_end, panel2base_start)

start_left = pt.exponential_coordinates_from_transform(left2base_start)
end_left = pt.exponential_coordinates_from_transform(left2base_end)
start_right = pt.exponential_coordinates_from_transform(right2base_start)
end_right = pt.exponential_coordinates_from_transform(right2base_end)
start_left, end_left = smooth_exponential_coordinates(np.array([start_left, end_left]))
start_right, end_right = smooth_exponential_coordinates(np.array([start_right, end_right]))

T = np.linspace(0, 1, n_steps)
left_trajectory = start_left[np.newaxis] + T[:, np.newaxis] * (end_left[np.newaxis] - start_left[np.newaxis])
right_trajectory = start_right[np.newaxis] + T[:, np.newaxis] * (end_right[np.newaxis] - start_right[np.newaxis])

"""# axis-angle SLERP
import pytransform3d.batch_rotations as pbr
a_start = np.hstack((rotation_axis, (0,)))
a_end = np.hstack((rotation_axis, (panel_rotation_angle,)))
A = np.array([pr.axis_angle_slerp(a_start, a_end, t) for t in np.linspace(0, 1, n_steps)])
Rs = pbr.matrices_from_compact_axis_angles(A[:, :3] * A[:, 3, np.newaxis])
start2current = np.empty((n_steps, 4, 4))
start2current[:] = np.eye(4)
start2current[:, :3, :3] = Rs
left_trajectory = np.empty((n_steps, 4, 4))
right_trajectory = np.empty((n_steps, 4, 4))
for t in range(n_steps):
    left2panel_current = pt.concat(left2panel_start, start2current[t])
    left2base_current = pt.concat(left2panel_current, panel2base_start)
    left_trajectory[t] = left2base_current
    right2panel_current = pt.concat(right2panel_start, start2current[t])
    right2base_current = pt.concat(right2panel_current, panel2base_start)
    right_trajectory[t] = right2base_current
left_trajectory = ptr.exponential_coordinates_from_transforms(left_trajectory)
right_trajectory = ptr.exponential_coordinates_from_transforms(right_trajectory)
#"""

print("Imitation...")
dt = 0.01
execution_time = (n_steps - 1) * dt
T = np.linspace(0, execution_time, n_steps)
Y = np.hstack((smooth_exponential_coordinates(left_trajectory),
               smooth_exponential_coordinates(right_trajectory)))
#dmp = DMP(n_dims=Y.shape[1], execution_time=execution_time, dt=dt, n_weights_per_dim=10)
#dmp.imitate(T, Y)
#_, Y = dmp.open_loop()

########################################################################################################################
# DMP export

########################################################################################################################

left_trajectory = Y[:, :6]
right_trajectory = Y[:, 6:]

left_trajectory = ptr.transforms_from_exponential_coordinates(left_trajectory)
right_trajectory = ptr.transforms_from_exponential_coordinates(right_trajectory)

print("Inverse kinematics...")
random_state = np.random.RandomState(0)
left_joint_trajectory = left_arm.inverse_trajectory(left_trajectory, q0_left, random_state=random_state)
right_joint_trajectory = right_arm.inverse_trajectory(right_trajectory, q0_right, random_state=random_state)

########################################################################################################################
# Data export
lwp2ltcp = kin.tm.get_transform("ALWristPitch_Link", "LTCP_Link")
rwp2rtcp = kin.tm.get_transform("ARWristPitch_Link", "RTCP_Link")

lwp_trajectory = np.array([pt.concat(lwp2ltcp, ltcp2base) for ltcp2base in left_trajectory])
rwp_trajectory = np.array([pt.concat(rwp2rtcp, rtcp2base) for rtcp2base in right_trajectory])

left_trajectory_pq = ptr.pqs_from_transforms(lwp_trajectory)
right_trajectory_pq = ptr.pqs_from_transforms(rwp_trajectory)

dmp = DualCartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)
Y_pq = np.empty((len(Y), 14))
Y_pq[:, :7] = left_trajectory_pq
Y_pq[:, 7:] = right_trajectory_pq
Y_pq[:, 3:7] = smooth_quaternion_trajectory(Y_pq[:, 3:7])
Y_pq[:, 10:14] = smooth_quaternion_trajectory(Y_pq[:, 10:14])
dmp.imitate(T, Y_pq)

write_yaml("rh5_dual_arm_dmp.yaml", dmp)
write_json("rh5_dual_arm_dmp.json", dmp)
write_pickle("rh5_dual_arm_dmp.pickle", dmp)

import pandas as pd
data_export = np.hstack((
    left_trajectory_pq,
    right_trajectory_pq
))
df = pd.DataFrame(
    data=data_export, columns=[
    "left_pos_x", "left_pos_y", "left_pos_z", "left_ori_w", "left_ori_x", "left_ori_y", "left_ori_z",
    "right_pos_x", "right_pos_y", "right_pos_z", "right_ori_w", "right_ori_x", "right_ori_y", "right_ori_z"],
    index=pd.Index(np.linspace(0, execution_time, len(left_trajectory_pq)), name="Time"))
df.to_csv("rh5_dual_arm_rotate_panel_posquat.csv")

data_export = np.hstack((left_joint_trajectory, right_joint_trajectory))
df = pd.DataFrame(
    data=data_export, columns=left_arm.joint_names + right_arm.joint_names,
    index=pd.Index(np.linspace(0, execution_time, len(left_trajectory_pq)), name="Time"))
df.to_csv("rh5_dual_arm_rotate_panel_joints.csv")# float_format="%.20f"
########################################################################################################################

#"""
import matplotlib.pyplot as plt
from movement_primitives.plot import plot_trajectory_in_rows
axes = plot_trajectory_in_rows(np.hstack((left_joint_trajectory, right_joint_trajectory)))
for i, jn in enumerate(left_arm.joint_names + right_arm.joint_names):
    joint_limits = kin.tm._joints[jn][-2]
    axes[i].plot([0, len(left_joint_trajectory)], [joint_limits[0]] * 2, c="r")
    axes[i].plot([0, len(left_joint_trajectory)], [joint_limits[1]] * 2, c="r")
plt.show()
#"""

fig = pv.figure()
fig.plot_transform(s=0.3)

# "solar_panels/solar_panel_02/meshes/stl/base link.stl"
# "solar_panels/solar_panel_03/meshes/stl/base link.stl"
panel_mesh = fig.plot_mesh("solar_panels/solar_panel_02/meshes/stl/base link.stl", A2B=panel2base_start)

graph = fig.plot_graph(
    kin.tm, "RH5_Root_Link", show_visuals=True, show_collision_objects=False, show_frames=True, s=0.1,
    whitelist=["ALWristPitch_Link", "ARWristPitch_Link", "LTCP_Link", "RTCP_Link"])

fig.plot_transform(panel2base_start, s=0.2)

fig.plot_transform(left2base_start, s=0.15)
fig.plot_transform(right2base_start, s=0.15)
fig.plot_transform(left_trajectory[-1], s=0.15)
fig.plot_transform(right_trajectory[-1], s=0.15)

#pv.Trajectory(left_trajectory, s=0.05).add_artist(fig)
#pv.Trajectory(right_trajectory, s=0.05).add_artist(fig)

pv.Trajectory(lwp_trajectory, s=0.05).add_artist(fig)
pv.Trajectory(rwp_trajectory, s=0.05).add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, len(left_joint_trajectory), loop=True,
    fargs=(graph, left_arm, right_arm, left_joint_trajectory, right_joint_trajectory, panel_mesh))
fig.show()

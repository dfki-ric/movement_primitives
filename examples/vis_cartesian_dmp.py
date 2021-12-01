"""
====================
Cartesian DMP on UR5
====================

A trajectory is created manually, imitated with a Cartesian DMP, converted
to a joint trajectory by inverse kinematics, and executed with a UR5.
"""
print(__doc__)

import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.kinematics import Kinematics
from movement_primitives.dmp import CartesianDMP


def animation_callback(step, graph, chain, joint_trajectory):
    chain.forward(joint_trajectory[step])
    graph.set_data()
    return graph


rotation_angle = np.deg2rad(45)
n_steps = 201

with open("examples/data/urdf/ur5.urdf", "r") as f:
    kin = Kinematics(f.read(), mesh_path="examples/data/urdf/")
chain = kin.create_chain(
    ["ur5_shoulder_pan_joint", "ur5_shoulder_lift_joint", "ur5_elbow_joint",
     "ur5_wrist_1_joint", "ur5_wrist_2_joint", "ur5_wrist_3_joint"],
    "ur5_base_link", "ur5_tool0")

q0 = np.array([0.0, -0.5, 0.8, -0.5, 0, 0])
chain.forward(q0)

ee2base_start = kin.tm.get_transform("ur5_tool0", "ur5_base_link")

rotation_axis = -pr.unity
start2end = pt.rotate_transform(np.eye(4), pr.matrix_from_compact_axis_angle(
    rotation_axis * rotation_angle))
ee2base_end = pt.concat(ee2base_start, start2end)

start = pt.exponential_coordinates_from_transform(ee2base_start)
end = pt.exponential_coordinates_from_transform(ee2base_end)

T = np.linspace(0, 1, n_steps)
trajectory = start[np.newaxis] + T[:, np.newaxis] * (end[np.newaxis] - start[np.newaxis])

dt = 0.01
execution_time = (n_steps - 1) * dt
T = np.linspace(0, execution_time, n_steps)
Y = ptr.pqs_from_transforms(ptr.transforms_from_exponential_coordinates(trajectory))
dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=10)
dmp.imitate(T, Y)
_, Y = dmp.open_loop()

trajectory = ptr.transforms_from_pqs(Y)

random_state = np.random.RandomState(0)
joint_trajectory = chain.inverse_trajectory(
    trajectory, q0, random_state=random_state)

fig = pv.figure()
fig.plot_transform(s=0.3)

graph = fig.plot_graph(
    kin.tm, "ur5_base_link", show_visuals=False, show_collision_objects=True,
    show_frames=True, s=0.1, whitelist=["ur5_base_link", "ur5_tool0"])

fig.plot_transform(ee2base_start, s=0.15)
fig.plot_transform(ee2base_end, s=0.15)
fig.plot_transform(trajectory[-1], s=0.15)
fig.plot_transform(trajectory[-1], s=0.15)

pv.Trajectory(trajectory, s=0.05).add_artist(fig)

fig.view_init()
fig.animate(
    animation_callback, len(trajectory), loop=True,
    fargs=(graph, chain, joint_trajectory))
fig.show()

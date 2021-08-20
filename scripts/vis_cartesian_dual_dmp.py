import numpy as np
from movement_primitives.dmp import DualCartesianDMP, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.visualizer as vis
from pytransform3d.urdf import UrdfTransformManager
from movement_primitives.testing.simulation import SimulationMockup

dt = 0.001
int_dt = 0.001
execution_time = 1.0

desired_distance = np.array([  # right arm to left arm
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.2],
    [0.0, 0.0, 0.0, 1.0]
])
desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle([np.deg2rad(180), 0, 0])
ct = CouplingTermDualCartesianPose(desired_distance=desired_distance, couple_position=True, couple_orientation=True, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000)  # c2=10000 in simulation

rh5 = SimulationMockup(dt=dt)

Y = np.zeros((1001, 14))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
radius = 0.5

circle1 = radius * np.cos(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
circle2 = radius * np.sin(np.deg2rad(90) + np.deg2rad(90) * sigmoid)
Y[:, 0] = circle1
Y[:, 1] = 0.55
Y[:, 2] = circle2
R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(0)])
# introduces coupling error (default goal: -90; error at: -110)
R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-110)])
q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
q_end = -pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
for i, t in enumerate(sigmoid):
    Y[i, 3:7] = pr.quaternion_slerp(q_start, q_end, t)

circle1 = radius * np.cos(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
circle2 = radius * np.sin(np.deg2rad(270) + np.deg2rad(90) * sigmoid)
Y[:, 7] = circle1
Y[:, 8] = 0.55
Y[:, 9] = circle2
R_three_fingers_front = pr.matrix_from_axis_angle([0, 0, 1, 0.5 * np.pi])
R_to_center_start = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-180)])
R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-270)])
q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
q_end = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
for i, t in enumerate(sigmoid):
    Y[i, 10:] = pr.quaternion_slerp(q_start, q_end, t)


dmp = DualCartesianDMP(
    execution_time=execution_time, dt=dt,
    n_weights_per_dim=10, int_dt=int_dt, p_gain=0.0)
dmp.imitate(T, Y)
dmp.configure(start_y=Y[0], goal_y=Y[-1])

recorded_trajectories = []
lf = (1.0, 0.0)
controller_args = dict(k=1, c1=0.1, c2=10000, verbose=0)
couple_both = CouplingTermDualCartesianPose(
    desired_distance=desired_distance, couple_position=True, couple_orientation=True,
    lf=lf, **controller_args)
couple_orientation = CouplingTermDualCartesianPose(
    desired_distance=desired_distance, couple_position=False, couple_orientation=True,
    lf=lf, **controller_args)
couple_position = CouplingTermDualCartesianPose(
    desired_distance=desired_distance, couple_position=True, couple_orientation=False,
    lf=lf, **controller_args)
for coupling_term in [couple_both, couple_orientation, couple_position, None]:
    dmp.reset()

    rh5.goto_ee_state(Y[0])
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time, coupling_term=coupling_term)
    P = np.asarray(positions)

    recorded_trajectories.append(P)

tm = UrdfTransformManager(check=False)
with open("abstract-urdf-gripper/urdf/GripperLeft.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
with open("abstract-urdf-gripper/urdf/GripperRight.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
tm.add_transform("ALWristPitch_Link", "base", np.eye(4))
tm.add_transform("ARWristPitch_Link", "base", np.eye(4))
# assert tm.check_consistency()

fig = vis.figure()
fig.plot_basis(R=np.eye(3), s=0.1)
for P, c in zip(recorded_trajectories, [[0, 1, 0], [1, 0, 0], [1, 0.5, 0], [0, 0, 0]]):
    fig.plot_trajectory(P=P[:, :7], s=0.05, c=c)
    fig.plot_trajectory(P=P[:, 7:], s=0.05, c=c)
graph = fig.plot_graph(tm, "base", show_visuals=True, show_frames=True,
                       whitelist=["ALWristPitch_Link", "ARWristPitch_Link"], s=0.03)


def animation_callback(t, recorded_trajectories, tm, graph):
    k = t // (len(recorded_trajectories[0]) // 10)
    P = recorded_trajectories[k]
    t = t % (len(recorded_trajectories[0]) // 10)
    gripper_left2base = pt.transform_from_pq(P[10 * t, :7])
    gripper_right2base = pt.transform_from_pq(P[10 * t, 7:])
    tm.add_transform("ALWristPitch_Link", "base", gripper_left2base)
    tm.add_transform("ARWristPitch_Link", "base", gripper_right2base)
    graph.set_data()
    return graph


fig.view_init(elev=0, azim=90)
fig.animate(
    animation_callback, len(recorded_trajectories) * len(recorded_trajectories[0]) // 10,
    loop=True, fargs=(recorded_trajectories, tm, graph))
fig.show()

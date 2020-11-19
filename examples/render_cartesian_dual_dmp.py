import numpy as np
from dmp import DualCartesianDMP, CouplingTermDualCartesianDistance, CouplingTermDualCartesianOrientation, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import visualization as vis
from pytransform3d.urdf import UrdfTransformManager
from simulation import SimulationMockup

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
ct = CouplingTermDualCartesianPose(desired_distance=desired_distance, couple_position=True, couple_orientation=True, lf=(1.0, 0.0), k=1, c1=0.1, c2=1000)  # c2=10000 in simulation

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
for i, t in enumerate(T):
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
for i, t in enumerate(T):
    Y[i, 10:] = pr.quaternion_slerp(q_start, q_end, t)


tm = UrdfTransformManager()
with open("abstract-urdf-gripper/urdf/GripperLeft.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
with open("abstract-urdf-gripper/urdf/GripperRight.urdf", "r") as f:
    tm.load_urdf(f, mesh_path="abstract-urdf-gripper/urdf/")
tm.add_transform("ALWristPitch_Link", "base", np.eye(4))
tm.add_transform("ARWristPitch_Link", "base", np.eye(4))
#assert tm.check_consistency()


for coupling_term in [ct]:#[ct, None]:
    # TODO reset DMP properly
    dmp = DualCartesianDMP(
        execution_time=execution_time, dt=dt,
        n_weights_per_dim=10, int_dt=int_dt, k_tracking_error=0.0)
    dmp.imitate(T, Y)
    dmp.configure(start_y=Y[0], goal_y=Y[-1])

    #import time
    #if coupling_term is not None:
    #    time.sleep(20)
    rh5.goto_ee_state(Y[0])
    desired_positions, positions, desired_velocities, velocities = \
        rh5.step_through_cartesian(dmp, Y[0], np.zeros(12), execution_time, coupling_term=coupling_term)
    P = np.asarray(positions)
    V = np.asarray(velocities)

    fig = vis.Figure()
    vis.plot_basis(fig, R=np.eye(3), s=0.1)
    vis.plot_trajectory(fig, P=P[:, :7], s=0.05, c=[1, 0.5, 0], show_direction=False)
    vis.plot_trajectory(fig, P=P[:, 7:], s=0.05, c=[1, 0.5, 0], show_direction=False)
    for t in range(0, len(P), 500):
        gripper_left2base = pt.transform_from_pq(P[t, :7])
        gripper_right2base = pt.transform_from_pq(P[t, 7:])
        tm.add_transform("ALWristPitch_Link", "base", gripper_left2base)
        tm.add_transform("ARWristPitch_Link", "base", gripper_right2base)
        vis.show_urdf_transform_manager(
            fig, tm, "base", visuals=True, collision_objects=False, frames=True,
            whitelist=["ALWristPitch_Link", "ARWristPitch_Link"], s=0.03)
    #ax.plot(P[:, 0], P[:, 1], P[:, 2], color="orange", lw=1)
    #ax.plot(P[:, 7], P[:, 8], P[:, 9], color="orange", lw=1)
    #ppu.plot_vector(
    #    ax=ax, start=P[0, :3] + np.array([0, 0, 0.2]), direction=P[-1, :3] - P[0, :3],
    #    lw=0, color="orange")
    #ppu.plot_vector(
    #    ax=ax, start=P[0, 7:10] - np.array([0, 0, 0.2]), direction=P[-1, 7:10] - P[0, 7:10],
    #    lw=0, color="orange")
    fig.view_init(elev=0, azim=90)
    fig.show()

    """
    dP = np.asarray(desired_positions)
    dV = np.asarray(desired_velocities)
    for subplot_idx, plot_dim in enumerate(range(14)):
        plt.subplot(3, 7, 1 + subplot_idx)
        plt.plot(T, Y[:, plot_dim], label="Demo %d" % plot_dim, c="k")
        plt.scatter([[0, T[-1]]], [[Y[0, plot_dim], Y[-1, plot_dim]]], c="k")
        plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim], label="Actual %d" % plot_dim, c="r")
        plt.scatter([[0, execution_time]], [[P[0, plot_dim], P[-1, plot_dim]]], c="r")
        plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
        plt.scatter([[0, execution_time]], [[dP[0, plot_dim], dP[-1, plot_dim]]], c="g")
        plt.ylim((min(P[:, plot_dim] - 0.05), max(P[:, plot_dim]) + 0.05))
    for subplot_idx, plot_dim in enumerate(range(7)):
        plt.subplot(3, 7, 15 + subplot_idx)
        plt.plot(T, Y[:, plot_dim] - Y[:, 7 + plot_dim], label="Demo %d" % plot_dim, c="k")
        plt.plot(np.linspace(0, execution_time, len(P)), P[:, plot_dim] - P[:, 7 + plot_dim], label="Actual %d" % plot_dim, c="r")
        plt.plot(np.linspace(0, execution_time, len(dP)), dP[:, plot_dim] - dP[:, 7 + plot_dim], label="Desired %d" % plot_dim, c="g", ls="--")
        D = P[:, plot_dim] - P[:, 7 + plot_dim]
        plt.ylim((min(D - 0.05), max(D) + 0.05))
    plt.legend()
    plt.show()
    """

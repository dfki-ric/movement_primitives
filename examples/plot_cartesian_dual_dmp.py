import numpy as np
import matplotlib.pyplot as plt
from dmp import DualCartesianDMP, CouplingTermDualCartesianDistance, CouplingTermDualCartesianOrientation, CouplingTermDualCartesianPose
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
import pytransform3d.plot_utils as ppu
from pytransform3d.urdf import UrdfTransformManager
from simulation import SimulationMockup

dt = 0.001
execution_time = 1.0

desired_distance = np.array([  # right arm to left arm
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 0.0, 1.0]
])
desired_distance[:3, :3] = pr.matrix_from_compact_axis_angle([np.deg2rad(180), 0, 0])
ct = CouplingTermDualCartesianPose(desired_distance=desired_distance, lf=(1.0, 0.0), k=1, c1=0.1, c2=10000)

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
R_to_center_end = pr.matrix_from_axis_angle([1, 0, 0, np.deg2rad(-90)])
q_start = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_start))
q_end = pr.quaternion_from_matrix(R_three_fingers_front.dot(R_to_center_end))
# TODO SLERP
Y[:, 3:7] = np.linspace(1, 0, len(Y))[:, np.newaxis] * q_start + np.linspace(0, 1, len(Y))[:, np.newaxis] * q_end

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
# TODO SLERP
Y[:, 10:] = np.linspace(1, 0, len(Y))[:, np.newaxis] * q_start + np.linspace(0, 1, len(Y))[:, np.newaxis] * q_end


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
        n_weights_per_dim=10, int_dt=0.001, k_tracking_error=0.0)
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

    ax = pr.plot_basis(ax_s=0.8, s=0.1)
    ptr.plot_trajectory(ax=ax, P=P[:, :7], s=0.05, color="orange", show_direction=False)
    ptr.plot_trajectory(ax=ax, P=P[:, 7:], s=0.05, color="orange", show_direction=False)
    for t in range(0, len(P), 500):
        gripper_left2base = pt.transform_from_pq(P[t, :7])
        gripper_right2base = pt.transform_from_pq(P[t, 7:])
        tm.add_transform("ALWristPitch_Link", "base", gripper_left2base)
        tm.add_transform("ARWristPitch_Link", "base", gripper_right2base)
        #ax = tm.plot_frames_in(
        #    "base", s=0.1, whitelist=["ALWristPitch_Link", "ARWristPitch_Link"],
        #    ax=ax, show_name=False)
        ax = tm.plot_visuals(frame="base", ax=ax)
    #ax.plot(P[:, 0], P[:, 1], P[:, 2], color="orange", lw=1)
    #ax.plot(P[:, 7], P[:, 8], P[:, 9], color="orange", lw=1)
    #ppu.plot_vector(
    #    ax=ax, start=P[0, :3] + np.array([0, 0, 0.2]), direction=P[-1, :3] - P[0, :3],
    #    lw=0, color="orange")
    #ppu.plot_vector(
    #    ax=ax, start=P[0, 7:10] - np.array([0, 0, 0.2]), direction=P[-1, 7:10] - P[0, 7:10],
    #    lw=0, color="orange")
    ax.view_init(elev=10, azim=110)
    plt.show()

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

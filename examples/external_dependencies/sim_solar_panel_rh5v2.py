# git clone git@git.hb.dfki.de:models-robots/rh5v2_models/pybullet-urdf.git --branch develop --recursive
import numpy as np
import pybullet
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from pytransform3d.batch_rotations import smooth_quaternion_trajectory
from movement_primitives.dmp import DualCartesianDMP
from movement_primitives.kinematics import Kinematics
from movement_primitives.testing.simulation import RH5Simulation, draw_transform, _pybullet_pose, get_absolute_path
from mocap.cleaning import smooth_exponential_coordinates
from movement_primitives.io import write_json, write_yaml, write_pickle


def panel_pose(tcp_left, tcp_right):
    tcp_left_pos = tcp_left[:3, 3]
    tcp_right_pos = tcp_right[:3, 3]
    tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
    tcp_middle[0] += 0.025
    x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
    y_axis = -pr.norm_vector(0.5 * (tcp_left[:3, 2] + tcp_right[:3, 2]))
    R_panel = pr.matrix_from_two_vectors(x_axis, y_axis)
    return pt.transform_from(R_panel, tcp_middle)


panel_rotation_angle = np.deg2rad(30)
n_steps = 1001
dt = 0.001

#q0 = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8, -1.57, 1.25, 0, -1.75, 0, 0, 0.8])
#q0 = np.array([-1.57, 0.76, 0, -1.3, 0, 0, -0.55, -1.57, 0.76, 0, -1.3, 0, 0, -0.55])
# q0 = np.array([-1.57, 0.9, 0, -1.05, 0, 0, 0, -1.57, 0.9, 0, -1.05, -0.25, 0, 0])
# new position for closed gripper
q0 = np.array([-1.59, 0.81, 0, -1.11, -0.25, 0.055, 0.56, -1.59, 0.81, 0, -1.11, -0.25, -0.055, -0.56])

rh5 = RH5Simulation(dt=dt, gui=True, real_time=False,
                    urdf_path=get_absolute_path("pybullet-urdf/urdf/RH5v2.urdf", "models/robots/rh5v2_models"),
                    left_arm_path=get_absolute_path("pybullet-urdf/submodels/left_arm.urdf", "models/robots/rh5v2_models"),
                    right_arm_path=get_absolute_path("pybullet-urdf/submodels/right_arm.urdf", "models/robots/rh5v2_models"))
rh5.set_desired_joint_state(q0, position_control=True)
rh5.sim_loop(int(1.0 / dt))
ee_state = rh5.get_ee_state()
left2base_start = pt.transform_from_pq(ee_state[:7])
right2base_start = pt.transform_from_pq(ee_state[7:])
draw_transform(left2base_start, s=0.1, client_id=rh5.client_id)
draw_transform(right2base_start, s=0.1, client_id=rh5.client_id)

panel2base_start = panel_pose(left2base_start, right2base_start)
draw_transform(panel2base_start, s=0.1, client_id=rh5.client_id)
panel2base_start_pq = pt.pq_from_transform(panel2base_start)
p, q = _pybullet_pose(panel2base_start_pq)

pybullet.loadURDF(get_absolute_path("solar_panels/solar_panel_02/urdf/pb_solar_panel_02.urdf", "models/objects"), p, q)
#time.sleep(10)

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

t = np.linspace(0, 1, n_steps)
left_trajectory = start_left[np.newaxis] + t[:, np.newaxis] * (end_left[np.newaxis] - start_left[np.newaxis])
right_trajectory = start_right[np.newaxis] + t[:, np.newaxis] * (end_right[np.newaxis] - start_right[np.newaxis])
left_trajectory = ptr.transforms_from_exponential_coordinates(left_trajectory)
right_trajectory = ptr.transforms_from_exponential_coordinates(right_trajectory)

########################################################################################################################
# Data export
with open(get_absolute_path("pybullet-urdf/urdf/RH5v2.urdf", "models/robots/rh5v2_models"), "r") as f:
    kin = Kinematics(f.read(), mesh_path=get_absolute_path("pybullet-urdf/urdf/", "models/robots/rh5v2_models"))
#kin.tm.write_png("graph.png", "twopi")

lwp2ltcp = kin.tm.get_transform("ALWristFT_Link", "LTCP_Link")
rwp2rtcp = kin.tm.get_transform("ARWristFT_Link", "RTCP_Link")

lwp_trajectory = np.array([pt.concat(lwp2ltcp, ltcp2base) for ltcp2base in left_trajectory])
rwp_trajectory = np.array([pt.concat(rwp2rtcp, rtcp2base) for rtcp2base in right_trajectory])

left_trajectory_pq = ptr.pqs_from_transforms(lwp_trajectory)
right_trajectory_pq = ptr.pqs_from_transforms(rwp_trajectory)

dmp = DualCartesianDMP(execution_time=n_steps * dt, dt=dt, n_weights_per_dim=10)
Y_pq = np.empty((n_steps, 14))
Y_pq[:, :7] = left_trajectory_pq
Y_pq[:, 7:] = right_trajectory_pq
Y_pq[:, 3:7] = smooth_quaternion_trajectory(Y_pq[:, 3:7])
Y_pq[:, 10:14] = smooth_quaternion_trajectory(Y_pq[:, 10:14])
T = np.linspace(0, 1, n_steps)
dmp.imitate(T, Y_pq)

write_yaml("rh5v2_dual_arm_dmp.yaml", dmp)
write_json("rh5v2_dual_arm_dmp.json", dmp)
write_pickle("rh5v2_dual_arm_dmp.pickle", dmp)
########################################################################################################################

#draw_trajectory(left_trajectory, rh5.client_id, s=0.05, n_key_frames=5)
#draw_trajectory(right_trajectory, rh5.client_id, s=0.05, n_key_frames=5)

print("Imitation...")
T = np.linspace(0, (n_steps - 1) * dt, n_steps)
P = np.hstack((ptr.pqs_from_transforms(left_trajectory),
               ptr.pqs_from_transforms(right_trajectory)))
dmp = DualCartesianDMP(execution_time=T[-1], dt=dt, n_weights_per_dim=10)
dmp.imitate(T, P)

desired_positions, positions, desired_velocities, velocities = \
    rh5.step_through_cartesian(dmp, P[0], np.zeros(12), T[-1])

rh5.stop()
rh5.sim_loop()

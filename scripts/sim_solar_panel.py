import numpy as np
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
import pytransform3d.trajectories as ptr
from movement_primitives.dmp import DualCartesianDMP
from simulation import RH5Simulation, draw_transform, draw_trajectory, _pybullet_pose


def panel_pose(tcp_left, tcp_right):
    tcp_left_pos = tcp_left[:3, 3]
    tcp_right_pos = tcp_right[:3, 3]
    tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
    x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
    y_axis = -pr.norm_vector(0.5 * (tcp_left[:3, 1] + tcp_right[:3, 1]))
    R_panel = pr.matrix_from_two_vectors(x_axis, y_axis)
    return pt.transform_from(R_panel, tcp_middle)


panel_rotation_angle = np.deg2rad(60)
n_steps = 51
dt = 0.01

#q0 = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8, 1.57, -1.25, 0, 1.75, 0, 0, 0.8])
q0 = np.array([-1.57, 0.76, 0, -1.3, 0, 0, -0.55, 1.57, -0.76, 0, 1.3, 0, 0, -0.55])

rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)
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

# TODO update grippers to manipulate solar panel
#import pybullet
#pybullet.loadURDF("solar_panels/solar_panel_02/urdf/solar_panel_02.urdf", p, q)

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

t = np.linspace(0, 1, n_steps)
left_trajectory = start_left[np.newaxis] + t[:, np.newaxis] * (end_left[np.newaxis] - start_left[np.newaxis])
right_trajectory = start_right[np.newaxis] + t[:, np.newaxis] * (end_right[np.newaxis] - start_right[np.newaxis])
left_trajectory = ptr.transforms_from_exponential_coordinates(left_trajectory)
right_trajectory = ptr.transforms_from_exponential_coordinates(right_trajectory)

draw_trajectory(left_trajectory, rh5.client_id, s=0.05, n_key_frames=5)
draw_trajectory(right_trajectory, rh5.client_id, s=0.05, n_key_frames=5)

print("Imitation...")
T = np.linspace(0, (n_steps - 1) * dt, n_steps)
P = np.hstack((ptr.pqs_from_transforms(left_trajectory),
               ptr.pqs_from_transforms(right_trajectory)))
dmp = DualCartesianDMP(execution_time=T[-1], dt=dt, n_weights_per_dim=10)
dmp.imitate(T, P)

desired_positions, positions, desired_velocities, velocities = \
    rh5.step_through_cartesian(dmp, P[0], np.zeros(12), t[-1])

rh5.stop()
rh5.sim_loop()

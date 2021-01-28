import numpy as np
from simulation import RH5Simulation, draw_pose
import pybullet
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt


def load_panel(tm):
    tcp_left = tm.get_transform("LTCP_Link", "RH5_Root_Link")
    tcp_right = tm.get_transform("RTCP_Link", "RH5_Root_Link")
    tcp_left_pos = tcp_left[:3, 3]
    tcp_right_pos = tcp_right[:3, 3]
    tcp_middle = 0.5 * (tcp_left_pos + tcp_right_pos)
    x_axis = pr.norm_vector(tcp_right_pos - tcp_left_pos)
    y_axis = -pr.norm_vector(0.5 * (tcp_left[:3, 1] + tcp_right[:3, 1]))
    R = pr.matrix_from_two_vectors(x_axis, y_axis)
    q = pr.quaternion_xyzw_from_wxyz(pr.quaternion_from_matrix(R))
    panel = pybullet.loadURDF(
        "solar_panels/solar_panel_02/urdf/solar_panel_02.urdf", tcp_middle, q)
    return panel, tcp_middle


sim = RH5Simulation(dt=0.00001, gui=True)

q = np.array([-1.57, 0.9, 0, -1.3, 0, 0, -0.55, 1.57, -0.9, 0, 1.3, 0, 0, -0.55])
#q = np.array([-1.57, 1.25, 0, -1.75, 0, 0, 0.8, 1.57, -1.25, 0, 1.75, 0, 0, 0.8])
sim.set_desired_joint_state(q, position_control=True)
sim.sim_loop(10000)

js = sim.get_joint_state()
print(f"Joint positions: {np.round(js[0], 2)}")

ee_state = sim.get_ee_state()
draw_pose(ee_state[:7], s=0.1, client_id=sim.client_id)
draw_pose(ee_state[7:], s=0.1, client_id=sim.client_id)

print(f"End-effector states: {np.round(ee_state, 2)}")

q = sim.inverse_kinematics(ee_state)
print(f"Desired joint positions: {np.round(q, 2)}")

sim.set_desired_ee_state(ee_state, position_control=True)

sim.sim_loop(100)

js = sim.get_joint_state()
print(f"Actual joint positions: {np.round(js[0], 2)}")

ee_state[1] *= 0.8
ee_state[8] *= 0.8

draw_pose(ee_state[:7], s=0.1, client_id=sim.client_id)
draw_pose(ee_state[7:], s=0.1, client_id=sim.client_id)

sim.set_desired_ee_state(ee_state, position_control=True)

sim.sim_loop()

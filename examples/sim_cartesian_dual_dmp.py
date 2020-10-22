import numpy as np
from pytransform3d.rotations import quaternion_from_axis_angle
from simulation import RH5Simulation
import pybullet

dt = 0.001
rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)

a = np.array([0.0, 0.0, 1.0, 0.0 * np.pi])
orientation = quaternion_from_axis_angle(a)

x = np.array([-0.4, 0.6, 0.3] + orientation.tolist() +
              [0.4, 0.6, 0.3] + orientation.tolist())
q = rh5.inverse_kinematics(x)
print("desired")
print(q[:7])
print(q[7:])
rh5.set_desired_joint_state(q, position_control=True)
rh5.sim_loop(1000)
q, qd = rh5.get_joint_state()
print("actual")
print(q[:7])
print(q[7:])
print("desired ee")
print(x[:7])
print(x[7:])
print("actual ee")
x = rh5.get_ee_state()
print(x[:7])
print(x[7:])
rh5.sim_loop()
import numpy as np
from pytransform3d.rotations import quaternion_from_axis_angle
from simulation import RH5Simulation
import pybullet

dt = 0.01
rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)

a = np.array([1.0, 0.0, 0.0, 0.0 * np.pi])
orientation = quaternion_from_axis_angle(a)

x = np.array([-0.5, 0.6, 0.5] + orientation.tolist() +
              [0.5, 0.6, 0.5] + orientation.tolist())
X = np.empty((1000, len(x)))
X[:, :] = x
length = 0.3
X[:int(len(X) / 2), 0] += np.linspace(-0.5 * length, 0.5 * length, int(len(X) / 2))
X[int(len(X) / 2):, 0] += np.linspace(0.5 * length, -0.5 * length, int(len(X) / 2))
X[:int(len(X) / 2), 7] += np.linspace(-0.5 * length, 0.5 * length, int(len(X) / 2))
X[int(len(X) / 2):, 7] += np.linspace(0.5 * length, -0.5 * length, int(len(X) / 2))
q = rh5.inverse_kinematics(x)
print("desired")
print(np.round(q[:7], 3))
print(np.round(q[7:], 3))
rh5.set_desired_joint_state(q, position_control=True)
rh5.sim_loop(100)
q, qd = rh5.get_joint_state()
print("actual")
print(np.round(q[:7], 3))
print(np.round(q[7:], 3))
print("desired ee")
print(x[:7])
print(x[7:])
pybullet.addUserDebugLine(x[:3], [0, 0, 0], [1, 0, 0], 2)
pybullet.addUserDebugLine(x[7:10], [0, 0, 0], [1, 0, 0], 2)
print("actual ee")
x = rh5.get_ee_state()
print(np.round(x[:7], 3))
print(np.round(x[7:], 3))
pybullet.addUserDebugLine(x[:3], [0, 0, 0], [0, 1, 0], 2)
pybullet.addUserDebugLine(x[7:10], [0, 0, 0], [0, 1, 0], 2)

rh5.goto_ee_state(X[0], wait_time=1.0)
while True:
    for i in range(len(X)):
        rh5.set_desired_ee_state(X[i])
        rh5.step()
rh5.set_desired_ee_state(X[-1], position_control=True)
#rh5.stop()
rh5.sim_loop()
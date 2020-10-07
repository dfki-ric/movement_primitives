import matplotlib.pyplot as plt
import numpy as np
from simulation import UR5Simulation


X = np.zeros((10001, 7))
t = np.linspace(0, 1, len(X))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (t - 0.5)) + 1.0)
X[:, 0] = 0.5 + 0.1 * sigmoid
X[:, 1] = -0.5 + 1.0 * sigmoid
X[:, 2] = 0.45 + 0.1 * sigmoid
X[:, 3] = 1.0
#plt.plot(X[:, 0])
#plt.show()

ur5 = UR5Simulation(dt=0.001, real_time=True)

q = ur5.inverse_kinematics(X[0])
ur5.set_desired_joint_state(q, position_control=True)
ur5.sim_loop(1000)
ur5.stop()
ur5.sim_loop(1000)

P = []
for t in range(len(X)):
    ur5.set_desired_ee_state(X[t])
    ur5.step()
    P.append(ur5.get_ee_state())
P = np.asarray(P)

plt.plot(X[:, 1], label="Desired")
plt.plot(P[:, 1], label="Actual")
plt.show()

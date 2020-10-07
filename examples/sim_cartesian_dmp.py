import numpy as np
from dmp import DMP
from simulation import UR5Simulation



dt = 0.01

dmp = DMP(n_dims=7, execution_time=1.0, dt=0.001, n_weights_per_dim=10)
T = np.linspace(0.0, 1.0, 101)
Y = np.empty((len(T), 7))
Y[:, 0] = 0.3# + 0.1 * np.cos(np.pi * T)
Y[:, 1] = 0.3# + 0.1 * np.sin(np.pi * T)
Y[:, 2] = 0.0# + 0.1 * np.sin(2 * np.pi * T)
Y[:, 3] = 1.0
Y[:, 4] = 0.0
Y[:, 5] = 0.0
Y[:, 6] = 0.0
dmp.imitate(T, Y)
dmp.configure(start_y=Y[0], goal_y=Y[-1])
ur5 = UR5Simulation(dt=0.001, real_time=True)
print(Y[0])
q = ur5.inverse_kinematics(Y[0])
ur5.set_desired_joint_state(q, position_control=True)
ur5.sim_loop(1000)
ur5.stop()
#ur5.sim_loop()
print(ur5.get_ee_state())
positions = []
desired_positions = []
velocities = []
desired_velocities = []
last_v = np.zeros(7)
for i in range(1001):
    #last_p, last_v = ur5.get_ee_state()
    last_p = ur5.get_ee_state()
    p, v = dmp.step(last_p, last_v)
    print(np.linalg.norm(p - last_p))
    ur5.set_desired_ee_state(p)
    ur5.step()

    positions.append(last_p)
    desired_positions.append(p)
    velocities.append(last_v)
    desired_velocities.append(v)
    print("====")
    print(dmp.t)
    print(np.round(p, 2))
    print(np.round(last_p, 2))
    last_v = v
ur5.stop()

import matplotlib.pyplot as plt
P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plt.plot(P[:, 1], label="Actual")
plt.plot(dP[:, 1], label="Desired")
T, Y = dmp.open_loop(run_t=1.0)
plt.plot(Y[:, 1], label="Open loop")
plt.legend()
plt.show()
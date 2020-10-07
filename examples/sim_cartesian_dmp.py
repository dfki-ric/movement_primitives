import numpy as np
from dmp import DMP
from simulation import UR5Simulation



dt = 0.01

dmp = DMP(n_dims=7, execution_time=1.0, dt=0.001, n_weights_per_dim=10, int_dt=0.001)
Y = np.zeros((1001, 7))
T = np.linspace(0, 1, len(Y))
sigmoid = 0.5 * (np.tanh(1.5 * np.pi * (T - 0.5)) + 1.0)
Y[:, 0] = 0.6
Y[:, 1] = -0.2 + 0.4 * sigmoid
Y[:, 2] = 0.45
Y[:, 4] = 1.0
dmp.imitate(T, Y)
#dmp.forcing_term.weights[:, :] = 0.0
dmp.configure(start_y=Y[0], goal_y=Y[-1])

ur5 = UR5Simulation(dt=0.001, real_time=False)
ur5.goto_ee_state(Y[0], wait_time=1.0)
ur5.stop()

positions = []
desired_positions = []
velocities = []
desired_velocities = []
last_p = Y[0]
last_v = np.zeros(7)
for i in range(4 * len(Y)):
    _, _ = ur5.get_ee_state(return_velocity=True)
    p, v = dmp.step(last_p, last_v)
    ur5.set_desired_ee_state(p)
    ur5.step()

    positions.append(last_p)
    desired_positions.append(p)
    velocities.append(last_v)
    desired_velocities.append(v)
    """
    print("====")
    print(dmp.t)
    print(np.round(v, 2))
    print(np.round(last_v, 2))
    print(np.round(p, 2))
    print(np.round(last_p, 2))
    print("Dist:", np.linalg.norm(p - last_p))
    """
    last_v = v
    last_p = p
ur5.stop()

import matplotlib.pyplot as plt
P = np.asarray(positions)
dP = np.asarray(desired_positions)
V = np.asarray(velocities)
dV = np.asarray(desired_velocities)

plot_dim = 1
plt.plot(Y[:, plot_dim], label="Demo")
plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.plot(P[:, plot_dim], label="Actual")
plt.scatter([[0, len(P)]], [[P[0, plot_dim], P[-1, plot_dim]]])
plt.plot(dP[:, plot_dim], label="Desired")
plt.scatter([[0, len(dP)]], [[dP[0, plot_dim], dP[-1, plot_dim]]])
T, Y = dmp.open_loop(run_t=2.0)
plt.plot(Y[:, plot_dim], label="Open loop")
plt.scatter([[0, len(Y)]], [[Y[0, plot_dim], Y[-1, plot_dim]]])
plt.legend()
plt.show()
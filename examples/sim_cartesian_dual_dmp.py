import numpy as np
from simulation import RH5Simulation

dt = 0.001
rh5 = RH5Simulation(dt=dt, gui=True, real_time=False)

rh5.set_desired_joint_state(0.1 * np.zeros(14), position_control=True)
#rh5.sim_loop()
x = np.array([0.6, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.6, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
q = rh5.inverse_kinematics(x)
print(q)
rh5.set_desired_joint_state(q, position_control=True)
rh5.sim_loop()
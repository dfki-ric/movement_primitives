"""
===================
State-following DMP
===================
"""
print(__doc__)


import matplotlib.pyplot as plt
import numpy as np
from movement_primitives.dmp import StateFollowingDMP

start_y = np.array([0.0, 1.0])
dt = 0.001
execution_time = 3.0
n_viapoints = 8

dmp = StateFollowingDMP(n_dims=2, execution_time=execution_time, dt=dt, n_viapoints=n_viapoints)
dmp.forcing_term.viapoints[:, 0] = np.linspace(0, 1, n_viapoints)
dmp.forcing_term.viapoints[:, 1] = np.linspace(1, 2, n_viapoints)
dmp.configure(start_y=start_y)

T, Y = dmp.open_loop(run_t=1.5 * execution_time)

plt.figure(figsize=(10, 5))
ax = plt.subplot(121)
ax.set_xlabel("Time")
ax.set_ylabel("Position")
ax.plot(T, Y[:, 0], label="X")
ax.plot(T, Y[:, 1], label="Y")
ax.scatter(np.linspace(0, execution_time, n_viapoints), dmp.forcing_term.viapoints[:, 0], label="Viapoints X")
ax.scatter(np.linspace(0, execution_time, n_viapoints), dmp.forcing_term.viapoints[:, 1], label="Viapoints Y")
ax.legend()
ax = plt.subplot(122)
ax.set_xlabel("Position X")
ax.set_ylabel("Position Y")
ax.plot(Y[:, 0], Y[:, 1], label="Positions")
ax.scatter(dmp.forcing_term.viapoints[:, 0], dmp.forcing_term.viapoints[:, 1], label="Viapoints")
ax.legend()
plt.show()

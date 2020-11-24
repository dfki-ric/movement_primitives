import matplotlib.pyplot as plt
import numpy as np
from dmp import StateFollowingDMP


start_y = np.array([0.0, 1.0])
dt = 0.001
execution_time = 3.0
n_viapoints = 8

dmp = StateFollowingDMP(n_dims=2, execution_time=execution_time, dt=dt, n_viapoints=n_viapoints)
dmp.forcing_term.viapoints[:, 0] = np.linspace(0, 1, n_viapoints)
dmp.forcing_term.viapoints[:, 1] = np.linspace(1, 2, n_viapoints)
dmp.configure(start_y=start_y)

T, Y = dmp.open_loop(run_t=1.5 * execution_time)

ax = plt.subplot(121)
ax.plot(T, Y)
ax.scatter(np.linspace(0, execution_time, n_viapoints), dmp.forcing_term.viapoints[:, 0])
ax.scatter(np.linspace(0, execution_time, n_viapoints), dmp.forcing_term.viapoints[:, 1])
ax = plt.subplot(122)
ax.plot(Y[:, 0], Y[:, 1])
ax.scatter(dmp.forcing_term.viapoints[:, 0], dmp.forcing_term.viapoints[:, 1])
plt.show()

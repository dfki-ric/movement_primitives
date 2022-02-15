"""
=======================
Minimum Jerk Trajectory
=======================

An example for a minimum jerk trajectory is displayed in the following plot.
"""
print(__doc__)

import matplotlib.pyplot as plt
from movement_primitives.data import generate_minimum_jerk

X, Xd, Xdd = generate_minimum_jerk([0], [1])
plt.figure()
plt.subplot(311)
plt.ylabel("$x$")
plt.plot(X[:, 0])
plt.subplot(312)
plt.ylabel("$\dot{x}$")
plt.plot(Xd[:, 0])
plt.subplot(313)
plt.xlabel("$t$")
plt.ylabel("$\ddot{x}$")
plt.plot(Xdd[:, 0])
plt.show()

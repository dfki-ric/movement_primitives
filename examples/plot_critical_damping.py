"""
================
Critical Damping
================

The transformation system of a DMP converges to the goal and the convergence is
modeled as a spring-damper system. For an optimal convergence, the constants
defining the spring-damper system (spring constant k and damping coefficient c)
have to be set to critical damping for optimal convergence, that is, as quickly
as possible without overshooting. To illustrate this, we use a standalone
spring-damper system and explore several values for these parameters.
"""
print(__doc__)


import numpy as np
import matplotlib.pyplot as plt
from movement_primitives.spring_damper import SpringDamper


k = 100
start_y = np.zeros(1)
goal_y = np.ones(1)
for c in [10, 20, 40]:
    attractor = SpringDamper(n_dims=1, k=k, c=c, dt=0.01)
    attractor.configure(start_y=start_y, goal_y=goal_y)
    T, Y = attractor.open_loop(run_t=2.0)
    plt.plot(T, Y[:, 0], label=f"$k={k}, c={c}$")
plt.scatter(1.0, 1.0, marker="*", s=200, label="Goal")
plt.legend(loc="best")
plt.title(r"Condition for critical damping: $c = 2 \sqrt{k}$")
plt.show()

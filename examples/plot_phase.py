import matplotlib.pyplot as plt
import numpy as np
from dmp import canonical_system_alpha, phase

alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
t = np.linspace(0.0, 1.0, 1001)
z = phase(t, alpha_z, goal_t=1.0, start_t=0.0)

plt.plot(t, z)
plt.show()
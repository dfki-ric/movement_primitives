import matplotlib.pyplot as plt
import numpy as np
from dmp import canonical_system_alpha, phase, ForcingTerm


alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
forcing_term = ForcingTerm(n_dims=1, n_weights_per_dim=6, goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)
t = np.linspace(0.0, 1.0, 1001)
z = phase(t, alpha_z, goal_t=1.0, start_t=0.0)
activations = forcing_term._activations(z, normalized=True)

plt.plot(t, activations.T)
plt.show()
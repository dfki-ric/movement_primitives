import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.trajectories as ptr
from movement_primitives.spring_damper import SpringDamperOrientation


dt = 0.01
execution_time = 5.0

sd = SpringDamperOrientation(k=4.0, c=2 * np.sqrt(4), dt=dt, int_dt=0.001)
random_state = np.random.RandomState(42)
start = pr.random_quaternion(random_state)
goal = pr.random_quaternion(random_state)
goal = np.array([0.0, 0.0, 1.0, 0.0])
sd.configure(start_y=start, goal_y=goal)

T, Q = sd.open_loop(run_t=execution_time)
ax = pr.plot_basis(R=pr.matrix_from_quaternion(start), p=[-0.5, -0.5, 0], s=0.3, alpha=0.5, lw=3)
ax = pr.plot_basis(R=pr.matrix_from_quaternion(goal), p=[0.5, 0.5, 0], s=0.3, alpha=0.5, lw=3, ax=ax)
P = np.hstack((np.zeros((len(Q), 3)), Q))
P[:, 0] = np.linspace(-0.5, 0.5, len(P))
P[:, 1] = np.linspace(-0.5, 0.5, len(P))
ptr.plot_trajectory(P=P, s=0.2, ax=ax)
plt.show()
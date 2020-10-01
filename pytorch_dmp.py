import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class DMP(nn.Module):
    def __init__(
            self, n_task_dims, g, execution_time, dt,
            alpha_y=25.0, beta_y=6.25, alpha_z=8.33):
        super(DMP, self).__init__()

        self.n_task_dims = n_task_dims
        self.g = g
        self.execution_time = execution_time
        self.dt = dt
        self.alpha_y = alpha_y
        self.beta_y = beta_y
        self.alpha_z = alpha_z

        self.forcing_term = nn.Sequential(
            nn.Linear(1, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, n_task_dims),
        )

    def forward(self, x, t_start):
        vel = torch.zeros_like(x)

        times = torch.arange(t_start, self.execution_time + self.dt, self.dt)
        times = times.view(len(times), 1)
        z = phase(times, self.execution_time, self.dt, self.alpha_z)
        f = z * self.forcing_term(times)

        trajectories = []
        for i, t in enumerate(times):
            x, vel = dmp_step(
                f[i], x, vel,
                self.g, self.execution_time, self.dt,
                self.alpha_y, self.beta_y)
            trajectories.append(x)
        trajectories = torch.cat(trajectories, dim=1)
        return trajectories


def dmp_step(f, pos, vel, g, execution_time, dt, alpha_y, beta_y):
    next_acc = alpha_y / (execution_time ** 2) * (beta_y * (g - pos) - execution_time * vel) + f
    next_vel = vel + dt * next_acc
    next_pos = pos + dt * next_vel
    return next_pos, next_vel


def phase(t, execution_time, dt, alpha_z):
    b = max(1.0 - alpha_z * dt / execution_time, 1e-10)
    return torch.pow(b, (t / dt))


g = Tensor(np.ones(1))
execution_time = 1.0
dt = 0.005
alpha_y = 25.0
beta_y = alpha_y / 4.0

"""
position = Tensor(np.zeros(1))
velocity = Tensor(np.zeros(1))
P = []
for t in np.arange(0.0, execution_time + dt, dt):
    position, velocity = dmp_step(
        0.0, position, velocity, g, execution_time, dt,
        alpha_y, beta_y)
    P.append(np.array(position))
P = np.array(P)
"""

dmp = DMP(1, g, execution_time, dt)

trajectories = dmp(Tensor(np.zeros((1, 1))), 0.0)
P1 = np.array(trajectories.detach())

X = np.empty((1, 201))
X[0] = np.sin(0.5 * np.pi * np.linspace(0, 1, 201))

n_epochs = 1000
optimizer = torch.optim.Adam(dmp.parameters())
pbar = tqdm.trange(n_epochs)
for _ in pbar:
    #for i in range(n_train_samples // batch_size):
    desired_trajectories = Tensor(X)
    actual_trajectories = dmp(Tensor(np.zeros((1, 1))), 0.0)
    l = F.mse_loss(actual_trajectories, desired_trajectories, reduction="sum")
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    pbar.set_description("MSE %g; Epochs" % l.item())

trajectories = dmp(Tensor(np.zeros((1, 1))), 0.0)
P2 = np.array(trajectories.detach())

import matplotlib.pyplot as plt
plt.plot(P1[0])
plt.plot(X[0])
plt.plot(P2[0])
plt.show()
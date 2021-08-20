import numpy as np
from movement_primitives.promp import ProMP
from nose.tools import assert_less
from numpy.testing import assert_array_almost_equal


def test_imitate():
    n_weights_per_dim = 50

    promp = ProMP(n_dims=1, n_weights_per_dim=n_weights_per_dim)

    random_state = np.random.RandomState(10)
    n_demos = 10
    n_steps = 101
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Y = np.empty((n_demos, n_steps, 1))
    for demo_idx in range(n_demos):
        Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
        Y[demo_idx, :, 0] += random_state.randn(n_steps) * 0.01
    promp.imitate(T, Y)

    random_state = np.random.RandomState(0)
    samples = promp.sample_trajectories(T[0], 10, random_state)

    mean_trajectory = promp.mean_trajectory(T[0])
    assert_less(np.linalg.norm(np.mean(samples, axis=0) - mean_trajectory) / n_steps, 0.001)
    std_trajectory = np.sqrt(promp.var_trajectory(T[0]))
    assert_less(np.linalg.norm(np.std(samples, axis=0) - std_trajectory) / n_steps, 0.001)


def test_promp():
    n_weights_per_dim = 15

    promp = ProMP(n_dims=2, n_weights_per_dim=n_weights_per_dim)

    random_state = np.random.RandomState(10)
    n_demos = 20
    n_steps = 51
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Ys = np.empty((n_demos, n_steps, 2))
    for demo_idx in range(n_demos):
        Ys[demo_idx, :, 0] = np.sin(2 * np.pi * T[demo_idx] + random_state.randn() * 0.3)
        Ys[demo_idx, :, 1] = 0.5 + np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.5)
    promp.imitate(T, Ys)

    mean_trajectory = promp.mean_trajectory(T[0])

    random_state = np.random.RandomState(0)
    samples = promp.sample_trajectories(T[0], 20, random_state)

    assert_less(np.linalg.norm(np.mean(samples, axis=0) - mean_trajectory) / n_steps, 0.01)

    std_trajectory = np.sqrt(promp.var_trajectory(T[0]))
    assert_less(np.linalg.norm(np.std(samples, axis=0) - std_trajectory) / n_steps, 0.012)


def test_promp_velocities():
    n_weights_per_dim = 100

    promp = ProMP(n_dims=1, n_weights_per_dim=n_weights_per_dim)

    random_state = np.random.RandomState(10)
    n_demos = 10
    n_steps = 101
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Y = np.empty((n_demos, n_steps, 1))
    for demo_idx in range(n_demos):
        Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
        Y[demo_idx, :, 0] += random_state.randn(n_steps) * 0.01
    promp.imitate(T, Y)

    random_state = np.random.RandomState(0)
    samples = promp.sample_trajectories(T[0], 10, random_state)

    mean_trajectory = promp.mean_trajectory(T[0])
    var_trajectory = np.sqrt(promp.var_trajectory(T[0]))

    mean_velocities = promp.mean_velocities(T[0])
    var_velocities = np.sqrt(promp.var_velocities(T[0]))

    dt = T[0, 1] - T[0, 0]
    int_velocities = mean_trajectory[0] + np.cumsum(mean_velocities * dt)
    assert_array_almost_equal(int_velocities, mean_trajectory[:, 0], decimal=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    plt.figure()
    n_weights_per_dim = 50

    promp = ProMP(n_dims=1, n_weights_per_dim=n_weights_per_dim)

    random_state = np.random.RandomState(10)
    n_demos = 10
    n_steps = 101
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Y = np.empty((n_demos, n_steps, 1))
    for demo_idx in range(n_demos):
        Y[demo_idx] = np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.1)[:, np.newaxis]
        Y[demo_idx, :, 0] += random_state.randn(n_steps) * 0.01
    promp.imitate(T, Y, verbose=1)

    for demo_idx in range(n_demos):
        plt.plot(T[demo_idx, :], Y[demo_idx, :], c="k", alpha=0.1)

    random_state = np.random.RandomState(0)
    samples = promp.sample_trajectories(T[0], 10, random_state)

    mean_trajectory = promp.mean_trajectory(T[0])
    plt.plot(T[0], mean_trajectory, label="Reproduction", c="r", lw=3)
    var_trajectory = np.sqrt(promp.var_trajectory(T[0]))
    factor = 2
    plt.fill_between(
        T[0],
        mean_trajectory[:, 0] - factor * var_trajectory[:, 0],
        mean_trajectory[:, 0] + factor * var_trajectory[:, 0],
        alpha=0.3)
    for sample in samples:
        plt.plot(T[0], sample, c="g", alpha=0.3)

    plt.figure()

    n_weights_per_dim = 15

    promp = ProMP(n_dims=2, n_weights_per_dim=n_weights_per_dim)

    random_state = np.random.RandomState(10)
    n_demos = 20
    n_steps = 51
    T = np.empty((n_demos, n_steps))
    T[:, :] = np.linspace(0.0, 1.0, n_steps)
    Ys = np.empty((n_demos, n_steps, 2))
    for demo_idx in range(n_demos):
        Ys[demo_idx, :, 0] = np.sin(2 * np.pi * T[demo_idx] + random_state.randn() * 0.3)
        Ys[demo_idx, :, 1] = 0.5 + np.cos(2 * np.pi * T[demo_idx] + random_state.randn() * 0.5)
    promp.imitate(T, Ys, verbose=1)

    plt.subplot(121)
    for demo_idx in range(n_demos):
        plt.plot(Ys[demo_idx, :, 0], Ys[demo_idx, :, 1], c="k", alpha=0.3)

    mean_trajectory = promp.mean_trajectory(T[0])
    plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], label="Reproduction", c="r", lw=3)
    plt.scatter(mean_trajectory[:, 0], mean_trajectory[:, 1], c="r")

    random_state = np.random.RandomState(0)
    samples = promp.sample_trajectories(T[0], 20, random_state)

    for sample in samples:
        plt.plot(sample[:, 0], sample[:, 1], c="g", alpha=0.3)

    plt.legend(loc="best")

    plt.subplot(122)
    std_trajectory = np.sqrt(promp.var_trajectory(T[0]))
    factor = 2
    plt.fill_between(
        T[0],
        mean_trajectory[:, 0] - factor * std_trajectory[:, 0],
        mean_trajectory[:, 0] + factor * std_trajectory[:, 0],
        color="orange", alpha=0.5)
    plt.plot(T[0], mean_trajectory[:, 0], color="orange")
    for Y in Ys:
        plt.plot(T[0], Y[:, 0], c="k", alpha=0.3, ls="--")
    for sample in samples:
        plt.plot(T[0], sample[:, 0], c="blue", alpha=0.3)
    plt.fill_between(
        T[0],
        mean_trajectory[:, 1] - factor * std_trajectory[:, 1],
        mean_trajectory[:, 1] + factor * std_trajectory[:, 1],
        color="blue", alpha=0.5)
    plt.plot(T[0], mean_trajectory[:, 1], color="blue")
    for Y in Ys:
        plt.plot(T[0], Y[:, 1], c="k", alpha=0.3, ls="--")
    for sample in samples:
        plt.plot(T[0], sample[:, 1], c="orange", alpha=0.3)

    plt.show()

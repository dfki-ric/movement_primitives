import numpy as np
from pytransform3d import rotations as pr
from pytransform3d import trajectories as ptr
from movement_primitives.spring_damper import SpringDamper, SpringDamperOrientation
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_less


def test_spring_damper():
    start_y = np.array([0.0])
    goal_y = np.array([1.0])

    for k in [5.0, 10.0, 50.0]:
        for c_factor in [0.5, 1.0, 2.0]:
            c = c_factor * 2.0 * np.sqrt(k)
            sd = SpringDamper(n_dims=1, dt=0.01, k=k, c=c, int_dt=0.001)
            sd.configure(start_y=start_y, goal_y=goal_y)
            T, Y = sd.open_loop(run_t=10.0)
            assert_array_almost_equal(Y[-1], goal_y, decimal=2)


def test_spring_damper_steps():
    start_y = np.array([0.0])
    start_yd = np.array([0.0])
    goal_y = np.array([1.0])

    k = 100.0
    c = 2.0 * np.sqrt(k)
    sd = SpringDamper(n_dims=1, dt=0.01, k=k, c=c, int_dt=0.001)
    sd.configure(start_y=start_y, goal_y=goal_y, start_yd=start_yd)
    y = np.copy(start_y)
    yd = np.copy(start_yd)
    for i in range(100):
        y, yd = sd.step(y, yd)
    error = np.linalg.norm(goal_y - y)
    assert_less(error, 6e-4)


def test_spring_damper_quaternion():
    dt = 0.01
    execution_time = 5.0

    sd = SpringDamperOrientation(k=4.0, c=2 * np.sqrt(4), dt=dt, int_dt=0.001)
    random_state = np.random.RandomState(42)
    start = pr.random_quaternion(random_state)
    goal = np.array([0.0, 0.0, 1.0, 0.0])
    sd.configure(start_y=start, goal_y=goal)

    T, Q = sd.open_loop(run_t=execution_time)
    assert_array_almost_equal(Q[0], start, decimal=3)
    assert_array_almost_equal(Q[-1], goal, decimal=3)


def test_spring_damper_quaternion_steps():
    random_state = np.random.RandomState(42)
    start = pr.random_quaternion(random_state)
    start_d = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.0, 0.0, 1.0, 0.0])

    k = 100.0
    c = 2.0 * np.sqrt(k)
    sd = SpringDamperOrientation(dt=0.01, k=k, c=c, int_dt=0.001)
    sd.configure(start_y=start, goal_y=goal, start_yd=start_d)
    y = np.copy(start)
    yd = np.copy(start_d)
    for i in range(100):
        y, yd = sd.step(y, yd)
    error = np.linalg.norm(goal - y)
    assert_less(error, 1e-3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    plt.figure()
    start_y = np.array([0.0])
    goal_y = np.array([1.0])

    for k in [5.0, 10.0, 50.0]:
        for c_factor in [0.5, 1.0, 2.0]:
            c = c_factor * 2.0 * np.sqrt(k)
            sd = SpringDamper(n_dims=1, dt=0.01, k=k, c=c, int_dt=0.001)
            sd.configure(start_y=start_y, goal_y=goal_y)
            T, Y = sd.open_loop(run_t=10.0)
            plt.plot(T, Y, label="k = %d, c = %.1f" % (k, c))
    plt.plot([0.0, 10.0], [goal_y, goal_y])
    plt.legend()

    plt.figure()
    dt = 0.01
    execution_time = 5.0

    sd = SpringDamperOrientation(k=4.0, c=2 * np.sqrt(4), dt=dt, int_dt=0.001)
    random_state = np.random.RandomState(42)
    start = pr.random_quaternion(random_state)
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

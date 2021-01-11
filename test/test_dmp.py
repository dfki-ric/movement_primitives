import numpy as np
from dmp import DMP
from nose.tools import assert_almost_equal, assert_equal
from numpy.testing import assert_array_almost_equal


def test_dmp1d():
    start_y = np.array([0.0])
    goal_y = np.array([1.0])
    dt = 0.05

    execution_time = 1.0
    dmp = DMP(n_dims=1, execution_time=execution_time, dt=dt, n_weights_per_dim=6)
    dmp.configure(start_y=start_y, goal_y=goal_y)
    random_state = np.random.RandomState(0)
    dmp.forcing_term.weights = 200 * random_state.randn(*dmp.forcing_term.weights.shape)

    T, Y = dmp.open_loop(run_t=2 * execution_time)
    assert_almost_equal(T[0], 0.0)
    assert_almost_equal(T[-1], 2 * execution_time)

    assert_equal(Y.ndim, 2)
    assert_equal(Y.shape[0], 41)
    assert_equal(Y.shape[1], 1)
    assert_array_almost_equal(Y[0], start_y)
    assert_array_almost_equal(Y[-1], goal_y, decimal=3)


def test_dmp1d_imitation():
    execution_time = 1.0
    dt = 0.001

    dmp = DMP(n_dims=1, execution_time=execution_time, dt=dt, n_weights_per_dim=100)

    T_demo = np.arange(0.0, 1.0 + dt, dt)
    Y_demo = np.cos(2 * np.pi * T_demo)[:, np.newaxis]
    dmp.imitate(T_demo, Y_demo)
    dmp.configure(start_y=Y_demo[0], goal_y=Y_demo[-1])

    T, Y = dmp.open_loop()
    assert_array_almost_equal(Y, Y_demo, decimal=2)

    new_start = np.array([1.0])
    new_goal = np.array([0.5])
    dmp.configure(start_y=new_start, goal_y=new_goal)
    T, Y = dmp.open_loop()
    assert_array_almost_equal(Y[0], new_start, decimal=4)
    assert_array_almost_equal(Y[-1], new_goal, decimal=4)


def test_dmp2d_imitation():
    execution_time = 1.0
    dt = 0.0003

    dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt,
              n_weights_per_dim=100)

    T = np.arange(0.0, execution_time + dt, dt)
    Y_demo = np.empty((len(T), 2))
    Y_demo[:, 0] = np.cos(2.5 * np.pi * T)
    Y_demo[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
    dmp.imitate(T, Y_demo)

    dmp.configure(start_y=Y_demo[0], goal_y=Y_demo[-1])
    T, Y = dmp.open_loop()
    assert_array_almost_equal(Y, Y_demo, decimal=2)

    new_start = np.array([1.0, 1.5])
    new_goal = np.array([0.2, 0.3])
    dmp.configure(start_y=new_start, goal_y=new_goal)
    T, Y = dmp.open_loop(run_t=execution_time)
    assert_array_almost_equal(Y[0], new_start, decimal=4)
    assert_array_almost_equal(Y[-1], new_goal, decimal=3)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    start_y = np.array([0.0])
    goal_y = np.array([1.0])

    execution_time = 1.0
    dmp = DMP(n_dims=1, execution_time=execution_time, dt=0.001, n_weights_per_dim=6)
    dmp.configure(start_y=start_y, goal_y=goal_y)
    random_state = np.random.RandomState(0)
    dmp.forcing_term.weights = 200 * random_state.randn(*dmp.forcing_term.weights.shape)

    T, Y = dmp.open_loop(run_t=2 * execution_time)

    plt.figure()
    plt.plot(T, Y, label="DMP")
    plt.scatter([0], start_y, label="Start")
    plt.scatter([execution_time], goal_y, label="Goal")
    plt.scatter([2 * execution_time], goal_y, label="Goal")
    plt.xlabel("T")
    plt.ylabel("State")
    plt.legend()
    plt.tight_layout()

    dt = 0.01

    dmp = DMP(n_dims=1, execution_time=execution_time, dt=0.01, n_weights_per_dim=10)

    T = np.linspace(0.0, 1.0, 101)
    Y = np.cos(2 * np.pi * T)[:, np.newaxis]
    dmp.imitate(T, Y)

    plt.figure()
    plt.plot(T, Y, label="Demo")
    plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T, Y = dmp.open_loop()
    plt.plot(T, Y, label="Reproduction")
    plt.scatter([T[0], T[-1]], [Y[0], Y[-1]])

    dmp.configure(start_y=np.array([1.0]), goal_y=np.array([0.5]))
    T, Y = dmp.open_loop(run_t=2.0)
    plt.plot(T, Y, label="Adaptation")
    plt.scatter([0.0, execution_time], [1.0, 0.5])

    plt.xlabel("T")
    plt.ylabel("State")
    plt.tight_layout()
    plt.legend(loc="best")

    execution_time = 1.0
    dt = 0.01

    dmp = DMP(n_dims=2, execution_time=execution_time, dt=dt, n_weights_per_dim=10)

    T = np.linspace(0.0, execution_time, 101)
    Y = np.empty((len(T), 2))
    Y[:, 0] = np.cos(2.5 * np.pi * T)
    Y[:, 1] = 0.5 + np.cos(1.5 * np.pi * T)
    dmp.imitate(T, Y)

    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], label="Demo")

    dmp.configure(start_y=Y[0], goal_y=Y[-1])
    T, Y = dmp.open_loop()
    plt.scatter(Y[:, 0], Y[:, 1], label="Reproduction")

    dmp.configure(start_y=np.array([1.0, 1.5]), goal_y=np.array([0.2, 0.3]))
    T, Y = dmp.open_loop(run_t=1.0)
    plt.scatter(Y[:, 0], Y[:, 1], label="Adaptation")

    plt.xlabel("State dimension 1")
    plt.ylabel("State dimension 1")
    plt.legend(loc="best")
    plt.tight_layout()

    plt.show()

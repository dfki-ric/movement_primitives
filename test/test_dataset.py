import numpy as np
from movement_primitives.data import (
    generate_minimum_jerk, load_lasa, generate_1d_trajectory_distribution)
from numpy.testing import assert_array_almost_equal
import pytest


def test_minimum_jerk_boundaries():
    random_state = np.random.RandomState(1)
    x0 = random_state.randn(2)
    g = x0 + random_state.rand(2)

    X, Xd, Xdd = generate_minimum_jerk(x0, g)
    assert_array_almost_equal(X[0], x0)
    assert_array_almost_equal(Xd[0], np.zeros(2))
    assert_array_almost_equal(Xdd[0], np.zeros(2))
    assert_array_almost_equal(X[-1], g)
    assert_array_almost_equal(Xd[-1], np.zeros(2))
    assert_array_almost_equal(Xdd[-1], np.zeros(2))

    for t in range(X.shape[0]):
        for d in range(X.shape[1]):
            assert X[t, d] >= x0[d]
            assert X[t, d] <= g[d]

    with pytest.raises(ValueError, match="Shape .* must be equal"):
        generate_minimum_jerk(x0, np.zeros(1))


def test_lasa():
    T, X, Xd, Xdd, dt, shape_name = load_lasa(0)
    assert T.shape[0] == X.shape[0]
    assert T.shape[1] == X.shape[1]
    assert X.shape[2] == 2
    assert Xd.shape[0] == X.shape[0]
    assert Xd.shape[1] == X.shape[1]
    assert Xd.shape[2] == X.shape[2]
    assert Xdd.shape[0] == X.shape[0]
    assert Xdd.shape[1] == X.shape[1]
    assert Xdd.shape[2] == X.shape[2]
    assert shape_name == "Angle"


@pytest.mark.parametrize("n_demos", [1, 3])
@pytest.mark.parametrize("n_steps", [2, 5, 21])
def test_lasa_preserves_recorded_times(tmp_path, monkeypatch, n_demos, n_steps):
    import scipy.io
    from movement_primitives.data import _lasa

    dataset = tmp_path / "lasa_data" / "DataSet"
    dataset.mkdir(parents=True)
    demos = np.empty((1, n_demos), dtype=object)
    expected_times = []
    expected_positions = []
    expected_velocities = []
    expected_accelerations = []
    for i in range(n_demos):
        dt = 0.002 * (i + 1)
        times = np.arange(n_steps) * dt
        positions = np.vstack((times ** 2, 3.0 * times))
        velocities = np.vstack((2.0 * times, np.full(n_steps, 3.0)))
        accelerations = np.vstack((np.full(n_steps, 2.0), np.zeros(n_steps)))
        demos[0, i] = dict(pos=positions, t=times[None, :], vel=velocities,
                           acc=accelerations, dt=dt)
        expected_times.append(times)
        expected_positions.append(positions.T)
        expected_velocities.append(velocities.T)
        expected_accelerations.append(accelerations.T)
    scipy.io.savemat(dataset / "Synthetic.mat", dict(demos=demos))
    monkeypatch.setattr(_lasa, "get_common_dataset_path",
                        lambda: str(tmp_path) + "/")

    T, X, Xd, Xdd, dt, shape_name = load_lasa(0)

    np.testing.assert_allclose(T, expected_times)
    np.testing.assert_allclose(X, expected_positions)
    np.testing.assert_allclose(Xd, expected_velocities)
    np.testing.assert_allclose(Xdd, expected_accelerations)
    assert dt == 0.002
    assert shape_name == "Synthetic"


def test_toy1d():
    T1, X1 = generate_1d_trajectory_distribution(
        n_demos=1, n_steps=11, noise_per_step_range=0.0,
        initial_offset_range=0, final_offset_range=0)

    T2, X2 = generate_1d_trajectory_distribution(
        n_demos=1000, n_steps=11, noise_per_step_range=20.0,
        initial_offset_range=0, final_offset_range=0)

    assert_array_almost_equal(
        np.mean(X1, axis=0), np.mean(X2, axis=0), decimal=1)

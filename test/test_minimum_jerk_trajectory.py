import numpy as np
from movement_primitives.minimum_jerk_trajectory import MinimumJerkTrajectory
from numpy.testing import assert_array_almost_equal, assert_allclose
import pytest
from movement_primitives.data import generate_minimum_jerk


def test_step_through_minimum_jerk_trajectory():
    mjt = MinimumJerkTrajectory(3, 1.0, 0.01)
    mjt.configure(start_y=np.zeros(3), goal_y=np.ones(3))
    y = np.zeros(3)
    yd = np.zeros(3)

    y, yd = mjt.n_steps_open_loop(y, yd, 50)
    phase = 49 * mjt.dt / mjt.execution_time
    position = 10 * phase ** 3 - 15 * phase ** 4 + 6 * phase ** 5
    velocity = (30 * phase ** 2 - 60 * phase ** 3 + 30 * phase ** 4) / mjt.execution_time
    assert_array_almost_equal(y, position * np.ones(3))
    assert_array_almost_equal(yd, velocity)

    y, yd = mjt.n_steps_open_loop(y, yd, 1)
    assert_array_almost_equal(y, 0.5 * np.ones(3), decimal=8)
    assert_array_almost_equal(yd, 1.875, decimal=7)

    y, yd = mjt.n_steps_open_loop(y, yd, 50)
    assert_array_almost_equal(y, np.ones(3))
    assert_array_almost_equal(yd, np.zeros(3))

    mjt.reset()

    y, yd = mjt.n_steps_open_loop(y, yd, 101)
    assert_array_almost_equal(y, np.ones(3))
    assert_array_almost_equal(yd, np.zeros(3))


@pytest.mark.parametrize("execution_time,dt", [
    (1.0, 0.01), (1.0, 0.1), (1.0, 0.5), (0.2, 0.1), (2.0, 0.25)])
@pytest.mark.parametrize("start,goal", [
    ([0.0], [1.0]), ([1.0, -2.0, 0.5], [-3.0, 2.0, 0.5]),
    ([1.0, 2.0], [1.0, 2.0])])
def test_minimum_jerk_sample_times(execution_time, dt, start, goal):
    positions, velocities, accelerations = generate_minimum_jerk(
        start, goal, execution_time, dt)
    phase = np.arange(len(positions))[:, np.newaxis] * dt / execution_time
    delta = np.asarray(goal) - np.asarray(start)
    expected_positions = start + delta * (
        10 * phase ** 3 - 15 * phase ** 4 + 6 * phase ** 5)
    expected_velocities = delta * (
        30 * phase ** 2 - 60 * phase ** 3 + 30 * phase ** 4) / execution_time
    expected_accelerations = delta * (
        60 * phase - 180 * phase ** 2 + 120 * phase ** 3) / execution_time ** 2
    assert_allclose(positions, expected_positions, atol=1e-12)
    assert_allclose(velocities, expected_velocities, atol=1e-12)
    assert_allclose(accelerations, expected_accelerations, atol=1e-11)
    assert_allclose(positions[-1], goal, atol=1e-12)
    assert_allclose(velocities[-1], 0.0, atol=1e-12)
    assert_allclose(accelerations[-1], 0.0, atol=1e-11)

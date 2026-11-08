import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from movement_primitives.dmp import DMP, CartesianDMP
from movement_primitives.dmp._dmp import DMP_STEP_FUNCTIONS
from movement_primitives.dmp._cartesian_dmp import CARTESIAN_DMP_STEP_FUNCTIONS


@pytest.mark.parametrize("step_function", list(DMP_STEP_FUNCTIONS))
@pytest.mark.parametrize("velocity", [-0.7, 0.0, 1.0])
def test_dmp_open_loop_matches_steps_with_initial_velocity(step_function,
                                                         velocity):
    dmp = DMP(n_dims=2, dt=0.01)
    dmp.configure(start_y=np.array([0.2, -0.3]),
                  goal_y=np.array([0.2, -0.3]),
                  start_yd=np.array([velocity, -velocity]))
    initial_velocity = dmp.start_yd.copy()
    T, Y = dmp.open_loop(run_t=0.1, step_function=step_function)
    y, yd = dmp.start_y.copy(), dmp.start_yd.copy()
    expected = [y.copy()]
    for _ in T[1:]:
        y, yd = dmp.step(y, yd, step_function=step_function)
        expected.append(y)
    assert_allclose(Y, expected, atol=1e-14)
    assert_array_equal(dmp.start_yd, initial_velocity)


@pytest.mark.parametrize("step_function", list(DMP_STEP_FUNCTIONS))
@pytest.mark.parametrize("quaternion_step_function",
                         list(CARTESIAN_DMP_STEP_FUNCTIONS))
@pytest.mark.parametrize("velocity", [-0.7, 0.0, 1.0])
def test_cartesian_open_loop_matches_steps_with_initial_velocity(
        step_function, quaternion_step_function, velocity):
    dmp = CartesianDMP(dt=0.01)
    y = np.array([0.2, -0.3, 0.7, 1.0, 0.0, 0.0, 0.0])
    dmp.configure(start_y=y.copy(), goal_y=y.copy(),
                  start_yd=velocity * np.array([1., -1., 2., -1., 2., 1.]))
    initial_velocity = dmp.start_yd.copy()
    T, Y = dmp.open_loop(
        run_t=0.1, step_function=step_function,
        quaternion_step_function=quaternion_step_function)
    yd = dmp.start_yd.copy()
    expected = [y.copy()]
    for _ in T[1:]:
        y, yd = dmp.step(
            y, yd, step_function=DMP_STEP_FUNCTIONS[step_function],
            quaternion_step_function=CARTESIAN_DMP_STEP_FUNCTIONS[
                quaternion_step_function])
        expected.append(y)
    assert_allclose(Y, expected, atol=1e-14)
    assert_array_equal(dmp.start_yd, initial_velocity)

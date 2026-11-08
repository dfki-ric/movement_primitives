import numpy as np
import pytest
from numpy.testing import assert_allclose
from movement_primitives.dmp._dmp import DMP_STEP_FUNCTIONS


class TimeForcingTerm:
    """Expose time as phase to test a known nonautonomous acceleration."""
    def __init__(self, n_dims, power):
        self.n_dims = n_dims
        self.power = power
        self.queries = []

    def phase(self, t):
        self.queries.append(np.copy(t))
        return t

    def forcing_term(self, t):
        return np.tile(t ** self.power, (self.n_dims, 1))


@pytest.mark.parametrize("step_function", ["rk4", "rk4-cython"])
@pytest.mark.parametrize("last_t", [0.0, 0.3, -0.4])
@pytest.mark.parametrize("power", [0, 1, 2])
def test_rk4_integrates_time_dependent_acceleration(step_function, last_t,
                                                  power):
    if step_function not in DMP_STEP_FUNCTIONS:
        pytest.skip("Cython extension not available")
    dt = 0.1
    t = last_t + dt
    forcing_term = TimeForcingTerm(2, power)
    y = np.array([0.2, -0.3])
    yd = np.array([0.7, -0.5])
    # With alpha=0 and duration=1, the DMP is y'' = time**power.
    expected_y = y + dt * yd
    expected_y += (t * (t ** (power + 1) - last_t ** (power + 1))
                   / (power + 1)
                   - (t ** (power + 2) - last_t ** (power + 2))
                   / (power + 2))
    expected_yd = yd + (t ** (power + 1) - last_t ** (power + 1)) / (power + 1)
    zeros = np.zeros(2)
    DMP_STEP_FUNCTIONS[step_function](
        last_t, t, y, yd, zeros, zeros, zeros, zeros, zeros, zeros,
        goal_t=1.0, start_t=0.0, alpha_y=zeros, beta_y=zeros,
        forcing_term=forcing_term)
    assert_allclose(y, expected_y, atol=1e-14)
    assert_allclose(yd, expected_yd, atol=1e-14)


@pytest.mark.parametrize("step_function", ["rk4", "rk4-cython"])
@pytest.mark.parametrize("last_t", [0.0, 0.3, -0.4])
def test_rk4_forcing_queries_stay_inside_step(step_function, last_t):
    if step_function not in DMP_STEP_FUNCTIONS:
        pytest.skip("Cython extension not available")
    forcing_term = TimeForcingTerm(2, 0)
    zeros = np.zeros(2)
    t = last_t + 0.1
    DMP_STEP_FUNCTIONS[step_function](
        last_t, t, zeros.copy(), zeros.copy(), zeros, zeros, zeros,
        zeros, zeros, zeros, goal_t=1.0, start_t=0.0,
        alpha_y=zeros, beta_y=zeros, forcing_term=forcing_term)
    assert_allclose(forcing_term.queries[0], [last_t, last_t + 0.05, t],
                    atol=1e-14)

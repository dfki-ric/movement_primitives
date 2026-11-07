import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from movement_primitives.dmp import (
    CouplingTermDualCartesianDistance, DualCartesianDMP)


@pytest.mark.parametrize("lf", [(1, 1), (1, 0), (0, 1), (0, 0)])
@pytest.mark.parametrize("desired_distance", [0.0, 0.4, -0.4])
def test_distance_coupling_at_coincident_positions(lf, desired_distance):
    y = np.tile([0.2, -0.3, 0.7, 1.0, 0.0, 0.0, 0.0], 2)
    original = y.copy()
    ct = CouplingTermDualCartesianDistance(desired_distance, lf)
    with np.errstate(invalid="raise", divide="raise"):
        cd, cdd = ct.coupling(y)
    # No separation direction is defined at coincidence.
    assert_array_equal(cd, np.zeros(12))
    assert_array_equal(cdd, np.zeros(12))
    assert_array_equal(y, original)


@pytest.mark.parametrize("desired_distance", [0.0, 0.4, -0.4])
@pytest.mark.parametrize("separation", [1.0, 1e-14])
def test_nonzero_distance_coupling_keeps_analytic_force(
        desired_distance, separation):
    y = np.tile([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 2)
    direction = np.array([1.0, -2.0, 2.0]) / 3.0
    y[:3] = separation * direction
    ct = CouplingTermDualCartesianDistance(desired_distance, (1, 1), k=2.0,
                                           c1=3.0, c2=5.0)
    force = 2.0 * (abs(desired_distance) - separation) * direction
    cd, cdd = ct.coupling(y)
    assert_allclose(cd, np.r_[3.0 * force, np.zeros(3),
                              -3.0 * force, np.zeros(3)], atol=1e-28)
    assert_allclose(cdd, np.r_[5.0 * force, np.zeros(3),
                               -5.0 * force, np.zeros(3)], atol=1e-28)


def test_zero_distance_coupling_preserves_stationary_dual_rollout():
    y = np.tile([0.2, -0.3, 0.7, 1.0, 0.0, 0.0, 0.0], 2)
    dmp = DualCartesianDMP(execution_time=0.1, dt=0.01)
    dmp.configure(start_y=y.copy(), goal_y=y.copy())
    ct = CouplingTermDualCartesianDistance(0.0, (1, 1))
    with np.errstate(invalid="raise", divide="raise"):
        T, Y = dmp.open_loop(coupling_term=ct)
    assert_allclose(Y, np.tile(y, (len(T), 1)))

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from movement_primitives.dmp import DMP, DMPWithFinalVelocity


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("variant", ["regular", "smooth", "final", "allowed"])
def test_imitate_matches_double_precision_demonstration(dtype, variant):
    T = np.linspace(0.0, 1.3, 21)
    samples = np.arange(len(T))
    Y = np.column_stack((samples, samples ** 2)).astype(dtype)
    original = Y.copy()
    Y.flags.writeable = False
    kwargs = dict(n_dims=2, execution_time=T[-1], dt=0.05)
    imitate_kwargs = {}
    if variant == "final":
        cls = DMPWithFinalVelocity
    else:
        cls = DMP
        kwargs["smooth_scaling"] = variant == "smooth"
        imitate_kwargs["allow_final_velocity"] = variant == "allowed"
    actual = cls(**kwargs)
    expected = cls(**kwargs)
    actual.imitate(T, Y, **imitate_kwargs)
    expected.imitate(T, Y.astype(float), **imitate_kwargs)

    assert_allclose(actual.get_weights(), expected.get_weights(), rtol=1e-12)
    for name in ["start_y", "start_yd", "start_ydd", "goal_y", "goal_yd",
                 "goal_ydd"]:
        assert_allclose(getattr(actual, name), getattr(expected, name),
                        rtol=1e-12)
    actual_T, actual_Y = actual.open_loop()
    expected_T, expected_Y = expected.open_loop()
    assert_array_equal(actual_T, expected_T)
    assert_allclose(actual_Y, expected_Y, rtol=1e-12)
    assert_array_equal(Y, original)

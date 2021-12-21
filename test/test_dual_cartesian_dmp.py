from movement_primitives.dmp import DualCartesianDMP
from nose.tools import assert_raises_regexp


def test_invalid_step_function():
    dmp = DualCartesianDMP(
        execution_time=1.0, dt=0.01, n_weights_per_dim=10, int_dt=0.001)
    assert_raises_regexp(ValueError, "Step function", dmp.open_loop,
                         step_function="invalid")

import numpy as np
from numpy.testing import assert_array_almost_equal


def test_compare_python_cython():
    from copy import deepcopy
    from movement_primitives.dmp._dual_cartesian_dmp import dmp_step_dual_cartesian_python
    from movement_primitives.dmp_fast import dmp_step_dual_cartesian as dmp_step_dual_cartesian_cython
    kwargs = dict(
        last_t=1.999, t=2.0,
        current_y=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), current_yd=np.zeros([12]),
        goal_y=np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), goal_yd=np.zeros([12]), goal_ydd=np.zeros([12]),
        start_y=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), start_yd=np.zeros([12]), start_ydd=np.zeros([12]),
        goal_t=2.0, start_t=0.0, alpha_y=25.0, beta_y=6.25,
        forcing_term=lambda x: 10000 * np.ones((12, 1)), coupling_term=None, int_dt=0.0001,
        p_gain=0.0, tracking_error=None
    )
    kwargs_python = deepcopy(kwargs)
    dmp_step_dual_cartesian_python(**kwargs_python)
    kwargs_cython = deepcopy(kwargs)
    dmp_step_dual_cartesian_cython(**kwargs_cython)

    assert_array_almost_equal(kwargs_python["current_y"], kwargs_cython["current_y"])
    assert_array_almost_equal(kwargs_python["current_yd"], kwargs_cython["current_yd"])

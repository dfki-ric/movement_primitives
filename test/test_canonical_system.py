import numpy as np
from movement_primitives.dmp._canonical_system import canonical_system_alpha
from movement_primitives.dmp._canonical_system import phase as phase_python
import pytest


def test_phase_cython():
    from movement_primitives.dmp_fast import phase as phase_cython
    goal_t = 1.0
    start_t = 0.0
    int_dt = 0.001
    alpha = canonical_system_alpha(0.01, goal_t, start_t)
    for t in np.linspace(0, 1, 101):
        z_python = phase_python(t, alpha, goal_t, start_t)
        z_cython = phase_cython(t, alpha, goal_t, start_t)
        assert z_cython == pytest.approx(z_python)

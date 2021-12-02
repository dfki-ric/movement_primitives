import numpy as np
from movement_primitives.dmp._canonical_system import canonical_system_alpha, phase
from nose.tools import assert_raises_regexp
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_canonical_system_alpha_errors():
    assert_raises_regexp(
        ValueError, "Final phase must be > 0!",
        canonical_system_alpha, -0.01, 1.0, 0.0)
    assert_raises_regexp(
        ValueError, "Goal must be chronologically after start!",
        canonical_system_alpha, 0.01, 0.0, 1.0)


def test_phase():
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    t = np.linspace(0.0, 1.0, 11)
    z = phase(t, alpha_z, goal_t=1.0, start_t=0.0)
    assert_array_equal(z.shape, (11,))
    z_expected = np.array([
        1.0, 0.63095734, 0.39810717, 0.25118864, 0.15848932,
        0.1, 0.06309573, 0.03981072, 0.02511886, 0.01584893,
        0.01])
    assert_array_almost_equal(z, z_expected)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    t = np.linspace(0.0, 1.0, 1001)
    z = phase(t, alpha_z, goal_t=1.0, start_t=0.0)

    plt.plot(t, z)
    plt.show()

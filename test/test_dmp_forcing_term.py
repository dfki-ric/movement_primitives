import numpy as np
from dmp import canonical_system_alpha, ForcingTerm
from nose.tools import assert_equal


def test_forcing_term():
    n_dims = 2
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    forcing_term = ForcingTerm(
        n_dims=n_dims, n_weights_per_dim=6, goal_t=1.0, start_t=0.0,
        overlap=0.8, alpha_z=alpha_z)
    T = np.linspace(0.0, 1.0, 1001)
    random_state = np.random.RandomState(22)
    forcing_term.weights = random_state.randn(*forcing_term.weights.shape)
    f = forcing_term(T)

    assert_equal(f.ndim, 2)
    assert_equal(f.shape[0], n_dims)
    assert_equal(f.shape[1], len(T))


if __name__ == "__main__":
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    forcing_term = ForcingTerm(n_dims=2, n_weights_per_dim=6, goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)
    t = np.linspace(0.0, 1.0, 1001)
    random_state = np.random.RandomState(22)
    forcing_term.weights = random_state.randn(*forcing_term.weights.shape)
    f = forcing_term(t)


    import matplotlib.pyplot as plt
    for d in range(f.shape[0]):
        plt.plot(t, f[d])
    plt.show()

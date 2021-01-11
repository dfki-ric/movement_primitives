import numpy as np
from movement_primitives import canonical_system_alpha, phase, ForcingTerm
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


def test_activations():
    n_dims = 1
    n_weights_per_dim = 6
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    forcing_term = ForcingTerm(
        n_dims=n_dims, n_weights_per_dim=n_weights_per_dim,
        goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)

    T = np.linspace(0.0, 1.0, 1001)
    z = phase(T, alpha_z, goal_t=1.0, start_t=0.0)
    activations = forcing_term._activations(z, normalized=True)

    assert_equal(activations.shape[0], n_weights_per_dim)
    assert_equal(activations.shape[1], len(T))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)

    forcing_term = ForcingTerm(n_dims=2, n_weights_per_dim=6, goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)
    T = np.linspace(0.0, 1.0, 1001)
    random_state = np.random.RandomState(22)
    forcing_term.weights = random_state.randn(*forcing_term.weights.shape)
    f = forcing_term(T)

    plt.figure()
    for rbf_idx in range(f.shape[0]):
        plt.plot(T, f[rbf_idx], label="Forcing term, dimension #%d" % (rbf_idx + 1))

    plt.legend()
    plt.xlabel("T")
    plt.ylabel("Forcing term")
    plt.tight_layout()

    forcing_term = ForcingTerm(n_dims=1, n_weights_per_dim=6, goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)
    T = np.linspace(0.0, 1.0, 1001)
    z = phase(T, alpha_z, goal_t=1.0, start_t=0.0)
    activations = forcing_term._activations(z, normalized=True)

    plt.figure()
    for rbf_idx in range(activations.shape[0]):
        plt.plot(T, activations[rbf_idx], label="Activation, RBF #%d" % (rbf_idx + 1))

    plt.legend()
    plt.xlabel("T")
    plt.ylabel("Activations")
    plt.tight_layout()

    plt.show()

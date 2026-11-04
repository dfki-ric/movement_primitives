import numpy as np
import pytest
from numpy.testing import assert_allclose

from movement_primitives.dmp._canonical_system import canonical_system_alpha, phase
from movement_primitives.dmp._forcing_term import ForcingTerm


def test_invalid_arguments():
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    with pytest.raises(ValueError, match="The number of weights per dimension must be > 1!"):
        ForcingTerm(n_dims=2, n_weights_per_dim=-1, goal_t=1.0, start_t=0.0,
                    overlap=0.7, alpha_z=alpha_z)
    with pytest.raises(ValueError, match="Goal must be chronologically after start!"):
        ForcingTerm(n_dims=2, n_weights_per_dim=2, goal_t=0.0, start_t=1.0,
                    overlap=0.7, alpha_z=alpha_z)


def test_forcing_term():
    n_dims = 2
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    forcing_term = ForcingTerm(
        n_dims=n_dims, n_weights_per_dim=6, goal_t=1.0, start_t=0.0,
        overlap=0.8, alpha_z=alpha_z)
    T = np.linspace(0.0, 1.0, 1001)
    random_state = np.random.RandomState(22)
    forcing_term.weights_ = random_state.randn(*forcing_term.weights_.shape)
    f = forcing_term(T)

    assert f.ndim == 2
    assert f.shape[0] == n_dims
    assert f.shape[1] == len(T)


def test_activations():
    n_dims = 1
    n_weights_per_dim = 6
    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)
    forcing_term = ForcingTerm(
        n_dims=n_dims, n_weights_per_dim=n_weights_per_dim,
        goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)

    T = np.linspace(0.0, 1.0, 1001)
    z = phase(T, alpha_z, goal_t=1.0, start_t=0.0)
    activations = forcing_term._activations(z)

    assert activations.shape[0] == n_weights_per_dim
    assert activations.shape[1] == len(T)


@pytest.mark.parametrize("start", [-2.0, 0.0, 0.25, 2.0])
@pytest.mark.parametrize("duration", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("n_weights", [2, 6, 10])
def test_forcing_time_origin(start, duration, n_weights):
    alpha = -np.log(0.01)
    baseline = ForcingTerm(2, n_weights, duration, 0.0, 0.8, alpha)
    shifted = ForcingTerm(2, n_weights, start + duration, start, 0.8, alpha)
    weights = np.arange(2 * n_weights).reshape(2, n_weights) * 0.1
    baseline.weights_[:] = shifted.weights_[:] = weights
    assert_allclose(shifted.centers, baseline.centers, rtol=1e-12, atol=1e-14)
    assert_allclose(shifted.widths, baseline.widths, rtol=1e-12, atol=1e-14)
    times = np.linspace(0.0, duration, 31)
    assert_allclose(shifted(times + start), baseline(times), rtol=1e-12, atol=1e-14)
    assert_allclose(shifted.design_matrix(times + start),
                    baseline.design_matrix(times), rtol=1e-12, atol=1e-14)
    assert_allclose(shifted.centers[[0, -1]], [1.0, 0.01])


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha_z = canonical_system_alpha(goal_z=0.01, goal_t=1.0, start_t=0.0)

    forcing_term = ForcingTerm(n_dims=2, n_weights_per_dim=6, goal_t=1.0, start_t=0.0, overlap=0.8, alpha_z=alpha_z)
    T = np.linspace(0.0, 1.0, 1001)
    random_state = np.random.RandomState(22)
    forcing_term.weights_ = random_state.randn(*forcing_term.weights_.shape)
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
    activations = forcing_term._activations(z)

    plt.figure()
    for rbf_idx in range(activations.shape[0]):
        plt.plot(T, activations[rbf_idx], label="Activation, RBF #%d" % (rbf_idx + 1))

    plt.legend()
    plt.xlabel("T")
    plt.ylabel("Activations")
    plt.tight_layout()

    plt.show()

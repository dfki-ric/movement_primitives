import numpy as np
from movement_primitives.data import load_lasa
from movement_primitives.dmp_to_state_space_distribution import propagate_weight_distribution_to_state_space


def test_dmp_to_state_space_distribution():  # smoke test
    n_weights_per_dim = 2
    T_2d, X_2d = load_lasa(0)[:2]
    T = T_2d[:, ::10] / 1000.0
    X = np.empty((X_2d.shape[0], X_2d.shape[1] // 10, 14))
    X[:, :, :2] = X_2d[:, ::10]
    X[:, :, 2] = 0.0
    X[:, :, 3] = 1.0
    X[:, :, 4:7] = 0.0
    X[:, :, 7:9] = X_2d[:, ::10]
    X[:, :, 9] = 0.0
    X[:, :, 10] = 1.0
    X[:, :, 10:] = 0.0
    dataset = [(t, x) for t, x in zip(T, X)]

    mvn = propagate_weight_distribution_to_state_space(
        dataset, n_weights_per_dim, alpha=1e-3, kappa=10.0,
        dt=0.1, int_dt=0.01, verbose=0)

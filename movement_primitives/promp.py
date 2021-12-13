"""Probabilistic movement primitive."""
import numpy as np


class ProMP:
    """Probabilistic Movement Primitive (ProMP).

    ProMPs have been proposed first in [1] and have been used later in [2,3].
    The learning algorithm is a specialized form of the one presented in [4].

    Note that internally we represented trajectories with the task space
    dimension as the first axis and the time step as the second axis while
    the exposed trajectory interface is transposed. In addition, we internally
    only use the a 1d array representation to make handling of the covariance
    simpler.

    Parameters
    ----------
    n_dims : int
        State space dimensions.

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.

    References
    ----------
    [1] Paraschos et al.: Probabilistic movement primitives, NeurIPS (2013),
    https://papers.nips.cc/paper/2013/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf

    [3] Maeda et al.: Probabilistic movement primitives for coordination of
    multiple human–robot collaborative tasks, AuRo 2017,
    https://link.springer.com/article/10.1007/s10514-016-9556-2

    [2] Paraschos et al.: Using probabilistic movement primitives in robotics, AuRo (2018),
    https://www.ias.informatik.tu-darmstadt.de/uploads/Team/AlexandrosParaschos/promps_auro.pdf,
    https://link.springer.com/article/10.1007/s10514-017-9648-7

    [4] Lazaric et al.: Bayesian Multi-Task Reinforcement Learning, ICML (2010),
    https://hal.inria.fr/inria-00475214/document
    """
    def __init__(self, n_dims, n_weights_per_dim=10):
        self.n_dims = n_dims
        self.n_weights_per_dim = n_weights_per_dim

        self.n_weights = n_dims * n_weights_per_dim

        self.weight_mean = np.zeros(self.n_weights)
        self.weight_cov = np.eye(self.n_weights)

        self.centers = np.linspace(0, 1, self.n_weights_per_dim)

    def weights(self, T, Y, lmbda=1e-12):
        """Obtain ProMP weights by linear regression.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Y : array-like, shape (n_steps, n_dims)
            Demonstrated trajectory

        lmbda : float, optional (default: 1e-12)
            Regularization coefficient

        Returns
        -------
        weights : array, shape (n_steps * n_weights_per_dim)
            ProMP weights
        """
        activations = self._rbfs_nd_sequence(T).T
        weights = np.linalg.pinv(
            activations.T.dot(activations)
            + lmbda * np.eye(activations.shape[1])
        ).dot(activations.T).dot(Y.T.ravel())
        return weights

    def trajectory_from_weights(self, T, weights):
        """Generate trajectory from ProMP weights.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        weights : array-like, shape (n_steps * n_weights_per_dim)
            ProMP weights

        Returns
        -------
        Y : array, shape (n_steps, n_dims)
            Trajectory
        """
        return self._rbfs_nd_sequence(T).T.dot(weights).reshape(
            self.n_dims, len(T)).T

    def condition_position(self, y_mean, y_cov=None, t=0, t_max=1.0):
        """Condition ProMP on a specific position (see page 4 of [1]).

        Parameters
        ----------
        y_mean : array, shape (n_dims,)
            Position mean

        y_cov : array, shape (n_dims, n_dims), optional (default: 0)
            Covariance of position

        t : float, optional (default: 0)
            Time at which the activations of RBFs will be queried. Note that
            we internally normalize the time so that t_max == 1.

        t_max : float, optional (default: 1)
            Duration of the ProMP

        Returns
        -------
        conditional_promp : ProMP
            New conditional ProMP

        References
        ----------
        [1] Paraschos et al.: Probabilistic movement primitives, NeurIPS (2013),
        https://papers.nips.cc/paper/2013/file/e53a0a2978c28872a4505bdb51db06dc-Paper.pdf
        """
        Psi_t = _nd_block_diagonal(
            self._rbfs_1d_point(t, t_max)[:, np.newaxis], self.n_dims)
        if y_cov is None:
            y_cov = 0.0

        common_term = self.weight_cov.dot(Psi_t).dot(
            np.linalg.inv(y_cov + Psi_t.T.dot(self.weight_cov).dot(Psi_t)))

        # Equation (5)
        weight_mean = (
            self.weight_mean
            + common_term.dot(y_mean - Psi_t.T.dot(self.weight_mean)))
        # Equation (6)
        weight_cov = (
            self.weight_cov - common_term.dot(Psi_t.T).dot(self.weight_cov))

        conditional_promp = ProMP(self.n_dims, self.n_weights_per_dim)
        conditional_promp.from_weight_distribution(weight_mean, weight_cov)
        return conditional_promp

    def mean_trajectory(self, T):
        """Get mean trajectory of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        Y : array, shape (n_steps, n_dims)
            Mean trajectory
        """
        return self.trajectory_from_weights(T, self.weight_mean)

    def cov_trajectory(self, T):
        """Get trajectory covariance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        cov : array, shape (n_dims * n_steps, n_dims * n_steps)
            Covariance
        """
        activations = self._rbfs_nd_sequence(T)
        return activations.T.dot(self.weight_cov).dot(activations)

    def var_trajectory(self, T):
        """Get trajectory variance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        var : array, shape (n_steps, n_dims)
            Variance
        """
        return np.maximum(np.diag(self.cov_trajectory(T)).reshape(
            self.n_dims, len(T)).T, 0.0)

    def mean_velocities(self, T):
        """Get mean velocities of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        Yd : array, shape (n_steps, n_dims)
            Mean velocities
        """
        return self._rbfs_derivative_nd_sequence(
            T).T.dot(self.weight_mean).reshape(self.n_dims, len(T)).T

    def cov_velocities(self, T):
        """Get velocity covariance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        cov : array, shape (n_dims * n_steps, n_dims * n_steps)
            Covariance
        """
        activations = self._rbfs_derivative_nd_sequence(T)
        return activations.T.dot(self.weight_cov).dot(activations)

    def var_velocities(self, T):
        """Get velocity variance of ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        Returns
        -------
        var : array, shape (n_steps, n_dims)
            Variance
        """
        return np.maximum(np.diag(self.cov_velocities(T)).reshape(
            self.n_dims, len(T)).T, 0.0)

    def sample_trajectories(self, T, n_samples, random_state):
        """Sample trajectories from ProMP.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Time steps

        n_samples : int
            Number of trajectories that will be sampled

        random_state : np.random.RandomState
            State of random number generator

        Returns
        -------
        samples : array, shape (n_samples, n_steps, n_dims)
            Sampled trajectories
        """
        weight_samples = random_state.multivariate_normal(
            self.weight_mean, self.weight_cov, n_samples)
        samples = np.empty((n_samples, len(T), self.n_dims))
        for i in range(n_samples):
            samples[i] = self.trajectory_from_weights(T, weight_samples[i])
        return samples

    def from_weight_distribution(self, mean, cov):
        """Initialize ProMP from mean and covariance in weight space.

        Parameters
        ----------
        mean : array, shape (n_dims * n_weights_per_dim)
            Mean of weight distribution

        cov : array, shape (n_dims * n_weights_per_dim, n_dims * n_weights_per_dim)
            Covariance of weight distribution

        Returns
        -------
        self : ProMP
            This object
        """
        self.weight_mean = mean
        self.weight_cov = cov
        return self

    def imitate(self, Ts, Ys, n_iter=1000, min_delta=1e-5, verbose=0):
        """Learn ProMP from multiple demonstrations.

        Parameters
        ----------
        Ts : array, shape (n_demos, n_steps)
            Time steps of demonstrations

        Ys : array, shape (n_demos, n_steps, n_dims)
            Demonstrations

        n_iter : int, optional (default: 1000)
            Maximum number of iterations

        min_delta : float, optional (default: 1e-5)
            Minimum delta between means to continue iteration

        verbose : int, optional (default: 0)
            Verbosity level
        """
        # Section 3.2 of https://hal.inria.fr/inria-00475214/document
        # P = I
        # mu_0 = 0
        # k_0 = 0
        # nu_0 = 0
        # Sigma_0 = 0
        # alpha_0 = 0
        # beta_0 = 0
        gamma = 0.7

        n_demos = len(Ts)
        self.variance = 1.0

        means = np.zeros((n_demos, self.n_weights))
        covs = np.empty((n_demos, self.n_weights, self.n_weights))

        # Precompute constant terms in expectation-maximization algorithm

        # n_demos x n_steps*self.n_dims x n_steps*self.n_dims
        Hs = []
        for demo_idx in range(n_demos):
            n_steps = len(Ys[demo_idx])
            H_partial = np.eye(n_steps)
            for y in range(n_steps - 1):
                H_partial[y, y + 1] = -gamma
            H = _nd_block_diagonal(H_partial, self.n_dims)
            Hs.append(H)

        # n_demos x n_steps*n_dims
        Ys_rearranged = [Y.T.ravel() for Y in Ys]

        # n_demos x n_steps*self.n_dims
        Rs = []
        for demo_idx in range(n_demos):
            R = Hs[demo_idx].dot(Ys_rearranged[demo_idx])
            Rs.append(R)

        # n_demos
        # RR in original code
        RTRs = []
        for demo_idx in range(n_demos):
            RTR = Rs[demo_idx].T.dot(Rs[demo_idx])
            RTRs.append(RTR)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # x self.n_dims*self.n_steps
        # BH in original code
        PhiHTs = []
        for demo_idx in range(n_demos):
            PhiHT = self._rbfs_nd_sequence(Ts[demo_idx]).dot(Hs[demo_idx].T)
            PhiHTs.append(PhiHT)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # mean_esteps in original code
        PhiHTRs = []
        for demo_idx in range(n_demos):
            PhiHTR = PhiHTs[demo_idx].dot(Rs[demo_idx])
            PhiHTRs.append(PhiHTR)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # x self.n_dims*self.n_weights_per_dim
        # cov_esteps in original code
        PhiHTHPhiTs = []
        for demo_idx in range(n_demos):
            PhiHTHPhiT = PhiHTs[demo_idx].dot(PhiHTs[demo_idx].T)
            PhiHTHPhiTs.append(PhiHTHPhiT)

        n_samples = sum([Y.shape[0] for Y in Ys])

        for it in range(n_iter):
            weight_mean_old = self.weight_mean

            for demo_idx in range(n_demos):
                means[demo_idx], covs[demo_idx] = self._expectation(
                        PhiHTRs[demo_idx], PhiHTHPhiTs[demo_idx])

            self._maximization(
                means, covs, RTRs, PhiHTRs, PhiHTHPhiTs, n_samples)

            delta = np.linalg.norm(self.weight_mean - weight_mean_old)
            if verbose:
                print("Iteration %04d: delta = %g" % (it + 1, delta))
            if delta < min_delta:
                break

    def _rbfs_1d_point(self, t, t_max=1.0, overlap=0.7):
        """Radial basis functions for one dimension and a point.

        Parameters
        ----------
        t : float
            Time at which the activations of RBFs will be queried. Note that
            we internally normalize the time so that t_max == 1.

        t_max : float, optional (default: 1)
            Duration of the ProMP

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_weights_per_dim,)
            Activations of RBFs for each time step.
        """
        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]
        t = t / t_max

        activations = np.exp(-(t - self.centers[:]) ** 2 / (2.0 * h))
        activations /= activations.sum(axis=0)  # normalize activations for each step

        assert activations.ndim == 1
        assert activations.shape[0] == self.n_weights_per_dim

        return activations

    def _rbfs_nd_sequence(self, T, overlap=0.7):
        """Radial basis functions for n_dims dimensions and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_dims * n_weights_per_dim, n_dims * n_steps)
            Activations of RBFs for each time step and each dimension.
        """
        return _nd_block_diagonal(
            self._rbfs_1d_sequence(T, overlap), self.n_dims)

    def _rbfs_1d_sequence(self, T, overlap=0.7, normalize=True):
        """Radial basis functions for one dimension and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        normalize : bool, optional (default: True)
            Normalize activations to sum up to one in each step

        Returns
        -------
        activations : array, shape (n_weights_per_dim, n_steps)
            Activations of RBFs for each time step.
        """
        assert T.ndim == 1

        n_steps = len(T)

        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]
        T = np.atleast_2d(T)
        T /= np.max(T)

        activations = np.exp(
            -(T - self.centers[:, np.newaxis]) ** 2 / (2.0 * h))
        if normalize:
            activations /= activations.sum(axis=0)

        assert activations.shape[0] == self.n_weights_per_dim
        assert activations.shape[1] == n_steps

        return activations

    def _rbfs_derivative_nd_sequence(self, T, overlap=0.7):
        """Derivative of RBFs for n_dims dimensions and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_dims * n_weights_per_dim, n_dims * n_steps)
            Activations of derivative of RBFs for each time step and dimension.
        """
        return _nd_block_diagonal(
            self._rbfs_derivative_1d_sequence(T, overlap), self.n_dims)

    def _rbfs_derivative_1d_sequence(self, T, overlap=0.7):
        """Derivative of RBFs for one dimension and a sequence.

        Parameters
        ----------
        T : array-like, shape (n_steps,)
            Times at which the activations of RBFs will be queried. Note that
            we assume that T[0] == 0.0 and the times will be normalized
            internally so that T[-1] == 1.0.

        overlap : float, optional (default: 0.7)
            Indicates how much the RBFs are allowed to overlap.

        Returns
        -------
        activations : array, shape (n_weights_per_dim, n_steps)
            Activations of derivative of RBFs for each time step.
        """
        assert T.ndim == 1

        n_steps = len(T)

        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        rbfs = self._rbfs_1d_sequence(T, overlap, normalize=False)
        rbfs_sum_per_step = rbfs.sum(axis=0)

        # normalize time to interval [0, 1]
        T = np.atleast_2d(T)
        T /= np.max(T)

        rbfs_deriv = (self.centers[:, np.newaxis] - T) / h
        rbfs_deriv *= rbfs
        rbfs_deriv_sum_per_step = rbfs_deriv.sum(axis=0)
        rbfs_deriv = (
             rbfs_deriv * rbfs_sum_per_step
             - rbfs * rbfs_deriv_sum_per_step) / (rbfs_sum_per_step ** 2)

        assert rbfs_deriv.shape[0] == self.n_weights_per_dim
        assert rbfs_deriv.shape[1] == n_steps

        return rbfs_deriv

    def _expectation(self, PhiHTR, PhiHTHPhiT):
        cov = np.linalg.pinv(PhiHTHPhiT / self.variance
                             + np.linalg.pinv(self.weight_cov))
        mean = cov.dot(PhiHTR / self.variance
                       + np.linalg.pinv(self.weight_cov).dot(self.weight_mean))
        return mean, cov

    def _maximization(self, means, covs, RRs, PhiHTR, PhiHTHPhiTs, n_samples):
        M = len(means)

        self.weight_mean = np.mean(means, axis=0)

        centered = means - self.weight_mean
        self.weight_cov = centered.T.dot(centered)
        for i in range(len(covs)):
            self.weight_cov += covs[i]
        self.weight_cov /= M  # TODO what is d + 2?

        self.variance = 0.0
        for i in range(len(means)):
            # a trace is the same irrelevant of the order of matrix
            # multiplications, see:
            # https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf,
            # Equation 16
            self.variance += np.trace(PhiHTHPhiTs[i].dot(covs[i]))

            self.variance += RRs[i]
            self.variance -= 2.0 * PhiHTR[i].T.dot(means[i].T)
            self.variance += (means[i].dot(PhiHTHPhiTs[i].dot(means[i].T)))

        # TODO why these factors?
        self.variance /= (np.linalg.norm(means) * M * self.n_dims * n_samples
                          + 2.0)
        #self.variance /= self.n_dims * n_samples + 2.0


def _nd_block_diagonal(partial_1d, n_dims):
    """Replicates matrix n_dims times to form a block-diagonal matrix.

    We also accept matrices of rectangular shape. In this case the result is
    not officially called a block-diagonal matrix anymore.

    Parameters
    ----------
    partial_1d : array, shape (n_block_rows, n_block_cols)
        Matrix that should be replicated.

    n_dims : int
        Number of times that the matrix has to be replicated.

    Returns
    -------
    full_nd : array, shape (n_block_rows * n_dims, n_block_cols * n_dims)
        Block-diagonal matrix with n_dims replications of the initial matrix.
    """
    assert partial_1d.ndim == 2
    n_block_rows, n_block_cols = partial_1d.shape

    full_nd = np.zeros((n_block_rows * n_dims, n_block_cols * n_dims))
    for j in range(n_dims):
        full_nd[n_block_rows * j:n_block_rows * (j + 1),
                n_block_cols * j:n_block_cols * (j + 1)] = partial_1d
    return full_nd

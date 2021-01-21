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
        activations = self._bf(T).T
        weights = np.linalg.pinv(
            activations.T.dot(activations) + lmbda * np.eye(activations.shape[1])
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
        return self._bf(T).T.dot(weights).reshape(self.n_dims, len(T)).T

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
        cov : array, shape TODO
            Covariance
        """
        activations = self._bf(T)
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
        return np.diag(self.cov_trajectory(T)).reshape(self.n_dims, len(T)).T

    def sample_trajectories(self, T, n_samples, random_state):
        """Sample trajectories from ProMP.

        TODO
        """
        weight_samples = random_state.multivariate_normal(
            self.weight_mean, self.weight_cov, n_samples)
        samples = np.empty((n_samples, len(T), self.n_dims))
        for i in range(n_samples):
            samples[i] = self.trajectory_from_weights(T, weight_samples[i])
        return samples

    def from_weight_distribution(self, mean, cov):
        """Initialize ProMP from mean and covariance.

        TODO
        """
        self.weight_mean = mean
        self.weight_cov = cov
        return self

    def imitate(self, Ts, Ys, gamma=0.7, n_iter=1000, min_delta=1e-5, verbose=0):
        """Learn ProMP from demonstrations.

        TODO
        """
        # https://github.com/rock-learning/bolero/blob/master/src/representation/promp/implementation/src/Trajectory.cpp#L64
        # https://git.hb.dfki.de/COROMA/PropMP/-/blob/master/prop_mp.ipynb
        # Section 3.2 of https://hal.inria.fr/inria-00475214/document

        # P = I
        # mu_0 = 0
        # k_0 = 0
        # nu_0 = 0
        # Sigma_0 = 0
        # alpha_0 = 0
        # beta_0 = 0

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
            H = np.zeros((n_steps * self.n_dims, n_steps * self.n_dims))
            for j in range(self.n_dims):
                H[n_steps * j:n_steps * (j + 1), n_steps * j:n_steps * (j + 1)] = H_partial
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

        # n_demos x self.n_dims*self.n_weights_per_dim x self.n_dims*self.n_steps
        # BH in original code
        PhiHTs = []
        for demo_idx in range(n_demos):
            PhiHT = self._bf(Ts[demo_idx]).dot(Hs[demo_idx].T)
            PhiHTs.append(PhiHT)

        # n_demos x self.n_dims*self.n_weights_per_dim
        # mean_esteps in original code
        PhiHTRs = []
        for demo_idx in range(n_demos):
            PhiHTR = PhiHTs[demo_idx].dot(Rs[demo_idx])
            PhiHTRs.append(PhiHTR)

        # n_demos x self.n_dims*self.n_weights_per_dim x self.n_dims*self.n_weights_per_dim
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

            self._maximization(means, covs, RTRs, PhiHTRs, PhiHTHPhiTs, n_samples)

            delta = np.linalg.norm(self.weight_mean - weight_mean_old)
            if verbose:
                print("Iteration %04d: delta = %g" % (it + 1, delta))
            if delta < min_delta:
                break

    def _bf(self, T, overlap=0.7):
        """Radial basis functions for all dimensions.

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
            Activations of RBFs for each time step and in each dimension.
            All activations for dimension d can be found in
            activations[d * n_weights_per_dim:(d + 1) * n_weights_per_dim, d * n_steps:(d + 1) * n_steps]
            so that the inner indices run over time / basis function and the
            outer index over dimensions.
        """
        n_steps = len(T)
        ret = np.zeros((self.n_dims * self.n_weights_per_dim, self.n_dims * n_steps))
        for d in range(self.n_dims):
            ret[d * self.n_weights_per_dim:(d + 1) * self.n_weights_per_dim,
                d * n_steps:(d + 1) * n_steps] = self._rbfs(T, overlap)
        return ret

    def _rbfs(self, T, overlap=0.7):
        """Radial basis functions per dimension.

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
            Activations of RBFs for each time step.
        """
        assert T.ndim == 1

        n_steps = len(T)

        self.centers = np.linspace(0, 1, self.n_weights_per_dim)
        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]
        T = np.atleast_2d(T)
        T /= np.max(T)

        activations = np.exp(-(T - self.centers[:, np.newaxis]) ** 2 / (2.0 * h))
        activations /= activations.sum(axis=0)  # normalize activations for each step

        assert activations.shape[0] == self.n_weights_per_dim
        assert activations.shape[1] == n_steps

        return activations

    def _expectation(self, PhiHTR, PhiHTHPhiT):
        cov = np.linalg.pinv(PhiHTHPhiT / self.variance + np.linalg.pinv(self.weight_cov))
        mean = cov.dot(PhiHTR / self.variance + np.linalg.pinv(self.weight_cov).dot(self.weight_mean))
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
            # a trace is the same irrelevant of the order of matrix multiplications,
            # see: https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf, Equation 16
            self.variance += np.trace(PhiHTHPhiTs[i].dot(covs[i]))

            self.variance += RRs[i]
            self.variance -= 2.0 * PhiHTR[i].T.dot(means[i].T)
            self.variance += (means[i].dot(PhiHTHPhiTs[i].dot(means[i].T)))

        self.variance /= np.linalg.norm(means) * M * self.n_dims * n_samples + 2.0  # TODO why these factors?
        #self.variance /= self.n_dims * n_samples + 2.0


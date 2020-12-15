import numpy as np


class ProMPBase:
    def configure(self, last_t=None, t=None):  # TODO viapoints / conditioning
        if last_t is not None:
            self.last_t = last_t
        if t is not None:
            self.t = t

    def _initialize(self, n_pos_dims, n_vel_dims):
        self.last_t = None
        self.t = 0

        self.current_y = np.zeros(n_pos_dims)
        self.current_yd = np.zeros(n_vel_dims)


class ProMP(ProMPBase):
    def __init__(self, n_dims, execution_time, dt=0.01, n_weights_per_dim=10, int_dt=0.001):
        self.n_dims = n_dims
        self.execution_time = execution_time
        self.dt = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt

        self._initialize(n_dims, n_dims)

        self.configure()

    def mean_trajectory(self, T):
        activations = self._rbfs(T)
        trajectory = np.empty((len(T), self.n_dims))
        for d in range(self.n_dims):
            trajectory[:, d] = activations.T.dot(self.weight_mean.reshape(self.n_dims, self.n_weights_per_dim)[d])
        return trajectory

    def sample_trajectories(self, T, n_samples, random_state):
        weight_samples = random_state.multivariate_normal(self.weight_mean, self.weight_cov, n_samples).reshape(n_samples, self.n_dims, self.n_weights_per_dim)
        samples = np.empty((n_samples, len(T), self.n_dims))
        activations = self._rbfs(T)
        for i in range(n_samples):
            for d in range(self.n_dims):
                samples[i, :, d] = activations.T.dot(weight_samples[i, d])
        return samples

    def imitate(self, Ts, Ys, lmbda=1e-5):
        n_demos = len(Ts)

        weights = np.empty((n_demos, self.n_weights_per_dim, self.n_dims))
        for i in range(n_demos):
            T = Ts[i]
            Y = Ys[i]
            # n_steps x self.n_weights_per_dim
            Phi = self._rbfs(T).T
            weights[i] = np.linalg.inv(Phi.T.dot(Phi) + lmbda * np.eye(self.n_weights_per_dim)).dot(Phi.T).dot(Y)

        self.weight_mean = np.mean(weights, axis=0)

        # TODO dependend dimensions
        self.weight_cov = np.zeros((self.n_dims, self.n_weights_per_dim, self.n_weights_per_dim))
        for d in range(self.n_dims):
            for i in range(n_demos):
                diff = weights[i, :, d] - self.weight_mean[:, d]
                self.weight_cov[d] += np.outer(diff, diff)
        self.weight_cov /= n_demos

    def imitate_scmtl(self, Ts, Ys, gamma=0.7, n_iter=1000, min_delta=1e-5, verbose=0):
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
        n_weights = self.n_dims * self.n_weights_per_dim

        self.weight_mean = np.zeros(n_weights)
        self.weight_cov = np.eye(n_weights)
        self.variance = 1.0

        means = np.zeros((n_demos, n_weights))
        covs = np.empty((n_demos, n_weights, n_weights))

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
        vals = []  # TODO rename
        for demo_idx in range(n_demos):
            n_steps = Ys[demo_idx].shape[0]
            val = np.zeros((n_steps * self.n_dims))
            for j in range(self.n_dims):
                val[n_steps * j:n_steps * (j + 1)] = Ys[demo_idx, :, j]
            vals.append(val)

        # n_demos x n_dims*self.n_weights_per_dim x self.n_dims*n_steps
        BFs = []
        for demo_idx in range(n_demos):
            BF = self._bf(Ts[demo_idx]).T
            BFs.append(BF)

        # n_demos x n_steps*self.n_dims
        Rs = []
        for demo_idx in range(n_demos):
            R = Hs[demo_idx].dot(vals[demo_idx])
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
            PhiHT = BFs[demo_idx].dot(Hs[demo_idx].T)
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

        n_samples = 0
        for demo_idx in range(n_demos):  # TODO more efficient
            n_samples += Ys[demo_idx].shape[0]

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

    def _rbfs(self, t, overlap=0.7, normalized=True):
        self.centers = np.linspace(0, 1, self.n_weights_per_dim)
        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        # normalize time to interval [0, 1]
        t = np.atleast_2d(t)
        t /= np.max(t)

        squared_dist = (t - self.centers[:, np.newaxis]) ** 2
        activations = np.exp(-squared_dist / (2.0 * h))
        if normalized:
            activations /= activations.sum(axis=0)
        return activations

    def _bf(self, time):
        n_steps = len(time)
        ret = np.zeros((self.n_dims * n_steps, self.n_dims * self.n_weights_per_dim))
        for d in range(self.n_dims):
            ret[d * n_steps:(d + 1) * n_steps, d * self.n_weights_per_dim:(d + 1) * self.n_weights_per_dim] = self._rbfs(time).T
        return ret

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

        #self.variance /= np.linalg.norm(means) * M * self.n_dims * n_samples + 2.0  # TODO why these factors?
        self.variance /= n_samples + 2.0


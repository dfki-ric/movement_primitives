import numpy as np


class ProMPBase:
    def configure(self, last_t=None, t=None):  # TODO viapoints / conditioning
        if last_t is not None:
            self.last_t = last_t
        if t is not None:
            self.t = t

    def _initialize(self, n_pos_dims, n_vel_dims):
        self.last_t = None
        self.t = 0.0

        self.start_y = np.zeros(n_pos_dims)
        self.start_yd = np.zeros(n_vel_dims)
        self.start_ydd = np.zeros(n_vel_dims)

        self.goal_y = np.zeros(n_pos_dims)
        self.goal_yd = np.zeros(n_vel_dims)
        self.goal_ydd = np.zeros(n_vel_dims)

        self.initialized = False

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

    def step(self, last_y, last_yd, coupling_term=None):
        self.last_t = self.t
        self.t += self.dt

        raise NotImplementedError()

        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None):
        raise NotImplementedError()

    def imitate(self, Ts, Ys, n_iter=100):
        # https://github.com/rock-learning/bolero/blob/master/src/representation/promp/implementation/src/Trajectory.cpp#L64
        # Section 3.2 of https://hal.inria.fr/inria-00475214/document

        n_demos = len(Ts)
        n_weights = self.n_dims * self.n_weights_per_dim

        self.weight_mean = np.zeros(n_weights)
        self.weight_cov = np.eye(n_weights)
        self.variance = 1.0

        means = np.zeros((n_demos, 2 * self.n_weights_per_dim))
        covs = np.empty((n_demos, 2 * self.n_weights_per_dim, 2 * self.n_weights_per_dim))
        Hs = []
        for demo_idx in range(n_demos):
            n_steps = len(Ys[demo_idx])
            H_partial = np.eye(n_steps)
            # https://github.com/rock-learning/bolero/blob/master/src/representation/promp/implementation/src/Trajectory.cpp#L80
            for y in range(n_steps - 1):
                H_partial[y, y + 1] = -0.7
            H = np.zeros((n_steps * self.n_dims, n_steps * self.n_dims))
            for j in range(self.n_dims):
                H[n_steps * j:n_steps * (j + 1), n_steps * j:n_steps * (j + 1)] = H_partial
            Hs.append(H)

        vals = []  # TODO rename
        for demo_idx in range(n_demos):
            n_steps = Ys[demo_idx].shape[0]
            val = np.zeros((n_steps * self.n_dims, 1))
            for j in range(self.n_dims):
                val[n_steps * j:n_steps * (j + 1)] = Ys[demo_idx, :n_steps, j:j + 1]
            vals.append(val)

        BFs = []
        for i in range(n_demos):
            BF = self._bf(Ts[i]).T  # TODO why .T?
            BFs.append(BF)

        Rs = []
        for i in range(n_demos):
            R = Hs[i].dot(vals[i])
            Rs.append(R)

        RRs = []
        for i in range(n_demos):
            RR = Rs[i].T.dot(Rs[i])
            RRs.append(RR)

        BHs = []
        for i in range(n_demos):
            BH = BFs[i].dot(Hs[i].T)
            BHs.append(BH)

        mean_esteps = []
        for i in range(n_demos):
            mean_estep = BHs[i].dot(Rs[i])
            mean_esteps.append(mean_estep)

        cov_esteps = []
        for i in range(n_demos):
            cov_estep = BHs[i].dot(BHs[i].T)
            cov_esteps.append(cov_estep)

        # TODO more efficient
        n_samples = 0
        for i in range(n_demos):
            n_samples += Ys[i].shape[0]

        for it in range(n_iter):
            weight_mean_old = self.weight_mean
            for i in range(n_demos):
                means[i], covs[i] = self._expectation(mean_esteps[i] / self.variance, cov_esteps[i] / self.variance)
            self._maximization(means, covs, RRs, mean_esteps, cov_esteps, n_samples)
            if np.linalg.norm(self.weight_mean - weight_mean_old) < 1e-5:
                break

    def _rbfs(self, t, overlap=0.7, normalized=True):
        self.centers = np.linspace(0, 1, self.n_weights_per_dim)
        h = -1.0 / (8.0 * self.n_weights_per_dim ** 2 * np.log(overlap))

        t = np.atleast_2d(t)  # 1 x n_steps
        t /= np.max(t)

        squared_dist = (t - self.centers[:, np.newaxis]) ** 2  # 1 x n_steps x
        activations = np.exp(squared_dist / (2.0 * h))
        if normalized:
            activations /= activations.sum(axis=0)
        return activations

    def _bf(self, time):
        n_steps = len(time)
        ret = np.zeros((self.n_dims * n_steps, self.n_dims * self.n_weights_per_dim))
        for d in range(self.n_dims):
            ret[d * n_steps:(d + 1) * n_steps, d * self.n_weights_per_dim:(d + 1) * self.n_weights_per_dim] = self._rbfs(time).T
        return ret

    def _expectation(self, mean_estep, cov_estep):
        cov = np.linalg.pinv(cov_estep + np.linalg.pinv(self.weight_cov))
        mean = cov.dot(mean_estep + np.linalg.pinv(self.weight_cov).dot(self.weight_mean))[:, 0]
        return mean, cov

    def _maximization(self, means, covs, RRs, RHs, HHs, n_samples):
        self.weight_mean = np.mean(means, axis=0)[0]  # TODO should reduce 2d array to 1d array
        centered = means - self.weight_mean
        self.weight_cov = centered.T.dot(centered)
        for i in range(len(covs)):
            self.weight_cov += covs[i]
        self.weight_cov /= len(covs)
        self.variance = 0.0
        for i in range(len(means)):
            self.variance += np.trace(HHs[i].dot(covs[i]))
            self.variance += RRs[i][0, 0]
            self.variance -= 2.0 * RHs[i].T.dot(means[i].T)[0, 0]
            self.variance += (means[i].dot(HHs[i].dot(means[i].T)))[0, 0]
        self.variance /= np.linalg.norm(means) * len(means) * self.n_dims * n_samples + 2


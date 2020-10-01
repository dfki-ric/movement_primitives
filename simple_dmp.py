import numpy as np


class DMP(object):
    def __init__(self, pastor_mod=False):
        self.pastor_mod = pastor_mod
        # Transformation system
        self.alpha = 25.0             # = D = 20.0
        self.beta = self.alpha / 4.0  # = K / D = 100.0 / 20.0 = 5.0
        # Canonical system
        self.alpha_t = self.alpha / 3.0
        # Obstacle avoidance
        self.gamma_o = 1000.0
        self.beta_o = 20.0 / np.pi

    def phase(self, n_steps, t=None):
        """The phase variable replaces explicit timing.

        It starts with 1 at the beginning of the movement and converges
        exponentially to 0.
        """
        phases = np.exp(-self.alpha_t * np.linspace(0, 1, n_steps))
        if t is None:
            return phases
        else:
            return phases[t]

    def spring_damper(self, x0, g, tau, s, X, Xd):
        """The transformation system generates a goal-directed movement."""
        if self.pastor_mod:
            # Allows smooth adaption to goals, in the original version also the
            # forcing term is multiplied by a constant alpha * beta which you
            # can of course omit since the weights will simply be scaled
            mod = -self.beta * (g - x0) * s
        else:
            mod = 0.0
        return self.alpha * (self.beta * (g - X) - tau * Xd + mod) / tau ** 2

    def forcing_term(self, x0, g, tau, w, s, X, scale=False):
        """The forcing term shapes the movement based on the weights."""
        n_features = w.shape[1]
        f = np.dot(w, self._features(tau, n_features, s))
        if scale:
            f *= g - x0

        if X.ndim == 3:
            F = np.empty_like(X)
            F[:, :] = f
            return F
        else:
            return f

    def _features(self, tau, n_features, s):
        if n_features == 0:
            return np.array([])
        elif n_features == 1:
            return np.array([1.0])
        c = self.phase(n_features)
        h = np.diff(c)
        h = np.hstack((h, [h[-1]]))
        phi = np.exp(-h * (s - c) ** 2)
        return s * phi / phi.sum()

    def obstacle(self, o, X, Xd):
        """Obstacle avoidance is based on point obstacles."""
        if X.ndim == 1:
          X = X[np.newaxis, np.newaxis, :]
        if Xd.ndim == 1:
          Xd = Xd[np.newaxis, np.newaxis, :]

        C = np.zeros_like(X)
        R = np.array([[np.cos(np.pi / 2.0), -np.sin(np.pi / 2.0)],
                      [np.sin(np.pi / 2.0),  np.cos(np.pi / 2.0)]])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obstacle_diff = o - X[i, j]
                theta = (np.arccos(obstacle_diff.dot(Xd[i, j]) /
                                   (np.linalg.norm(obstacle_diff) *
                                    np.linalg.norm(Xd[i, j]) + 1e-10)))
                C[i, j] = (self.gamma_o * R.dot(Xd[i, j]) * theta *
                           np.exp(-self.beta_o * theta))

        return np.squeeze(C)

    def imitate(self, X, tau, n_features):
        n_steps, n_dims = X.shape
        dt = tau / float(n_steps - 1)
        g = X[:, -1]

        Xd = np.vstack((np.zeros((1, n_dims)), np.diff(X, axis=0) / dt))
        Xdd = np.vstack((np.zeros((1, n_dims)), np.diff(Xd, axis=0) / dt))

        F = tau * tau * Xdd - self.alpha * (self.beta * (g[:, np.newaxis] - X)
                                            - tau * Xd)

        design = np.array([self._features(tau, n_features, s)
                           for s in self.phase(n_steps)])
        #w = np.linalg.lstsq(design, F)[0].T
        from sklearn.linear_model import Ridge
        lr = Ridge(alpha=1.0, fit_intercept=False)
        lr.fit(design, F)
        w = lr.coef_

        return w


def trajectory(dmp, w, x0, g, tau, dt, o=None, shape=True, avoidance=False,
               verbose=0):
    """Generate trajectory from DMP in open loop."""
    if verbose >= 1:
        print("Trajectory with x0 = %s, g = %s, tau=%.2f, dt=%.3f"
              % (x0, g, tau, dt))

    x = x0.copy()
    xd = np.zeros_like(x, dtype=np.float64)
    xdd = np.zeros_like(x, dtype=np.float64)
    X = [x0.copy()]
    Xd = [xd.copy()]
    Xdd = [xdd.copy()]

    # Internally, we do Euler integration usually with a much smaller step size
    # than the step size required by the system
    internal_dt = min(0.001, dt)
    n_internal_steps = int(tau / internal_dt)
    steps_between_measurement = int(dt / internal_dt)

    # Usually we would initialize t with 0, but that results in floating point
    # errors for very small step sizes. To ensure that the condition t < tau
    # really works as expected, we add a constant that is smaller than
    # internal_dt.
    t = 0.5 * internal_dt
    ti = 0
    S = dmp.phase(n_internal_steps + 1)
    while t < tau:
        t += internal_dt
        ti += 1
        s = S[ti]

        x += internal_dt * xd
        xd += internal_dt * xdd

        sd = dmp.spring_damper(x0, g, tau, s, x, xd)
        f = dmp.forcing_term(x0, g, tau, w, s, x) if shape else 0.0
        C = dmp.obstacle(o, x, xd) if avoidance else 0.0
        xdd = sd + f + C

        if ti % steps_between_measurement == 0:
            X.append(x.copy())
            Xd.append(xd.copy())
            Xdd.append(xdd.copy())

    return np.array(X), np.array(Xd), np.array(Xdd)


def potential_field(dmp, t, v, w, x0, g, tau, dt, o, x_range, y_range,
                    n_tics):
    xx, yy = np.meshgrid(np.linspace(x_range[0], x_range[1], n_tics),
                         np.linspace(y_range[0], y_range[1], n_tics))
    x = np.array((xx, yy)).transpose((1, 2, 0))
    xd = np.empty_like(x)
    xd[:, :] = v

    n_steps = int(tau / dt)

    s = dmp.phase(n_steps, t)
    sd = dmp.spring_damper(x0, g, tau, s, x, xd)
    f = dmp.forcing_term(x0, g, tau, w, s, x)
    C = dmp.obstacle(o, x, xd)
    acc = sd + f + C
    return xx, yy, sd, f, C, acc


if __name__ == "__main__":
    import matplotlib.pyplot as plt


    x0 = np.array([0, 0], dtype=np.float64)
    g = np.array([1, 1], dtype=np.float64)
    tau = 1.0
    w = np.array([[-50.0, 100.0, 300.0],
                  [-200.0, -200.0, -200.0]])
    o = np.array([1.0, 0.5])
    dt = 0.01

    dmp = DMP()

    x_range = (-0.2, 1.2)
    y_range = (-0.2, 1.2)
    n_tics = 10

    G, _, _ = trajectory(dmp, w, x0, g, tau, dt, o, shape=False, avoidance=False)
    T, _, _ = trajectory(dmp, w, x0, g, tau, dt, o, shape=True, avoidance=False)
    O, _, _ = trajectory(dmp, w, x0, g, tau, dt, o, shape=True, avoidance=True)

    fig = plt.figure(figsize=(5, 5))

    plt.plot(G[:, 0], G[:, 1], lw=3, color="g", label="Goal-directed")
    plt.plot(T[:, 0], T[:, 1], lw=3, color="r", label="Shaped")
    plt.plot(O[:, 0], O[:, 1], lw=3, color="black", label="Obstacle avoidance")

    plt.plot(x0[0], x0[1], "o", color="b", markersize=10)
    plt.plot(g[0], g[1], "o", color="g", markersize=10)
    plt.plot(o[0], o[1], "o", color="y", markersize=10)
    plt.show()
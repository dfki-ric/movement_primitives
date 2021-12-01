import warnings
import numpy as np
import pytransform3d.rotations as pr
from ._base import DMPBase
from ._canonical_system import canonical_system_alpha
from ._forcing_term import ForcingTerm
from ._dmp import dmp_imitate
from ._cartesian_dmp import dmp_quaternion_imitation


class DualCartesianDMP(DMPBase):
    """Dual cartesian dynamical movement primitive.

    Each of the two Cartesian DMPs handles orientation and position separately.
    The orientation is represented by a quaternion. The quaternion DMP is
    implemented according to

    A. Ude, B. Nemec, T. Petric, J. Murimoto:
    Orientation in Cartesian space dynamic movement primitives (2014),
    IEEE International Conference on Robotics and Automation (ICRA),
    pp. 2997-3004, doi: 10.1109/ICRA.2014.6907291,
    https://ieeexplore.ieee.org/document/6907291

    While the dimension of the state space is 14, the dimension of the
    velocity, acceleration, and forcing term is 12.

    Parameters
    ----------
    execution_time : float
        Execution time of the DMP.

    dt : float, optional (default: 0.01)
        Time difference between DMP steps.

    n_weights_per_dim : int, optional (default: 10)
        Number of weights of the function approximator per dimension.

    int_dt : float, optional (default: 0.001)
        Time difference for Euler integration.

    p_gain : float, optional (default: 0)
        Gain for proportional controller of DMP tracking error.
        The domain is [0, execution_time**2/dt].

    Attributes
    ----------
    dt_ : float
        Time difference between DMP steps. This value can be changed to adapt
        the frequency.
    """
    def __init__(self, execution_time, dt=0.01,
                 n_weights_per_dim=10, int_dt=0.001, p_gain=0.0):
        super(DualCartesianDMP, self).__init__(14, 12)
        self.execution_time = execution_time
        self.dt_ = dt
        self.n_weights_per_dim = n_weights_per_dim
        self.int_dt = int_dt
        self.p_gain = p_gain
        alpha_z = canonical_system_alpha(
            0.01, self.execution_time, 0.0, self.int_dt)
        self.forcing_term = ForcingTerm(
            12, self.n_weights_per_dim, self.execution_time, 0.0, 0.8,
            alpha_z)

        self.alpha_y = 25.0
        self.beta_y = self.alpha_y / 4.0

    def step(self, last_y, last_yd, coupling_term=None):
        """DMP step.

        Parameters
        ----------
        last_y : array, shape (14,)
            Last state.

        last_yd : array, shape (12,)
            Last time derivative of state (velocity).

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        Returns
        -------
        y : array, shape (14,)
            Next state.

        yd : array, shape (12,)
            Next time derivative of state (velocity).
        """
        assert len(last_y) == self.n_dims
        assert len(last_yd) == 12

        self.last_t = self.t
        self.t += self.dt_

        if not self.initialized:
            self.current_y = np.copy(self.start_y)
            self.current_yd = np.copy(self.start_yd)
            self.initialized = True

        tracking_error = self.current_y - last_y
        for ops in (slice(3, 7), slice(10, 14)):
            tracking_error[ops] = pr.concatenate_quaternions(
                self.current_y[ops], pr.q_conj(last_y[ops]))
        self.current_y[:], self.current_yd[:] = last_y, last_yd
        dmp_step_dual_cartesian(
            self.last_t, self.t, self.current_y, self.current_yd,
            self.goal_y, self.goal_yd, self.goal_ydd,
            self.start_y, self.start_yd, self.start_ydd,
            self.execution_time, 0.0,
            self.alpha_y, self.beta_y,
            self.forcing_term, coupling_term,
            self.int_dt,
            self.p_gain, tracking_error)

        return np.copy(self.current_y), np.copy(self.current_yd)

    def open_loop(self, run_t=None, coupling_term=None):
        """Run DMP open loop.

        Parameters
        ----------
        run_t : float, optional (default: execution_time)
            Run time of DMP. Can be shorter or longer than execution_time.

        coupling_term : object, optional (default: None)
            Coupling term that will be added to velocity.

        Returns
        -------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, 14)
            State at each step.
        """
        if run_t is None:
            run_t = self.execution_time
        T = [0.0]
        Y = [np.copy(self.start_y)]
        y = np.copy(self.start_y)
        yd = np.copy(self.start_yd)
        while self.t < run_t:
            y, yd = self.step(y, yd, coupling_term)
            T.append(self.t)
            Y.append(np.copy(self.current_y))
        return np.array(T), np.vstack(Y)

    def imitate(self, T, Y, regularization_coefficient=0.0,
                allow_final_velocity=False):
        """Imitate demonstration.

        Parameters
        ----------
        T : array, shape (n_steps,)
            Time for each step.

        Y : array, shape (n_steps, 14)
            State at each step.

        regularization_coefficient : float, optional (default: 0)
            Regularization coefficient for regression.

        allow_final_velocity : bool, optional (default: False)
            Allow a final velocity.
        """
        self.forcing_term.weights[:3, :] = dmp_imitate(
            T, Y[:, :3],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity)[0]
        self.forcing_term.weights[3:6, :] = dmp_quaternion_imitation(
            T, Y[:, 3:7],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity)[0]
        self.forcing_term.weights[6:9, :] = dmp_imitate(
            T, Y[:, 7:10],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity)[0]
        self.forcing_term.weights[9:12, :] = dmp_quaternion_imitation(
            T, Y[:, 10:14],
            n_weights_per_dim=self.n_weights_per_dim,
            regularization_coefficient=regularization_coefficient,
            alpha_y=self.alpha_y, beta_y=self.beta_y,
            overlap=self.forcing_term.overlap,
            alpha_z=self.forcing_term.alpha_z,
            allow_final_velocity=allow_final_velocity)[0]

        self.configure(start_y=Y[0], goal_y=Y[-1])

    def get_weights(self):
        """Get weight vector of DMP.

        Returns
        -------
        weights : array, shape (12 * n_weights_per_dim,)
            Current weights of the DMP.
        """
        return self.forcing_term.weights.ravel()

    def set_weights(self, weights):
        """Set weight vector of DMP.

        Parameters
        ----------
        weights : array, shape (12 * n_weights_per_dim,)
            New weights of the DMP.
        """
        self.forcing_term.weights[:, :] = weights.reshape(
            -1, self.n_weights_per_dim)


pps = [0, 1, 2, 7, 8, 9]
pvs = [0, 1, 2, 6, 7, 8]


def dmp_step_dual_cartesian_python(
        last_t, t,
        current_y, current_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term, coupling_term=None, int_dt=0.001,
        p_gain=0.0, tracking_error=None):
    """Integrate bimanual Cartesian DMP for one step with Euler integration."""
    if t <= start_t:
        current_y[:] = start_y
        current_yd[:] = start_yd

    execution_time = goal_t - start_t

    current_ydd = np.empty_like(current_yd)

    cd, cdd = np.zeros_like(current_yd), np.zeros_like(current_ydd)

    current_t = last_t
    while current_t < t:
        dt = int_dt
        if t - current_t < int_dt:
            dt = t - current_t
        current_t += dt

        if coupling_term is not None:
            cd[:], cdd[:] = coupling_term.coupling(current_y, current_yd)

        f = forcing_term(current_t).squeeze()
        if tracking_error is not None:
            cdd[pvs] += p_gain * tracking_error[pps] / dt
            for ops, ovs in ((slice(3, 7), slice(3, 6)), (slice(10, 14), slice(9, 12))):
                cdd[ovs] += p_gain * pr.compact_axis_angle_from_quaternion(tracking_error[ops]) / dt

        # position components
        current_ydd[pvs] = (
            alpha_y * (beta_y * (goal_y[pps] - current_y[pps])
                       + execution_time * goal_yd[pvs]
                       - execution_time * current_yd[pvs])
            + goal_ydd[pvs] * execution_time ** 2 + f[pvs] + cdd[pvs]) / execution_time ** 2
        current_yd[pvs] += dt * current_ydd[pvs] + cd[pvs] / execution_time
        current_y[pps] += dt * current_yd[pvs]

        # orientation components
        for ops, ovs in ((slice(3, 7), slice(3, 6)), (slice(10, 14), slice(9, 12))):
            current_ydd[ovs] = (
                alpha_y * (beta_y * pr.compact_axis_angle_from_quaternion(
                                   pr.concatenate_quaternions(goal_y[ops], pr.q_conj(current_y[ops])))
                           - execution_time * current_yd[ovs]) + f[ovs] + cdd[ovs]) / execution_time ** 2
            current_yd[ovs] += dt * current_ydd[ovs] + cd[ovs] / execution_time
            current_y[ops] = pr.concatenate_quaternions(
                pr.quaternion_from_compact_axis_angle(dt * current_yd[ovs]), current_y[ops])


try:
    from dmp_fast import dmp_step_dual_cartesian
except ImportError:
    warnings.warn(
        "Could not import fast dual cartesian DMP. "
        "Build Cython extension if you want it.",
        UserWarning)
    dmp_step_dual_cartesian = dmp_step_dual_cartesian_python

# uncomment to overwrite cython version with python version:
#dmp_step_dual_cartesian = dmp_step_dual_cartesian_python

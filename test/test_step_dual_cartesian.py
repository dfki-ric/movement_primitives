import numpy as np
from pytransform3d import rotations as pr
from numpy.testing import assert_array_almost_equal


pps = [0, 1, 2, 7, 8, 9]
pvs = [0, 1, 2, 6, 7, 8]


def dmp_step_dual_cartesian_python(
        last_t, t,
        current_y, current_yd,
        goal_y, goal_yd, goal_ydd,
        start_y, start_yd, start_ydd,
        goal_t, start_t, alpha_y, beta_y,
        forcing_term, coupling_term=None, int_dt=0.001,
        k_tracking_error=0.0, tracking_error=None):
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
        # TODO handle tracking error of orientation correctly
        if tracking_error is not None:
            cdd[pvs] += k_tracking_error * tracking_error[pps] / dt

        # position components
        current_ydd[pvs] = (alpha_y * (beta_y * (goal_y[pps] - current_y[pps]) + execution_time * goal_yd[pvs] - execution_time * current_yd[pvs]) + goal_ydd[pvs] * execution_time ** 2 + f[pvs] + cdd[pvs]) / execution_time ** 2
        current_yd[pvs] += dt * current_ydd[pvs] + cd[pvs] / execution_time
        current_y[pps] += dt * current_yd[pvs]

        # TODO handle tracking error of orientation correctly
        # orientation components
        for ops, ovs in ((slice(3, 7), slice(3, 6)), (slice(10, 14), slice(9, 12))):
            current_ydd[ovs] = (alpha_y * (beta_y * pr.compact_axis_angle_from_quaternion(pr.concatenate_quaternions(goal_y[ops], pr.q_conj(current_y[ops]))) - execution_time * current_yd[ovs]) + f[ovs] + cdd[ovs]) / execution_time ** 2
            current_yd[ovs] += dt * current_ydd[ovs] + cd[ovs] / execution_time
            current_y[ops] = pr.concatenate_quaternions(pr.quaternion_from_compact_axis_angle(dt * current_yd[ovs]), current_y[ops])


def test_compare_python_cython():
    from copy import deepcopy
    from dmp_fast import dmp_step_dual_cartesian as dmp_step_dual_cartesian_cython
    kwargs = dict(
        last_t=1.999, t=2.0,
        current_y=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), current_yd=np.zeros([12]),
        goal_y=np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), goal_yd=np.zeros([12]), goal_ydd=np.zeros([12]),
        start_y=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), start_yd=np.zeros([12]), start_ydd=np.zeros([12]),
        goal_t=2.0, start_t=0.0, alpha_y=25.0, beta_y=6.25,
        forcing_term=lambda x: 10000 * np.ones((12, 1)), coupling_term=None, int_dt=0.0001,
        k_tracking_error=0.0, tracking_error=None
    )
    kwargs_python = deepcopy(kwargs)
    dmp_step_dual_cartesian_python(**kwargs_python)
    kwargs_cython = deepcopy(kwargs)
    dmp_step_dual_cartesian_cython(**kwargs_cython)

    assert_array_almost_equal(kwargs_python["current_y"], kwargs_cython["current_y"])
    assert_array_almost_equal(kwargs_python["current_yd"], kwargs_cython["current_yd"])

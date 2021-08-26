import numpy as np
from numpy.testing import assert_array_almost_equal
from pytransform3d.rotations import compact_axis_angle_from_quaternion as compact_axis_angle_from_quaternion_python
from nose import SkipTest


def test_compact_axis_angle_from_quaternion():
    try:
        from dmp_fast import compact_axis_angle_from_quaternion as compact_axis_angle_from_quaternion_cython
    except ImportError:
        raise SkipTest("Cython extension is not available")

    q = np.array([-9.99988684e-01, -4.73967909e-03,  3.10305586e-04,  2.67831240e-04])
    axis_angle_python = compact_axis_angle_from_quaternion_python(q.copy())
    axis_angle_cython = compact_axis_angle_from_quaternion_cython(q.copy())
    assert_array_almost_equal(axis_angle_python, axis_angle_cython)

    q2 = np.array([9.89307860e-01,  1.45783469e-01, -4.06641953e-03,  7.76339940e-04])
    axis_angle_2_python = compact_axis_angle_from_quaternion_python(q2.copy())
    axis_angle_2_cython = compact_axis_angle_from_quaternion_cython(q2.copy())
    assert_array_almost_equal(axis_angle_2_python, axis_angle_2_cython)

    q3 = np.array([1.00000001e-00, 1.15428500e-11, -7.24072746e-11,  1.28322353e-12])
    axis_angle_3_python = compact_axis_angle_from_quaternion_python(q3.copy())
    axis_angle_3_cython = compact_axis_angle_from_quaternion_cython(q3.copy())
    assert_array_almost_equal(axis_angle_3_python, axis_angle_3_cython)

    q4 = np.array([-9.99999999e-01, 1.15428500e-11, -7.24072746e-11,  1.28322353e-12])
    axis_angle_4_python = compact_axis_angle_from_quaternion_python(q4.copy())
    axis_angle_4_cython = compact_axis_angle_from_quaternion_cython(q4.copy())
    assert_array_almost_equal(axis_angle_4_python, axis_angle_4_cython)

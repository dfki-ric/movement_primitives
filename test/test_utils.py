import numpy as np
import pytest

from movement_primitives.utils import ensure_1d_array, check_1d_array_length


def test_ensure_1d_array_float():
    a = ensure_1d_array(5.0, 6, "a")
    assert a.ndim == 1
    assert a.shape[0] == 6


def test_ensure_1d_array():
    a = ensure_1d_array(np.ones(7), 7, "a")
    assert a.ndim == 1
    assert a.shape[0] == 7


def test_ensure_1d_array_wrong_size():
    with pytest.raises(
            ValueError,
            match=r"a has incorrect shape, expected \(8,\) got \(7,\)"):
        ensure_1d_array(np.ones(7), 8, "a")


def test_ensure_1d_array_wrong_shape():
    with pytest.raises(
            ValueError,
            match=r"a has incorrect shape, expected \(7,\) got \(1, 7\)"):
        ensure_1d_array(np.ones((1, 7)), 7, "a")


def test_check_1d_array_length_correct():
    check_1d_array_length([0, 1], "a", 2)


def test_check_1d_array_length_2_vs_1():
    with pytest.raises(
            ValueError,
            match=r"Expected a with 1 element, got 2."):
        check_1d_array_length([0, 2], "a", 1)


def test_check_1d_array_length_1_vs_2():
    with pytest.raises(
            ValueError,
            match=r"Expected b with 2 elements, got 1."):
        check_1d_array_length([0], "b", 2)

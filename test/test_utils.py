import numpy as np
import pytest

from movement_primitives.utils import ensure_1d_array


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

import os
from movement_primitives import data
from nose.tools import assert_equal
from nose import SkipTest


def test_load_dataset():
    if not os.path.exists("data/kuka/20200129_peg_in_hole/csv_processed/"
                          "01_peg_in_hole_both_arms/"):
        raise SkipTest("Test data not available.")
    if not data.mocap_available:
        raise SkipTest("mocap library not available.")

    dataset = data.load_kuka_dataset(
        "data/kuka/20200129_peg_in_hole/csv_processed/"
        "01_peg_in_hole_both_arms/*.csv")
    assert_equal(len(dataset), 10)

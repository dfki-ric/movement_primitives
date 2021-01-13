from movement_primitives.data import load_kuka_dataset
from nose.tools import assert_equal


def test_load_dataset():
    dataset = load_kuka_dataset(
        "data/kuka/20200129_peg_in_hole/csv_processed/"
        "01_peg_in_hole_both_arms/*.csv")
    assert_equal(len(dataset), 10)

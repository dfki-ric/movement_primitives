"""Tools for loading datasets."""
from ._lasa import load_lasa
from ._minimum_jerk import generate_minimum_jerk
from ._toy_1d import generate_1d_trajectory_distribution


__all__ = [
    "load_lasa",
    "generate_minimum_jerk",
    "generate_1d_trajectory_distribution"]


try:
    from ._mocap import (
        smooth_dual_arm_trajectories_pq, smooth_single_arm_trajectories_pq,
        transpose_dataset, load_mia_demo, load_kuka_demo, load_rh5_demo,
        load_kuka_dataset)
    mocap_available = True
except ImportError:
    mocap_available = False
    import warnings
    warnings.warn("mocap library is not available")


__all__ += [
    "smooth_dual_arm_trajectories_pq", "smooth_single_arm_trajectories_pq",
    "load_mia_demo", "load_kuka_demo", "load_rh5_demo",
    "load_kuka_dataset"]

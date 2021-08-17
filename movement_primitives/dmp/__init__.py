"""Dynamical movement primitive variants."""
from ._dmp import DMP, dmp_transformation_system
from ._cartesian_dmp import CartesianDMP
from ._dual_cartesian_dmp import DualCartesianDMP
from ._coupling_terms import (
    CouplingTerm, CouplingTermCartesianDistance, CouplingTermCartesianPosition,
    CouplingTermDualCartesianOrientation, CouplingTermDualCartesianPose,
    CouplingTermDualCartesianDistance, CouplingTermDualCartesianTrajectory)
from ._state_following_dmp import StateFollowingDMP


__all__ = [
    "DMP", "dmp_transformation_system", "CartesianDMP", "DualCartesianDMP",
    "CouplingTerm", "CouplingTermCartesianDistance",
    "CouplingTermCartesianPosition", "CouplingTermDualCartesianOrientation",
    "CouplingTermDualCartesianPose", "CouplingTermDualCartesianDistance",
    "CouplingTermDualCartesianTrajectory"]

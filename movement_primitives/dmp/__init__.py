"""Dynamical movement primitive variants."""
from ._dmp import DMP, dmp_transformation_system
from ._dmp_with_final_velocity import DMPWithFinalVelocity
from ._cartesian_dmp import CartesianDMP
from ._dual_cartesian_dmp import DualCartesianDMP
from ._coupling_terms import (
    CouplingTerm, CouplingTermObstacleAvoidance, CouplingTermCartesianDistance,
    CouplingTermCartesianPosition, CouplingTermDualCartesianOrientation,
    CouplingTermDualCartesianPose, CouplingTermDualCartesianDistance,
    CouplingTermDualCartesianTrajectory)
from ._state_following_dmp import StateFollowingDMP
from ._canonical_system import canonical_system_alpha, phase


__all__ = [
    "DMP", "dmp_transformation_system", "DMPWithFinalVelocity", "CartesianDMP",
    "DualCartesianDMP", "CouplingTerm", "CouplingTermObstacleAvoidance",
    "CouplingTermCartesianDistance", "CouplingTermCartesianPosition",
    "CouplingTermDualCartesianOrientation", "CouplingTermDualCartesianPose",
    "CouplingTermDualCartesianDistance", "CouplingTermDualCartesianTrajectory",
    "canonical_system_alpha", "phase"]

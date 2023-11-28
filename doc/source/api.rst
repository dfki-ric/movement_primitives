.. _api:

=================
API Documentation
=================

This is the detailed documentation of all public classes and functions.
You can also search for specific modules, classes, or functions in the
:ref:`genindex`.

.. contents:: :local:
    :depth: 1


:mod:`movement_primitives.dmp`
==============================

.. automodule:: movement_primitives.dmp

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~movement_primitives.dmp.DMP
   ~movement_primitives.dmp.DMPWithFinalVelocity
   ~movement_primitives.dmp.CartesianDMP
   ~movement_primitives.dmp.DualCartesianDMP
   ~movement_primitives.dmp.DMPBase
   ~movement_primitives.dmp.WeightParametersMixin
   ~movement_primitives.dmp.CouplingTermPos1DToPos1D
   ~movement_primitives.dmp.CouplingTermObstacleAvoidance2D
   ~movement_primitives.dmp.CouplingTermObstacleAvoidance3D
   ~movement_primitives.dmp.CouplingTermPos3DToPos3D
   ~movement_primitives.dmp.CouplingTermDualCartesianPose
   ~movement_primitives.dmp.CouplingTermObstacleAvoidance3D
   ~movement_primitives.dmp.CouplingTermDualCartesianDistance
   ~movement_primitives.dmp.CouplingTermDualCartesianTrajectory

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.dmp.dmp_transformation_system
   ~movement_primitives.dmp.canonical_system_alpha
   ~movement_primitives.dmp.phase
   ~movement_primitives.dmp.obstacle_avoidance_acceleration_2d
   ~movement_primitives.dmp.obstacle_avoidance_acceleration_3d


:mod:`movement_primitives.promp`
================================

.. automodule:: movement_primitives.promp

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~movement_primitives.promp.ProMP


:mod:`movement_primitives.io`
=============================

.. automodule:: movement_primitives.io

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.io.write_pickle
   ~movement_primitives.io.read_pickle
   ~movement_primitives.io.write_yaml
   ~movement_primitives.io.read_yaml
   ~movement_primitives.io.write_json
   ~movement_primitives.io.read_json


:mod:`movement_primitives.base`
===============================

.. automodule:: movement_primitives.base

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~movement_primitives.base.PointToPointMovement


:mod:`movement_primitives.data`
===============================

.. automodule:: movement_primitives.data

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.data.load_lasa
   ~movement_primitives.data.generate_minimum_jerk
   ~movement_primitives.data.generate_1d_trajectory_distribution


:mod:`movement_primitives.kinematics`
=====================================

.. automodule:: movement_primitives.kinematics

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~movement_primitives.kinematics.Kinematics
   ~movement_primitives.kinematics.Chain


:mod:`movement_primitives.plot`
===============================

.. automodule:: movement_primitives.plot

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.plot.plot_trajectory_in_rows
   ~movement_primitives.plot.plot_distribution_in_rows


:mod:`movement_primitives.visualization`
========================================

.. automodule:: movement_primitives.visualization

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.visualization.plot_pointcloud
   ~movement_primitives.visualization.to_ellipsoid


:mod:`movement_primitives.dmp_potential_field`
==============================================

.. automodule:: movement_primitives.dmp_potential_field

.. autosummary::
   :toctree: _apidoc/

   ~movement_primitives.dmp_potential_field.plot_potential_field_2d


:mod:`movement_primitives.spring_damper`
========================================

.. automodule:: movement_primitives.spring_damper

.. autosummary::
   :toctree: _apidoc/
   :template: class.rst

   ~movement_primitives.spring_damper.SpringDamper
   ~movement_primitives.spring_damper.SpringDamperOrientation

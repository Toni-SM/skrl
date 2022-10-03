Omniverse Isaac Gym utils
=========================

.. contents:: Table of Contents
    :depth: 2
    :local:
    :backlinks: none

.. raw:: html

    <hr>

Control of robotic manipulators
-------------------------------

Inverse kinematics using damped least squares method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation attempts to unify under a single and reusable function the whole set of procedures used to calculate the inverse kinematics of a robotic manipulator shown originally in Isaac Gym's example: Franka IK Picking (:literal:`franka_cube_ik_osc.py`) but this time for Omniverse Isaac Gym

:math:`\Delta\theta = J^T (JJ^T + \lambda^2 I)^{-1} \, \vec{e}`

where

| :math:`\qquad \Delta\theta \;` is the change in joint angles
| :math:`\qquad J \;` is the Jacobian
| :math:`\qquad \lambda \;` is a non-zero damping constant
| :math:`\qquad \vec{e} \;` is the Cartesian pose error (position and orientation)

API
"""

.. autofunction:: skrl.utils.omniverse_isaacgym_utils.ik

.. raw:: html

    <hr>

OmniIsaacGymEnvs-like environment instance
------------------------------------------

API
"""

.. autofunction:: skrl.utils.omniverse_isaacgym_utils.get_env_instance

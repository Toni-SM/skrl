Omniverse Isaac Gym utils
=========================

Utilities for ease of programming of Omniverse Isaac Gym environments.

.. raw:: html

    <br><hr>

Control of robotic manipulators
-------------------------------

.. raw:: html

    <br>

Inverse kinematics using damped least squares method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation attempts to unify under a single and reusable function the whole set of procedures used to calculate the inverse kinematics of a robotic manipulator shown originally in Isaac Gym's example: Franka IK Picking (:literal:`franka_cube_ik_osc.py`) but this time for Omniverse Isaac Gym

:math:`\Delta\theta = J^T (JJ^T + \lambda^2 I)^{-1} \, \vec{e}`

where

| :math:`\qquad \Delta\theta \;` is the change in joint angles
| :math:`\qquad J \;` is the Jacobian
| :math:`\qquad \lambda \;` is a non-zero damping constant
| :math:`\qquad \vec{e} \;` is the Cartesian pose error (position and orientation)

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.utils.omniverse_isaacgym_utils.ik

.. raw:: html

    <br>

OmniIsaacGymEnvs-like environment instance
------------------------------------------

Instantiate a VecEnvBase-based object compatible with OmniIsaacGymEnvs for use outside of the OmniIsaacGymEnvs implementation.

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.utils.omniverse_isaacgym_utils.get_env_instance

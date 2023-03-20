Omniverse Isaac Gym utils
=========================

Utilities for ease of programming of Omniverse Isaac Gym environments.

.. raw:: html

    <br><hr>

Control of robotic manipulators
-------------------------------

.. raw:: html

    <br>

Differential inverse kinematics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation attempts to unify under a single and reusable function the whole set of procedures used to compute the inverse kinematics of a robotic manipulator, originally shown in the Isaac Orbit framework's task space controllers section, but this time for Omniverse Isaac Gym.

:math:`\Delta\theta =` :guilabel:`scale` :math:`J^\dagger \, \vec{e}`

where

| :math:`\qquad \Delta\theta \;` is the change in joint angles
| :math:`\qquad \vec{e} \;` is the Cartesian pose error (position and orientation)
| :math:`\qquad J^\dagger \;` is the pseudoinverse of the Jacobian estimated as follows:

The pseudoinverse of the Jacobian (:math:`J^\dagger`) is estimated as follows:

* Tanspose: :math:`\; J^\dagger = J^T`
* Pseduoinverse: :math:`\; J^\dagger = J^T(JJ^T)^{-1}`
* Damped least-squares: :math:`\; J^\dagger = J^T(JJ^T \, +` :guilabel:`damping`:math:`{}^2 I)^{-1}`
* Singular-vale decomposition: See `buss2004introduction <https://mathweb.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf>`_ (section 6)

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

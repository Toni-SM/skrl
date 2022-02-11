Isaac Gym utils
===============

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

.. https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf

:math:`\Delta\theta = J^T (JJ^T + \lambda^2 I)^{-1} \, \vec{e}`

where

| :math:`\qquad \Delta\theta \;` is the change in joint angles
| :math:`\qquad J \;` is the Jacobian
| :math:`\qquad \lambda \;` is a non-zero damping constant
| :math:`\qquad \vec{e} \;` is the Cartesian pose error (position and orientation)

API
"""

.. autofunction:: skrl.utils.isaacgym_utils.ik

.. raw:: html

   <hr>

Web viewer for headless development
-----------------------------------

API
"""

.. autoclass:: skrl.utils.isaacgym_utils.WebViewer
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _route_index, _route_stream, _stream
   :members:
   
   .. automethod:: __init__

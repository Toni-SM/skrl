Isaac Gym utils
===============

Utilities for ease of programming of Isaac Gym environments.

.. raw:: html

    <br><hr>

Control of robotic manipulators
-------------------------------

.. raw:: html

    <br>

Inverse kinematics using damped least squares method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This implementation attempts to unify under a single and reusable function the whole set of procedures used to calculate the inverse kinematics of a robotic manipulator shown in Isaac Gym's example: Franka IK Picking (:literal:`franka_cube_ik_osc.py`)

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

.. autofunction:: skrl.utils.isaacgym_utils.ik

.. raw:: html

    <br>

Web viewer for development without X server
-------------------------------------------

This library provides an API for instantiating a lightweight web viewer useful, mostly, for designing Isaac Gym environments in remote workstations or docker containers without X server

.. raw:: html

    <br>

Gestures and actions
^^^^^^^^^^^^^^^^^^^^

+---------------------------------------------------------+------------+--------------+
| Gestures / actions                                      | Key        | Mouse        |
+=========================================================+============+==============+
| Orbit (rotate view around a point)                      | :kbd:`Alt` | Left click   |
+---------------------------------------------------------+------------+--------------+
| Pan (rotate view around itself)                         |            | Right click  |
+---------------------------------------------------------+------------+--------------+
| Walk mode (move view linearly to the current alignment) |            | Middle click |
+---------------------------------------------------------+------------+--------------+
| Zoom in/out                                             |            | Scroll wheel |
+---------------------------------------------------------+------------+--------------+
| Freeze view                                             | :kbd:`V`   |              |
+---------------------------------------------------------+------------+--------------+
| Toggle image type (color, depth)                        | :kbd:`T`   |              |
+---------------------------------------------------------+------------+--------------+

Watch an animation of the gestures and actions in the following video

.. raw:: html

    <video width="100%" controls autoplay>
        <source src="https://user-images.githubusercontent.com/22400377/157323911-40729895-6175-48d2-85d7-c1b30fe0ee9c.mp4" type="video/mp4">
    </video>
    <br>


.. raw:: html

    <br>

Requirements
^^^^^^^^^^^^

The web viewer is build on top of `Flask <https://flask.palletsprojects.com>`_ and requires the following package to be installed

.. code-block:: bash

    pip install Flask

Also, to be launched in Visual Studio Code (Preview in Editor), the `Live Preview - VS Code Extension <https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server>`_ must be installed

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. tab:: Snippet

        .. literalinclude:: ../../snippets/isaacgym_utils.py
            :language: python
            :linenos:
            :emphasize-lines: 4, 8, 56, 65-68

.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.utils.isaacgym_utils.WebViewer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _route_index, _route_stream, _route_input_event, _stream
    :members:

    .. automethod:: __init__

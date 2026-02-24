:tocdepth: 5

Trainers
========

.. toctree::
    :hidden:

    Sequential <trainers/sequential>
    Parallel <trainers/parallel>
    Step <trainers/step>

Trainers are responsible for orchestrating and managing the training/evaluation of agents
and their interactions with the environment.

|br| |hr|

Implemented trainers
--------------------

The following table lists the implemented trainers and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Trainers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Sequential trainer <trainers/sequential>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Parallel trainer <trainers/parallel>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Step trainer <trainers/step>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|br| |hr|

Base class
----------

Base class and configuration for trainer implementations.

API
^^^

|

PyTorch
"""""""

.. automodule:: skrl.trainers.torch
.. autosummary::
    :nosignatures:

    TrainerCfg
    Trainer

.. autoclass:: skrl.trainers.torch.TrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.torch.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
"""

.. automodule:: skrl.trainers.jax
.. autosummary::
    :nosignatures:

    TrainerCfg
    Trainer

.. autoclass:: skrl.trainers.jax.TrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.jax.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

Warp
""""

.. automodule:: skrl.trainers.warp
.. autosummary::
    :nosignatures:

    TrainerCfg
    Trainer

.. autoclass:: skrl.trainers.warp.TrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.warp.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

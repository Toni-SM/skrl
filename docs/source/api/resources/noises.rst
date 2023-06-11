Noises
======

.. toctree::
    :hidden:

    Gaussian noise <noises/gaussian>
    Ornstein-Uhlenbeck <noises/ornstein_uhlenbeck>

Definition of the noises used by the agents during the exploration stage. All noises inherit from a base class that defines a uniform interface.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Noises
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Gaussian <noises/gaussian>` noise
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Ornstein-Uhlenbeck <noises/ornstein_uhlenbeck>` noise |_2|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Inheritance

        .. literalinclude:: ../../snippets/noises.py
            :language: python
            :start-after: [torch-start-base-class]
            :end-before: [torch-end-base-class]


.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.resources.noises.torch.base.Noise
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _update
    :members:

    .. automethod:: __init__

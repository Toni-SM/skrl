Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::
            
    .. tab:: Inheritance

        View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/snippets/trainer.py>`_

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :linenos:

API
^^^

.. autoclass:: skrl.trainers.torch.base.Trainer
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _pre_interaction, _post_interaction, _setup_agents
   :members:
   
   .. automethod:: __init__
   .. automethod:: __str__

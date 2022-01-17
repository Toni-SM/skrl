Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../snippets/agent.py
    :language: python
    :linenos:

API
^^^

.. autoclass:: skrl.agents.torch.base.Agent
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
   .. automethod:: __str__

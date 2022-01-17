Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../snippets/model.py
    :language: python
    :linenos:

API
^^^

.. autoclass:: skrl.models.torch.base.Model
   :undoc-members:
   :show-inheritance:
   :private-members: _get_space_size
   :members:
   
   .. automethod:: __init__

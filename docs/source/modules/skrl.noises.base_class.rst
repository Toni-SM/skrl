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

        View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/snippets/noise.py>`_

        .. literalinclude:: ../snippets/noise.py
            :language: python
            :linenos:

API
^^^

.. autoclass:: skrl.noises.torch.base.Noise
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _update
   :members:
   
   .. automethod:: __init__

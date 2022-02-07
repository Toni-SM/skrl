File post-processing
====================

This library provides an implementation for quickly loading exported memory files to inspect their contents in future post-processing steps. See the section :ref:`Examples <examples>` for a real use case

Basic usage
^^^^^^^^^^^

.. tabs::
            
    .. tab:: PyTorch (.pt)

        .. literalinclude:: ../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-torch]
            :end-before: [end-torch]

    .. tab:: NumPy (.npz)

        .. literalinclude:: ../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-numpy]
            :end-before: [end-numpy]

    .. tab:: Comma-separated values (.csv)

        .. literalinclude:: ../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-csv]
            :end-before: [end-csv]

API
^^^

.. autoclass:: skrl.utils.postprocessing.MemoryFileIterator
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _format_numpy, _format_torch, _format_csv
   :members:
   
   .. automethod:: __init__
   .. automethod:: __iter__
   .. automethod:: __next__
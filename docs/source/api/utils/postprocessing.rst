File post-processing
====================

Utilities for processing files generated during training/evaluation.

.. raw:: html

    <br><hr>

Exported memories
-----------------

This library provides an implementation for quickly loading exported memory files to inspect their contents in future post-processing steps. See the section :ref:`Library utilities (skrl.utils module) <library_utilities>` for a real use case

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. tab:: PyTorch (.pt)

        .. literalinclude:: ../../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-memory_file_iterator-torch]
            :end-before: [end-memory_file_iterator-torch]

    .. tab:: NumPy (.npz)

        .. literalinclude:: ../../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-memory_file_iterator-numpy]
            :end-before: [end-memory_file_iterator-numpy]

    .. tab:: Comma-separated values (.csv)

        .. literalinclude:: ../../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-6
            :start-after: [start-memory_file_iterator-csv]
            :end-before: [end-memory_file_iterator-csv]

.. raw:: html

    <br>

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

.. raw:: html

    <br>

Tensorboard files
-----------------

This library provides an implementation for quickly loading Tensorboard files to inspect their contents in future post-processing steps. See the section :ref:`Library utilities (skrl.utils module) <library_utilities>` for a real use case

.. raw:: html

    <br>

Requirements
^^^^^^^^^^^^

This utility requires the `TensorFlow <https://www.tensorflow.org/>`_ package to be installed to load and parse Tensorboard files:

.. code-block:: bash

    pip install tensorflow

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. tab:: Tensorboard (events.out.tfevents.*)

        .. literalinclude:: ../../snippets/utils_postprocessing.py
            :language: python
            :linenos:
            :emphasize-lines: 1, 5-7
            :start-after: [start-tensorboard_file_iterator-list]
            :end-before: [end-tensorboard_file_iterator-list]

.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.utils.postprocessing.TensorboardFileIterator
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
    .. automethod:: __iter__
    .. automethod:: __next__

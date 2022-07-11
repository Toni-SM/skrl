Installation
============

.. raw:: html

    <hr>
    
Prerequisites
-------------

**skrl** requires Python 3.6 or higher and the following libraries (they will be installed automatically):

    * `gym <https://www.gymlibrary.ml>`_
    * `torch <https://pytorch.org>`_ 1.8.0 or higher
    * `tensorboard <https://www.tensorflow.org/tensorboard>`_

.. raw:: html

    <hr>

Library Installation
--------------------

Python Package Index (PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install **skrl** with pip, execute:

    .. code-block:: bash
        
        pip install skrl

GitHub repository
^^^^^^^^^^^^^^^^^

Clone or download the library from its GitHub repository (https://github.com/Toni-SM/skrl)

    .. code-block:: bash
        
        git clone https://github.com/Toni-SM/skrl.git
        cd skrl

* **Install in editable/development mode** (links the package to its original location allowing any modifications to be reflected directly in its Python environment)

    .. code-block:: bash
        
        pip install -e .

* **Install in the current Python site-packages directory** (modifications to the code downloaded from GitHub will not be reflected in your Python environment)

    .. code-block:: bash
        
        pip install .

.. raw:: html

    <hr>

Troubleshooting
---------------

To ask questions or discuss about the library visit skrl's GitHub discussions

.. centered:: https://github.com/Toni-SM/skrl/discussions

Bug detection and/or correction, feature requests and everything else are more than welcome. Come on, open a new issue!

.. centered:: https://github.com/Toni-SM/skrl/issues

Known issues
------------

1. When using the parallel trainer with PyTorch 1.12

    See PyTorch issue `#80831 <https://github.com/pytorch/pytorch/issues/80831>`_

    .. code-block:: text
        
        AttributeError: 'Adam' object has no attribute '_warned_capturable_if_run_uncaptured'

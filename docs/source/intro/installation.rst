Installation
============

In this section, you will find the steps to install the library, troubleshoot known issues, review changes between versions, and more.

.. raw:: html

    <br><hr>

**Dependencies**
----------------

**skrl** requires Python 3.6 or higher and the following libraries (they will be installed automatically):

    * `gym <https://www.gymlibrary.dev>`_ / `gymnasium <https://gymnasium.farama.org/>`_
    * `tqdm <https://tqdm.github.io>`_
    * `packaging <https://packaging.pypa.io>`_
    * `tensorboard <https://www.tensorflow.org/tensorboard>`_

Machine learning (ML) framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to the specific ML frameworks, the following libraries are required:

PyTorch
"""""""

    * `torch <https://pytorch.org>`_ 1.9.0 or higher

JAX
"""

    * `jax <https://jax.readthedocs.io>`_ / `jaxlib <https://jax.readthedocs.io>`_ 0.4.3 or higher
    * `flax <https://flax.readthedocs.io>`_
    * `optax <https://optax.readthedocs.io>`_

.. raw:: html

    <br><hr>

**Library Installation**
------------------------

.. raw:: html

    <br>

Python Package Index (PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install **skrl** with pip, execute:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. code-block:: bash

            pip install skrl["torch"]

    .. group-tab:: |_4| |jax| |_4|

        .. warning::

            JAX installs its CPU version if not specified. For GPU/TPU versions visit the JAX
            `installation <https://github.com/google/jax#installation>`_ page before proceeding with the steps described below.

        .. code-block:: bash

            pip install skrl["jax"]

    .. group-tab:: All ML frameworks

        .. code-block:: bash

            pip install skrl["all"]

    .. group-tab:: No ML framework

        .. code-block:: bash

            pip install skrl

.. raw:: html

    <br>

GitHub repository
^^^^^^^^^^^^^^^^^

Clone or download the library from its GitHub repository (https://github.com/Toni-SM/skrl)

    .. code-block:: bash

        git clone https://github.com/Toni-SM/skrl.git
        cd skrl

* **Install in editable/development mode** (links the package to its original location allowing any modifications to be reflected directly in its Python environment)

    .. tabs::

        .. group-tab:: |_4| |pytorch| |_4|

            .. code-block:: bash

                pip install -e .["torch"]

        .. group-tab:: |_4| |jax| |_4|

            .. warning::

                JAX installs its CPU version if not specified. For GPU/TPU versions visit the JAX
                `installation <https://github.com/google/jax#installation>`_ page before proceeding with the steps described below.

            .. code-block:: bash

                pip install -e .["jax"]

        .. group-tab:: All ML frameworks

            .. code-block:: bash

                pip install -e .["all"]

        .. group-tab:: No ML framework

            .. code-block:: bash

                pip install -e .

* **Install in the current Python site-packages directory** (modifications to the code downloaded from GitHub will not be reflected in your Python environment)

    .. tabs::

        .. group-tab:: |_4| |pytorch| |_4|

            .. code-block:: bash

                pip install .["torch"]

        .. group-tab:: |_4| |jax| |_4|

            .. warning::

                JAX installs its CPU version if not specified. For GPU/TPU versions visit the JAX
                `installation <https://github.com/google/jax#installation>`_ page before proceeding with the steps described below.

            .. code-block:: bash

                pip install .["jax"]

        .. group-tab:: All ML frameworks

            .. code-block:: bash

                pip install .["all"]

        .. group-tab:: No ML framework

            .. code-block:: bash

                pip install .

.. raw:: html

    <br><hr>

**Discussions and issues**
--------------------------

To ask questions or discuss about the library visit skrl's GitHub discussions

.. centered:: https://github.com/Toni-SM/skrl/discussions

Bug detection and/or correction, feature requests and everything else are more than welcome. Come on, open a new issue!

.. centered:: https://github.com/Toni-SM/skrl/issues

.. raw:: html

    <br><hr>

**Known issues and troubleshooting**
------------------------------------

1. When using the parallel trainer with PyTorch 1.12.

    See PyTorch issue `#80831 <https://github.com/pytorch/pytorch/issues/80831>`_

    .. code-block:: text

        AttributeError: 'Adam' object has no attribute '_warned_capturable_if_run_uncaptured'

2. When installing the JAX version in Python 3.7 (e.g. OmniIsaacGymEnvs or Isaac Orbit on Isaac Sim 2022.2.1 and earlier).

    .. code-block:: text

        ERROR: Ignored the following versions that require a different python version: 0.4.0 Requires-Python >=3.8; ...
        ERROR: Could not find a version that satisfies the requirement jax>=0.4.3; extra == "jax" (from skrl[jax]) (from versions: 0.0, ..., 0.3.25)
        ERROR: No matching distribution found for jax>=0.4.3; extra == "jax"

    JAX support for Python 3.7 is up to version 0.3.25, while skrl requires ``jax>=0.4.3``.
    Furthermore, ``jaxlib<=0.3.25`` builds are only available up to NVIDIA CUDA 11 and cuDNN 8.2 versions.

    However, it is possible to use **skrl** under these circumstances, subject to the following points:

    * Install JAX, Flax and Optax manually using ``pip install jax flax optax`` and ignore the installation errors for skrl.

    * The ``jax.Device = jax.xla.Device`` statement is required by skrl to support ``jax<0.4.3``.

    * Overload models ``__hash__`` method to avoid :literal:`"TypeError: Failed to hash Flax Module"`.

3. When training/evaluating using JAX in Python 3.7 (e.g. OmniIsaacGymEnvs or Isaac Orbit on Isaac Sim 2022.2.1 and earlier).

    .. code-block:: text

        TypeError: Failed to hash Flax Module. The module probably contains unhashable attributes

    Overload the ``__hash__`` method for each defined model to avoid this issue:

    .. code-block:: python

        def __hash__(self):
            return id(self)

4. When training/evaluating using JAX with the NVIDIA Isaac Gym Preview, Isaac Orbit or Omniverse Isaac Gym environments.

    .. code-block:: text

        PxgCudaDeviceMemoryAllocator fail to allocate memory XXXXXX bytes!! Result = 2
        RuntimeError: CUDA error: an illegal memory access was encountered

    NVIDIA environments use PyTorch as a backend, and both PyTorch (for CUDA kernels, among others) and JAX preallocate GPU memory,
    which can lead to out-of-memory (OOM) problems. Reduce or disable GPU memory preallocation as indicated in JAX
    `GPU memory allocation <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`_ to avoid this issue. For example:

    .. code-block:: bash

        export XLA_PYTHON_CLIENT_MEM_FRACTION=.50  # lowering preallocated GPU memory to 50%

.. raw:: html

    <br><hr>

**Changelog**
-------------

.. literalinclude:: ../../../CHANGELOG.md
    :language: markdown

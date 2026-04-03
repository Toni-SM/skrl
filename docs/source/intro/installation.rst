Installation
============

In this section, you will find the steps to install the library, troubleshoot known issues, review changes between versions, and more.

|br| |hr|

Dependencies
------------

* General dependencies: `gymnasium <https://gymnasium.farama.org/>`_, `packaging <https://packaging.pypa.io>`_,
  `tensorboard <https://www.tensorflow.org/tensorboard>`_ and `tqdm <https://tqdm.github.io>`_.

* ML framework-specific dependencies:

.. list-table::
    :header-rows: 1

    * - Dependencies
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - Python
      - ``>= 3.10``
      - ``>= 3.10``
      - ``>= 3.10``
    * - Packages
      - `torch <https://pytorch.org>`_ ``>= 1.11``
      - `jax <https://jax.readthedocs.io>`_ / `jaxlib <https://jax.readthedocs.io>`_ ``>= 0.4.31``
        |br| `flax <https://flax.readthedocs.io>`_ ``>= 0.9.0``
        |br| `optax <https://optax.readthedocs.io>`_
      - `warp-lang <https://nvidia.github.io/warp>`_ ``>= 1.12``

.. warning::

    It is **recommended to install JAX manually before proceeding to install the skrl dependencies**, as JAX installs
    its CPU version by default. Visit the JAX `installation <https://jax.readthedocs.io/en/latest/installation.html>`_
    page before proceeding with the steps described below.

|br| |hr|

Library Installation
--------------------

|

Python Package Index (PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install *skrl* from PyPI, execute:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. code-block:: bash

            pip install skrl[torch]

    .. group-tab:: |_4| |jax| |_4|

        .. warning::

            It is **recommended to install JAX manually before proceeding to install the skrl dependencies**, as JAX installs its CPU version by default.
            Visit the JAX `installation <https://jax.readthedocs.io/en/latest/installation.html>`_ page before proceeding with the next steps.

        .. code-block:: bash

            pip install skrl[jax]

    .. group-tab:: |_4| |warp| |_4|

        .. code-block:: bash

            pip install skrl[warp]

    .. group-tab:: All ML frameworks

        .. code-block:: bash

            pip install skrl[all]

    .. group-tab:: No ML framework

        .. code-block:: bash

            pip install skrl

|

GitHub repository
^^^^^^^^^^^^^^^^^

To install *skrl* from the GitHub repository, follow one of the following options:

From Git
""""""""

Install, in the Python environment, the development version from the ``develop`` branch,
or the stable version (latest published version on PyPI) from the ``main`` branch:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: Development version

                .. code-block:: bash

                    pip install "skrl[torch] @ git+https://github.com/Toni-SM/skrl.git@develop"

            .. group-tab:: Stable version

                .. code-block:: bash

                    pip install "skrl[torch] @ git+https://github.com/Toni-SM/skrl.git@main"

    .. group-tab:: |_4| |jax| |_4|

        .. warning::

            It is **recommended to install JAX manually before proceeding to install the skrl dependencies**,
            as JAX installs its CPU version by default.
            Visit the JAX `installation <https://jax.readthedocs.io/en/latest/installation.html>`_
            page before proceeding with the next steps.

        .. tabs::

            .. group-tab:: Development version

                .. code-block:: bash

                    pip install "skrl[jax] @ git+https://github.com/Toni-SM/skrl.git@develop"

            .. group-tab:: Stable version

                .. code-block:: bash

                    pip install "skrl[jax] @ git+https://github.com/Toni-SM/skrl.git@main"

    .. group-tab:: |_4| |warp| |_4|

        .. tabs::

            .. group-tab:: Development version

                .. code-block:: bash

                    pip install "skrl[warp] @ git+https://github.com/Toni-SM/skrl.git@develop"

            .. group-tab:: Stable version

                .. code-block:: bash

                    pip install "skrl[warp] @ git+https://github.com/Toni-SM/skrl.git@main"

    .. group-tab:: All ML frameworks

        .. tabs::

            .. group-tab:: Development version

                .. code-block:: bash

                    pip install "skrl[all] @ git+https://github.com/Toni-SM/skrl.git@develop"

            .. group-tab:: Stable version

                .. code-block:: bash

                    pip install "skrl[all] @ git+https://github.com/Toni-SM/skrl.git@main"

    .. group-tab:: No ML framework

        .. tabs::

            .. group-tab:: Development version

                .. code-block:: bash

                    pip install git+https://github.com/Toni-SM/skrl.git@develop

            .. group-tab:: Stable version

                .. code-block:: bash

                    pip install git+https://github.com/Toni-SM/skrl.git@main

Editable installation
"""""""""""""""""""""

The editable installation is useful when you want to modify the library (e.g.: add new features, fix bugs, etc.),
and test the changes immediately without reinstalling it. In this mode, the library is linked to
its original location, allowing any modifications to be reflected directly in the Python environment.

Clone or download the library from its GitHub repository:

.. code-block:: bash

    git clone https://github.com/Toni-SM/skrl.git
    cd skrl

Then, install the library in editable/development mode:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. code-block:: bash

            pip install -e .[torch]

    .. group-tab:: |_4| |jax| |_4|

        .. warning::

            It is **recommended to install JAX manually before proceeding to install the skrl dependencies**,
            as JAX installs its CPU version by default.
            Visit the JAX `installation <https://jax.readthedocs.io/en/latest/installation.html>`_
            page before proceeding with the next steps.

        .. code-block:: bash

            pip install -e .[jax]

    .. group-tab:: |_4| |warp| |_4|

        .. code-block:: bash

            pip install -e .[warp]

    .. group-tab:: All ML frameworks

        .. code-block:: bash

            pip install -e .[all]

    .. group-tab:: No ML framework

        .. code-block:: bash

            pip install -e .

|br| |hr|

Discussions and issues
----------------------

To ask questions or discuss about the library visit *skrl*'s GitHub discussions.

.. centered:: https://github.com/Toni-SM/skrl/discussions

Bug detection and/or correction, feature requests and everything else are more than welcome.
|br| Come on, open a new issue!

.. centered:: https://github.com/Toni-SM/skrl/issues

|br| |hr|

Known issues and troubleshooting
--------------------------------

#. When using the parallel trainer with PyTorch 1.12.

    See PyTorch issue `#80831 <https://github.com/pytorch/pytorch/issues/80831>`_

    .. code-block:: text

        AttributeError: 'Adam' object has no attribute '_warned_capturable_if_run_uncaptured'

#. When training/evaluating using JAX with the NVIDIA Isaac Lab (and Isaac Gym) environments.

    .. code-block:: text

        PxgCudaDeviceMemoryAllocator fail to allocate memory XXXXXX bytes!! Result = 2
        RuntimeError: CUDA error: an illegal memory access was encountered

    NVIDIA environments use PyTorch as a backend, and both PyTorch (for CUDA kernels, among others) and JAX preallocate GPU memory,
    which can lead to out-of-memory (OOM) problems. Reduce or disable GPU memory preallocation as indicated in JAX
    `GPU memory allocation <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`_ to avoid this issue. For example:

    .. code-block:: bash

        export XLA_PYTHON_CLIENT_MEM_FRACTION=.50  # lowering preallocated GPU memory to 50%

|br| |hr|

Changelog
---------

.. literalinclude:: ../../../CHANGELOG.md
    :language: markdown

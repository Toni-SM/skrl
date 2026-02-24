:tocdepth: 4

Running standard scaler
=======================

Standardize input features by removing the mean and scaling to unit variance.

|br| |hr|

Algorithm
---------

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - mean (:math:`\bar{x}`), standard deviation (:math:`\sigma`), variance (:math:`\sigma^2`)
|   - running mean (:math:`\bar{x}_t`), running variance (:math:`\sigma^2_t`)
|

**Standardization by centering and scaling**

| :math:`\text{clip}((x - \bar{x}_t) / (\sqrt{\sigma^2} \;+` :guilabel:`epsilon` :math:`), -c, c) \qquad` with :math:`c` as :guilabel:`clip_threshold`
|

**Scale back the data to the original representation (inverse transform)**

| :math:`\sqrt{\sigma^2_t} \; \text{clip}(x, -c, c) + \bar{x}_t \qquad` with :math:`c` as :guilabel:`clip_threshold`
|

**Update the running mean and variance** (See `parallel algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm>`_)

| :math:`\delta \leftarrow x - \bar{x}_t`
| :math:`n_T \leftarrow n_t + n`
| :math:`M2 \leftarrow (\sigma^2_t n_t) + (\sigma^2 n) + \delta^2 \dfrac{n_t n}{n_T}`
| :green:`# update internal variables`
| :math:`\bar{x}_t \leftarrow \bar{x}_t + \delta \dfrac{n}{n_T}`
| :math:`\sigma^2_t \leftarrow \dfrac{M2}{n_T}`
| :math:`n_t \leftarrow n_T`

|

Usage
-----

The preprocessor usage is defined in each agent's configuration.
The preprocessor class is set under the :literal:`"<type>_preprocessor"` key and its arguments are set under
the :literal:`"<type>_preprocessor_kwargs"` key, as a Python dictionary.

The following examples show how to set the preprocessors for an agent:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../../snippets/preprocessors.py
            :language: python
            :emphasize-lines: 2, 6-11
            :start-after: [start-running-standard-scaler-torch]
            :end-before: [end-running-standard-scaler-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../../snippets/preprocessors.py
            :language: python
            :emphasize-lines: 2, 6-11
            :start-after: [start-running-standard-scaler-jax]
            :end-before: [end-running-standard-scaler-jax]

    .. group-tab:: |_4| |warp| |_4|

        .. literalinclude:: ../../../snippets/preprocessors.py
            :language: python
            :emphasize-lines: 2, 6-11
            :start-after: [start-running-standard-scaler-warp]
            :end-before: [end-running-standard-scaler-warp]

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.resources.preprocessors.torch.running_standard_scaler
.. autosummary::
    :nosignatures:

    RunningStandardScaler

.. autoclass:: skrl.resources.preprocessors.torch.running_standard_scaler.RunningStandardScaler
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __call__

|

JAX
^^^

.. automodule:: skrl.resources.preprocessors.jax.running_standard_scaler
.. autosummary::
    :nosignatures:

    RunningStandardScaler

.. autoclass:: skrl.resources.preprocessors.jax.running_standard_scaler.RunningStandardScaler
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __call__

|

Warp
^^^^

.. automodule:: skrl.resources.preprocessors.warp.running_standard_scaler
.. autosummary::
    :nosignatures:

    RunningStandardScaler

.. autoclass:: skrl.resources.preprocessors.warp.running_standard_scaler.RunningStandardScaler
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __call__

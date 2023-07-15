Running standard scaler
=======================

Standardize input features by removing the mean and scaling to unit variance.

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

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

.. raw:: html

    <br>

Usage
-----

The preprocessors usage is defined in each agent's configuration dictionary.

The preprocessor class is set under the :literal:`"<variable>_preprocessor"` key and its arguments are set under the :literal:`"<variable>_preprocessor_kwargs"` key as a keyword argument dictionary. The following examples show how to set the preprocessors for an agent:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. code-block:: python
            :emphasize-lines: 2, 5-8

            # import the preprocessor class
            from skrl.resources.preprocessors.torch import RunningStandardScaler

            cfg = DEFAULT_CONFIG.copy()
            cfg["state_preprocessor"] = RunningStandardScaler
            cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
            cfg["value_preprocessor"] = RunningStandardScaler
            cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

    .. group-tab:: |_4| |jax| |_4|

        .. code-block:: python
            :emphasize-lines: 2, 5-8

            # import the preprocessor class
            from skrl.resources.preprocessors.jax import RunningStandardScaler

            cfg = DEFAULT_CONFIG.copy()
            cfg["state_preprocessor"] = RunningStandardScaler
            cfg["state_preprocessor_kwargs"] = {"size": env.observation_space}
            cfg["value_preprocessor"] = RunningStandardScaler
            cfg["value_preprocessor_kwargs"] = {"size": 1}

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.resources.preprocessors.torch.running_standard_scaler.RunningStandardScaler
    :private-members: _parallel_variance, _get_space_size
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.resources.preprocessors.jax.running_standard_scaler.RunningStandardScaler
    :undoc-members:
    :private-members: _parallel_variance, _get_space_size
    :members:

    .. automethod:: __init__
    .. automethod:: __call__

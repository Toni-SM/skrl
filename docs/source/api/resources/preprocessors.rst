Preprocessors
=============

.. toctree::
    :hidden:

    Running standard scaler <preprocessors/running_standard_scaler>

Basic usage
-----------

The preprocessors usage is defined in each agent's configuration dictionary.

The preprocessor class is set under the :literal:`"<variable>_preprocessor"` key and its arguments are set under the :literal:`"<variable>_preprocessor_kwargs"` key as a keyword argument dictionary. The following examples show how to set the preprocessors for an agent:

.. tabs::

    .. tab:: Running standard scaler

        .. code-block:: python
            :emphasize-lines: 5-8

            # import the preprocessor class
            from skrl.resources.preprocessors.torch import RunningStandardScaler

            cfg = DEFAULT_CONFIG.copy()
            cfg["state_preprocessor"] = RunningStandardScaler
            cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
            cfg["value_preprocessor"] = RunningStandardScaler
            cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}

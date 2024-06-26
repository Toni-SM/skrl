Saving, loading and logging
===========================

In this section, you will find the information you need to log data with TensorBoard or Weights & Biases and to save and load checkpoints and memories to and from persistent storage.

.. raw:: html

    <br><hr>

**TensorBoard integration**
---------------------------

`TensorBoard <https://www.tensorflow.org/tensorboard>`_ is used for tracking and visualizing metrics and scalars (coefficients, losses, etc.). The tracking and writing of metrics and scalars is the responsibility of the agents (**can be customized independently for each agent using its configuration dictionary**).

.. .. admonition:: |jax|
.. note::

    A standalone JAX installation does not include any package for writing events to Tensorboard. In this case it is necessary to install (if not installed) one of the following frameworks/packages:

    * `PyTorch <https://pytorch.org/get-started/locally>`_
    * `TensorFlow <https://www.tensorflow.org/install>`_
    * `TensorboardX <https://github.com/lanpa/tensorboardX#install>`_

.. raw:: html

    <br>

Configuration
^^^^^^^^^^^^^

Each agent offers the following parameters under the :literal:`"experiment"` key:

.. literalinclude:: ../snippets/data.py
    :language: python
    :emphasize-lines: 5-7
    :start-after: [start-data-configuration]
    :end-before: [end-data-configuration]

* **directory**: directory path where the data generated by the experiments (a subdirectory) are stored. If no value is set, the :literal:`runs` folder (inside the current working directory) will be used (and created if it does not exist).

* **experiment_name**: name of the experiment (subdirectory). If no value is set, it will be the current date and time and the agent's name (e.g. :literal:`22-01-09_22-48-49-816281_DDPG`).

* **write_interval**: interval for writing metrics and values to TensorBoard. A value equal to or less than 0 disables tracking and writing to TensorBoard. If set to ``"auto"`` (default value), the interval will be defined to collect 100 samples throughout training/evaluation (``timesteps / 100``).

.. raw:: html

    <br>

Tracked metrics/scales visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To visualize the tracked metrics/scales, during or after the training, TensorBoard can be launched using the following command in a terminal:

.. code-block:: bash

    tensorboard --logdir=PATH_TO_RUNS_DIRECTORY

.. image:: ../_static/imgs/data_tensorboard.jpg
    :width: 100%
    :align: center
    :alt: TensorBoard panel

|

The following table shows the metrics/scales tracked by each agent ([**+**] all the time, [**-**] only when such a function is enabled in the agent's configuration):

+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Tag        |Metric / Scalar     |.. centered:: A2C |.. centered:: AMP |.. centered:: CEM |.. centered:: DDPG|.. centered:: DDQN|.. centered:: DQN |.. centered:: PPO |.. centered:: Q-learning |.. centered:: SAC |.. centered:: SARSA |.. centered:: TD3 |.. centered:: TRPO|
+===========+====================+==================+==================+==================+==================+==================+==================+==================+=========================+==================+====================+==================+==================+
|Coefficient|Entropy coefficient |                  |                  |                  |                  |                  |                  |                  |                         |.. centered:: +   |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Return threshold    |                  |                  |.. centered:: +   |                  |                  |                  |                  |                         |                  |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Mean disc. returns  |                  |                  |.. centered:: +   |                  |                  |                  |                  |                         |                  |                    |                  |                  |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Episode    |Total timesteps     |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +          |.. centered:: +   |.. centered:: +     |.. centered:: +   |.. centered:: +   |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Exploration|Exploration noise   |                  |                  |                  |.. centered:: +   |                  |                  |                  |                         |                  |                    |.. centered:: +   |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Exploration epsilon |                  |                  |                  |                  |.. centered:: +   |.. centered:: +   |                  |                         |                  |                    |                  |                  |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Learning   |Learning rate       |.. centered:: +   |.. centered:: +   |.. centered:: --  |                  |.. centered:: --  |.. centered:: --  |.. centered:: --  |                         |                  |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Policy learning rate|                  |                  |                  |.. centered:: --  |                  |                  |                  |                         |.. centered:: --  |                    |.. centered:: --  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Critic learning rate|                  |                  |                  |.. centered:: --  |                  |                  |                  |                         |.. centered:: --  |                    |.. centered:: --  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Return threshold    |                  |                  |                  |                  |                  |                  |                  |                         |                  |                    |                  |.. centered:: --  |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Loss       |Critic loss         |                  |                  |                  |.. centered:: +   |                  |                  |                  |                         |.. centered:: +   |                    |.. centered:: +   |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Entropy loss        |.. centered:: --  |.. centered:: --  |                  |                  |                  |                  |.. centered:: --  |                         |.. centered:: --  |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Discriminator loss  |                  |.. centered:: +   |                  |                  |                  |                  |                  |                         |                  |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Policy loss         |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |                  |                  |.. centered:: +   |                         |.. centered:: +   |                    |.. centered:: +   |.. centered:: +   |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Q-network loss      |                  |                  |                  |                  |.. centered:: +   |.. centered:: +   |                  |                         |                  |                    |                  |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Value loss          |.. centered:: +   |.. centered:: +   |                  |                  |                  |                  |.. centered:: +   |                         |                  |                    |                  |.. centered:: +   |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Policy     |Standard deviation  |.. centered:: +   |.. centered:: +   |                  |                  |                  |                  |.. centered:: +   |                         |                  |                    |                  |.. centered:: +   |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Q-network  |Q1                  |                  |                  |                  |.. centered:: +   |                  |                  |                  |                         |.. centered:: +   |                    |.. centered:: +   |                  |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Q2                  |                  |                  |                  |                  |                  |                  |                  |                         |.. centered:: +   |                    |.. centered:: +   |                  |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Reward     |Instantaneous reward|.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +          |.. centered:: +   |.. centered:: +     |.. centered:: +   |.. centered:: +   |
+           +--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|           |Total reward        |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +   |.. centered:: +          |.. centered:: +   |.. centered:: +     |.. centered:: +   |.. centered:: +   |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+
|Target     |Target              |                  |                  |                  |.. centered:: +   |.. centered:: +   |.. centered:: +   |                  |                         |.. centered:: +   |                    |.. centered:: +   |                  |
+-----------+--------------------+------------------+------------------+------------------+------------------+------------------+------------------+------------------+-------------------------+------------------+--------------------+------------------+------------------+

.. raw:: html

    <br>

Tracking custom metrics/scales
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Tracking custom data attached to the agent's control and timing logic (recommended)**

  Although the TensorBoard's writing control and timing logic is controlled by the base class Agent, it is possible to track custom data. The :literal:`track_data` method can be used (see :doc:`Agent <../api/agents>` class for more details), passing as arguments the data identification (tag) and the scalar value to be recorded.

  For example, to track the current CPU usage, the following code can be used:

  .. code-block:: python

      # assuming agent is an instance of an Agent subclass
      agent.track_data("Resource / CPU usage", psutil.cpu_percent())

* **Tracking custom data directly to Tensorboard**

  It is also feasible to access directly to the `SummaryWriter <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter>`_ instance through the :literal:`writer` property if it is desired to write directly to Tensorboard, avoiding the base class's control and timing logic.

  For example, to write directly to TensorBoard:

  .. code-block:: python

      # assuming agent is an instance of an Agent subclass
      agent.writer.add_scalar("Resource / CPU usage", psutil.cpu_percent(), global_step=1000)

.. raw:: html

    <br><hr>

**Weights & Biases integration**
--------------------------------

`Weights & Biases (wandb) <https://wandb.ai>`_ is also supported for tracking and visualizing metrics and scalars. Its configuration is responsibility of the agents (**can be customized independently for each agent using its configuration dictionary**).

Follow the steps described in Weights & Biases documentation (`Set up wandb <https://docs.wandb.ai/quickstart#1.-set-up-wandb>`_) to login to the :literal:`wandb` library on the current machine.

.. note::

    The :literal:`wandb` library is not installed by default. Install it in a Python 3 environment using pip as follows:

    .. code-block:: bash

        pip install wandb

.. raw:: html

    <br>

Configuration
^^^^^^^^^^^^^

Each agent offers the following parameters under the :literal:`"experiment"` key. Visit the Weights & Biases documentation for more details about the configuration parameters.

.. literalinclude:: ../snippets/data.py
    :language: python
    :emphasize-lines: 12-13
    :start-after: [start-data-configuration]
    :end-before: [end-data-configuration]

* **wandb**: whether to enable support for Weights & Biases.

* **wandb_kwargs**: keyword argument dictionary used to parameterize the `wandb.init <https://docs.wandb.ai/ref/python/init>`_ function. If no values are provided for the following parameters, the following values will be set for them:

  * :literal:`"name"`: will be set to the name of the experiment directory.

  * :literal:`"sync_tensorboard"`:  will be set to :literal:`True`.

  * :literal:`"config"`: will be updated with the configuration dictionaries of both the agent (and its models) and the trainer. The update will be done even if a value has been set for the parameter.

.. raw:: html

    <br><hr>

**Checkpoints**
---------------

.. raw:: html

    <br>

Saving checkpoints
^^^^^^^^^^^^^^^^^^

The checkpoints are saved in the :literal:`checkpoints` subdirectory of the experiment's directory (its path can be customized using the options described in the previous subsection). The checkpoint name is the key referring to the agent (or models, optimizers and preprocessors) and the current timestep (e.g. :literal:`runs/22-01-09_22-48-49-816281_DDPG/checkpoints/agent_2500.pt`).

The checkpoint management, as in the previous case, is the responsibility of the agents (**can be customized independently for each agent using its configuration dictionary**).

.. literalinclude:: ../snippets/data.py
    :language: python
    :emphasize-lines: 9,10
    :start-after: [start-data-configuration]
    :end-before: [end-data-configuration]

* **checkpoint_interval**: interval for checkpoints. A value equal to or less than 0 disables the checkpoint creation. If set to ``"auto"`` (default value), the interval will be defined to collect 10 checkpoints throughout training/evaluation (``timesteps / 10``).

* **store_separately**: if set to :literal:`True`, all the modules that an agent contains (models, optimizers, preprocessors, etc.) will be saved each one in a separate file. By default (:literal:`False`) the modules are grouped in a dictionary and stored in the same file.

**Checkpointing the best models**

The best models, attending the mean total reward, will be saved in the :literal:`checkpoints` subdirectory of the experiment's directory. The checkpoint name is the word :literal:`best` and the key referring to the model (e.g. :literal:`runs/22-01-09_22-48-49-816281_DDPG/checkpoints/best_agent.pt`).

The best models are updated internally on each TensorBoard writing interval :literal:`"write_interval"` and they are saved on each checkpoint interval :literal:`"checkpoint_interval"`. The :literal:`"store_separately"` key specifies whether the best modules are grouped and stored together or separately.

.. raw:: html

    <br>

Loading checkpoints
^^^^^^^^^^^^^^^^^^^

Checkpoints can be loaded (e.g. to resume or continue training) for each of the instantiated agents (or models) independently via the :literal:`.load(...)` method (`Agent.load <../modules/skrl.agents.base_class.html#skrl.agents.torch.base.Agent.load>`_ or `Model.load <../modules/skrl.models.base_class.html#skrl.models.torch.base.Model.load>`_). It accepts the path (relative or absolute) of the checkpoint to load as the only argument. The checkpoint will be dynamically mapped to the device specified as argument in the class constructor (internally the torch load's :literal:`map_location` method is used during loading).

.. note::

    The agents or models instances must have the same architecture/structure as the one used to save the checkpoint. The current implementation load the model's state-dict directly.

.. note::

    Warnings such as :literal:`[skrl:WARNING] Cannot load the <module> module. The agent doesn't have such an instance` can be ignored without problems during evaluation. The reason for this is that during the evaluation not all components, such as optimizers or other models apart from the policy, may be defined.

The following code snippets show how to load the checkpoints through the instantiated agent (recommended) or models. See the :doc:`Examples <examples>` section for showcases about how to checkpoints and use them to continue the training or evaluate experiments.

.. tabs::

    .. tab:: Agent (recommended)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 12
                    :start-after: [start-checkpoint-load-agent-torch]
                    :end-before: [end-checkpoint-load-agent-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 12
                    :start-after: [start-checkpoint-load-agent-jax]
                    :end-before: [end-checkpoint-load-agent-jax]

    .. tab:: Model

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 22
                    :start-after: [start-checkpoint-load-model-torch]
                    :end-before: [end-checkpoint-load-model-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 22
                    :start-after: [start-checkpoint-load-model-jax]
                    :end-before: [end-checkpoint-load-model-jax]

In addition, it is possible to load, through the library utilities, trained agent checkpoints from the Hugging Face Hub (`huggingface.co/skrl <https://huggingface.co/skrl>`_). See the :doc:`Hugging Face integration <../api/utils/huggingface>` for more information.

.. tabs::

    .. tab:: Agent (from Hugging Face Hub)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 2, 13-14
                    :start-after: [start-checkpoint-load-huggingface-torch]
                    :end-before: [end-checkpoint-load-huggingface-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 2, 13-14
                    :start-after: [start-checkpoint-load-huggingface-jax]
                    :end-before: [end-checkpoint-load-huggingface-jax]

.. raw:: html

    <br>

Migrating external checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is possible to load checkpoints generated with external reinforcement learning libraries into skrl agents (or models) via the :literal:`.migrate(...)` method (`Agent.migrate <../modules/skrl.agents.base_class.html#skrl.agents.torch.base.Agent.migrate>`_ or `Model.migrate <../modules/skrl.models.base_class.html#skrl.models.torch.base.Model.migrate>`_).

.. note::

    In some cases it will be necessary to specify a parameter mapping, especially in ambiguous models (where 2 or more parameters, for source or current model, have equal shape). Refer to the respective method documentation for more details in these cases.

The following code snippets show how to migrate checkpoints from other libraries to the agents or models implemented in skrl:

.. tabs::

    .. tab:: Agent

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 12
                    :start-after: [start-checkpoint-migrate-agent-torch]
                    :end-before: [end-checkpoint-migrate-agent-torch]

    .. tab:: Model

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/data.py
                    :language: python
                    :emphasize-lines: 22, 25, 28-29
                    :start-after: [start-checkpoint-migrate-model-torch]
                    :end-before: [end-checkpoint-migrate-model-torch]

.. raw:: html

    <br><hr>

**Memory export/import**
------------------------

.. raw:: html

    <br>

Exporting memories
^^^^^^^^^^^^^^^^^^

Memories can be automatically exported to files at each filling cycle (before data overwriting is performed). Its activation, the output files' format and their path can be modified through the constructor parameters when an instance is created.

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../snippets/data.py
            :language: python
            :emphasize-lines: 7-9
            :start-after: [start-export-memory-torch]
            :end-before: [end-export-memory-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../snippets/data.py
            :language: python
            :emphasize-lines: 7-9
            :start-after: [start-export-memory-jax]
            :end-before: [end-export-memory-jax]

* **export**: enable or disable the memory export (default is disabled).

* **export_format**: the format of the exported memory (default is :literal:`"pt"`). Supported formats are PyTorch (:literal:`"pt"`), NumPy (:literal:`"np"`) and Comma-separated values (:literal:`"csv"`).

* **export_directory**: the directory where the memory will be exported (default is :literal:`"memory"`).

.. raw:: html

    <br>

Importing memories
^^^^^^^^^^^^^^^^^^

TODO :red:`(coming soon)`

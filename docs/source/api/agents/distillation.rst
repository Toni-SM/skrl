:tocdepth: 4

Teacher-Student Distillation
============================

Distillation trains a **student** policy, which observes only the deployable observations, to imitate a frozen
**teacher** policy, which typically also observes privileged states that are available in simulation but not on
the real system.

Since the student is the model that acts in the environment, the teacher labels the state distribution visited by
the student itself. This online (DAgger-like) behavior cloning avoids the compounding-error problem of imitating
a teacher only on the states the teacher itself would have visited.

.. note::

    The agent's ``policy`` model **is the student**. The role is named ``policy`` because it is the model that
    acts in the environment and the only one whose parameters are optimized, which also keeps its checkpoint
    interchangeable with the other skrl agents.

|br| |hr|

Algorithm
---------

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - student (policy) function approximator (:math:`\pi_\theta`), teacher function approximator (:math:`\pi_{\phi}`)
|   - observations (:math:`o`), states (:math:`s`), actions (:math:`a`)
|   - loss (:math:`L`)

|

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :green:`# the student acts (stochastically) in the environment`
| :math:`a \leftarrow \pi_\theta(o)`
| :green:`# the teacher labels the visited observation/state with its (deterministic) action`
| :math:`a_{_{teacher}} \leftarrow \pi_{\phi}(o, s)`

|

Learning algorithm
""""""""""""""""""

|
| :literal:`update(...)`
| **FOR** each epoch in :guilabel:`learning_epochs` **DO**
|     **FOR** each mini-batch in :guilabel:`mini_batches` up to :guilabel:`rollouts` **DO**
|         :green:`# sample the stored observations/states and teacher actions`
|         :math:`o, s, a_{_{teacher}} \leftarrow` observations, states, teacher actions
|         :green:`# compute the student's deterministic actions`
|         :math:`\hat{a} \leftarrow \pi_\theta(o)`
|         :green:`# compute the behavior cloning loss`
|         :math:`L_{\pi_\theta} \leftarrow \text{loss}(\hat{a},\; a_{_{teacher}})` according to :guilabel:`loss_type`
|         :green:`# optimization step (only the student's parameters are optimized)`
|         reset :math:`\text{optimizer}_\theta`
|         :math:`\nabla_{\theta} L_{\pi_\theta}`
|         :math:`\lVert \nabla_{\theta} \rVert` clipped to :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_\theta`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\theta (\text{optimizer}_\theta)`

|

Usage
-----

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-distillation]
                    :end-before: [torch-end-distillation]

|

Loading the teacher
^^^^^^^^^^^^^^^^^^^

The teacher's parameters must be loaded before training. Set the :guilabel:`teacher_checkpoint` configuration
value to the path of a checkpoint file, which can be either of the following:

* A whole-agent checkpoint (e.g.: :literal:`agent_48000.pt`), in which case the ``teacher`` or the ``policy``
  module is used, in that order. This makes a checkpoint written by a PPO (or any other agent) run directly
  usable as a teacher.
* A single model's state dict (e.g.: :literal:`policy_48000.pt`), as written when the
  :guilabel:`experiment.store_separately` option is enabled.

Alternatively, call the agent's :py:meth:`~skrl.agents.torch.distillation.Distillation.load_teacher` method, or
load the model directly via :literal:`agent.models["teacher"].load(path)`. If the agent does not load the teacher
itself, a warning is logged when the agent is initialized.

The teacher's parameters are frozen and the model is kept in evaluation mode for the whole run.

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``DISTILLATION_CFG``
      - :py:class:`~skrl.agents.torch.distillation.DISTILLATION_CFG`
      -
      -

|

Spaces
^^^^^^

The implementation supports the following `Gymnasium spaces <https://gymnasium.farama.org/api/spaces>`_:

.. list-table::
    :header-rows: 1

    * - Gymnasium spaces
      - .. centered:: Observation
      - .. centered:: Action
    * - Discrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - MultiDiscrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. warning::

    The student's actions are regressed onto the teacher's actions, which is only meaningful for continuous
    action spaces. Instantiating the agent with a ``Discrete`` or ``MultiDiscrete`` action space raises a
    :py:class:`ValueError`.

|

Models
^^^^^^

The implementation uses 2 continuous function approximators.
These function approximators (models) must be collected in a dictionary and passed to the constructor of the class
under the argument :literal:`models`.

.. note::

    The ``policy`` model is the **student**: the model that acts in the environment and the only one whose
    parameters are optimized. The ``teacher`` model is frozen and never optimized.

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_\theta(o)`
      - Student (the distilled policy)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Gaussian <models_gaussian>` /
        |br| :ref:`MultivariateGaussian <models_multivariate_gaussian>` /
        |br| :ref:`Deterministic <models_deterministic>`
    * - :math:`\pi_{\phi}(o, s)`
      - Teacher (frozen, typically privileged)
      - :literal:`"teacher"`
      - observation / state
      - action
      - :ref:`Gaussian <models_gaussian>` /
        |br| :ref:`MultivariateGaussian <models_multivariate_gaussian>` /
        |br| :ref:`Deterministic <models_deterministic>`

The privileged information available to the teacher is expressed using skrl's distinction between the
environment's observations and states. Typically, the student's network input is defined as
:literal:`OBSERVATIONS` while the teacher's is defined as :literal:`STATES` or
:literal:`concatenate([OBSERVATIONS, STATES])`, e.g.:

.. code-block:: yaml

    models:
      separate: True
      policy:  # the student: deployable observations only
        class: GaussianMixin
        input: OBSERVATIONS
        layers: [256, 128, 64]
        activations: elu
      teacher:  # frozen, privileged: observations and states
        class: GaussianMixin
        input: concatenate([OBSERVATIONS, STATES])
        layers: [512, 256, 128]
        activations: elu

Defining both models over the same input (without a state space) is also valid, and corresponds to distilling a
large teacher into a smaller student rather than removing privileged information.

.. note::

    The regression is performed between the student's and the teacher's *deterministic* actions (the mean actions
    for Gaussian-like models). Consequently, a Gaussian student's standard deviation receives no gradient from the
    behavior cloning loss: the exploration noise applied while collecting transitions stays at the value defined by
    the model's :literal:`initial_log_std`. Adjust that value to control how widely the student explores around the
    teacher's behavior.

|

Features
^^^^^^^^

Support for advanced features is described in the following table:

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - RNN support
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Mixed precision
      - Automatic mixed precision
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Distributed
      - Single Program Multi Data (SPMD) multi-GPU
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.agents.torch.distillation
.. autosummary::
    :nosignatures:

    DISTILLATION_CFG
    Distillation

.. autoclass:: skrl.agents.torch.distillation.DISTILLATION_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.agents.torch.distillation.Distillation
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

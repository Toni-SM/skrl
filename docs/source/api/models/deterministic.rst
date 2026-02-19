:tocdepth: 4

.. _models_deterministic:

Deterministic model
===================

Deterministic models run **continuous-domain deterministic** policies.

|br| |hr|

*skrl* provides a Python mixin (:literal:`DeterministicMixin`) to assist in the creation of these types of models,
allowing users to have full control over the function approximator definitions and architectures.
Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

.. include:: common-jax.rst

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/deterministic_model.py
            :language: python
            :start-after: [start-definition-torch]
            :end-before: [end-definition-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/deterministic_model.py
            :language: python
            :start-after: [start-definition-jax]
            :end-before: [end-definition-jax]

    .. group-tab:: |_4| |warp| |_4|

        .. literalinclude:: ../../snippets/deterministic_model.py
            :language: python
            :start-after: [start-definition-warp]
            :end-before: [end-definition-warp]

|

Concept
-------

.. image:: ../../_static/imgs/model_deterministic-light.svg
    :width: 65%
    :align: center
    :class: only-light
    :alt: Deterministic model

.. image:: ../../_static/imgs/model_deterministic-dark.svg
    :width: 65%
    :align: center
    :class: only-dark
    :alt: Deterministic model

|

Usage
-----

* Multi-Layer Perceptron (**MLP**)
* Convolutional Neural Network (**CNN**)
* Recurrent Neural Network (**RNN**)
* Gated Recurrent Unit RNN (**GRU**)
* Long Short-Term Memory RNN (**LSTM**)

.. tabs::

    .. tab:: MLP

        .. image:: ../../_static/imgs/model_deterministic_mlp-light.svg
            :width: 35%
            :align: center
            :class: only-light

        .. image:: ../../_static/imgs/model_deterministic_mlp-dark.svg
            :width: 35%
            :align: center
            :class: only-dark

        |

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-mlp-sequential-torch]
                            :end-before: [end-mlp-sequential-torch]

                    .. group-tab:: nn.functional

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-mlp-functional-torch]
                            :end-before: [end-mlp-functional-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. tabs::

                    .. group-tab:: setup-style

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-mlp-setup-jax]
                            :end-before: [end-mlp-setup-jax]

                    .. group-tab:: compact-style

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-mlp-compact-jax]
                            :end-before: [end-mlp-compact-jax]

            .. group-tab:: |_4| |warp| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-mlp-sequential-warp]
                            :end-before: [end-mlp-sequential-warp]

    .. tab:: CNN

        .. image:: ../../_static/imgs/model_deterministic_cnn-light.svg
            :width: 100%
            :align: center
            :class: only-light

        .. image:: ../../_static/imgs/model_deterministic_cnn-dark.svg
            :width: 100%
            :align: center
            :class: only-dark

        |

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-cnn-sequential-torch]
                            :end-before: [end-cnn-sequential-torch]

                    .. group-tab:: nn.functional

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-cnn-functional-torch]
                            :end-before: [end-cnn-functional-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. tabs::

                    .. group-tab:: setup-style

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-cnn-setup-jax]
                            :end-before: [end-cnn-setup-jax]

                    .. group-tab:: compact-style

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-cnn-compact-jax]
                            :end-before: [end-cnn-compact-jax]

    .. tab:: RNN

        .. image:: ../../_static/imgs/model_deterministic_rnn-light.svg
            :width: 90%
            :align: center
            :class: only-light

        .. image:: ../../_static/imgs/model_deterministic_rnn-dark.svg
            :width: 90%
            :align: center
            :class: only-dark

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input_size} \\
                H_{out} ={} & \text{hidden_size}
            \end{aligned}

        |hr|

        .. include:: common-rnn.rst

        |

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-rnn-sequential-torch]
                            :end-before: [end-rnn-sequential-torch]

                    .. group-tab:: nn.functional

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-rnn-functional-torch]
                            :end-before: [end-rnn-functional-torch]

    .. tab:: GRU

        .. image:: ../../_static/imgs/model_deterministic_rnn-light.svg
            :width: 90%
            :align: center
            :class: only-light

        .. image:: ../../_static/imgs/model_deterministic_rnn-dark.svg
            :width: 90%
            :align: center
            :class: only-dark

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input_size} \\
                H_{out} ={} & \text{hidden_size}
            \end{aligned}

        |hr|

        .. include:: common-rnn.rst

        |

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-gru-sequential-torch]
                            :end-before: [end-gru-sequential-torch]

                    .. group-tab:: nn.functional

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-gru-functional-torch]
                            :end-before: [end-gru-functional-torch]

    .. tab:: LSTM

        .. image:: ../../_static/imgs/model_deterministic_rnn-light.svg
            :width: 90%
            :align: center
            :class: only-light

        .. image:: ../../_static/imgs/model_deterministic_rnn-dark.svg
            :width: 90%
            :align: center
            :class: only-dark

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                D ={} & 2 \text{ if bidirectional=True otherwise } 1 \\
                H_{in} ={} & \text{input_size} \\
                H_{cell} ={} & \text{hidden_size} \\
                H_{out} ={} & \text{proj_size if } \text{proj_size}>0 \text{ otherwise hidden_size} \\
            \end{aligned}

        |hr|

        .. include:: common-rnn.rst

        |

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. tabs::

                    .. group-tab:: nn.Sequential

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-lstm-sequential-torch]
                            :end-before: [end-lstm-sequential-torch]

                    .. group-tab:: nn.functional

                        .. literalinclude:: ../../snippets/deterministic_model.py
                            :language: python
                            :start-after: [start-lstm-functional-torch]
                            :end-before: [end-lstm-functional-torch]

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.models.torch.deterministic
.. autosummary::
    :nosignatures:

    DeterministicMixin

.. autoclass:: skrl.models.torch.deterministic.DeterministicMixin
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
|

JAX
^^^

.. automodule:: skrl.models.jax.deterministic
.. autosummary::
    :nosignatures:

    DeterministicMixin

.. autoclass:: skrl.models.jax.deterministic.DeterministicMixin
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
|

Warp
^^^^

.. automodule:: skrl.models.warp.deterministic
.. autosummary::
    :nosignatures:

    DeterministicMixin

.. autoclass:: skrl.models.warp.deterministic.DeterministicMixin
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

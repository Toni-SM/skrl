Manual training
===============

Train agents by manually controlling the training/evaluation loop.

.. raw:: html

    <br><hr>

Concept
-------

.. image:: ../../_static/imgs/manual_trainer-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Manual trainer

.. image:: ../../_static/imgs/manual_trainer-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Manual trainer

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: Training

                .. literalinclude:: ../../snippets/trainer.py
                    :language: python
                    :start-after: [pytorch-start-manual-training]
                    :end-before: [pytorch-end-manual-training]

            .. group-tab:: Evaluation

                .. literalinclude:: ../../snippets/trainer.py
                    :language: python
                    :start-after: [pytorch-start-manual-evaluation]
                    :end-before: [pytorch-end-manual-evaluation]

    .. group-tab:: |_4| |jax| |_4|

        .. tabs::

            .. group-tab:: Training

                .. literalinclude:: ../../snippets/trainer.py
                    :language: python
                    :start-after: [jax-start-manual-training]
                    :end-before: [jax-end-manual-training]

            .. group-tab:: Evaluation

                .. literalinclude:: ../../snippets/trainer.py
                    :language: python
                    :start-after: [jax-start-manual-evaluation]
                    :end-before: [jax-end-manual-evaluation]

.. raw:: html

    <br>

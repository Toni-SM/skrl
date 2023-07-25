Hugging Face integration
========================

The Hugging Face (HF) Hub is a platform for building, training, and deploying ML models, as well as accessing a variety of datasets and metrics for further analysis and validation.

.. raw:: html

    <br><hr>

Integration
-----------

.. raw:: html

    <br>

Download model from HF Hub
^^^^^^^^^^^^^^^^^^^^^^^^^^

Several skrl-trained models (agent checkpoints) for different environments/tasks of Gym/Gymnasium, Isaac Gym, Omniverse Isaac Gym, etc. are available in the Hugging Face Hub

These models can be used as comparison benchmarks, for collecting environment transitions in memory (for offline reinforcement learning, e.g.) or for pre-initialization of agents for performing similar tasks, among others

Visit the `skrl organization on the Hugging Face Hub <https://huggingface.co/skrl>`_ to access publicly available models!

.. raw:: html

    <br>

API
---

.. autofunction:: skrl.utils.huggingface.download_model_from_huggingface

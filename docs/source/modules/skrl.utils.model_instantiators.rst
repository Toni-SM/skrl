Model instantiators
===================

Basic usage
^^^^^^^^^^^

TODO: add snippet

API
^^^

.. autoclass:: skrl.utils.model_instantiators.Shape
    
    .. py:property:: ONE

        Flag to indicate that the model's input/output has shape (1,)
        
        This flag is useful for the definition of critic models, where the critic's output is a scalar  

    .. py:property:: STATES

        Flag to indicate that the model's input/output is the state (observation) space of the environment
        It is an alias for :py:attr:`OBSERVATIONS`
    
    .. py:property:: OBSERVATIONS

        Flag to indicate that the model's input/output is the observation space of the environment
    
    .. py:property:: ACTIONS

        Flag to indicate that the model's input/output is the action space of the environment
    
    .. py:property:: STATES_ACTIONS

        Flag to indicate that the model's input/output is the combination (concatenation) of the state (observation) and action spaces of the environment

.. autofunction:: skrl.utils.model_instantiators.categorical_model

.. autofunction:: skrl.utils.model_instantiators.gaussian_model

.. autofunction:: skrl.utils.model_instantiators.deterministic_model

The following points are relevant in the definition of recurrent models:

* The ``.get_specification()`` method must be overwritten to return a dictionary with key ``"rnn"``. Under such key, a sub-dictionary must contain the following items:
    * The sequence length (under sub-key ``"sequence_length"``).
    * A list of the dimensions for each initial hidden/cell state (under sub-key ``"sizes"``).

* The ``.compute()`` method's ``inputs`` parameter may include the following items:
    * ``"observations"``: observations of the environment.
    * ``"states"``: state of the environment.
    * ``"taken_actions"``: actions taken by the policy for the given observations and/or states, if applicable.
    * ``"terminated"``: episode termination status for sampled environment transitions.
        This key is only defined during the training process.
    * ``"rnn"``: list of initial hidden states ordered according to the model specification.

* The ``.compute()`` method should include, under the ``"rnn"`` key of the returned dictionary,
  a list of each final hidden/cell state (when applicable).

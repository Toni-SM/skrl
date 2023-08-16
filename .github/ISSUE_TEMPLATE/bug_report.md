---
name: Bug report
about: Submit a bug report
labels:
- bug
body:
- type: markdown
  attributes:
    value: >
      ## Your help in making skrl better is greatly appreciated!

      * Please ensure that the issue hasn't already been reported by using the [issue search](https://github.com/Toni-SM/skrl/search?q=is%3Aissue&type=issues).

      * The issue (and its solution) is not listed in the skrl documentation's [troubleshooting](https://skrl.readthedocs.io/en/latest/intro/installation.html#known-issues-and-troubleshooting) section.

      * For questions, please consider [open a discussion](https://github.com/Toni-SM/skrl/discussions).
      <br>
- type: textarea
  attributes:
    label: Description
    description: >-
      A clear and concise description of the bug/issue. Try to provide a minimal example to reproduce it (error/log messages are also helpful).
    placeholder: |
      Markdown formatting might be applied to the text.

      ```python
      # use triple backticks for code-blocks or error/log messages
      ```
  validations:
    required: true
- type: dropdown
  attributes:
    label: What skrl version are you using?
    description: The skrl version can be obtained with the command `pip show skrl`.
    options:
      - unknown
      - 1.0.0
      - 1.0.0-rc2
      - 1.0.0-rc1
      - 0.10.2 or 0.10.1
      - 0.10.0 or earlier
  validations:
    required: true
- type: input
  attributes:
    label: What ML framework/library version are you using?
    description: The version can be obtained with the command `pip show torch` or `pip show jax flax optax`.
    placeholder: PyTorch version, JAX/Flax/Optax version, etc.
- type: input
  attributes:
    label: Additional system information
    placeholder: Python version, OS (Linux/Windows/Mac/WSL), etc.
---

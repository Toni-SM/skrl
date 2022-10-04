
First of all, **thank you**... For what? Because you are dedicating some time to reading these guidelines and possibly thinking about contributing

<hr>

### I just want to ask a question!

If you have a question, please do not open an issue for this. Instead, use the following resources for it (you will get a faster response):

- [skrl's GitHub discussions](https://github.com/Toni-SM/skrl/discussions), a place to ask questions and discuss about the project

- [Isaac Gym's forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322), a place to post your questions, find past answers, or just chat with other members of the community about Isaac Gym topics

- [Omniverse Isaac Sim's forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/simulation/69), a place to post your questions, find past answers, or just chat with other members of the community about Omniverse Isaac Sim/Gym topics

### I have found a (good) bug. What can I do?

Open an issue on [skrl's GitHub issues](https://github.com/Toni-SM/skrl/issues) and describe the bug. If possible, please provide some of the following items:

- Minimum code that reproduces the bug...
- or the exact steps to reproduce it
- The error log or a screenshot of it
- A link to the source code of the library that you are using (some problems may be due to the use of older versions. If possible, always use the latest version)
- Any other information that you think may be useful or help to reproduce/describe the problem

### I want to contribute, but I don't know how

There is a [board](https://github.com/users/Toni-SM/projects/2/views/8) containing relevant future implementations which can be a good starting place to identify contributions. Please consider the following points

#### Notes about contributing

- Try to **communicate your change first** to [discuss](https://github.com/Toni-SM/skrl/discussions) the implementation if you want to add a new feature or change an existing one
- Modify only the minimum amount of code required and the files needed to make the change
- Use the provided [pre-commit](https://pre-commit.com/) hooks to format the code. Install it by running `pre-commit install` in the root of the repository, running it periodically using `pre-commit run --all` helps reducing commit errors
- Changes that are cosmetic in nature (code formatting, removing whitespace, etc.) or that correct grammatical, spelling or typo errors, and that do not add anything substantial to the functionality of the library will generally not be accepted as a pull request
  - The only exception are changes that results from the use of the pre-commit hooks

#### Coding conventions

**skrl** is designed with a focus on modularity, readability, simplicity and transparency of algorithm implementation. The file system structure groups components according to their functionality. Library components only inherit (and must inherit) from a single base class (no multilevel or multiple inheritance) that provides a uniform interface and implements common functionality that is not tied to the implementation details of the algorithms

Read the code a little bit and you will understand it at first glance... Also

- Use 4 indentation spaces
- Follow, as much as possible, the PEP8 Style Guide for Python code
- Document each module, class, function or method using the reStructuredText format
- Annotate all functions, both for the parameters and for the return value
- Follow the commit message style guide for Git described in https://commit.style
  - Capitalize (the first letter) and omit any trailing punctuation
  - Write it in the imperative tense
  - Aim for about 50 (or 72) characters
- Add import statements at the top of each module as follows:

  ```ini
  function annotation (e.g. typing)
  # insert an empty line
  python libraries and other libraries (e.g. gym, numpy, time, etc.)
  # insert an empty line
  machine learning framework modules (e.g. torch, torch.nn)
  # insert an empty line
  skrl components
  ```

<hr>

Thank you once again,

Toni

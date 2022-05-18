
First of all, **thank you**... For what? Because you are dedicating some time to reading these guidelines and possibly thinking about contributing

<hr>

### I don't want to contribute (for now), I just want to ask a question!

If you have a question, please do not open an issue for this. Instead, use the following resources for it (you will get a faster response):

- [skrl's GitHub discussions](https://github.com/Toni-SM/skrl/discussions), a place to ask questions and discuss about the project

- [Isaac Gym's forum](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/isaac-gym/322), , a place to post your questions, find past answers, or just chat with other members of the community about Isaac Gym topics

### I have found a (good) bug. What can I do?

Open an issue on [skrl's GitHub issues](https://github.com/Toni-SM/skrl/issues) and describe the bug. If possible, please provide some of the following items:

- Minimum code that reproduces the bug...
- or the exact steps to reproduce it 
- The error log or a screenshot of it
- A link to the source code of the library that you are using (some problems may be due to the use of older versions. If possible, always use the latest version)
- Any other information that you think may be useful or help to reproduce/describe the problem

Note: Changes that are cosmetic in nature (code formatting, removing whitespace, etc.) or that correct grammatical, spelling or typo errors, and that do not add anything substantial to the functionality of the library will generally not be accepted as a pull request

### I want to contribute, but I don't know how

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

<hr>

Thank you once again,

Toni
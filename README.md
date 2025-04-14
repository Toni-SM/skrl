[![pypi](https://img.shields.io/pypi/v/skrl)](https://pypi.org/project/skrl)
[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20models-hugging%20face-F8D521">](https://huggingface.co/skrl)
![discussions](https://img.shields.io/github/discussions/Toni-SM/skrl)
<br>
[![license](https://img.shields.io/github/license/Toni-SM/skrl)](https://github.com/Toni-SM/skrl)
<span>&nbsp;&nbsp;&nbsp;&nbsp;</span>
[![docs](https://readthedocs.org/projects/skrl/badge/?version=latest)](https://skrl.readthedocs.io/en/latest/?badge=latest)
[![pre-commit](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml)
[![pytest-torch](https://github.com/Toni-SM/skrl/actions/workflows/tests-torch.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/tests-torch.yml)
[![pytest-jax](https://github.com/Toni-SM/skrl/actions/workflows/tests-jax.yml/badge.svg)](https://github.com/Toni-SM/skrl/actions/workflows/tests-jax.yml)

<br>
<p align="center">
  <a href="https://skrl.readthedocs.io">
  <img width="300rem" src="https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/_static/data/logo-light-mode.png">
  </a>
</p>
<h2 align="center" style="border-bottom: 0 !important;">SKRL - Reinforcement Learning library</h2>
<br>

**skrl** is an open-source modular library for Reinforcement Learning written in Python (on top of [PyTorch](https://pytorch.org/) and [JAX](https://jax.readthedocs.io)) and designed with a focus on modularity, readability, simplicity, and transparency of algorithm implementation. In addition to supporting the OpenAI [Gym](https://www.gymlibrary.dev), Farama [Gymnasium](https://gymnasium.farama.org) and [PettingZoo](https://pettingzoo.farama.org), Google [DeepMind](https://github.com/deepmind/dm_env) and [Brax](https://github.com/google/brax), among other environment interfaces, it allows loading and configuring NVIDIA [Isaac Lab](https://isaac-sim.github.io/IsaacLab/index.html) (as well as [Isaac Gym](https://developer.nvidia.com/isaac-gym/) and [Omniverse Isaac Gym](https://github.com/isaac-sim/OmniIsaacGymEnvs)) environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run.

<br>

### Please, visit the documentation for usage details and examples

<strong>https://skrl.readthedocs.io</strong>

<br>

> **Note:** This project is under **active continuous development**. Please make sure you always have the latest version. Visit the [develop](https://github.com/Toni-SM/skrl/tree/develop) branch or its [documentation](https://skrl.readthedocs.io/en/develop) to access the latest updates to be released.

<br>

### Citing this library

To cite this library in publications, please use the following reference:

```bibtex
@article{serrano2023skrl,
  author  = {Antonio Serrano-Muñoz and Dimitrios Chrysostomou and Simon Bøgh and Nestor Arana-Arexolaleiba},
  title   = {skrl: Modular and Flexible Library for Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {254},
  pages   = {1--9},
  url     = {http://jmlr.org/papers/v24/23-0112.html}
}
```

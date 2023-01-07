from __future__ import absolute_import
from __future__ import print_function

import os
import setuptools


root_dir = os.path.dirname(os.path.realpath(__file__))

# dependencies
INSTALL_REQUIRES = [
    "gym",
    "gymnasium",
    "torch>=1.8",
    "tensorboard",
    "wandb",
    "tqdm",
    "packaging"
]

# installation
setuptools.setup(
    name="skrl",
    author="Toni-SM",
    version=open(os.path.join(root_dir, "skrl", "version.txt")).read(),
    description="Modular and flexible library for Reinforcement Learning",
    long_description=open(os.path.join(root_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords=["reinforcement", "machine", "learning", "rl"],
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/Toni-SM/skrl",
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={'': ['version.txt']},
    include_package_data=True,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    zip_safe=False,
    project_urls={
        "Documentation": "https://skrl.readthedocs.io",
        "Repository": "https://github.com/Toni-SM/skrl",
        "Bug Tracker": "https://github.com/Toni-SM/skrl/issues",
        "Discussions": "https://github.com/Toni-SM/skrl/discussions",
    }
)

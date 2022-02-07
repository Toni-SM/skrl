from __future__ import absolute_import
from __future__ import print_function

import os
import setuptools


root_dir = os.path.dirname(os.path.realpath(__file__))

# dependencies
INSTALL_REQUIRES = [
    "gym",
    "torch",
    "tensorboard",
]

# installation
setuptools.setup(
    name="skrl",
    author="Toni-SM",
    version=open(os.path.join(root_dir, "skrl", "version.txt")).read(),
    description="Another Reinforcement Learning library :)",
    long_description=open(os.path.join(root_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    keywords=["rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    url="https://github.com/Toni-SM/skrl",
    packages=setuptools.find_packages("."),
    classifiers=[
        "Natural Language :: English",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    zip_safe=False,
)

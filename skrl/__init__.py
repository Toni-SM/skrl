import os

# read library version from file
path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(path, "r") as file:
    __version__ = file.read().strip()

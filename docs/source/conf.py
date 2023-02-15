import os
import sys

# skrl library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("[DOCS] skrl library path: {}".format(sys.path[0]))

try:
    import isaacgym
except Exception as e:
    print("[DOCS] Isaac Gym import failed: {}".format(e))
import skrl

# -- Project information

project = "skrl"
copyright = "2021, Toni-SM"
author = "Toni-SM"

release = skrl.__version__
version = skrl.__version__

master_doc = "index"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx_copybutton"
]

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "gym": ("https://www.gymlibrary.dev/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

pygments_style = "tango"
pygments_dark_style = "zenburn"

intersphinx_disabled_domains = ["std"]
templates_path = ["_templates"]
rst_prolog = """
.. include:: <s5defs.txt>

"""

# HTML output

html_theme = "furo"
html_title = f"<div style='text-align: center;'><strong>{project}</strong> ({version})</div>"

html_static_path = ["_static"]
html_favicon = "_static/data/favicon.ico"
html_css_files = ["css/s5defs-roles.css",
                  "css/skrl.css"]

html_theme_options = {
    # logo
    "light_logo": "data/logo-light-mode.png",
    "dark_logo": "data/logo-dark-mode.png",
    # edit button
    "source_repository": "https://github.com/Toni-SM/skrl",
    "source_branch": "../tree/main",
    "source_directory": "docs/source",
    # css
    "light_css_variables": {
        "color-brand-primary": "#FF4800",
        "color-brand-content": "#FF4800",
    },
    "dark_css_variables": {
        "color-brand-primary": "#EAA000",
        "color-brand-content": "#EAA000",
    },
}

# EPUB output

epub_show_urls = "footnote"

# copybutton ext

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

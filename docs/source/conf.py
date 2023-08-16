import os
import sys

# skrl library
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("[DOCS] skrl library path: {}".format(sys.path[0]))

import skrl


# project information
project = "skrl"
copyright = "2021, Toni-SM"
author = "Toni-SM"

if skrl.__version__ != "unknown":
    release = version = skrl.__version__
else:
    release = version = "1.0.0"

master_doc = "index"

# general configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "sphinx_copybutton",
    "notfound.extension",
]

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "gym": ("https://www.gymlibrary.dev/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "optax": ("https://optax.readthedocs.io/en/latest/", None),
}

pygments_style = "tango"
pygments_dark_style = "zenburn"

intersphinx_disabled_domains = ["std"]
templates_path = ["_templates"]
rst_prolog = """
.. include:: <s5defs.txt>

.. |_1| unicode:: 0xA0
    :trim:

.. |_2| unicode:: 0xA0 0xA0
    :trim:

.. |_3| unicode:: 0xA0 0xA0 0xA0
    :trim:

.. |_4| unicode:: 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |_5| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |jax| image:: /_static/data/logo-jax.svg
    :width: 28

.. |pytorch| image:: /_static/data/logo-torch.svg
    :width: 16

.. |br| raw:: html

            <br>

"""

# HTML output
html_theme = "furo"
html_title = f"<div style='text-align: center;'><strong>{project}</strong> ({version})</div>"
html_scaled_image_link = False

html_static_path = ["_static"]
html_favicon = "_static/data/favicon.ico"
html_css_files = ["css/skrl.css", "css/s5defs-roles.css"]

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

# autodoc ext
autodoc_mock_imports = [
    "gym",
    "gymnasium",
    "torch",
    "jax",
    "jaxlib",
    "flax",
    "optax",
    "tensorboard",
    "tqdm",
    "packaging",
    "isaacgym",
]

# copybutton ext
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# notfound ext
notfound_template = "404.rst"
notfound_context = {
    "title": "Page Not Found",
    "body": """
<h1>Page Not Found</h1>
<p>Sorry, we couldn't find that page in skrl.</p>
<p>Try using the search box or go to the homepage.</p>
""",
}

# suppress warning messages
suppress_warnings = [
    "ref.python",  # more than one target found for cross-reference
]

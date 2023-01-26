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

project = 'skrl'
copyright = '2021, Toni-SM'
author = 'Toni-SM'

release = skrl.__version__
version = skrl.__version__

master_doc = 'index'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx_tabs.tabs',
    'sphinx_copybutton'
]

# generate links to the documentation of objects in external projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'gym': ('https://www.gymlibrary.dev/', None),
    'gymnasium': ('https://gymnasium.farama.org/', None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

intersphinx_disabled_domains = ['std']
templates_path = ['_templates']
rst_prolog = """
 .. include:: <s5defs.txt>

"""

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
html_logo = '_static/data/skrl-up.png'
html_favicon = "_static/data/favicon.ico"
html_static_path = ['_static']
html_css_files = ['css/s5defs-roles.css',
                  'css/skrl.css']

# -- Options for EPUB output

epub_show_urls = 'footnote'

# -- Options for copybutton ext

copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

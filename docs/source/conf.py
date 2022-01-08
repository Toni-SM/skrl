# Configuration file for the Sphinx documentation builder.

# Import isaacgym hee to solve the problem of the following error: 
#   PyTorch was imported before isaacgym modules. 
#   Please import torch after isaacgym modules.
try:
    import isaacgym
except Exception as e:
    print("Using Isaac Gym Failed: {}".format(e))

# -- Project information

project = 'skrl'
copyright = '2021, Toni-SM'
author = 'Toni-SM'

release = '0.1'
version = '0.1.0'

master_doc = 'index'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

rst_prolog = """
 .. include:: <s5defs.txt>

"""

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

html_logo = '_static/data/skrl-up.png'

html_static_path = ['_static']

html_css_files = ['css/s5defs-roles.css',
                  'css/skrl.css']

# -- Options for EPUB output
epub_show_urls = 'footnote'

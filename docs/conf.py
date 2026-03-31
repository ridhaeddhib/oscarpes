# Configuration file for the Sphinx documentation builder.
import os
import sys

# Make the src-layout package importable without installing.
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# -- Project information -------------------------------------------------------
project   = 'OSCARpes'
copyright = '2024, OSCARpes contributors'
author    = 'OSCARpes contributors'
release   = '3.0.0'
version   = '3.0'

# -- General configuration -----------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',      # NumPy / Google docstrings
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
]

autodoc_mock_imports = ['ase2sprkkr']

autosummary_generate = True
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
    'member-order':     'bysource',
}
napoleon_google_docstring = False
napoleon_numpy_docstring  = True

intersphinx_mapping = {
    'python':  ('https://docs.python.org/3', None),
    'numpy':   ('https://numpy.org/doc/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- HTML output ---------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_title = 'OSCARpes v3'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
}

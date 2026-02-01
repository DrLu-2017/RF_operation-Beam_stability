import sys, os
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ALBuMS'
copyright = '2025, Alexis Gamelin, Vadim Gubaidulin, Murilo B. Alves, Teresia Olsson'
author = 'Alexis Gamelin, Vadim Gubaidulin, Murilo B. Alves, Teresia Olsson'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../../"))

extensions = []

templates_path = ['_templates']
exclude_patterns = []


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'myst_parser',
]
autodoc_default_options = {
    'member-order': 'bysource',  # Keep the order of members as they appear in the source code
    'undoc-members': True,      # Do not show undocumented members
    'private-members': True,    # Do not show private members
    'special-members': '__init__', # Document __init__ method
    'show-inheritance': False,   # Do not show base classes
}
add_module_names = False
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

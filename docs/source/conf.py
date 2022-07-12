# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../../'))
# sys.path.insert(0, os.path.abspath('../AMRL'))


# -- Project information -----------------------------------------------------

project = 'Atom manipulation with reinforcement learning ( AMRL )'
copyright = '2022, I-Ju Chen, Markus Aapro, Abraham Kipnis'
author = 'I-Ju Chen, Markus Aapro, Abraham Kipnis'

# The full version, including alpha/beta/rc tags
release = '0.0.0'
master_doc = 'index'



def run_apidoc(app):
    """Generate API documentation"""
    from sphinx.ext import apidoc
    max_depth = '10'
    apidoc.main([
        '../../AMRL',
        '-o', '.',
        '-d', max_depth,
    ])

def setup(app):
    app.connect('builder-inited', run_apidoc)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
'sphinx.ext.autodoc',
'sphinx.ext.viewcode',
'sphinx.ext.napoleon',
'sphinx.ext.autosummary'
]



# Generate automatic API documentation
autosummary_generate = True
autosummary_imported_members = True
autodoc_default_flags = ['members']
# https://github.com/readthedocs/readthedocs.org/issues/8554
autodoc_mock_imports = ['win32com'] # since ReadTheDocs compiles on Linux
# See http://blog.rtwilson.com/how-to-make-your-sphinx-documentation-compile-with-readthedocs-when-youre-using-numpy-and-scipy/
# import mock
#
# MOCK_MODULES = ['numpy', 'numpy.ma', 'scipy', 'scipy.stats']
# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.Mock()
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_logo = '..\..\..\logo.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

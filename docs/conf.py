# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import tf_pwa


# -- Project information -----------------------------------------------------

project = 'TFPWA'
copyright = "2020, "
author = ""


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.mathjax',
  'sphinx.ext.doctest',
  'sphinx.ext.graphviz'
]

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

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


autodoc_mock_imports = ["tensorflow", "iminuit"]

from tf_pwa.experimental import extra_amp
from tf_pwa.amp import get_config, PARTICLE_MODEL


def add_indent(s, number=2):
    ret = ""
    for i in s.split("\n"):
        ret += " "*number + i +"\n"
    return ret

def gen_particle_model():
    particle_model_doc = """
--------------------------
Available Resonances Model
--------------------------

"""
    idx = 1
    for k, v in get_config(PARTICLE_MODEL).items():
        n = len(k)
        doc_i = v.__doc__
        if v.__doc__ is None and v.get_amp.__doc__ is None:
            continue
        if v.__doc__ is None:
            doc_i = v.get_amp.__doc__

        particle_model_doc +=  "\n- {}. {}\n\n".format(idx, k)
        idx += 1
        particle_model_doc += add_indent(doc_i) + "\n\n"

    with open(os.path.dirname(os.path.abspath(__file__)) + "/particle_model.rst", "w") as f:
        f.write(particle_model_doc)

gen_particle_model()

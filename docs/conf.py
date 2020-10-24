"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import shutil
import subprocess

from tf_pwa.amp import PARTICLE_MODEL, get_config

# -- Project information -----------------------------------------------------
project = "TFPWA"
copyright = "2020, Yi Jiang"
author = "Yi Jiang"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.graphviz",
]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
source_suffix = [
    ".rst",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "TFPWA"

# -- Options for API ---------------------------------------------------------
autodoc_mock_imports = [
    "iminuit",
    "tensorflow",
]

# -- Generate API skeleton ----------------------------------------------------
shutil.rmtree("api", ignore_errors=True)
subprocess.call(
    " ".join(
        [
            "sphinx-apidoc",
            "../tf_pwa/",
            "-o api/",
            "--force",
            "--no-toc",
            "--templatedir _templates",
            "--separate",
        ]
    ),
    shell=True,
)


def add_indent(s, number=2):
    ret = ""
    for i in s.split("\n"):
        ret += " " * number + i + "\n"
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

        particle_model_doc += "\n- {}. {}\n\n".format(idx, k)
        idx += 1
        particle_model_doc += add_indent(doc_i) + "\n\n"

    with open(
        os.path.dirname(os.path.abspath(__file__)) + "/particle_model.rst", "w"
    ) as f:
        f.write(particle_model_doc)


gen_particle_model()

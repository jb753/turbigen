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
import turbigen
import datetime

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------


project = "turbigen"
copyright = turbigen.__copyright__
author = "James Brind"

# The full version, including alpha/beta/rc tags
release = turbigen.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "autodocsumm",
    "sphinxcontrib.programoutput",
    "matplotlib.sphinxext.plot_directive",
    "sphinxarg.ext",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = ["svg", "pdf"]
plot_formats = ["svg", "pdf"]

add_module_names = False
autoclass_content = "init"
autodoc_member_order = "bysource"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "furo"
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "description": f"Version {release}",
    "fixed_sidebar": True,
}

# Macros for LaTeX
mathjax3_config = {
    "tex": {
        "macros": {
            "Ma": r"{M\kern-.1ema}",
            "Rey": r"{R\kern-.1eme}",
            "htr": r"{\mathit{HTR}}",
            "inn": r"{\mathrm{in}}",
            "out": r"{\mathrm{out}}",
            "rel": r"{\mathrm{rel}}",
            "Chi": r"{\hat{\chi}}",
            "TE": r"{\mathrm{TE}}",
            "LE": r"{\mathrm{LE}}",
            "dee": r"\mathrm{d}",
            "RR": r"{\mathit{RR}}",
            "VmR": r"{\mathit{VR}}",
        },
    }
}

bibtex_bibfiles = ["refs.bib"]
bibtex_reference_style = "author_year"

# Update copyright year
current_year = datetime.datetime.now().year

# Optional: if you want to support a range like 2020–2025
start_year = 2023
if current_year == start_year:
    copyright_year = str(start_year)
else:
    copyright_year = f"{start_year}–{current_year}"

# Define the replacement
rst_epilog = f"""
.. |ProjectVersion| replace:: {release}
.. |copyright_year| replace:: {copyright_year}
"""

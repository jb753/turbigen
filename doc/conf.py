# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import turbigen

# -- Tutorial figures --------------------------------------------------------
# The Plotting section of the tutorial embeds the plots from step 5's report.
# The reader is shown `turbigen report input.yaml`; the SVGs it needs come from
# the --svg variant, rendered here so that step is not part of the walkthrough.

# Which report processors the tutorial shows, mapped to the static file each is
# embedded as. Keyed on processor type rather than figure index so a reordering
# of turbigen.post.STANDARD does not silently break the page. Kept in step with
# test_tutorial.py, which checks these names against the page and the report.
TUTORIAL_FIGURES = {
    "triangle": "tut_step5_triangle.svg",
    "annulus": "tut_step5_annulus.svg",
    "sections": "tut_step5_sections.svg",
}


def _render_tutorial_figures():
    here = Path(__file__).parent
    step5 = here.parent / "tutorial" / "step5"
    static = here / "_static"
    static.mkdir(exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-c",
            "from turbigen.cli import main; raise SystemExit(main())",
            "report",
            "--svg",
            "input.yaml",
        ],
        cwd=step5,
        check=True,
        capture_output=True,
    )

    # report --svg writes post_<NN>_<type>_<fig>.svg; match on <type>.
    for svg in step5.glob("post_*.svg"):
        figure_type = svg.stem.split("_")[2]
        if figure_type in TUTORIAL_FIGURES:
            shutil.move(str(svg), str(static / TUTORIAL_FIGURES[figure_type]))

    missing = [
        name for name in TUTORIAL_FIGURES.values() if not (static / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"the tutorial report produced no figure for {missing}")


_render_tutorial_figures()

# -- Project information -----------------------------------------------------


project = "turbigen"
# Stated here rather than read off the package, which no longer carries a
# __copyright__ for this to import.
copyright = f"{datetime.date.today().year}, James Brind"
author = "James Brind"

# The full version, including alpha/beta/rc tags
release = turbigen.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.programoutput",
    "sphinxarg.ext",
    "turbigen_schema",
]

# The mean line and its designer build on ember, so their inherited flow-field
# API is documented there, not here. This must point at the docs for the exact
# ember release pinned in pyproject.toml -- a `:meth:` in a design docstring
# resolves against this inventory, so a mismatched version silently drops or
# misdirects the link. Keep the two in sync: bump both together.
intersphinx_mapping = {
    "ember": ("https://ember-cfd.org/0.4.0/", None),
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Both documented modules are written in NumPy-style docstrings.
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_member_order = "bysource"

# A method in the sidebar is listed as `forward()`, not `MeanLineDesign.forward()`.
# The class it belongs to is the entry directly above it, so repeating the name
# on every child only costs the width that would have shown the method itself.
toc_object_entries_show_parents = "hide"
# Inherited members are ember's; a mention in the prose links to its docs
# rather than pulling the whole Block API onto turbigen's page.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# Local extensions, which are not installed anywhere.
sys.path.insert(0, str(Path(__file__).parent / "_ext"))

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

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

# Loaded after the theme, so it can override the theme.
html_css_files = ["custom.css"]

html_theme_options = {
    "description": f"Version {release}",
    "fixed_sidebar": True,
}

# Docs are published as one directory per version, so a build has to know the
# slug it will be served under: the rolling master build lives at "master", not
# at the release string. The picker template reads this to mark the current
# entry.
html_context = {
    "turbigen_version_slug": os.environ.get("TURBIGEN_DOCS_VERSION", release)
}

# Setting html_sidebars replaces alabaster's theme-level default wholesale, so
# its stock blocks are re-listed here around the version picker.
html_sidebars = {
    "**": [
        "about.html",
        "searchfield.html",
        "navigation.html",
        "versions.html",
        "relations.html",
        "donate.html",
    ]
}

# html_baseurl is deliberately unset. Without it Sphinx emits only relative
# links, so one build tree serves correctly both from a GitHub Pages project
# subpath and from a site apex, with no rebuild in between.

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

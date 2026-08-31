# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import turbigen

# -- Tutorial figures --------------------------------------------------------
# The Plotting section of the tutorial embeds the plots from step 5's report.
# The reader is shown `turbigen report input.yaml`; the SVGs it needs come from
# the --svg variant, rendered here so that step is not part of the walkthrough.


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

    for src, dst in {
        "post_00_triangle_0.svg": "tut_step5_triangle.svg",
        "post_01_annulus_0.svg": "tut_step5_annulus.svg",
        "post_02_sections_0.svg": "tut_step5_sections.svg",
    }.items():
        shutil.move(str(step5 / src), str(static / dst))


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
    "sphinxcontrib.bibtex",
    "sphinxcontrib.programoutput",
    "sphinxarg.ext",
]

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

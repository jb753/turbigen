"""A directive that shows one example: its input, its log and its figures.

``.. turbigen-example:: turbine_cascade`` expands to the download link, the
configuration file, the transcript of the run and the plots it drew.

Nothing here generates a page. The predecessor wrote a complete rst file per
example, pasting in the YAML and the log, which put a second copy of both under
`doc/examples/` --- gitignored, so a clean checkout built an examples index with
no pages under it and no error to say so. Including the files instead means the
committed page is the prose, the run is the build product, and the two cannot
disagree.

The YAML shown is the source in `examples/`, comments and all, rather than the
copy the run resolved into its own directory: the comments are the half of an
example worth reading, and a resolved config has none.

Missing output is normal --- the examples cost minutes of CFD and are not run by
an ordinary documentation build --- so it produces a note saying so rather than
an error. That is what lets `make doc` work from a fresh clone.
"""

import os
from pathlib import Path

from docutils.parsers.rst import directives
from sphinx.util.docutils import SphinxDirective

DOC_DIR = Path(__file__).resolve().parent.parent
ROOT = DOC_DIR.parent
INPUT_DIR = ROOT / "examples"
BUILD_DIR = DOC_DIR / "_examples"

LOG_NAME = "log_turbigen.txt"


class TurbigenExample(SphinxDirective):
    """Show the input, log and figures of one example."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {"width": directives.unchanged}

    def run(self):
        name = self.arguments[0]
        width = self.options.get("width", "100%")

        source = INPUT_DIR / f"{name}.yaml"
        if not source.exists():
            raise self.error(
                f"turbigen-example: no such example {name!r}; expected "
                f"{source.relative_to(ROOT)}"
            )

        # Rebuild this page when a file it shows changes, which is what makes
        # `make doc-dev` redraw as soon as a run finishes.
        self.env.note_dependency(str(source))

        out_dir = BUILD_DIR / name
        log = out_dir / LOG_NAME
        figures = sorted(out_dir.glob("post_*.svg"))

        lines = [
            f":download:`Download this example <{self._relative(source)}>`",
            "",
            "Input file",
            "==========",
            "",
            f".. literalinclude:: {self._relative(source)}",
            "   :language: yaml",
            "",
        ]

        if log.exists():
            self.env.note_dependency(str(log))
            lines += [
                "Log output",
                "==========",
                "",
                f".. literalinclude:: {self._relative(log)}",
                "   :language: none",
                "",
            ]
        else:
            lines += [
                ".. note::",
                "",
                "   This example has not been run in this checkout, so the log",
                "   and plots below it are missing. Run",
                "   ``make generate-examples`` to produce them.",
                "",
            ]

        if figures:
            lines += ["Plots", "=====", ""]
            for figure in figures:
                self.env.note_dependency(str(figure))
                lines += [
                    f".. image:: {self._relative(figure)}",
                    f"   :width: {width}",
                    "",
                ]

        # Section headings, so the input, the log and the plots get their own
        # entries in the page's contents rather than running together.
        return self.parse_text_to_nodes("\n".join(lines), allow_section_headings=True)

    def _relative(self, path):
        """Return `path` as written from the page being built.

        Relative rather than root-anchored because half of what is shown lives
        outside the documentation directory: `examples/` is source that the
        program itself runs, and a path from the Sphinx root cannot leave it.
        """
        here = DOC_DIR / Path(self.env.docname).parent

        return Path(os.path.relpath(path, here)).as_posix()


def setup(app):
    app.add_directive("turbigen-example", TurbigenExample)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }

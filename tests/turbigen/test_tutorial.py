"""Tests that the tutorial still describes what turbigen does.

The tutorial is what a new user types first, so a tutorial that no longer runs
is worse than none: it fails in the hands of the one reader least able to tell
a stale page from a broken program. It went stale in the usual way, by being
prose with the code pasted into it --- a `code-block` is a copy, and a copy
drifts silently. One of them had drifted far enough to caption a block of YAML
as a Python file.

So the page quotes files instead of restating them, and this reads the same
files. The tutorial is written in stages and each stage is a case directory of
its own, complete and runnable at the point the page leaves it; `step1` is the
skeleton, where the design exists but does nothing yet.

What is asserted is only what the page claims. `step1` claims that the files
load and that the run stops because `forward` is not implemented, so that is
the assertion --- not that a mean line comes out, which the page does not yet
promise.

`step5` is where the page leaves the reader, with a designed machine and a
report drawn from it. Its transcripts are output, not prose --- `program-output`
runs the command as the build proceeds --- so the drift that matters is the
command changing what it prints, which the build catches with `-W` but the
commit does not. These run the same commands and check the same claims.

Test cases:
- test_plugin_is_found_in_the_step_directory: the case carries its own plugins
- test_the_skeleton_stops_at_forward: it loads, and stops where the page says
- test_a_design_transcript_exits_as_the_page_says: every :returncode: still holds
- test_the_report_produces_the_figures_the_page_depends_on: and they are drawn
- test_the_plotting_section_embeds_the_figures_the_build_generates: names agree
- test_tutorial_literalincludes_resolve: every quotation still has a source
- test_tutorial_program_outputs_resolve: every transcript is run somewhere
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from turbigen import Config, plugins

RUN_TURBIGEN = [sys.executable, "-c", "from turbigen.cli import main; raise SystemExit(main())"]
"""Invoke the CLI in a fresh interpreter; there is no ``turbigen.__main__``."""

ROOT = Path(__file__).resolve().parents[2]
STEP1 = ROOT / "tutorial" / "step1"
STEP5 = ROOT / "tutorial" / "step5"
TUTORIAL_RST = ROOT / "doc" / "tutorial.rst"
CONF_PY = ROOT / "doc" / "conf.py"


def test_plugin_is_found_in_the_step_directory():
    """The case directory holds the plugins it uses, and no ancestor's.

    Discovery walks up from the config file and takes the first match, so a
    case with no plugin directory of its own silently runs on whatever happens
    to sit above it --- passing on the machine that has one and failing on the
    machine that does not. The tutorial tells the reader to copy the directory
    somewhere and run it, which is only true if the match is inside.
    """
    found = plugins.find_plugin_dir(STEP1 / "input.yaml")

    assert found == STEP1 / "turbigen_plugins"


FIGURE = re.compile(r"^\.\. figure:: */_static/(tut_step5_\w+\.svg)", re.MULTILINE)
"""An SVG the Plotting section embeds, by basename."""


def test_the_plotting_section_embeds_the_figures_the_build_generates():
    """`conf.py` renders step 5's report to these SVGs; the page embeds them by
    name. The two lists have to agree or the page shows broken images, and the
    documentation is not built on every commit to catch it.
    """
    embedded = set(FIGURE.findall(TUTORIAL_RST.read_text()))
    generated = set(re.findall(r"tut_step5_\w+\.svg", CONF_PY.read_text()))

    assert embedded, "the Plotting section embeds no figures"
    assert embedded == generated


def test_the_report_produces_the_figures_the_page_depends_on(tmp_path):
    """`conf.py` keys the figures on report-processor type: a plot that stops
    drawing, or a `type` that gets renamed, leaves the page with a gap. Run the
    command the build runs and check the figures come out.

    A subprocess, and a copy of the step: the report writes beside its input,
    and step 5 carries the `fan` plugin that step 1 also registers.
    """
    case_dir = shutil.copytree(STEP5, tmp_path / "step5")
    result = subprocess.run(
        [*RUN_TURBIGEN, "report", "--svg", str(case_dir / "input.yaml")],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (case_dir / "post.pdf").is_file()

    wanted = set(re.findall(r"tut_step5_(\w+)\.svg", CONF_PY.read_text()))
    drawn = {svg.stem.split("_")[2] for svg in case_dir.glob("post_*.svg")}

    assert wanted <= drawn, f"the report drew {drawn}, the page needs {wanted}"


def test_the_skeleton_stops_at_forward():
    """The skeleton loads, and designing stops where the page says it does.

    The page shows this traceback as the reward for having got the files right,
    so it is a claim about behaviour like any other. Loading has to get as far
    as calling `forward`: a config rejected earlier would print a different
    error and the page would be wrong about it.
    """
    config = Config.from_file(STEP1 / "input.yaml")

    with pytest.raises(NotImplementedError, match="Implement the forward method"):
        config.design()


LITERALINCLUDE = re.compile(
    r"^\.\. literalinclude:: *(\S+)\n((?:^ +:.*\n|^\s*\n(?=^ +:))*)",
    re.MULTILINE,
)
"""One directive: its path, and the block of options indented beneath it."""

ANCHORS = ("start-at", "start-after", "end-at", "end-before", "pyobject")
"""Options naming text that must occur in the file being quoted."""


def test_tutorial_literalincludes_resolve():
    """Every file the tutorial quotes exists, and every anchor is in it.

    `sphinx-build -W` fails on these too, but the documentation is not built on
    every commit and the tests are. An anchor is a comment or a statement in
    someone else's file: rewording one is an ordinary edit that silently takes
    a paragraph of the tutorial with it.
    """
    text = TUTORIAL_RST.read_text()
    directives = LITERALINCLUDE.findall(text)

    assert directives, f"no literalinclude directives in {TUTORIAL_RST.name}"

    for path, options in directives:
        source = (TUTORIAL_RST.parent / path).resolve()
        assert source.is_file(), f"{TUTORIAL_RST.name} quotes missing {path}"

        quoted = source.read_text()
        for name, value in re.findall(r"^ +:(\S+): *(.*?)\s*$", options, re.MULTILINE):
            if name not in ANCHORS:
                continue
            needle = value.split(".")[-1] if name == "pyobject" else value
            assert needle in quoted, f"{path} has no {name} `{value}`"


PROGRAM_OUTPUT = re.compile(
    r"^\.\. program-output:: *(.+)\n((?:^ +:.*\n|^\s*\n(?=^ +:))*)",
    re.MULTILINE,
)
"""One directive: the command line, and the block of options beneath it."""


def test_tutorial_program_outputs_resolve():
    """Every transcript in the page is run in a directory that exists.

    The transcripts are output, not prose: `program-output` runs the command
    while the documentation builds, so what the page shows is what the command
    said. That only holds while the `:cwd:` it is run in is still there, and a
    step directory is exactly the kind of thing a later stage renames.
    """
    text = TUTORIAL_RST.read_text()
    directives = PROGRAM_OUTPUT.findall(text)

    assert directives, f"no program-output directives in {TUTORIAL_RST.name}"

    for command, options in directives:
        cwd = dict(re.findall(r"^ +:(\S+): *(.*?)\s*$", options, re.MULTILINE)).get(
            "cwd"
        )
        assert cwd, f"`{command}` is run wherever the build happens to be"

        directory = (TUTORIAL_RST.parent / cwd).resolve()
        assert directory.is_dir(), f"`{command}` is run in missing {cwd}"


def _options(block):
    return dict(re.findall(r"^ +:(\S+): *(.*?)\s*$", block, re.MULTILINE))


def _design_transcripts():
    """Yield ``(cwd, returncode)`` for every ``turbigen design`` block."""
    for command, block in PROGRAM_OUTPUT.findall(TUTORIAL_RST.read_text()):
        if command.strip() == "turbigen design input.yaml":
            opts = _options(block)
            yield opts["cwd"], int(opts["returncode"])


DESIGN_TRANSCRIPTS = list(_design_transcripts())


def test_the_tutorial_shows_a_design_transcript_for_every_step():
    assert len(DESIGN_TRANSCRIPTS) >= 4


@pytest.mark.parametrize("cwd, returncode", DESIGN_TRANSCRIPTS)
def test_a_design_transcript_exits_as_the_page_says(cwd, returncode):
    """Each `turbigen design` block declares a :returncode:, and the build
    fails if the command disagrees. A code change that makes step 3 stop
    failing, or step 4 start, rewrites these transcripts without anyone
    touching the page.

    A subprocess each, as the build does: the steps carry a plugin apiece, all
    registering the same `fan` type, so one interpreter cannot load two.
    """
    case = (TUTORIAL_RST.parent / cwd).resolve() / "input.yaml"
    result = subprocess.run(
        [*RUN_TURBIGEN, "design", str(case)], capture_output=True, text=True
    )
    assert result.returncode == returncode, result.stderr

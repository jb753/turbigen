"""Tests that the shipped examples still describe a machine.

The examples are documentation, and documentation that no longer loads is worse
than none: it is what a new user types first. They went stale unnoticed through
the config rewrite precisely because nothing here read them --- every key that
was renamed, removed or turned into a ClassVar left them a little further from
running, and the first sign of it was the documentation build producing nothing.

Designing is where the line is drawn. It is pure numpy and costs milliseconds,
so the whole set can be checked on every commit, and it exercises what actually
rots: the schema, the ``type:`` names, and the arithmetic each design stage does
with the values. What it deliberately does not check is meshing or solving,
which cost minutes; `doc/generate_examples.py` runs those, and the documentation
shows the result.

Test cases:
- test_there_are_examples: the glob would otherwise pass by finding nothing
- test_an_example_designs: every example loads and designs
"""

from pathlib import Path

import pytest

from turbigen import Config

EXAMPLES = sorted((Path(__file__).resolve().parents[2] / "examples").glob("*.yaml"))


def test_there_are_examples():
    assert EXAMPLES


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_an_example_designs(path):
    """Load one example and design it, as far as its config goes.

    `design` runs every stage the config describes and stops there, so an
    example with no blades is not required to grow any: what is under test is
    that the file says something the current code understands.
    """
    machine = Config.from_file(path).design()

    assert machine.mean_line is not None

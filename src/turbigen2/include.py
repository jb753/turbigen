"""Assembling one config document out of several files.

A top-level ``include:`` names files whose keys are merged underneath this
one's, so a site's ``job:`` block, a standard ``solver:`` and a family's
``mesh:`` can be written once and shared.

One rule, everywhere: **later beats earlier, the including file beats
everything it includes, and mappings merge exactly one level deep.**

Depth one is a deliberate stopping point, not a limitation waiting to be
lifted. It covers what a file actually wants --- ``solver: {n_step: 100}`` over
an included solver block --- while refusing to splice two mappings that declare
different ``type:`` keys into one node with ``psi`` from one design and ``span``
from another, which would fail validation somewhere unrelated to the mistake.
Lists replace wholesale, since merging ``blades:`` by index is a trap and the
row you meant is never the row you would get.

The package this replaces has two rules under one word: its includes merge into
each other by plain ``dict.update``, so a second one defining ``solver:`` wins
wholesale, while the including file merges a level deep. The asymmetry is the
loop being written twice rather than a decision, and there is no spelling that
tells you which you are getting.

Ambiguity is refused rather than resolved, in the two places it can arise. A
key written twice in one file raises, at any depth. And two files in the *same*
``include:`` list both setting a top-level key raises, because siblings have no
order of precedence worth relying on --- where a file overriding what it
includes is a hierarchy, and is the whole reason the merge is a level deep.
"""

import logging
from pathlib import Path

import yaml

import ember.yaml_util

logger = logging.getLogger("turbigen")

_COMPOSER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
"""Loader used to look at a file's key structure before it is built.

Only the shape is wanted, so this deliberately does *not* go through
:func:`ember.yaml_util._float_loader`: scientific notation matters to values
and a duplicate key is a question about keys. Composing is a second parse of
the same text, which is why it is worth using the C loader where there is one
--- the pure-Python one is the difference between 0.15 s and 7.7 s on a
multi-megabyte scalar.
"""

INCLUDE_KEY = "include"
"""Top-level key naming the files to merge underneath this one."""

DROPPED_FROM_INCLUDES = ("result",)
"""Keys an included file may hold but may not contribute.

Only ``result:``, and the reason is worth stating. Including a finished
``output.yaml`` to inherit a converged design is a legitimate thing to want,
but its answer belongs to the run that computed it. Carried through, a new case
would claim a solution it never marched --- and
:mod:`turbigen2.database` decides what counts as a sample by reading
``result:``, so one inherited answer poisons every warm start that globs it.

Dropped rather than refused, because the workflow is worth supporting. This
duplicates the string in :data:`turbigen2.case.RESULT_KEY`, which cannot be
imported here without a cycle; `test_include.py` asserts the two agree.
"""


def merge(base, overlay):
    """Return `base` with `overlay` laid over it, merging mappings one deep.

    Neither argument is modified, so a document can be merged into several
    others without the first result aliasing the second.

    Parameters
    ----------
    base : dict
        What is being overridden.
    overlay : dict
        What overrides it.

    Returns
    -------
    merged : dict

    """
    merged = dict(base)

    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = {**current, **value}
        else:
            merged[key] = value

    return merged


def read(path):
    """Read a config document, resolving any ``include:`` it names.

    Parameters
    ----------
    path : Path or str
        File to read.

    Returns
    -------
    data : dict
        The merged document, with no ``include:`` key left in it. Nothing is
        validated here: this returns the raw mapping a `Config` is built from,
        so an override applies to the assembled document rather than to
        whichever file happened to define the key.

    """
    return _read(Path(path), ())


def _read(path, chain):
    """Return the document at `path`, merged under the files it includes.

    `chain` is the resolved paths currently being read, outermost first. It is
    the *current chain* rather than everything seen so far, so a diamond --- two
    files that both include a third --- resolves, while a loop is caught.
    """
    path = path.resolve()

    if path in chain:
        loop = " -> ".join(str(link) for link in (*chain, path))
        raise ValueError(f"Config files include each other in a loop: {loop}.")

    data = _load(path)
    names = _names(data.pop(INCLUDE_KEY, []), path)

    merged = {}
    claimed = {}
    for name in names:
        # Relative to the file that named it, never to the working directory.
        # That is what lets a case directory and its fragments be copied
        # anywhere together, and it is why the same file cannot design two
        # different machines depending on where it was invoked from. Plugin
        # discovery is anchored the same way and for the same reason.
        child = (path.parent / name).resolve()
        if not child.is_file():
            raise ValueError(
                f"Cannot find the config file {name!r} included by {path}; "
                f"looked for it at {child}."
            )

        logger.debug(f"Including {child} from {path}")
        contribution = _dropped(_read(child, (*chain, path)), child)

        for key in contribution:
            if key in claimed:
                raise ValueError(
                    f"Both {claimed[key]!r} and {name!r}, included by {path}, "
                    f"set the top-level key {key!r}. Files in one include: "
                    f"list have no precedence between them, so set {key!r} in "
                    f"{path.name} itself to say what it should be."
                )
            claimed[key] = name

        merged = merge(merged, contribution)

    return merge(merged, data)


def _load(path):
    """Return the mapping in `path`, refusing a key it writes twice.

    Checked before the document is built, because building it is what loses
    the evidence: every YAML loader keeps the last of a repeated key and says
    nothing, so ``n_step`` set twice is a setting silently ignored. The old
    package wrote this check and then wired it to `read_yaml_list` only, so no
    config has ever been checked by it.
    """
    with open(path, "r") as stream:
        _check_unique(yaml.compose(stream, Loader=_COMPOSER), path)

    data = ember.yaml_util.read_yaml(path)

    # An empty file is a mapping of nothing, not a None to trip over three
    # frames later.
    return {} if data is None else data


def _check_unique(node, path, where=""):
    """Raise if any mapping under `node` writes a key twice."""
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if key.value in seen:
                raise ValueError(
                    f"{path} sets {key.value!r} twice"
                    f"{' in ' + where if where else ' at the top level'}, "
                    f"on line {key.start_mark.line + 1}. Only the last would "
                    f"be used, so the earlier one is a setting that would go "
                    f"quietly missing."
                )
            seen.add(key.value)
            _check_unique(
                value, path, f"{where}.{key.value}" if where else str(key.value)
            )

    elif isinstance(node, yaml.SequenceNode):
        for i, value in enumerate(node.value):
            _check_unique(value, path, f"{where}[{i}]")


def _names(names, path):
    """Return the include list of `path`, checking it is one."""
    if isinstance(names, str):
        raise ValueError(
            f"The include: in {path} is a single file name, {names!r}. It "
            f"takes a list, so write it as `include: [{names}]`."
        )

    if not isinstance(names, (list, tuple)):
        raise ValueError(
            f"The include: in {path} is {names!r}, which is not a list of "
            f"config file names."
        )

    return names


def _dropped(data, path):
    """Return `data` without the keys an included file may not contribute."""
    kept = dict(data)

    for key in DROPPED_FROM_INCLUDES:
        if kept.pop(key, None) is not None:
            logger.debug(
                f"Ignoring the {key!r} of {path}, which belongs to the run "
                f"that wrote it rather than to the design including it."
            )

    return kept

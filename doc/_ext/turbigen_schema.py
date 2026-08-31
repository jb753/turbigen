"""The configuration reference, generated from the config classes themselves.

A config file is a tree of :class:`~turbigen.node.Node` classes, every one of
which already carries a docstring on itself and on each of its fields. Writing
that out by hand would copy prose that exists, into a listing nothing checks,
so this walks the tree instead and emits the reference from the code that reads
the file. A key renamed, defaulted differently or removed changes this page in
the same commit.

The walk starts at :class:`turbigen.config.Config` and follows field
annotations, so the page is ordered as a file is written rather than
alphabetically or by module. A family --- a key whose mapping has alternatives selected by
``type:`` --- becomes a section with one subsection per alternative.

Field docstrings do not survive to runtime: Python evaluates the string literal
after an annotated assignment and throws it away. They are read back out of the
source with :class:`~sphinx.pycode.ModuleAnalyzer`, the machinery autodoc uses
for the same purpose, which works on any importable module --- turbigen's own
and ember's alike.
"""

import dataclasses
import inspect
import re
import types
import typing
from typing import get_args, get_origin

from docutils import nodes
from docutils.statemachine import StringList
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles

from turbigen import config as turbigen_config
from turbigen.node import Node

SECTION_CHARACTERS = "-~\"'+"
"""Underline character for each depth, starting one below the page's sections.

The prose on the page uses ``^``, so every generated key nests inside the
section the directive is written in and the whole reference is one entry in the
sidebar rather than fifteen.
"""

SCALAR_NAMES = {
    bool: ("true or false", "true or false"),
    int: ("integer", "integers"),
    float: ("float", "floats"),
    str: ("text", "text"),
}
"""What an annotation is called in a YAML file, singular and plural.

YAML's own names for these, which happen to be Python's as well --- a file has
``!!float`` and ``!!int`` in it, so "float" is what the format calls the thing
and not a detail of the program reading it. What is rendered rather than
printed is the shape around them: ``tuple[float, ...]`` names a type YAML
cannot express, and is a list here. So every annotation the walk meets goes
through this table or through :func:`value_text`, never through ``str()``.
"""

COUNTS = {2: "two", 3: "three", 4: "four"}
"""How a fixed-length sequence says how long it is."""

UNKNOWN = ("value", "values")
"""What an annotation this module does not recognise is called."""

OWN_KEYS = "mapping of your own keys"
"""What a bare ``dict`` is called.

The one mapping this page cannot list the keys of, because they are the
reader's to invent --- a design variable named by its path, in `batch.bounds`
and `batch.values`. Said in the value rather than left to the description, so
that the absence of a link reads as a fact about the key instead of as an
omission.
"""

HOME = "turbigen"
"""Top-level package whose docstrings are this page's to print in full.

A field can be documented in another package --- every setting under `solver:`
is ember's --- and such a description is quoted rather than owned. Only its
first paragraph is taken, for two reasons. It is the part that says what the
key does, where the rest is rationale that belongs beside the code it explains:
ember's `mix_reflective` runs to forty lines on what a reflective plane gives
up, which is worth reading and is not a table cell. And it is the part least
likely to reach for a cross-reference this project cannot resolve --- a
citation into a bibliography that is not ours, say.
"""

CITATION = re.compile(r":cite:(?:[tp]:)?`([^`]+)`")
"""A citation role, which resolves only against the citing project's own bib."""


def label_of(path):
    """Return the cross-reference label for the key at `path`.

    One definition, used both to claim a label and to ask whether a claimed one
    belongs to this path, so the two spellings cannot drift apart.
    """
    return "config-" + path.replace(".", "-").lower()


def strip_none(annotation):
    """Split ``X | None`` into ``(X, True)``; anything else is ``(it, False)``."""
    if get_origin(annotation) not in (typing.Union, types.UnionType):
        return annotation, False

    args = get_args(annotation)
    rest = [arg for arg in args if arg is not type(None)]
    optional = len(rest) < len(args)
    return (rest[0] if len(rest) == 1 else annotation), optional


def sequence_member(annotation):
    """Return ``(element annotation, fixed length or None)`` for a sequence.

    ``(None, None)`` for anything that is not one. A length comes back only for
    a sequence written out in full --- ``tuple[float, float]`` is two numbers,
    where ``tuple[float, ...]`` is any many.
    """
    if get_origin(annotation) not in (tuple, list):
        return None, None

    args = get_args(annotation)
    members = [arg for arg in args if arg is not Ellipsis]
    if not members or not all(member is members[0] for member in members):
        return None, None

    return members[0], (None if Ellipsis in args else len(members))


def mapping_in(annotation):
    """Return the class whose keys an annotation can hold, or None.

    Looks through an optional and through a sequence, so ``Mesher | None`` and
    ``tuple[Post, ...]`` both give up the class they are written around.
    """
    annotation, _ = strip_none(annotation)
    if isinstance(annotation, type) and issubclass(annotation, Node):
        return annotation

    member, _ = sequence_member(annotation)
    return mapping_in(member) if member is not None else None


def yaml_value(value):
    """Return `value` written the way a config file would write it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(yaml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{}" if not value else str(value)
    if isinstance(value, str):
        return repr(value)
    return str(value)


def default_text(field):
    """Return the default of a key, or nothing where it has none.

    Nothing, rather than the word "required", because the two are the same
    fact: a key with a default may be left out and takes it, and a key without
    one has to be written. Saying both put a word on 45 rows that the next
    column already answers, so the page states the rule once instead.
    """
    if field.default is not dataclasses.MISSING:
        return f"default ``{yaml_value(field.default)}``"
    if field.default_factory is not dataclasses.MISSING:
        return f"default ``{yaml_value(field.default_factory())}``"
    return ""


def nullable_text(annotation, field, value):
    """Return whether ``null`` is worth naming as a value of this key.

    Only where the default is not itself ``null``. A key that defaults to
    ``null`` already says both things it has to --- that it may be left out,
    and what it is when it is --- so calling it optional as well restates the
    default in a second vocabulary. Nothing is required and nullable at once,
    so there is no third case to cover.

    What is left is the handful where ``null`` means something other than the
    default: ember's relaxation factors, where it turns a boundary treatment
    off rather than choosing a number for it. There, ``null`` is a value the
    reader may write, and one they could not guess from a default of 0.05.
    """
    _, nullable = strip_none(annotation)
    if not nullable:
        return ""

    default = field.default
    if default is dataclasses.MISSING or default is None:
        return ""

    # A comma where the value is already a choice, so that a boolean reads
    # "true or false, or null" rather than as three alternatives in a row.
    return ", or ``null``" if " or " in value else " or ``null``"


def value_text(annotation, many=False):
    """Return what a config file may write for `annotation`.

    Everything in a file is a key, so a key holding further keys is a mapping
    like any other and is described as one. Where those keys are listed is the
    Key cell's business --- it is the name that links --- so this says only
    what may be written, and a mapping the user fills with keys of their own
    (a bare ``dict``, so ``batch.bounds``) says that instead.
    """
    # Whether it may be written null is a question about the default, and is
    # answered beside it by `nullable_text`; here that is looked through to
    # reach the type the key actually takes.
    annotation, _ = strip_none(annotation)

    node = annotation if isinstance(annotation, type) else None
    if node is not None and issubclass(node, Node):
        return "mappings" if many else "mapping"

    member, length = sequence_member(annotation)
    if member is not None:
        counted = f"{COUNTS[length]} " if length in COUNTS else ""
        word = "lists" if many else "list"
        return f"{word} of {counted}{value_text(member, True)}"

    if annotation is dict:
        return "mappings of your own keys" if many else OWN_KEYS

    return SCALAR_NAMES.get(annotation, UNKNOWN)[1 if many else 0]


def attribute_docs(module):
    """Return the attribute docstrings in `module`, keyed ``(class, field)``.

    Empty when the module cannot be parsed. That is not treated as an error
    here --- the caller falls back to a base class, and `test_config_docs.py`
    is what refuses a field left with nothing anywhere.
    """
    try:
        analyzer = ModuleAnalyzer.for_module(module)
        analyzer.analyze()
    except PycodeError:
        return {}
    return analyzer.find_attr_docs()


class Schema:
    """A walk over the config tree, writing reStructuredText as it goes.

    One instance per directive invocation. It remembers the label of every
    class it has written, so a mapping reachable by two paths is documented once
    and linked to the second time --- which is also what stops a label being
    emitted twice, and the docs build with ``-W``.
    """

    def __init__(self):
        self.lines = []
        self.labels = {}
        self.docs = {}
        self.dependencies = set()

    #
    # READING THE CLASSES
    #

    def attr_docs(self, module):
        """Return `module`'s attribute docstrings, analysing it once."""
        if module not in self.docs:
            self.docs[module] = attribute_docs(module)
        return self.docs[module]

    def describe(self, cls, name):
        """Return the description of field `name` on `cls`, and who wrote it.

        Walks the MRO, because a subclass restating a field to change its
        default writes no new docstring: the meaning is unchanged and only the
        number moved. ``H.yplus`` is documented on ``Mesher``, ``Ember.cfl`` on
        ember's own solver, ``Deviation.gain`` on ``Iterator``. Taking the
        first description found up the MRO is what lets those restatements stay
        as bare in the source as they read.

        Returns ``(lines, owner)``, with `owner` the class the description came
        from, or ``([], None)`` when nothing anywhere carries one.
        """
        for base in cls.__mro__:
            found = self.attr_docs(base.__module__).get((base.__qualname__, name))
            if found:
                return list(found), base
        return [], None

    def moved_default(self, cls, name, owner):
        """Return the default `owner` gives field `name`, when it differs here.

        None when it does not differ, when the description is the class's own,
        or when either end has no default. This is the one thing the MRO
        fallback would otherwise swallow: `Ember` restates four of ember's
        settings purely to lower them, and a reader given ember's description
        under turbigen's number should be told that the two disagree, since
        that number is the one thing the borrowed description does not cover.

        Only across packages, though. A default lowered from a base class of
        our own --- `Deviation.clip` against `Iterator.clip` --- is not news to
        anybody: the value column already gives the default that applies, and
        what some base class would otherwise have said is an implementation
        detail of a file format that has no base classes in it.
        """
        if owner is None or owner is cls or owner.__module__.startswith(HOME):
            return None

        here = field_named(cls, name)
        there = field_named(owner, name)
        if here is None or there is None:
            return None
        if dataclasses.MISSING in (here.default, there.default):
            return None
        if here.default == there.default:
            return None

        return there.default

    #
    # RENDERING A VALUE
    #

    def choices_text(self, annotation):
        """Return the ``type:`` a key chooses between, if it chooses at all.

        Written after the default rather than beside the value, because a
        reader who has decided to write the key at all wants the alternatives
        last, where they read as what to write next --- and because "mapping,
        one of ``perfect``, ``real``, required" puts the two facts that matter
        either side of a list that can run to six names.
        """
        held = mapping_in(annotation)
        alternatives = held.options() if held is not None else {}
        if not alternatives:
            return ""

        names = ", ".join(f"``{name}``" for name in sorted(alternatives))
        stripped, _ = strip_none(annotation)
        each = "each " if sequence_member(stripped)[0] is not None else ""
        return f"; {each}one of {names}"

    def label_for(self, cls, path):
        """Return the label of `cls`, claiming `path` for it if it is new.

        A class documented once keeps the label of the path that reached it
        first, so a second reference links to where it was written instead of
        writing it again under a second name.
        """
        return self.labels.setdefault(cls, label_of(path))

    #
    # WALKING
    #

    def emit(self, cls, title, path, depth):
        """Write the section for `cls`, and for every mapping under it."""
        source = inspect.getsourcefile(cls)
        if source:
            self.dependencies.add(source)

        self.line(f".. _{self.label_for(cls, path)}:")
        self.line("")
        self.line(title)
        self.line(SECTION_CHARACTERS[depth] * len(title))
        self.line("")
        self.class_docstring(cls)

        alternatives = cls.options()
        if alternatives:
            self.family(alternatives, path, depth)
        else:
            self.keys(cls, path, depth)

    def family(self, alternatives, path, depth):
        """Write one subsection per alternative of a family.

        The alternatives are linked, because this is the sentence that exists
        to say what the choices are, and the sections describing them are the
        one part of the reference no table of contents reaches --- the sidebar
        stops at the keys, one level above. Six of them under `post_process:`
        is enough that landing on the right one beats reading past the rest.

        The link is claimed here rather than by :meth:`emit` below, which is
        only an ordering: both spell the path the same way, so the label the
        sentence points at is the one the section will carry.
        """
        names = ", ".join(
            f":ref:`{name} <{self.label_for(member, f'{path}.{name}')}>`"
            for name, member in sorted(alternatives.items())
        )

        # A reference carries no inline markup with it, so the monospace comes
        # from `custom.css` matching this paragraph -- the same arrangement,
        # and the same reason, as the key column of a `config-keys` table.
        self.line(".. rst-class:: config-choices")
        self.line("")
        self.line(f"Takes one of {names} as its ``type:``.")
        self.line("")

        for name, member in sorted(alternatives.items()):
            self.emit(member, f"type: {name}", f"{path}.{name}", depth + 1)

    def keys(self, cls, path, depth):
        """Write the key table of `cls`, then the mappings its keys hold."""
        fields = dataclasses.fields(cls)
        if not fields:
            self.line("This mapping takes no keys.")
            self.line("")
            return

        hints = typing.get_type_hints(cls)

        self.line(".. list-table::")
        self.line("   :header-rows: 1")
        self.line("   :widths: 18 24 58")
        self.line("   :class: config-keys")
        self.line("")
        self.row("Key", "Value", "Description")

        nested = []
        for field in fields:
            annotation = hints.get(field.name, field.type)
            here = f"{path}.{field.name}" if path else field.name
            value = value_text(annotation)

            # Claim the label before the Key cell asks for it, so that the
            # name links to the section this key opens -- or, for a mapping
            # already written under another key, to where it was written.
            held = mapping_in(annotation)
            if held is not None:
                self.label_for(held, here)

            self.row(
                self.key_text(field.name, held),
                self.value_cell(annotation, field, value),
                self.description(cls, field.name),
            )

            # Claimed by this key, rather than by an earlier one that reached
            # the same class, so the mapping is written where it was first named
            # and every other mention of it is a link.
            if held is not None and self.labels.get(held) == label_of(here):
                nested.append((held, here))

        self.line("")

        for held, here in nested:
            self.emit(held, f"{here.rsplit('.', 1)[-1]}:", here, depth + 1)

    def key_text(self, name, held):
        """Return the Key cell: the key as written, linked where it leads on.

        A key holding further keys is the row a reader wants to follow, so the
        name itself is the link rather than a second one in the value beside
        it. With the colon it is written as the file writes it, and as the
        page's prose names it --- a key, rather than a Python attribute that
        happens to share the spelling.

        A reference carries no inline markup with it, so the monospace comes
        from ``custom.css`` matching the first column of a ``config-keys``
        table. That is what keeps a linked key and a plain one looking alike.
        """
        label = self.labels.get(held) if held is not None else None
        if label is None:
            return f"``{name}:``"
        return f":ref:`{name}: <{label}>`"

    def value_cell(self, annotation, field, value):
        """Return the Value cell: what may be written, and what it defaults to.

        The parts that exist, joined --- a key with no default contributes
        nothing rather than the word "required", for the reason on
        :func:`default_text`.
        """
        written = f"{value}{nullable_text(annotation, field, value)}"
        default = default_text(field)
        if default:
            written = f"{written}, {default}"
        return written + self.choices_text(annotation)

    def description(self, cls, name):
        """Return the description of a key, as lines, with any note it earns."""
        lines, owner = self.describe(cls, name)

        if owner is not None and not owner.__module__.startswith(HOME):
            lines = quoted(lines)

        moved = self.moved_default(cls, name, owner)

        if moved is not None:
            package = owner.__module__.split(".")[0]
            note = (
                f"turbigen changes this default; {package}'s own is "
                f"``{yaml_value(moved)}``."
            )
            lines = lines + ["", note]

        return lines or ["*Undocumented.*"]

    #
    # WRITING
    #

    def line(self, text):
        self.lines.append(text)

    def row(self, key, value, description):
        """Write one row of a list-table.

        `description` may be several lines and may hold blank ones, so that a
        docstring written as two paragraphs stays two paragraphs. Continuation
        lines are indented to the cell rather than joined onto the first, which
        is what keeps a bullet list or a literal block inside a docstring
        rendering as itself.
        """
        if isinstance(description, str):
            description = [description]

        self.line(f"   * - {key}")
        self.line(f"     - {value}")
        self.line(f"     - {description[0]}")
        for line in description[1:]:
            self.line(f"       {line}" if line else "")

    def class_docstring(self, cls):
        """Write the docstring of `cls`."""
        text = inspect.getdoc(cls)
        if not text:
            return
        self.lines.extend(text.splitlines())
        self.line("")


def quoted(lines):
    """Return `lines` as a description borrowed from another project.

    Its first paragraph, with any citation left as the bare key it names --- see
    :data:`HOME` for why both.
    """
    paragraph = []
    for line in lines:
        if not line.strip():
            break
        paragraph.append(CITATION.sub(r"``\1``", line))
    return paragraph


def field_named(cls, name):
    """Return the dataclass field `name` of `cls`, or None."""
    if not dataclasses.is_dataclass(cls):
        return None
    for field in dataclasses.fields(cls):
        if field.name == name:
            return field
    return None


class ConfigDirective(SphinxDirective):
    """Write the reference for a config key and everything under it.

    With no argument the whole file is documented, from the top-level keys
    down. With one --- the name of a class in :mod:`turbigen.config` --- only
    that key is, which is what a split into a page per family would use.
    """

    has_content = False
    optional_arguments = 1
    final_argument_whitespace = False

    def run(self):
        schema = Schema()

        if self.arguments:
            root = getattr(turbigen_config, self.arguments[0])
            schema.emit(root, f"{self.arguments[0]}:", self.arguments[0], 0)
        else:
            # The top-level keys are the file itself rather than a key in it,
            # so they are written without a section of their own and their
            # paths are bare -- `fluid`, not `config.fluid`, which is how the
            # page and the command line both spell them.
            schema.keys(turbigen_config.Config, "", -1)

        for path in schema.dependencies:
            self.env.note_dependency(path)

        node = nodes.section()
        node.document = self.state.document
        content = StringList(schema.lines, source="turbigen-config")
        nested_parse_with_titles(self.state, content, node)

        return node.children


def setup(app):
    app.add_directive("turbigen-config", ConfigDirective)
    return {"version": "1.0", "parallel_read_safe": True}

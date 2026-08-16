"""The one serialisation protocol used by every part of a config file.

A :class:`Node` is a frozen dataclass that knows how to build itself from a
plain dict and turn back into one. Everything in a config file is a Node, so
loading a file is one recursive call and writing it back is its mirror image.

Subclassing does three things automatically:

* reserved names (``type``, ``n_row``) become class variables rather than
  fields, so they never reach ``__init__`` or the config file;
* the class becomes a frozen, keyword-only dataclass, so no decorator is
  needed;
* declaring a ``type`` registers the class, so no registration call is needed.

Fields are keyword-only because a config file is a mapping, so nothing is ever
built positionally. That also means a family base may carry a defaulted field
without preventing its members from declaring required ones: with positional
fields, ``Mesher.yplus = 30.0`` would make every later ``forward`` parameter
default-only.

A *family* is a direct subclass of Node with alternatives of its own --- a
:class:`~turbigen.fluid.Fluid` or a
:class:`~turbigen.design.MeanLineDesign`. Members of a family declare a
``type`` and are chosen by a matching ``type:`` key in the file. The dispatch
lives here, once, instead of being written again for every family.

Writing a new alternative is therefore one class::

    class PerfectFluid(Fluid):
        type: ClassVar[str] = "perfect"

        cp: float
        gamma: float
        mu: float
        Pr: float = 0.7

Fields holding a Node are followed when loading and dumping, as are fields
holding a sequence of them, written ``tuple[Post, ...]``. An optional stage is
written ``Mesher | None``, and ``tuple[float, ...]`` gets its elements
converted like a scalar field.

A field annotation is used, not just documented: a value out of a config file
is converted to the annotated type and rejected if it cannot be. So ``cp`` is a
float even when the file quoted it, and ``mu: null`` is an error rather than a
``None`` that surfaces as an ``AttributeError`` several stages later.
"""

import dataclasses
import types
import typing
from typing import (
    ClassVar,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
)

RESERVED = ("type", "n_row")
"""Names that are always class-level, never dataclass fields."""

_REGISTRY: dict[type, dict[str, type]] = {}
"""{family root: {type name: class}}."""


def _family_root(cls):
    """Return the direct subclass of Node that `cls` belongs to."""
    for base in reversed(cls.__mro__):
        if base is not Node and Node in base.__bases__:
            return base
    return None


def _as_classvar(annotation):
    """Wrap `annotation` in ClassVar unless it already is one."""
    if isinstance(annotation, str):
        # A stringised annotation, from `from __future__ import annotations`.
        return annotation if "ClassVar" in annotation else f"ClassVar[{annotation}]"
    if get_origin(annotation) is ClassVar:
        return annotation
    return ClassVar[annotation]


def _to_config(value):
    """Convert a field value into something a config file can hold."""
    if isinstance(value, Node):
        return value.to_dict()
    if isinstance(value, (tuple, list)):
        # YAML has no tuple; store as a list and restore on the way back in.
        # Recurse, so that a sequence of Nodes dumps its members properly.
        return [_to_config(item) for item in value]
    return value


def _type_name(annotation):
    """Return a readable name for an annotation, for an error message."""
    return getattr(annotation, "__name__", None) or str(annotation)


def _strip_none(annotation):
    """Split ``X | None`` into ``(X, True)``; anything else is ``(it, False)``.

    A union is not a class, so it fails every ``issubclass`` test below: left
    alone, a field annotated ``Mesher | None`` would take a raw dict straight
    out of the file and hand it on as though it were a Node. Stripping the
    ``None`` first is what makes the honest annotation for an optional stage
    behave exactly like the bare class.
    """
    if get_origin(annotation) not in (typing.Union, types.UnionType):
        return annotation, False

    args = get_args(annotation)
    rest = tuple(arg for arg in args if arg is not type(None))
    allows_none = len(rest) < len(args)

    # Only a two-member union collapses to a single annotation. A wider one is
    # left alone, so its values pass through unconverted rather than being
    # forced into whichever member happened to come first.
    return (rest[0] if len(rest) == 1 else annotation), allows_none


def _sequence_member(annotation):
    """Return the element annotation of ``tuple[X, ...]`` or ``list[X]``.

    ``tuple[X, Y]`` is recognised only when every member is the same, since
    there is one annotation to convert each element against. A bare ``tuple``
    has no element type and so returns None, leaving its contents untouched.
    """
    if get_origin(annotation) not in (tuple, list):
        return None

    args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
    if args and all(arg is args[0] for arg in args):
        return args[0]
    return None


SCALARS = (bool, int, float, str)
"""Scalar annotations a value out of a config file is converted against."""


def _to_scalar(annotation, value, where):
    """Convert `value` to the scalar type `annotation` names.

    Only the types a YAML file yields are handled, and the caller has already
    checked that `annotation` is one of them.
    """
    if annotation is bool:
        # Checked first, and excluded everywhere below, because bool is a
        # subclass of int -- so `gamma: true` would otherwise become 1.0.
        if isinstance(value, bool):
            return value
    elif annotation is int:
        if isinstance(value, bool):
            pass
        elif isinstance(value, int):
            return value
        elif isinstance(value, float) and value.is_integer():
            return int(value)
        elif isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
    elif annotation is float:
        if isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            # A quoted number in a config file is a slip, not a different
            # intention, so accept it rather than failing on a rendering detail.
            try:
                return float(value)
            except ValueError:
                pass
    elif annotation is str:
        if isinstance(value, str):
            return value

    raise ValueError(
        f"{where} must be {_type_name(annotation)}, got {value!r} "
        f"({type(value).__name__})."
    )


def _from_config(annotation, value, where):
    """Convert a value out of a config file into a field value."""
    annotation, allows_none = _strip_none(annotation)

    if value is None:
        if not allows_none:
            # Only a field whose type admits None may be null. Omitting the key
            # is how a default is taken; writing null is how an optional stage
            # is turned off, and it is an error anywhere else. Catching it here
            # is the difference between naming the field and an AttributeError
            # from inside forward().
            raise ValueError(
                f"{where} is null, but its type {_type_name(annotation)} does "
                f"not allow it. Omit the key to take the default."
            )
        # An optional stage that was not configured. Written out as null and
        # read back as None, so the round trip holds for a config that omits
        # part of the pipeline.
        return None

    if isinstance(annotation, type) and issubclass(annotation, Node):
        return annotation.from_dict(value)

    member = _sequence_member(annotation)
    if member is not None:
        if isinstance(value, str) or not isinstance(value, (list, tuple)):
            raise ValueError(
                f"{where} must be a sequence of {_type_name(member)}, got "
                f"{value!r} ({type(value).__name__})."
            )
        # Sequences are held as tuples so that a Node stays hashable and
        # compares equal whether it was built from a file or by hand.
        return tuple(
            _from_config(member, item, f"{where}[{i}]") for i, item in enumerate(value)
        )

    if annotation in SCALARS:
        # Checked before the sequence fallback below, so that a list handed to
        # a scalar field is an error rather than silently becoming a tuple.
        return _to_scalar(annotation, value, where)

    if isinstance(value, list):
        # An annotation this module does not understand, such as a bare `tuple`.
        # Sequences are held as tuples so that a Node stays hashable and
        # compares equal whether it was built from a file or by hand.
        return tuple(value)
    return value


@dataclass_transform(frozen_default=True, kw_only_default=True)
class Node:
    """Base for every mapping that can appear in a config file."""

    type: ClassVar[str | None] = None
    """Name this class is selected by, for members of a family."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Reserved names describe the class, not an instance, so keep them out
        # of the dataclass fields. Accept the natural `n_row: int = 2` as well
        # as the explicit `n_row: ClassVar[int] = 2`.
        annotations = cls.__dict__.get("__annotations__")
        if annotations:
            for name in RESERVED:
                if name in annotations:
                    annotations[name] = _as_classvar(annotations[name])

        # Keyword-only, because everything is built from a mapping. It also
        # frees a family base to carry a defaulted field -- with positional
        # fields, Mesher.yplus would forbid a required field on every mesher
        # written afterwards.
        dataclasses.dataclass(frozen=True, kw_only=True)(cls)

        # Registering here rather than in a decorator means a plugin author
        # writes one class and nothing else.
        declared = cls.__dict__.get("type")
        if declared:
            root = _family_root(cls)
            if root is None:
                raise TypeError(
                    f"{cls.__name__} declares type={declared!r} but does not "
                    f"belong to a family; it must subclass a direct subclass "
                    f"of Node."
                )
            existing = _REGISTRY.setdefault(root, {}).get(declared)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"{root.__name__} type {declared!r} is already registered "
                    f"by {existing.__name__}."
                )
            _REGISTRY[root][declared] = cls

    #
    # LOADING AND DUMPING
    #

    @classmethod
    def options(cls):
        """Return the registered alternatives for this family, by name."""
        return dict(_REGISTRY.get(cls, {}))

    @classmethod
    def from_dict(cls, data):
        """Build an instance of `cls`, or of one of its family, from a dict."""
        if not isinstance(data, dict):
            raise TypeError(
                f"{cls.__name__} must be given a mapping, got {type(data).__name__}."
            )

        data = dict(data)

        # If this is a family root, hand off to the member named by `type`.
        family = _REGISTRY.get(cls)
        if family:
            name = data.pop("type", None)
            if name is None:
                raise ValueError(
                    f"{cls.__name__} needs a 'type' key, one of {sorted(family)}."
                )
            if name not in family:
                raise ValueError(
                    f"Unknown {cls.__name__} type {name!r}. "
                    f"Available types: {sorted(family)}."
                )
            return family[name].from_dict(data)

        # Otherwise build this class directly.
        if "type" in data:
            declared = data.pop("type")
            if declared != cls.type:
                raise ValueError(
                    f"{cls.__name__} has type {cls.type!r}, got {declared!r}."
                )

        hints = get_type_hints(cls)
        fields = dataclasses.fields(cls)

        kwargs = {}
        for field in fields:
            if field.name in data:
                kwargs[field.name] = _from_config(
                    hints.get(field.name),
                    data.pop(field.name),
                    f"{cls.__name__}.{field.name}",
                )

        if data:
            raise ValueError(
                f"Unknown key(s) for {cls.__name__}: {sorted(data)}. "
                f"Valid keys: {sorted(f.name for f in fields)}."
            )

        # A missing required field raises TypeError from the dataclass, which
        # already names it. Deliberately not caught: swallowing TypeError here
        # would hide genuine errors raised inside a subclass constructor.
        return cls(**kwargs)

    def to_dict(self):
        """Return a dict holding everything needed to rebuild this instance."""
        data = {}
        if self.type is not None:
            data["type"] = self.type
        for field in dataclasses.fields(self):
            data[field.name] = _to_config(getattr(self, field.name))
        return data


#
# PATHS THROUGH A TREE
#


def flatten(node):
    """Return every leaf of `node`, keyed by the path that reaches it.

    Mappings are joined with a dot and sequences indexed with brackets, so a
    recamber angle is ``blades[0].sections[1].dchi_TE`` and a loss coefficient
    ``mean_line.Ys[0]``.

    This is the only definition of how a path is spelled. Anything naming one
    --- an :class:`~turbigen.iterate.Iterator` declaring which leaves it owns,
    a predictor deciding which leaves are design variables --- is written
    against this function rather than against its own idea of the convention,
    so the two cannot drift apart in spelling while agreeing in content.

    Every leaf is returned, not only the numeric ones. What counts as a design
    variable is the caller's business; what the tree contains is this
    function's.
    """
    leaves = {}
    _leaves(node.to_dict(), "", leaves)
    return leaves


def _leaves(value, prefix, leaves):
    """Collect the leaves of `value` into `leaves`, under `prefix`."""
    if isinstance(value, dict):
        for key, item in value.items():
            _leaves(item, f"{prefix}.{key}" if prefix else str(key), leaves)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _leaves(item, f"{prefix}[{index}]", leaves)
    else:
        leaves[prefix] = value


def parse_path(path):
    """Return the segments of `path`, the inverse of how :func:`flatten` spells one.

    Mapping keys come back as strings and sequence indices as integers, so
    ``mean_line.Ys[0]`` gives ``("mean_line", "Ys", 0)`` and is ready to walk a
    raw config dict with.

    Both spellings are accepted. ``Ys[0]`` is what `flatten` writes; ``Ys.0``
    is what ``--set`` has always taken, and a dotted integer is unambiguous
    because a mapping key that is a bare number cannot appear in a config ---
    every one of them is a field name. Accepting both is what lets a design
    variable be named identically in ``batch:``, in ``database:`` and on the
    command line, rather than each growing its own translation.
    """
    segments = []
    for part in str(path).split("."):
        name, _, rest = part.partition("[")
        if name:
            segments.append(int(name) if name.lstrip("-").isdigit() else name)
        elif not rest:
            raise ValueError(f"The path {path!r} has an empty segment.")

        while rest:
            index, bracket, rest = rest.partition("]")
            if not bracket:
                raise ValueError(f"The path {path!r} has an unclosed bracket.")
            try:
                segments.append(int(index))
            except ValueError:
                raise ValueError(
                    f"The path {path!r} indexes with {index!r}, which is not a "
                    "whole number."
                ) from None
            # Whatever follows a closing bracket is another index, so anything
            # else -- `Ys[0]x` -- is a typo rather than a name.
            if rest and not rest.startswith("["):
                raise ValueError(
                    f"The path {path!r} has {rest!r} after a closing bracket."
                )
            rest = rest[1:] if rest else ""

    if not segments:
        raise ValueError("A path must name at least one segment.")

    return tuple(segments)


def set_by_path(data, path, value):
    """Set `value` at `path` in the raw dict `data`, in place.

    Works on the dict a config is built from rather than on a `Node`, because
    both callers --- a ``--set`` override and a batch design variable --- are
    changing a value on the way *in*, before anything is validated. That is
    what makes a mistyped key an error from the strict unknown-key check
    instead of a silent no-op.

    Intermediate containers are created as the next segment implies, so a path
    may reach into a section the file leaves out entirely.
    """
    segments = parse_path(path)

    target = data
    for segment, following in zip(segments[:-1], segments[1:]):
        child = _child(target, segment)
        if not isinstance(child, (dict, list)):
            child = [] if isinstance(following, int) else {}
            _set_child(target, segment, child)
        target = child

    _set_child(target, segments[-1], value)


def _child(data, segment):
    """Return data[segment] if it is there, else None, whatever the types."""
    if isinstance(segment, int):
        if isinstance(data, list) and segment < len(data):
            return data[segment]
        return None
    return data.get(segment) if isinstance(data, dict) else None


def _set_child(data, segment, value):
    """Assign data[segment] = value, growing a list with None as needed."""
    if isinstance(segment, int):
        if not isinstance(data, list):
            raise ValueError(f"Cannot index a {type(data).__name__} with {segment}.")
        data.extend([None] * (segment + 1 - len(data)))
        data[segment] = value
    else:
        if not isinstance(data, dict):
            raise ValueError(
                f"Cannot set the key {segment!r} on a {type(data).__name__}."
            )
        data[segment] = value

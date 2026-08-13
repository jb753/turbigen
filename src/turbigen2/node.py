"""The one serialisation protocol used by every part of a config file.

A :class:`Node` is a frozen dataclass that knows how to build itself from a
plain dict and turn back into one. Everything in a config file is a Node, so
loading a file is one recursive call and writing it back is its mirror image.

Subclassing does three things automatically:

* reserved names (``type``, ``n_row``) become class variables rather than
  fields, so they never reach ``__init__`` or the config file;
* the class becomes a frozen dataclass, so no decorator is needed;
* declaring a ``type`` registers the class, so no registration call is needed.

A *family* is a direct subclass of Node with alternatives of its own --- a
:class:`~turbigen2.fluid.Fluid` or a
:class:`~turbigen2.design.MeanLineDesign`. Members of a family declare a
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
holding a sequence of them, written ``tuple[Post, ...]``.
"""

import dataclasses
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


def _node_member(annotation):
    """Return the Node subclass a sequence annotation holds, if it holds one.

    Recognises ``tuple[Post, ...]`` and ``list[Post]``, which is how a config
    holds a list of alternatives such as the post-processors.
    """
    if get_origin(annotation) not in (tuple, list):
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and issubclass(arg, Node):
            return arg
    return None


def _from_config(annotation, value):
    """Convert a value out of a config file into a field value."""
    if value is None:
        # An optional stage that was not configured. Written out as null and
        # read back as None, so the round trip holds for a config that omits
        # part of the pipeline.
        return None

    if isinstance(annotation, type) and issubclass(annotation, Node):
        return annotation.from_dict(value)

    member = _node_member(annotation)
    if member is not None:
        return tuple(member.from_dict(item) for item in value)

    if isinstance(value, list):
        # Sequences are held as tuples so that a Node stays hashable and
        # compares equal whether it was built from a file or by hand.
        return tuple(value)
    return value


@dataclass_transform(frozen_default=True)
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

        dataclasses.dataclass(frozen=True)(cls)

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
                    hints.get(field.name), data.pop(field.name)
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

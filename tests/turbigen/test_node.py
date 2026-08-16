"""Tests for the Node serialisation protocol.

The protocol replaces fourteen hand-written construction idioms and thirteen
serialisation special cases in the old config with one pair of methods. The
test that matters most is therefore the round trip: because nodes are frozen
dataclasses they compare by value, so a single assertion covers every node type
at once. That is the guarantee the old code had no way to make.
"""

import dataclasses
from typing import ClassVar

import pytest

from turbigen import Config, Fluid, MeanLineDesign, Mesher, PerfectFluid, node
from turbigen.node import Node

CASE = {
    "fluid": {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5},
    "mean_line": {
        "type": "axial_turbine",
        "psi": 1.6,
        "phi2": 0.8,
        "Ma2": 0.9,
        "fac_Ma3_rel": 0.8,
        "mdot": 10.0,
        "Ys": [0.05, 0.05],
        "r_rms": 0.3,
    },
}


#
# THE PROTOCOL
#


def test_subclass_is_a_frozen_dataclass_without_a_decorator():
    """Node applies @dataclass(frozen=True) itself, so a plugin is one class."""
    assert dataclasses.is_dataclass(PerfectFluid)

    fluid = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fluid.cp = 999.0


def test_round_trip_is_exact():
    """from_dict(to_dict(x)) == x, for the whole tree at once."""
    config = Config.from_dict(CASE)

    assert Config.from_dict(config.to_dict()) == config


def test_dump_includes_resolved_defaults():
    """A written config records every value the design used, not just typed ones.

    Without this an archived config stops reproducing its machine as soon as a
    default changes.
    """
    dumped = Config.from_dict(CASE).to_dict()

    assert dumped["fluid"]["Pr"] == 0.7
    assert dumped["mean_line"]["Po1"] == 1e5
    assert dumped["mean_line"]["To1"] == 300.0
    assert dumped["mean_line"]["zeta"] == [1.0, 1.0]


def test_sequences_round_trip_as_tuples():
    """YAML lists become tuples, so a node stays hashable and compares equal."""
    config = Config.from_dict(CASE)

    assert config.mean_line.Ys == (0.05, 0.05)
    assert config.to_dict()["mean_line"]["Ys"] == [0.05, 0.05]
    assert hash(config) is not None


def test_nested_nodes_are_built_and_dumped_recursively():
    config = Config.from_dict(CASE)

    assert isinstance(config.fluid, PerfectFluid)
    assert isinstance(config.mean_line, MeanLineDesign)
    assert config.to_dict()["fluid"]["type"] == "perfect"


#
# OPTIONAL FIELDS
#
# An annotation like `Mesher | None` is a union, not a class, so it fails every
# issubclass test in the protocol. Left unhandled it does not raise: the raw
# dict out of the file lands in the field, and the failure surfaces stages later
# as a dict with no methods. So the honest annotation for an optional stage has
# to be tested, not just the bare one.
#


class WithOptional(Node):
    fluid: Fluid | None = None
    fluids: tuple[Fluid, ...] = ()


def test_optional_node_field_is_built():
    node = WithOptional.from_dict({"fluid": CASE["fluid"]})

    assert isinstance(node.fluid, PerfectFluid)


def test_optional_node_field_accepts_null():
    """Writing null is how an omitted stage round-trips through a file."""
    assert WithOptional.from_dict({"fluid": None}).fluid is None
    assert WithOptional.from_dict(WithOptional().to_dict()) == WithOptional()


def test_null_is_refused_where_the_type_does_not_allow_it():
    """`mu: null` names the field, rather than failing inside forward()."""
    bad = dict(CASE["fluid"], mu=None)

    with pytest.raises(ValueError, match="PerfectFluid.mu is null"):
        Fluid.from_dict(bad)


def test_config_optional_stages_round_trip():
    """The stages Config declares optional really are optional."""
    config = Config.from_dict(CASE)

    assert config.annulus is None
    assert config.mesh is None
    assert Config.from_dict(config.to_dict()) == config


#
# VALUE CONVERSION
#
# The annotations are used, not merely documented. Without this a field holds
# whatever the file happened to contain: `cp: '1005'` gives a string, and
# `cp * 2` silently returns '10051005'.
#


class Scalars(Node):
    x: float = 0.0
    n: int = 0
    flag: bool = False
    name: str = ""
    xs: tuple[float, ...] = ()


@pytest.mark.parametrize(
    "given,expect",
    [
        ({"x": 1}, 1.0),
        ({"x": "1.5"}, 1.5),
        ({"x": 1.5}, 1.5),
    ],
)
def test_a_float_field_holds_a_float(given, expect):
    value = Scalars.from_dict(given).x

    assert isinstance(value, float)
    assert value == expect


def test_an_int_field_accepts_a_whole_float():
    assert Scalars.from_dict({"n": 4.0}).n == 4


def test_sequence_elements_are_converted():
    assert Scalars.from_dict({"xs": [1, "2", 3.0]}).xs == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    "given,match",
    [
        ({"x": "lots"}, "Scalars.x must be float"),
        # bool is a subclass of int, so `flag: true` for a number has to be
        # excluded deliberately rather than by isinstance.
        ({"x": True}, "Scalars.x must be float"),
        ({"n": 1.5}, "Scalars.n must be int"),
        ({"flag": 1}, "Scalars.flag must be bool"),
        ({"name": 3}, "Scalars.name must be str"),
        # A list for a scalar field must not fall through the tuple conversion.
        ({"x": [1.0, 2.0]}, "Scalars.x must be float"),
        ({"xs": 1.0}, r"Scalars.xs must be a sequence"),
        ({"xs": [1.0, "x"]}, r"Scalars.xs\[1\] must be float"),
    ],
)
def test_a_value_of_the_wrong_type_names_the_field(given, match):
    with pytest.raises(ValueError, match=match):
        Scalars.from_dict(given)


#
# KEYWORD-ONLY FIELDS
#


def test_a_family_member_may_require_a_field_after_an_inherited_default():
    """Mesher.yplus has a default; a mesher must still be able to require one.

    With positional fields this is a TypeError at class definition, so the first
    defaulted field on any family base would forbid required fields in every
    member written afterwards -- which is every mesher after the first.
    """

    class Layered(Mesher):
        type: ClassVar[str] = "layered"
        n_layer: int

    built = Layered.from_dict({"type": "layered", "n_layer": 4})
    assert (built.n_layer, built.yplus) == (4, 30.0)

    with pytest.raises(TypeError, match="n_layer"):
        Layered()


#
# RESERVED NAMES
#


class Family(Node):
    pass


class PlainSpelling(Family):
    type = "plain"
    n_row: int = 2
    x: float = 1.0


class ClassVarSpelling(Family):
    type: ClassVar[str] = "classvar"
    n_row: ClassVar[int] = 3
    x: float = 1.0


@pytest.mark.parametrize("cls", [PlainSpelling, ClassVarSpelling])
def test_reserved_names_are_not_fields(cls):
    """`type` and `n_row` describe the class, however they are spelled."""
    assert [f.name for f in dataclasses.fields(cls)] == ["x"]


@pytest.mark.parametrize("cls", [PlainSpelling, ClassVarSpelling])
def test_reserved_names_are_rejected_by_init(cls):
    with pytest.raises(TypeError, match="n_row"):
        cls(x=1.0, n_row=5)


def test_n_row_does_not_reach_the_config_file():
    """The row count comes from the design class, so a file cannot contradict it."""
    assert "n_row" not in PlainSpelling(x=1.0).to_dict()
    assert PlainSpelling.n_row == 2


def test_type_does_reach_the_config_file():
    """`type` is the discriminator, so it must be written back."""
    assert PlainSpelling(x=1.0).to_dict()["type"] == "plain"


#
# DISPATCH AND ERRORS
#


def test_unknown_type_lists_the_available_ones():
    with pytest.raises(ValueError, match="Unknown Fluid type 'nonsense'") as excinfo:
        Fluid.from_dict({"type": "nonsense"})

    assert "perfect" in str(excinfo.value)


def test_missing_type_is_reported():
    with pytest.raises(ValueError, match="needs a 'type' key"):
        Fluid.from_dict({"cp": 1005.0})


def test_unknown_key_is_rejected():
    """A typo in a config file fails loudly rather than being ignored."""
    bad = dict(CASE, fluid=dict(CASE["fluid"], gama=1.4))

    with pytest.raises(ValueError, match="Unknown key.*gama"):
        Config.from_dict(bad)


def test_missing_required_field_names_it():
    with pytest.raises(TypeError, match="mu"):
        Fluid.from_dict({"type": "perfect", "cp": 1005.0, "gamma": 1.4})


def test_wrong_type_for_a_named_subclass_is_rejected():
    with pytest.raises(ValueError, match="has type 'perfect'"):
        PerfectFluid.from_dict(
            {"type": "something_else", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}
        )


def test_non_mapping_is_rejected():
    with pytest.raises(TypeError, match="must be given a mapping"):
        Fluid.from_dict([1, 2, 3])


def test_duplicate_type_is_refused():
    with pytest.raises(ValueError, match="already registered"):

        class Clash(Family):
            type: ClassVar[str] = "plain"


#
# REGISTRY
#


def test_options_lists_the_family():
    """The built-ins are registered.

    A subset check, not equality: the registry is deliberately open, so any
    plugin or test that defines a design adds to it.
    """
    assert {"axial_turbine", "turbine_cascade"} <= set(MeanLineDesign.options())
    assert "perfect" in Fluid.options()


def test_family_root_is_the_direct_subclass_of_node():
    """Alternatives register under their own family, not globally."""
    assert "plain" in Family.options()
    assert "plain" not in Fluid.options()
    assert "plain" not in MeanLineDesign.options()
    assert not set(Fluid.options()) & set(MeanLineDesign.options())


#
# FILE IO
#


def test_config_round_trips_through_a_file(tmp_path):
    config = Config.from_dict(CASE)
    path = tmp_path / "config.yaml"

    config.to_file(path)

    assert Config.from_file(path) == config


#
# PATHS THROUGH A TREE
#
# One spelling, shared by everything that names a leaf: an iterator declaring
# what it owns, and a predictor deciding what is a design variable.
#


def test_flatten_spells_a_nested_leaf():
    leaves = node.flatten(Config.from_dict(CASE))

    assert leaves["fluid.cp"] == 1005.0
    assert leaves["mean_line.psi"] == 1.6


def test_flatten_indexes_a_sequence():
    leaves = node.flatten(Config.from_dict(CASE))

    assert leaves["mean_line.Ys[0]"] == 0.05
    assert leaves["mean_line.Ys[1]"] == 0.05


def test_flatten_keeps_no_branches():
    """Only leaves, so a path never names something with children."""
    leaves = node.flatten(Config.from_dict(CASE))

    assert "mean_line" not in leaves
    assert "mean_line.Ys" not in leaves
    assert not any(isinstance(value, (dict, list)) for value in leaves.values())


def test_flatten_reaches_every_leaf_of_a_file():
    """A round trip through a file changes nothing about what is in the tree."""
    config = Config.from_dict(CASE)

    assert node.flatten(Config.from_dict(config.to_dict())) == node.flatten(config)


def test_parse_path_inverts_flatten():
    """Every path `flatten` writes walks back to the leaf it came from."""
    config = Config.from_dict(CASE)
    data = config.to_dict()

    for path, value in node.flatten(config).items():
        walked = data
        for segment in node.parse_path(path):
            walked = walked[segment]
        assert walked == value


def test_parse_path_takes_both_spellings():
    """Brackets from `flatten`, dotted integers from `--set`."""
    assert node.parse_path("mean_line.Ys[0]") == ("mean_line", "Ys", 0)
    assert node.parse_path("mean_line.Ys.0") == ("mean_line", "Ys", 0)


@pytest.mark.parametrize("path", ["", "a..b", "a[0", "a[x]", "a[0]x", "a[]"])
def test_a_malformed_path_is_refused(path):
    with pytest.raises(ValueError):
        node.parse_path(path)


def test_set_by_path_builds_what_a_path_implies():
    """So a value can be set in a section the file leaves out."""
    data = {}

    node.set_by_path(data, "mean_line.Ys[1]", 0.06)

    assert data == {"mean_line": {"Ys": [None, 0.06]}}

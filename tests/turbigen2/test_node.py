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

from turbigen2 import Config, Fluid, MeanLineDesign, PerfectFluid
from turbigen2.node import Node

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

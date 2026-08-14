"""Tests for serialising a mean line.

A mean line is a result, not a config node, so it has no `type` and never
appears among a config's own keys. It does serialise, because a run's answer
has to be readable back without repeating the CFD that produced it.

Two things make the pair worth testing rather than assuming. Which quantities
constitute a complete mean line is not obvious --- four of the twelve data keys
it inherits are deliberately never set and raise on read. And the state has to
survive being rebuilt against a *different* fluid, which rules out the
conserved variables.

Test cases:
- test_unset_keys_really_are_unset: why STATE is eight and not twelve
- test_round_trip_reproduces_every_property: to_dict then from_dict
- test_round_trip_survives_a_different_datum: the trap that cost 105 K
- test_state_is_dimensional_and_readable: what lands in the file
- test_missing_state_is_reported: a truncated file names what it lacks
- test_odd_station_count_is_rejected: two stations per row, always
"""

import numpy as np
import pytest

from test_blade import build
from turbigen2 import MeanLine, PerfectFluid

FLUID = PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5)


@pytest.fixture(scope="module")
def mean_line():
    return build().design().mean_line


def test_unset_keys_really_are_unset(mean_line):
    """The reason MeanLine.STATE is eight quantities and not twelve.

    A mean line has no axial or tangential coordinate, no turbulent viscosity
    and no wall distance -- the annulus supplies position, and the rest are
    properties of a grid. Reading them raises, so a serialiser that walked the
    data keys would fail rather than write junk.
    """
    for key in ("x", "t", "mu_turb", "wdist"):
        assert key in mean_line._data_keys
        with pytest.raises(ValueError):
            getattr(mean_line.flat, key)

    assert set(MeanLine.STATE).isdisjoint({"x", "t", "mu_turb", "wdist"})


def test_round_trip_reproduces_every_property(mean_line):
    restored = MeanLine.from_dict(mean_line.to_dict(), FLUID.eos())

    assert restored.shape == mean_line.shape
    for name in ("P", "T", "Vx", "Vr", "Vt", "r", "Am", "Omega", "Po", "To", "Ma"):
        np.testing.assert_allclose(
            np.asarray(getattr(restored.flat, name), dtype=float),
            np.asarray(getattr(mean_line.flat, name), dtype=float),
            rtol=1e-6,
            err_msg=f"{name} did not survive the round trip",
        )
    assert restored.eta_tt == pytest.approx(mean_line.eta_tt, rel=1e-6)


def test_round_trip_survives_a_different_datum(mean_line):
    """The state is stored dimensionally, so the datum cannot reinterpret it.

    Conserved energy is measured from where internal energy is zero. Stored as
    conserved and rebuilt against another datum it would come back a hundred
    kelvin out, looking perfectly well formed -- which is what happened in
    `guess.py` before it transferred P, T and V instead.
    """
    shifted = FLUID.eos().change_datum(P_dtm=3.0e5, T_dtm=800.0)
    assert shifted.T_dtm != FLUID.eos().T_dtm

    restored = MeanLine.from_dict(mean_line.to_dict(), shifted)

    np.testing.assert_allclose(
        np.asarray(restored.flat.T, dtype=float),
        np.asarray(mean_line.flat.T, dtype=float),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(restored.flat.P, dtype=float),
        np.asarray(mean_line.flat.P, dtype=float),
        rtol=1e-6,
    )


def test_state_is_dimensional_and_readable(mean_line):
    """What lands in the file is plain numbers a person can check."""
    data = mean_line.to_dict()

    assert set(data) == set(MeanLine.STATE)
    assert all(isinstance(v, list) for v in data.values())
    assert all(len(v) == 2 * mean_line.n_row for v in data.values())

    # Pascals and kelvin, not non-dimensional ratios.
    assert 1e4 < data["P"][0] < 1e6
    assert 100.0 < data["T"][0] < 2000.0


def test_missing_state_is_reported(mean_line):
    data = mean_line.to_dict()
    del data["Omega"]

    with pytest.raises(ValueError, match="missing.*Omega"):
        MeanLine.from_dict(data, FLUID.eos())


def test_odd_station_count_is_rejected():
    """A mean line has an inlet and an outlet for every row."""
    data = {key: [1.0, 2.0, 3.0] for key in MeanLine.STATE}

    with pytest.raises(ValueError, match="even number"):
        MeanLine.from_dict(data, FLUID.eos())

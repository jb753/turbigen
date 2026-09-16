"""Tests for building a POD basis for modal inlet profiles.

No CFD: the profiles are synthetic combinations of three spanwise shapes, so
the basis they give is known to span them exactly, and what is under test is
that the file keeps that promise and the contract `bconds.PodModes` reads.

Test cases:
- test_a_basis_is_read_back_as_written: the file contract, closed at the walls
- test_the_modes_span_the_profiles_they_came_from: exactly, level aside
- test_the_modes_carry_no_level: a profile redistributes
- test_the_wall_stations_are_not_decomposed: trimmed before, not after
- test_rebuilding_gives_the_same_file: the hash is a property of the runs
- test_more_modes_than_profiles_are_refused: an SVD cannot invent them
"""

import numpy as np
import pytest

from turbigen import bconds, pod

N_SPAN = 60
N_RUN = 12
N_MODE = 3


def stations():
    """Cut stations clustered towards both walls, strictly inside the span."""
    return 0.5 * (1.0 - np.cos(np.pi * (np.arange(N_SPAN) + 0.5) / N_SPAN))


def shapes(spf):
    """Three spanwise shapes: a bow, a hub layer and a casing layer."""
    return np.array([np.cos(np.pi * spf), np.exp(-spf / 0.05), np.exp(-(1 - spf) / 0.05)])


def profiles(seed=0):
    """Synthetic exit profiles, each a combination of the three shapes."""
    rng = np.random.default_rng(seed)
    base = shapes(stations())
    return {
        "DPo": rng.normal(size=(N_RUN, 3)) @ base,
        "DTo": rng.normal(size=(N_RUN, 3)) @ base,
        "DAlpha": 5.0 * rng.normal(size=(N_RUN, 3)) @ base,
    }


def basis_file(directory, n_mode=N_MODE):
    """Write a basis built from `profiles` under `directory`; return path, hash."""
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / "basis.npz").resolve()
    sha = pod.write_basis(path, pod.build_basis(stations(), profiles(), n_mode))
    return str(path), sha


def test_a_basis_is_read_back_as_written(tmp_path):
    path, sha = basis_file(tmp_path)

    modes = bconds.PodModes(path, sha)

    assert modes.spf[0] == 0.0 and modes.spf[-1] == 1.0
    assert modes.table["DPo"].shape == (N_MODE, N_SPAN - 2 + 2)
    assert modes.identity == sha


def test_the_modes_span_the_profiles_they_came_from(tmp_path):
    modes = bconds.PodModes(*basis_file(tmp_path))
    inner = stations()[1:-1]

    for name, values in profiles().items():
        for profile in values[:, 1:-1]:
            coefficients = modes.fit(name, inner, profile, N_MODE)
            shape = np.asarray(coefficients) @ modes.modes(name, inner, N_MODE)
            # Equal up to the level, which the fit drops.
            residual = profile - shape
            assert np.ptp(residual) < 1e-8 * np.ptp(profile)


def test_the_modes_carry_no_level(tmp_path):
    modes = bconds.PodModes(*basis_file(tmp_path))
    weight = bconds.face_weights(modes.spf)

    for table in modes.table.values():
        assert np.abs(table @ weight).max() < 1e-12


def test_the_wall_stations_are_not_decomposed():
    spiked = profiles()
    for values in spiked.values():
        values[:, [0, -1]] = 1e3

    clean = pod.build_basis(stations(), profiles(), N_MODE)
    dirty = pod.build_basis(stations(), spiked, N_MODE)

    assert dirty["DPo"] == pytest.approx(clean["DPo"])


def test_rebuilding_gives_the_same_file(tmp_path):
    _, first = basis_file(tmp_path / "a")
    _, second = basis_file(tmp_path / "b")

    assert first == second


def test_more_modes_than_profiles_are_refused():
    with pytest.raises(ValueError, match="at most"):
        pod.build_basis(stations(), profiles(), N_RUN + 1)

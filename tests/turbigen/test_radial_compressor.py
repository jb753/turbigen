"""The radial compressor mean-line design."""

import dataclasses

import numpy as np
import pytest

from turbigen.config import Config
from turbigen.design import DesignError, _check_round_trip

FLUID = {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5, "Pr": 0.72}

DATUM = {
    "PR_tt": 2.0,
    "phi1": 0.6,
    "Ma1_rel": 0.6,
    "htr1": 0.5,
    "DH": 1.0,
    "Alpha2_rel": -60.0,
    "mdot": 5.0,
    "Ys": 0.07,
}


def build(**kwargs):
    return Config.from_dict(
        {"fluid": FLUID, "mean_line": {"type": "radial_compressor", **DATUM, **kwargs}}
    )


@pytest.fixture(scope="module")
def datum():
    config = build()
    return config, config.design().mean_line


def test_round_trips(datum):
    config, ml = datum
    _check_round_trip(config.mean_line, ml)
    inverted = config.mean_line.backward(ml)
    for field in dataclasses.fields(config.mean_line):
        np.testing.assert_allclose(
            np.asarray(inverted[field.name], dtype=float),
            getattr(config.mean_line, field.name),
            rtol=1e-3,
            err_msg=field.name,
        )


def test_conserves_mass(datum):
    _, ml = datum
    np.testing.assert_allclose(ml.flat.mdot, DATUM["mdot"], rtol=1e-4)


def test_inlet_is_axial_and_exit_radial(datum):
    _, ml = datum
    np.testing.assert_allclose(ml.Beta.ravel(), [0.0, 90.0], atol=1e-6)
    assert float(np.squeeze(ml.inlet.Alpha)) == pytest.approx(0.0, abs=1e-6)


def test_compresses(datum):
    _, ml = datum
    assert ml.outlet.Po / ml.inlet.Po > 1.0
    assert 0.0 < ml.eta_tt < 1.0


def test_euler_work(datum):
    """With no inlet swirl, the work is the exit blade speed times swirl."""
    _, ml = datum
    work = float(np.squeeze(ml.outlet.ho - ml.inlet.ho))
    euler = float(np.squeeze(ml.outlet.U * ml.outlet.Vt))
    assert work == pytest.approx(euler, rel=1e-4)


def test_matches_the_published_datum():
    """Inlet geometry and shaft speed fixed only by the inlet variables.

    Brind (2024) Fig. 16 lists 15915 rpm, a hub radius of 0.0672 m and a
    casing radius of 0.134 m for the datum at 1 bar, 300 K and 5 kg/s. The
    exit radius depends on the loss as well, so it is checked loosely.
    """
    config = build()
    ml = config.design().mean_line
    out = config.mean_line.backward(ml)
    rpm = float(np.squeeze(out["Omega"])) * 60.0 / 2.0 / np.pi
    assert rpm == pytest.approx(15915.0, rel=2e-3)
    assert float(np.squeeze(out["rh1"])) == pytest.approx(0.0672, rel=5e-3)
    assert float(np.squeeze(out["rc1"])) == pytest.approx(0.134, rel=5e-3)
    assert float(np.squeeze(out["r2"])) == pytest.approx(0.222, rel=2e-2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"PR_tt": 0.9},
        {"phi1": 0.0},
        {"htr1": 1.0},
        {"DH": -1.0},
        {"Alpha2_rel": 90.0},
    ],
)
def test_refuses_unphysical_variables(kwargs):
    with pytest.raises(DesignError):
        build(**kwargs).design()


def test_refuses_an_exit_inside_the_inducer():
    with pytest.raises(DesignError, match="casing radius"):
        build(
            PR_tt=1.5, Ma1_rel=0.9, htr1=0.2, phi1=1.0, DH=1.4, Alpha2_rel=-20.0
        ).design()

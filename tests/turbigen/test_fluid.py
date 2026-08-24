"""Tests for the working fluid nodes (turbigen.fluid).

The Node protocol itself is tested in test_node.py; these are about the
equations of state that use it. A real gas is the interesting case: its
coefficients are the only nested sequence anywhere in a config file, and it is
the only fluid with a domain outside which it does not exist.
"""

import ember.fluid
import pytest

from turbigen import Fluid, PerfectFluid, RealFluid

# An order-2 fit to the van der Waals gas of ember's own test suite, over the
# box below. Written out rather than fitted here so that the test needs neither
# a fit nor a property library, and small enough to read: a real design uses
# order 8 or so and about fifty coefficients.
ALPHA = [
    [1.0246335762792034, 0.004312092679063582, -0.0004123519704467727],
    [0.02564476668302332, 0.004136791105261169, 0.0],
    [0.0018978167928593785, 0.0, 0.0],
]
BETA = [147.62290181807342, 3.9188329391484373, -0.18721253522834344]

# Constant transport surfaces: a single Legendre term, order 0, whose value is
# 1.0 everywhere in normalised coordinates -- so mu and kappa come out equal to
# mu_c and kappa_c at every state, matching what the old mu/Pr fields meant
# before ember fitted transport as surfaces of their own.
DELTA = [[1.0]]
GAMMA = [[1.0]]

CASE = {
    "type": "real",
    "alpha": ALPHA,
    "beta": BETA,
    "delta": DELTA,
    "gamma": GAMMA,
    "rho_lim": [1.0, 50.0],
    "u_lim": [3.0e5, 4.0e5],
    "Rgas": 51.2,
    "mu_c": 1.8e-5,
    "kappa_c": 2.5e-2,
    # The centre of the box above, which is where the fitting script puts it.
    "P_dtm": 334563.125,
    "T_dtm": 250.27320861816406,
}


@pytest.fixture
def real():
    """The fitted gas above, as a config node."""
    return Fluid.from_dict(CASE)


def test_dispatches_from_a_config_file(real):
    """`type: real` selects the class, with no registration call anywhere."""
    assert isinstance(real, RealFluid)
    assert "real" in Fluid.options()


def test_round_trip_is_exact(real):
    """from_dict(to_dict(x)) == x, coefficient surface and all.

    One assertion covers the nested conversion because a node is a frozen
    dataclass and so compares by value.
    """
    assert RealFluid.from_dict(real.to_dict()) == real


def test_coefficients_are_tuples(real):
    """The surface is held as tuples, not the lists that came out of the file.

    A node has to compare equal and hash the same whether it was read from a
    config file or written by hand, which a list would break.
    """
    assert real.alpha == tuple(tuple(row) for row in ALPHA)
    assert real.beta == tuple(BETA)
    assert real.rho_lim == (1.0, 50.0)

    assert hash(real) == hash(RealFluid.from_dict(CASE))


def test_a_ragged_surface_is_refused():
    """Every row of the surface is converted, not just the outer sequence."""
    bad = dict(CASE, alpha=[[1.0, 0.0], "nonsense"])

    with pytest.raises(ValueError, match=r"RealFluid.alpha\[1\]"):
        Fluid.from_dict(bad)


def test_eos_builds_an_ember_real_fluid(real):
    """The node hands its coefficients to ember unchanged."""
    eos = real.eos()

    assert isinstance(eos, ember.fluid.RealFluid)
    assert eos.rho_lim_nd == (1.0, 50.0)

    # Evaluated against ember built directly from the same numbers, so this
    # says the node loses nothing on the way through rather than merely that
    # it returns some fluid.
    direct = ember.fluid.RealFluid(
        alpha=ALPHA,
        beta=BETA,
        delta=DELTA,
        gamma=GAMMA,
        rho_lim=(1.0, 50.0),
        u_lim=(3.0e5, 4.0e5),
        Rgas=51.2,
        mu_c=1.8e-5,
        kappa_c=2.5e-2,
        P_dtm=334563.125,
        T_dtm=250.27320861816406,
    )
    rho, u = 25.5, 3.5e5
    assert float(eos.get_P(rho, u)) == float(direct.get_P(rho, u))
    assert float(eos.get_T(rho, u)) == float(direct.get_T(rho, u))


def test_the_reference_scales_are_not_config(real):
    """A file sets the fit, never the non-dimensionalisation.

    The scales are derived from the design by MeanLine.referenced_fluid and
    replaced before anything reads them, so a value for one in a config file
    could only ever be ignored.
    """
    dumped = real.to_dict()
    for scale in ("rho_ref", "V_ref", "Rgas_ref"):
        assert scale not in dumped

    with pytest.raises(ValueError, match="rho_ref"):
        Fluid.from_dict(dict(CASE, rho_ref=1.2))


def test_the_datum_is_config_unlike_a_perfect_gas(real):
    """A real gas may be told its datum; a perfect gas has nowhere to say it.

    A perfect gas is defined everywhere, so its datum is free and
    MeanLine.referenced_fluid is the only thing that picks one. A fitted
    surface exists only inside its box, so the datum has to land in there and
    the config is allowed a say.
    """
    assert real.to_dict()["P_dtm"] == CASE["P_dtm"]
    assert "P_dtm" not in PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5).to_dict()


def test_the_datum_may_be_omitted():
    """Left out, the datum comes from the middle of the fit box.

    Which is inside the box for any fit, whereas a fixed pressure and
    temperature is only inside some of them.
    """
    without = {k: v for k, v in CASE.items() if k not in ("P_dtm", "T_dtm")}
    fluid = Fluid.from_dict(without)

    assert fluid.P_dtm is None
    assert fluid.to_dict()["P_dtm"] is None

    # The box centre, which for this fit is nowhere near the 1 bar and 300 K
    # that ember used to fall back on. Loose, because the two numbers are not
    # the same measurement: the datum above is read off the fitted surface,
    # while CASE carries what the gas being fitted actually does there, and an
    # order-2 fit separates them by about half a percent. Tightening this
    # would be asserting the quality of the fit, which is ember's business.
    eos = fluid.eos()
    assert float(eos.P_dtm) == pytest.approx(CASE["P_dtm"], rel=1e-2)
    assert float(eos.T_dtm) == pytest.approx(CASE["T_dtm"], rel=1e-2)


def test_a_given_datum_still_wins(real):
    """An explicit datum is passed through rather than replaced by the box."""
    moved = Fluid.from_dict(dict(CASE, T_dtm=260.0))

    assert float(moved.eos().T_dtm) == pytest.approx(260.0)
    assert float(real.eos().T_dtm) == pytest.approx(CASE["T_dtm"], rel=1e-5)


def test_datum_outside_the_fit_box_raises(real):
    """A datum off the box is refused, by ember, naming the box.

    Deliberately not caught or reworded here. turbigen places the datum from
    the mean state of a design, which has no idea where the coefficients were
    fitted, so this is the report that the two do not overlap -- and widening
    the box at fit time is the only fix.
    """
    with pytest.raises(Exception) as excinfo:
        real.eos().change_datum(P_dtm=1e5, T_dtm=2.0e4)

    assert "box" in str(excinfo.value).lower()


def test_perfect_fluid_still_dispatches():
    """The family gained a member without disturbing the one it had."""
    fluid = Fluid.from_dict(
        {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}
    )

    assert isinstance(fluid, PerfectFluid)
    assert isinstance(fluid.eos(), ember.fluid.PerfectFluid)

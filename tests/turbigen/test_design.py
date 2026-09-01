"""Tests for MeanLineDesign, its solver, and the built-in designs.

`backward` is the single definition of what a design variable means, and it
also supplies the residual that `forward` is driven by. So the tests that
matter most are the round trips: build a mean line from design variables, read
them back, and require the same numbers. That is the check the old
free-function architecture could not enforce, and which turbine_cascade's loss
coefficient silently failed by four orders of magnitude.
"""

from typing import ClassVar

import numpy as np
import pytest

import turbigen_ref.meanline_new
import turbigen_ref.plugins
from turbigen import Config, DesignError, MeanLineDesign, PerfectFluid
from turbigen.design import check_round_trip
from turbigen.designs.axial_turbine import AxialTurbine

GAMMA, CP = 1.4, 1005.0
RGAS = CP * (GAMMA - 1.0) / GAMMA


@pytest.fixture
def air():
    return PerfectFluid(cp=CP, gamma=GAMMA, mu=1.8e-5, Pr=0.72).eos()


class Uniform(MeanLineDesign):
    """A one-row design with an analytically known answer.

    Static state is fixed, so the speed of sound is fixed, and the axial
    velocity that achieves a target Mach number is exactly ``Ma * a``.
    """

    type: ClassVar[str] = "_test_uniform"
    n_row: ClassVar[int] = 1

    Ma: float
    P1: float = 1e5
    T1: float = 300.0
    guess: float = 50.0

    def _build(self, ml):
        def build(Vx):
            ml.set_r(0.5)
            ml.set_Am(1.0)
            ml.set_P_T(self.P1, self.T1)
            ml.set_Vx(Vx)
            ml.set_Vr(0.0)
            ml.set_Vt(0.0)

        return build

    def forward(self, fluid):
        ml = self.allocate(fluid)
        self.solve_for(
            ml,
            self._build(ml),
            unknowns={"Vx": self.guess},
            targets={"Ma": self.Ma},
            name="uniform",
        )
        return ml

    def backward(self, ml):
        return {"Ma": ml.outlet.Ma, "P1": ml.inlet.P, "T1": ml.inlet.T}


#
# solve_for
#


def test_solve_for_finds_the_analytic_answer(air):
    ml = Uniform(Ma=0.6).design(air)

    a = np.sqrt(GAMMA * RGAS * 300.0)
    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-4)
    assert float(ml.flat.Vx[0]) == pytest.approx(0.6 * a, rel=1e-3)


@pytest.mark.parametrize("guess", [1.0, 50.0, 300.0, 900.0])
def test_solve_for_is_robust_to_the_initial_guess(air, guess):
    ml = Uniform(Ma=0.6, guess=guess).design(air)

    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-4)


def test_solve_for_leaves_the_mean_line_at_the_solution(air):
    """The mean line must match the answer, not the last trial evaluated.

    The old hand-rolled loop exited with Omega from the new blade speed but
    velocities and areas from the previous iterate, leaving a state that was
    self-inconsistent at the level of the convergence tolerance.
    """
    ml = Uniform(Ma=0.6).design(air)

    a = np.sqrt(GAMMA * RGAS * 300.0)
    np.testing.assert_allclose(ml.flat.Vx, 0.6 * a, rtol=1e-3)


def test_solve_for_rejects_an_underdetermined_system(air):
    class TwoUnknownsOneTarget(Uniform):
        type: ClassVar[str] = "_test_underdetermined"

        def forward(self, fluid):
            ml = self.allocate(fluid)

            def build(Vx, Vt):
                ml.set_r(0.5)
                ml.set_Am(1.0)
                ml.set_P_T(1e5, 300.0)
                ml.set_Vx(Vx)
                ml.set_Vr(0.0)
                ml.set_Vt(Vt)

            self.solve_for(
                ml, build, unknowns={"Vx": 100.0, "Vt": 10.0}, targets={"Ma": self.Ma}
            )
            return ml

    with pytest.raises(DesignError, match="underdetermined"):
        TwoUnknownsOneTarget(Ma=0.6).design(air)


def test_solve_for_reports_an_unreachable_target(air):
    class FixedVx(Uniform):
        type: ClassVar[str] = "_test_unreachable"

        def forward(self, fluid):
            ml = self.allocate(fluid)

            def build(unused):
                ml.set_r(0.5)
                ml.set_Am(1.0)
                ml.set_P_T(1e5, 300.0)
                ml.set_Vx(100.0)  # ignores the unknown entirely
                ml.set_Vr(0.0)
                ml.set_Vt(0.0)

            self.solve_for(ml, build, unknowns={"unused": 1.0}, targets={"Ma": self.Ma})
            return ml

    with pytest.raises(DesignError, match="did not converge") as excinfo:
        FixedVx(Ma=0.9).design(air)

    assert "history" in str(excinfo.value)


def test_solve_for_rejects_an_unknown_target_key(air):
    class BadTarget(Uniform):
        type: ClassVar[str] = "_test_bad_target"

        def forward(self, fluid):
            ml = self.allocate(fluid)
            self.solve_for(
                ml,
                self._build(ml),
                unknowns={"Vx": 100.0},
                targets={"nonexistent": 1.0},
            )
            return ml

    with pytest.raises(DesignError, match="not returned by backward"):
        BadTarget(Ma=0.6).design(air)


def test_solve_for_survives_an_infeasible_trial(air):
    """A trial the design cannot evaluate must not abort the solve.

    A trust-region solver probes unphysical states as a matter of course; those
    should be rejected and the radius shrunk, not raised through.
    """

    class Fragile(Uniform):
        type: ClassVar[str] = "_test_fragile"
        tripped = []  # class level: the design itself is frozen

        def backward(self, ml):
            # Fail once, the first time the solver steps away from the guess.
            # Failing only once leaves the Jacobian around the starting point
            # intact, which is what lets the solver retreat and try again.
            if not self.tripped and float(ml.outlet.Vx) > 150.0:
                self.tripped.append(True)
                raise ValueError("unphysical")
            return super().backward(ml)

    Fragile.tripped.clear()
    # The answer is Vx = 0.6 * a ~ 208, so reaching it must cross 150.
    ml = Fragile(Ma=0.6, guess=100.0).design(air)

    assert Fragile.tripped, "the infeasible branch was never exercised"
    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-3)


def test_design_rejects_a_bad_row_count(air):
    class NoRows(Uniform):
        type: ClassVar[str] = "_test_no_rows"
        n_row: ClassVar[int] = 0

    with pytest.raises(DesignError, match="n_row"):
        NoRows(Ma=0.6).design(air)


#
# BUILT-IN DESIGNS
#

CASES = {
    "turbine_cascade": {
        "span": [0.05, 0.05],
        "Alpha": [0.0, 70.0],
        "Ma2": 0.8,
        "Ys": 0.05,
    },
    "axial_turbine": {
        "psi": 1.6,
        "phi2": 0.8,
        "Ma2": 0.9,
        "fac_Ma3_rel": 0.8,
        "mdot": 10.0,
        "Ys": [0.05, 0.05],
        "r_rms": 0.3,
    },
}

FLUID = {"type": "perfect", "cp": CP, "gamma": GAMMA, "mu": 1.8e-5, "Pr": 0.72}


def build_config(name):
    return Config.from_dict(
        {"fluid": FLUID, "mean_line": {"type": name, **CASES[name]}}
    )


@pytest.fixture(params=sorted(CASES))
def designed(request):
    """A designed mean line for each built-in design."""
    config = build_config(request.param)
    return request.param, config, config.design().mean_line


def test_builtin_design_round_trips(designed):
    """Every design variable comes back out of backward()."""
    import dataclasses

    name, config, ml = designed
    inverted = config.mean_line.backward(ml)

    for field in dataclasses.fields(config.mean_line):
        key = field.name
        assert key in inverted, f"{name}: backward() omits design variable {key!r}"
        got = inverted[key]
        assert got is not None, f"{name}: backward() returns None for {key!r}"
        np.testing.assert_allclose(
            np.asarray(got, dtype=float),
            np.asarray(getattr(config.mean_line, key), dtype=float),
            rtol=1e-3,
            err_msg=f"{name}: design variable {key!r} does not round trip",
        )


def test_builtin_design_passes_its_own_check(designed):
    _, config, ml = designed
    check_round_trip(config.mean_line, ml)


def test_builtin_design_conserves_mass(designed):
    name, _, ml = designed
    mdot = ml.flat.mdot

    assert np.all(np.isfinite(mdot)), f"{name}: non-finite mass flow {mdot}"
    np.testing.assert_allclose(mdot, mdot[0], rtol=5e-3)


def test_builtin_design_is_physical(designed):
    name, _, ml = designed

    assert np.all(ml.P > 0.0), f"{name}: non-positive static pressure"
    assert ml.PR_tt > 1.0, f"{name}: not expanding"
    assert 0.0 < ml.eta_tt <= 1.0, f"{name}: eta_tt = {ml.eta_tt}"


def test_axial_turbine_is_a_repeating_stage():
    config = build_config("axial_turbine")
    inverted = config.mean_line.backward(config.design().mean_line)

    assert float(inverted["Alpha1"]) == pytest.approx(
        float(inverted["Alpha3"]), abs=1e-2
    )


def test_axial_turbine_stator_is_stationary():
    ml = build_config("axial_turbine").design().mean_line

    assert np.all(ml[:, 0].Omega == 0.0)
    assert np.all(ml[:, 1].Omega > 0.0)


#
# EQUIVALENCE WITH THE EXISTING IMPLEMENTATION
#


@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_the_turbigen_implementation(name):
    """turbigen must design the same machine as the package it replaces."""
    new = build_config(name).design().mean_line

    old_config = turbigen_ref.meanline_new.MeanLineConfig.from_dict(
        {"type": name, "n_row": new.n_row, **CASES[name]}
    )
    old_config.set_nominal(PerfectFluid(cp=CP, gamma=GAMMA, mu=1.8e-5, Pr=0.72).eos())
    old = old_config.nominal

    for prop in ("Po", "To", "Ma", "Ma_rel", "Alpha", "s", "mdot"):
        np.testing.assert_allclose(
            getattr(new, prop),
            getattr(old, prop),
            rtol=1e-3,
            err_msg=f"{name}: {prop} differs from the turbigen implementation",
        )
    assert new.eta_tt == pytest.approx(old.eta_tt, rel=1e-3)


#
# THE FRAMEWORK METHOD
#
# `design` validates and freezes; `forward` is the part an author writes. The
# two are separate so that every design gets the check and the freeze without
# having to remember them.
#


def test_design_freezes_what_it_returns(designed):
    """Frozen at the earliest opportunity, so no later stage can write to it."""
    _, _, ml = designed

    assert ml.frozen
    assert ml.flat.frozen and ml[:, 0].frozen

    with pytest.raises(ValueError, match="frozen"):
        ml.flat.set_Vx(10.0)


def test_design_refuses_a_mean_line_that_does_not_invert():
    """The check runs inside design(), not only when a test calls it.

    That is what lets the rest of the package treat a nominal mean line that
    exists as the design that was asked for, with no third state in between.
    """

    class Wrong(AxialTurbine):
        type: ClassVar[str] = "wrong_backward"

        def backward(self, ml):
            # Deliberately misreport a variable that `forward` sets directly.
            # A solve_for *target* such as psi would not do: backward is the
            # single definition of what a variable means, so corrupting one
            # consistently just changes what the design asks for, and the
            # solver hits the new meaning. The round trip catches forward and
            # backward disagreeing, not backward being wrong.
            out = super().backward(ml)
            return {**out, "mdot": out["mdot"] * 1.5}

    config = build_config("axial_turbine")
    design = Wrong(**CASES["axial_turbine"])

    with pytest.raises(DesignError, match="mdot"):
        design.design(config.fluid.eos())


#
# REFERENCE SCALES
#


def test_get_referenced_fluid_does_not_touch_the_mean_line(designed):
    """It returns a fluid rather than applying one, which is what lets it run
    on a frozen mean line and lets the caller choose what to put it on."""
    _, _, ml = designed
    before = ml.fluid

    referenced = ml.get_referenced_fluid()

    assert ml.fluid is before
    assert referenced is not before


def test_get_referenced_fluid_scales_from_the_design(designed):
    _, _, ml = designed
    flat = ml.flat

    referenced = ml.get_referenced_fluid()

    assert referenced.rho_ref == pytest.approx(float(flat.rho.mean()), rel=1e-6)
    assert referenced.V_ref == pytest.approx(float(flat.V.mean()), rel=1e-6)

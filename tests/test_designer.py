"""Tests for the Designer protocol and its target-matching solver.

The point of the Designer class is that `backward` is the single definition of
what a design variable means, and `forward` is driven by it. So the tests that
matter most are the round trips: build a mean line from design variables, read
them back, and require the same numbers. That is the check the previous
free-function design could not enforce, and which `turbine_cascade`'s loss
coefficient silently failed by four orders of magnitude.
"""

import numpy as np
import pytest

import ember.fluid
import turbigen.designer as D
import turbigen.plugins
from turbigen.designer import Designer, DesignError
from turbigen.meanline_new import MeanLine, MeanLineConfig

GAMMA, CP = 1.4, 1005.0
RGAS = CP * (GAMMA - 1.0) / GAMMA


@pytest.fixture
def air():
    return ember.fluid.PerfectFluid(cp=CP, gamma=GAMMA, mu=1.8e-5, Pr=0.72)


@pytest.fixture
def clean_registry():
    """Restore the designer registry after a test registers into it."""
    reg = turbigen.plugins.get_registry()["designer"]
    before = dict(reg)
    yield reg
    reg.clear()
    reg.update(before)


class Uniform(Designer):
    """A one-row designer with an analytically known answer.

    Static state is fixed, so the speed of sound is fixed, and the axial
    velocity that achieves a target Mach number is exactly ``Ma * a``.
    """

    n_row = 1

    def forward(self, ml, Ma, P1=1e5, T1=300.0, guess=50.0):
        def build(Vx):
            ml.set_r(0.5)
            ml.set_Am(1.0)
            ml.set_P_T(P1, T1)
            ml.set_Vx(Vx)
            ml.set_Vr(0.0)
            ml.set_Vt(0.0)

        self.solved = self.solve_for(
            ml, build, unknowns={"Vx": guess}, targets={"Ma": Ma}, name="uniform"
        )

    def backward(self, ml):
        return {"Ma": ml.outlet.Ma, "P1": ml.inlet.P, "T1": ml.inlet.T}


def make(designer, air, n_row=1):
    ml = MeanLine(n_row)
    ml.set_fluid(air)
    designer.name = designer.name or type(designer).__name__
    return ml


#
# solve_for
#


def test_solve_for_finds_the_analytic_answer(air):
    d = Uniform()
    ml = make(d, air)

    d.forward(ml, Ma=0.6)

    a = np.sqrt(GAMMA * RGAS * 300.0)
    assert d.solved["Vx"] == pytest.approx(0.6 * a, rel=1e-3)
    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-4)


@pytest.mark.parametrize("guess", [1.0, 50.0, 300.0, 900.0])
def test_solve_for_is_robust_to_the_initial_guess(air, guess):
    d = Uniform()
    ml = make(d, air)

    d.forward(ml, Ma=0.6, guess=guess)

    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-4)


def test_solve_for_leaves_the_mean_line_at_the_solution(air):
    """The mean line must match the returned unknowns, not the last trial.

    The old hand-rolled loop exited with Omega taken from the new blade speed
    but velocities and areas from the previous iterate, leaving a state that
    was self-inconsistent at the level of the convergence tolerance.
    """
    d = Uniform()
    ml = make(d, air)

    d.forward(ml, Ma=0.6)

    # Rebuilding at the reported answer must change nothing.
    Vx_reported = d.solved["Vx"]
    assert float(ml.Vx.ravel()[0]) == pytest.approx(Vx_reported, rel=1e-6)


def test_solve_for_rejects_an_underdetermined_system(air):
    class TwoUnknownsOneTarget(Uniform):
        def forward(self, ml, Ma):
            def build(Vx, Vt):
                ml.set_r(0.5)
                ml.set_Am(1.0)
                ml.set_P_T(1e5, 300.0)
                ml.set_Vx(Vx)
                ml.set_Vr(0.0)
                ml.set_Vt(Vt)

            self.solve_for(
                ml, build, unknowns={"Vx": 100.0, "Vt": 10.0}, targets={"Ma": Ma}
            )

    d = TwoUnknownsOneTarget()
    ml = make(d, air)

    with pytest.raises(DesignError, match="underdetermined"):
        d.forward(ml, Ma=0.6)


def test_solve_for_reports_an_unreachable_target(air):
    """A target that cannot be met raises, with the history in the message."""
    d = Uniform()
    ml = make(d, air)

    class FixedVx(Uniform):
        def forward(self, ml, Ma):
            def build(unused):
                ml.set_r(0.5)
                ml.set_Am(1.0)
                ml.set_P_T(1e5, 300.0)
                ml.set_Vx(100.0)  # ignores the unknown entirely
                ml.set_Vr(0.0)
                ml.set_Vt(0.0)

            self.solve_for(ml, build, unknowns={"unused": 1.0}, targets={"Ma": Ma})

    d = FixedVx()
    ml = make(d, air)

    with pytest.raises(DesignError, match="did not converge") as excinfo:
        d.forward(ml, Ma=0.9)

    assert "history" in str(excinfo.value)


def test_solve_for_rejects_an_unknown_target_key(air):
    class BadTarget(Uniform):
        def forward(self, ml, Ma):
            def build(Vx):
                ml.set_r(0.5)
                ml.set_Am(1.0)
                ml.set_P_T(1e5, 300.0)
                ml.set_Vx(Vx)
                ml.set_Vr(0.0)
                ml.set_Vt(0.0)

            self.solve_for(
                ml, build, unknowns={"Vx": 100.0}, targets={"nonexistent": 1.0}
            )

    d = BadTarget()
    ml = make(d, air)

    with pytest.raises(DesignError, match="not returned by backward"):
        d.forward(ml, Ma=0.6)


def test_solve_for_survives_an_infeasible_trial(air):
    """A trial state the designer cannot evaluate must not abort the solve.

    A trust-region solver probes unphysical states as a matter of course; those
    trials should be rejected and the radius shrunk, not raised through.
    """

    class Fragile(Uniform):
        triggered = False

        def backward(self, ml):
            # Fail once, the first time the solver steps away from the guess,
            # as a fragile diagnostic in a real designer does. Failing only
            # once leaves the Jacobian around the starting point intact, which
            # is what lets the solver retreat and try a shorter step.
            if not self.triggered and float(ml.outlet.Vx) > 150.0:
                self.triggered = True
                raise ValueError("unphysical")
            return super().backward(ml)

    d = Fragile()
    ml = make(d, air)

    # The answer is Vx = 0.6 * a ~ 208, so reaching it must cross 150.
    d.forward(ml, Ma=0.6, guess=100.0)

    assert d.triggered, "the infeasible branch was never exercised"
    assert float(ml.outlet.Ma) == pytest.approx(0.6, rel=1e-3)


#
# Parameters and defaults
#


def test_design_params_reads_the_forward_signature():
    params = D.design_params(Uniform())

    assert set(params) == {"Ma", "P1", "T1", "guess"}
    assert params["Ma"] is D.REQUIRED
    assert params["P1"] == 1e5


def test_resolve_defaults_fills_in_every_value():
    resolved = D.resolve_defaults(Uniform(), {"Ma": 0.5})

    # A written config must record what the design used, not only what the
    # user typed, or an archived config stops reproducing its machine.
    assert resolved == {"Ma": 0.5, "P1": 1e5, "T1": 300.0, "guess": 50.0}


def test_resolve_defaults_rejects_a_missing_required_variable():
    with pytest.raises(ValueError, match="Missing required design variables"):
        D.resolve_defaults(Uniform(), {"P1": 2e5})


def test_resolve_defaults_rejects_an_unknown_variable():
    with pytest.raises(ValueError, match="Unexpected design variables"):
        D.resolve_defaults(Uniform(), {"Ma": 0.5, "nonsense": 1.0})


#
# Registration
#


def test_register_designer_accepts_a_valid_designer(clean_registry):
    @turbigen.plugins.register_designer("_test_ok")
    class Good(Uniform):
        pass

    assert clean_registry["_test_ok"].n_row == 1
    assert clean_registry["_test_ok"].name == "_test_ok"


def test_register_designer_rejects_a_non_designer(clean_registry):
    with pytest.raises(TypeError, match="subclass of"):

        @turbigen.plugins.register_designer("_test_bad")
        class NotADesigner:
            pass


def test_register_designer_rejects_a_bad_n_row(clean_registry):
    with pytest.raises(ValueError, match="n_row"):

        @turbigen.plugins.register_designer("_test_nrow")
        class BadRows(Uniform):
            n_row = 0


def test_register_designer_rejects_a_bad_forward_signature(clean_registry):
    with pytest.raises(TypeError, match="must be named 'ml'"):

        @turbigen.plugins.register_designer("_test_sig")
        class BadSig(Uniform):
            def forward(self, mean_line, Ma):
                pass


def test_register_designer_rejects_a_duplicate_name(clean_registry):
    @turbigen.plugins.register_designer("_test_dup")
    class First(Uniform):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @turbigen.plugins.register_designer("_test_dup")
        class Second(Uniform):
            pass


#
# Round trips through the built-in designers
#
# forward() then backward() must return the design variables it was given.
# This is the check that the previous architecture could not enforce.
#

CASES = {
    "turbine_cascade": {
        "n_row": 1,
        "vars": {
            "span": [0.05, 0.05],
            "Alpha": [0.0, 70.0],
            "Ma2": 0.8,
            "Ys": 0.05,
        },
    },
    "axial_turbine": {
        "n_row": 2,
        "vars": {
            "psi": 1.6,
            "phi2": 0.8,
            "Ma2": 0.9,
            "fac_Ma3_rel": 0.8,
            "mdot": 10.0,
            "Ys": [0.05, 0.05],
            "r_rms": 0.3,
        },
    },
}


@pytest.fixture(params=sorted(CASES))
def designed(request, air):
    """A designed mean line for each built-in designer."""
    name = request.param
    case = CASES[name]
    config = MeanLineConfig.from_dict(
        {"type": name, "n_row": case["n_row"], **case["vars"]}
    )
    config.set_nominal(air)
    return name, config


def test_builtin_designer_round_trips(designed):
    """Every design variable comes back out of backward()."""
    name, config = designed
    inverted = config.designer.backward(config.nominal)

    for key, nominal in config.design_vars.items():
        assert key in inverted, f"{name}: backward() omits design variable '{key}'"
        got = inverted[key]
        assert got is not None, f"{name}: backward() returns None for '{key}'"
        np.testing.assert_allclose(
            np.asarray(got, dtype=float),
            np.asarray(nominal, dtype=float),
            rtol=1e-3,
            err_msg=f"{name}: design variable '{key}' does not round trip",
        )


def test_builtin_designer_passes_its_own_check(designed):
    _, config = designed
    config.check_nominal()


def test_builtin_designer_conserves_mass(designed):
    name, config = designed
    mdot = config.nominal.flat.mdot

    assert np.all(np.isfinite(mdot)), f"{name}: non-finite mass flow {mdot}"
    np.testing.assert_allclose(mdot, mdot[0], rtol=5e-3)


def test_builtin_designer_is_physical(designed):
    """Pressure falls through a turbine and the efficiency is sane."""
    name, config = designed
    ml = config.nominal

    assert np.all(ml.P > 0.0), f"{name}: non-positive static pressure"
    assert ml.PR_tt > 1.0, f"{name}: not expanding"
    assert 0.0 < ml.eta_tt <= 1.0, f"{name}: eta_tt = {ml.eta_tt}"


def test_axial_turbine_is_a_repeating_stage(air):
    """The inlet and outlet yaw angles match, which forward solves for."""
    config = MeanLineConfig.from_dict(
        {"type": "axial_turbine", "n_row": 2, **CASES["axial_turbine"]["vars"]}
    )
    config.set_nominal(air)
    inverted = config.designer.backward(config.nominal)

    assert float(inverted["Alpha1"]) == pytest.approx(
        float(inverted["Alpha3"]), abs=1e-2
    )


def test_axial_turbine_stator_is_stationary(air):
    config = MeanLineConfig.from_dict(
        {"type": "axial_turbine", "n_row": 2, **CASES["axial_turbine"]["vars"]}
    )
    config.set_nominal(air)
    ml = config.nominal

    assert np.all(ml.row(0).Omega == 0.0)
    assert np.all(ml.row(1).Omega > 0.0)


#
# Config wiring
#


def test_config_resolves_defaults(air):
    config = MeanLineConfig.from_dict(
        {"type": "turbine_cascade", "n_row": 1, **CASES["turbine_cascade"]["vars"]}
    )

    # Defaults the user did not type are stored, so to_dict round trips them.
    assert config.design_vars["htr"] == 0.95
    assert config.design_vars["Po1"] == 1e5
    assert config.to_dict()["To1"] == 300.0


def test_config_does_not_mutate_the_input_dict():
    source = {
        "type": "turbine_cascade",
        "n_row": 1,
        **CASES["turbine_cascade"]["vars"],
    }
    before = dict(source)

    MeanLineConfig.from_dict(source)

    assert source == before


def test_config_rejects_a_row_count_the_designer_does_not_support():
    with pytest.raises(ValueError, match="designs 1 row"):
        MeanLineConfig.from_dict(
            {"type": "turbine_cascade", "n_row": 3, **CASES["turbine_cascade"]["vars"]}
        )


def test_config_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="Unknown mean_line type"):
        MeanLineConfig.from_dict({"type": "no_such_designer", "n_row": 1})

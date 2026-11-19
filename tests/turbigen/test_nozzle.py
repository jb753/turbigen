"""Tests for throttling with a choked exit nozzle morphed into the mesh.

No CFD. The gas dynamics are checked against textbook values, the morph on a
coarse meshed grid by reading its nodes back, and the sizing and the sweep
against stand-in runs whose throat passes a known fraction of the
one-dimensional flow, as `test_chic.py` drives its march.

Test cases:
- test_sonic_ratios: A/A* is one and p/Po the critical ratio at Ma=1
- test_critical_flux_is_rho_a_at_the_throat: against the definition
- test_the_area_profile_has_its_minimum_at_the_throat: and meets the duct
- test_no_nozzle_leaves_the_grid_alone: morph is a no-op without one
- test_the_morph_only_moves_the_exit_duct: upstream of the fade is untouched
- test_the_morph_reaches_the_exit_area: sonic throat, expanded to nozzle_Ma
- test_the_morph_keeps_the_mid_radius: each wall moves towards the other
- test_a_throat_wider_than_the_duct_is_refused: there would be no throat
- test_a_nozzle_cannot_also_be_throttled: one or the other
- test_a_nozzle_must_expand_supersonically: or it is not choked
- test_the_exit_pressure_is_below_choke: isentropic at nozzle_Ma
- test_the_throat_is_sized_one_dimensionally: and not solved
- test_a_datum_that_does_not_converge_stops_the_speedline: no datum, no sweep
- test_a_nozzle_sweep_closes_the_throat: by step, from the sized area
- test_a_nozzle_sweep_needs_a_sized_throat: and says so
- test_at_takes_a_nozzle_off: a pressure sweep is a pressure at exit
- test_the_nozzle_table_names_the_closure: in the heading and the bracket
- test_a_nozzle_choke_leg_is_a_back_pressure: the throat comes off
- test_a_speed_needs_a_nozzle: and must be positive
- test_the_speed_scales_every_row: blocks turn at the fraction asked for
- test_a_nozzle_keeps_its_speed: placing a throat leaves the speed alone
- test_a_speedline_departs_from_the_design_throat: soft started from it
- test_a_leg_must_be_swept: stall and choke not both off
- test_the_choke_leg_runs_the_open_duct: from where the datum left it
- test_the_choke_leg_departs_on_the_scale_of_every_other_DP_adjust: at the exit
- test_DP_adjust_at_inverts_the_exit_pressure: one scale, both ways
- test_a_restart_beside_a_converged_design_point_is_the_datum: no solve
- test_a_restart_that_is_not_this_design_point_is_refused: loudly
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from test_bconds import meshed
from test_blade import build

from turbigen import Chic, bconds, chic, loop, nozzle
from turbigen.bconds import OperatingPoint
from turbigen.solver import Ember

GAMMA = 1.4


def nozzle_point(area, **kwargs):
    """An operating point choking the exit with a throat of `area`."""
    return OperatingPoint(nozzle_area=area, **kwargs)


def outlet_xr(grid, machine):
    """Return (s, eta) of every node, against the outlet line, and the line."""
    centre, t, n, span, area = nozzle.outlet_line(machine.annulus)
    xr = np.concatenate([np.array(b.xrt)[..., :2].reshape(-1, 2) for b in grid])
    return xr, (xr - centre) @ n, centre, t, n, span, area


#
# GAS DYNAMICS
#


def test_sonic_ratios():
    assert nozzle.area_ratio(1.0, GAMMA) == pytest.approx(1.0)
    assert nozzle.p_ratio(1.0, GAMMA) == pytest.approx(0.528282, rel=1e-5)
    # Either side of the throat needs more area
    assert nozzle.area_ratio(1.1, GAMMA) > 1.0
    assert nozzle.area_ratio(0.9, GAMMA) > 1.0


def test_critical_flux_is_rho_a_at_the_throat():
    cp, Po, To = 1005.0, 1.5e5, 320.0
    R = cp * (GAMMA - 1.0) / GAMMA
    T = To * 2.0 / (GAMMA + 1.0)
    P = Po * nozzle.p_ratio(1.0, GAMMA)
    rho_a = P / (R * T) * np.sqrt(GAMMA * R * T)

    assert nozzle.critical_flux(Po, To, cp, GAMMA) == pytest.approx(rho_a)


def test_the_area_profile_has_its_minimum_at_the_throat():
    length = 0.1
    s = np.linspace(0.0, length, 1001)
    duct = 1.0 + 0.2 * s
    area = nozzle.area_profile(s, duct, length, 0.9, 0.93)

    assert s[np.argmin(area)] == pytest.approx(nozzle.THROAT * length, abs=1e-3)
    assert area.min() == pytest.approx(0.9)
    assert area[0] == pytest.approx(0.93)
    assert area[-1] == pytest.approx(duct[-1])


#
# THE MORPH
#


def test_no_nozzle_leaves_the_grid_alone():
    machine, grid = meshed()
    before = [np.array(b.xrt) for b in grid]

    assert nozzle.morph(grid, machine, OperatingPoint()) is None
    assert nozzle.morph(grid, machine, None) is None
    for b, xrt in zip(grid, before):
        assert np.array_equal(np.array(b.xrt), xrt)


def test_the_morph_only_moves_the_exit_duct():
    machine, grid = meshed()
    xr0, s, *_, span, _ = outlet_xr(grid, machine)

    nozzle.morph(grid, machine, nozzle_point(0.9))

    xr1, *_ = outlet_xr(grid, machine)
    moved = np.any(xr1 != xr0, axis=1)
    assert moved.any()
    assert not moved[s >= span].any()


def test_the_morph_reaches_the_exit_area():
    machine, grid = meshed()
    op = nozzle_point(0.9, nozzle_Ma=1.2)

    nozzle.morph(grid, machine, op)

    xr, s, centre, t, n, span, area = outlet_xr(grid, machine)
    gamma = float(machine.mean_line.outlet.gamma)

    def annulus_area(near):
        r = xr[np.abs(s - near) < 1e-6 * span, 1]
        return np.pi * (r.max() ** 2 - r.min() ** 2)

    exit_area = 0.9 * area * nozzle.area_ratio(1.2, gamma)
    assert annulus_area(0.0) == pytest.approx(exit_area, rel=1e-3)


def test_the_morph_keeps_the_mid_radius():
    machine, grid = meshed()
    xr0, s0, *_, span, _ = outlet_xr(grid, machine)
    at_exit = np.abs(s0) < 1e-6 * span
    mid0 = 0.5 * (xr0[at_exit, 1].min() + xr0[at_exit, 1].max())

    nozzle.morph(grid, machine, nozzle_point(0.9))

    xr1, *_ = outlet_xr(grid, machine)
    mid1 = 0.5 * (xr1[at_exit, 1].min() + xr1[at_exit, 1].max())
    assert mid1 == pytest.approx(mid0, rel=1e-6)


def test_a_throat_wider_than_the_duct_is_refused():
    machine, grid = meshed()
    with pytest.raises(ValueError, match="no throat"):
        nozzle.morph(grid, machine, nozzle_point(1.2))


#
# THE OPERATING POINT
#


def test_a_nozzle_cannot_also_be_throttled():
    with pytest.raises(ValueError, match="not both"):
        OperatingPoint(nozzle_area=0.9, mdot_adjust=0.0)


def test_a_nozzle_must_expand_supersonically():
    with pytest.raises(ValueError, match="supersonic"):
        OperatingPoint(nozzle_area=0.9, nozzle_Ma=0.95)


def test_the_exit_pressure_is_below_choke():
    """From the inlet stagnation pressure, which a choked nozzle's own is
    above at any speed, so that it is choked at every speed."""
    machine = build().design()
    Po_in = float(machine.mean_line.inlet.Po)
    gamma = float(machine.mean_line.outlet.gamma)

    P = bconds.exit_pressure(machine, nozzle_point(0.9, nozzle_Ma=1.1))

    assert P == pytest.approx(Po_in * nozzle.p_ratio(1.1, gamma))
    assert P < Po_in * nozzle.p_ratio(1.0, gamma)


#
# SIZING AND SWEEPING
#


CD = 0.98
"""What the stand-in's throat passes, as a fraction of the 1D flow."""


def chic_config(**kwargs):
    return dataclasses.replace(build(), chic=Chic(nozzle=True, **kwargs))


def design_result(config):
    """A design point at the machine's own outlet state."""
    outlet = config.design().mean_line.outlet
    return SimpleNamespace(actual=SimpleNamespace(outlet=outlet))


def choking_runner(config, log=None, limit=0.0, cap=np.inf):
    """Return a `run` whose throat passes CD of the 1D flow, converging while
    the throat is open wider than `limit` of the outlet area, and the flow
    held to `cap` as a choked fan would hold it."""
    machine = config.design()
    outlet = machine.mean_line.outlet
    *_, area_out = nozzle.outlet_line(machine.annulus)
    flux = nozzle.critical_flux(
        float(outlet.Po), float(outlet.To), float(outlet.cp), float(outlet.gamma)
    )

    def run(config_now, _):
        area = config_now.operating_point.nozzle_area
        if log is not None:
            log.append(area)
        mdot = min(CD * flux * area * area_out, cap)
        state = SimpleNamespace(
            mdot=mdot,
            Po=outlet.Po,
            To=outlet.To,
            cp=outlet.cp,
            gamma=outlet.gamma,
            P=outlet.P,
        )
        # A duct that changes nothing, carried down to the exit
        actual = SimpleNamespace(
            outlet=state, PR_ts=1.0, interpolate_to=lambda annulus, m: [state]
        )
        return SimpleNamespace(converged=area > limit, actual=actual)

    return run


def test_the_throat_is_sized_one_dimensionally():
    config = chic_config()
    design = design_result(config)

    area = loop.sized_throat(config, design)

    one_d = nozzle.throat_fraction(design.actual.outlet, config.design().annulus)
    assert area == pytest.approx(one_d)


def test_a_datum_that_does_not_converge_stops_the_speedline(tmp_path, monkeypatch):
    config = chic_config(speed=0.9)
    stand_in = choking_runner(config, limit=np.inf)
    calls = []

    def solve(config_now, point_dir, restart_path=None, soft=False, **_):
        calls.append(config_now.operating_point)
        return stand_in(config_now, None)

    monkeypatch.setattr(loop.pipeline, "solve", solve)

    points, bracket = loop.characteristic(
        config, tmp_path, design=design_result(config)
    )

    assert len(calls) == 1
    assert bracket is None
    assert [p.stood for p in points] == [False]


def test_a_nozzle_sweep_closes_the_throat():
    config = chic_config(step_coarse=0.05, step_fine=0.01)
    sized = chic.at_nozzle(config, 0.8)
    seen = []

    points, (lo, hi) = chic.sweep(sized, choking_runner(config, seen, limit=0.67))

    assert seen[:3] == pytest.approx([0.8 * 0.95, 0.8 * 0.90, 0.8 * 0.85])
    assert points[0].setting == pytest.approx(0.05)
    # Closed by 1 - 0.67/0.8 = 0.1625 at the stand-in's limit
    assert lo < 1.0 - 0.67 / 0.8 < hi
    assert hi - lo == pytest.approx(0.01)


def test_a_nozzle_sweep_needs_a_sized_throat():
    with pytest.raises(ValueError, match="nozzle_area is unset"):
        chic.sweep(chic_config(), choking_runner(chic_config()))


def test_at_takes_a_nozzle_off():
    choked = dataclasses.replace(build(), operating_point=nozzle_point(0.9))

    moved = chic.at(choked, 0.1)

    assert moved.operating_point.nozzle_area is None
    assert moved.operating_point.DP_adjust == pytest.approx(0.1)


def test_the_nozzle_table_names_the_closure():
    config = chic_config(step_coarse=0.05, step_fine=0.02)
    config = chic.at_nozzle(config, 0.8)
    points, bracket = chic.sweep(config, choking_runner(config, limit=0.67))

    table = chic.format_table(points, bracket, chic.setting_name(config))

    assert "closure" in table
    assert "up to closure" in table


#
# TOWARDS CHOKE
#


def test_a_nozzle_choke_leg_is_a_back_pressure():
    """Lowered from the config's own DP_adjust, with no throat."""
    config = chic.at_nozzle(chic_config(step_coarse=0.05, step_fine=0.01), 0.8)
    config = dataclasses.replace(
        config,
        operating_point=dataclasses.replace(config.operating_point, DP_adjust=0.2),
    )
    seen = []

    def run(config_now, _):
        seen.append(config_now.operating_point)
        return SimpleNamespace(converged=config_now.operating_point.DP_adjust > 0.08)

    points, bracket = chic.sweep(config, run, direction=-1)

    assert all(op.nozzle_area is None for op in seen)
    assert [op.DP_adjust for op in seen[:2]] == pytest.approx([0.15, 0.10])
    assert bracket[0] > 0.08 > bracket[1]
    table = chic.format_table(points, bracket, chic.setting_name(config, direction=-1))
    assert "Towards choke, points stand down to DP_adjust" in table


#
# SPEEDLINES
#


@pytest.mark.parametrize(
    "speed, match", [(0.9, "needs chic.nozzle"), (-0.5, "positive")]
)
def test_a_speed_needs_a_nozzle(speed, match):
    with pytest.raises(ValueError, match=match):
        Chic(speed=speed)


def test_the_speed_scales_every_row():
    machine, grid = meshed()
    design = machine.mean_line.Omega[0]

    bconds.apply(grid, machine, OperatingPoint(speed=0.8))

    for blocks, Omega in zip(grid.rows, design):
        for block in blocks:
            assert float(block.Omega) == pytest.approx(0.8 * Omega)


def test_a_nozzle_keeps_its_speed():
    config = chic.at_speed(chic_config(), 0.8)

    assert chic.at_nozzle(config, 0.7).operating_point.speed == 0.8


def test_a_speedline_departs_from_the_design_throat(tmp_path, monkeypatch):
    """At another speed, the datum is the throat sized for the design point,
    run at that speed and soft started from the design point's field, and
    nothing is solved at the design speed."""
    config = dataclasses.replace(
        build(),
        chic=Chic(nozzle=True, step_coarse=0.05, step_fine=0.05, speed=0.9),
    )
    stand_in = choking_runner(config, limit=0.0)
    calls = []

    def solve(config_now, point_dir, restart_path=None, soft=False, **_):
        op = config_now.operating_point
        calls.append((op.speed, op.nozzle_area, soft, restart_path, point_dir))
        result = stand_in(config_now, None)
        # The speedline refuses its second step towards stall, closed by
        # 0.10 from the sized throat
        if len(calls) > 1:
            result.converged = op.nozzle_area > 0.93 * calls[0][1]
        return result

    monkeypatch.setattr(loop.pipeline, "solve", solve)

    design = design_result(config)
    points, _ = loop.characteristic(config, tmp_path, "design.npz", design=design)

    assert {c[0] for c in calls} == {0.9}
    assert all(c[4].parent == tmp_path for c in calls)

    # Its datum the sized throat, soft started from the design point's field,
    # and a point of the characteristic
    one_d = loop.sized_throat(config, design)
    assert calls[0][1] == pytest.approx(one_d)
    assert calls[0][2] is True
    assert calls[0][3] == "design.npz"
    assert points[0].setting == 0.0 and points[0].stood
    # The stall leg from the datum's field, hard started
    assert calls[1][3] == calls[0][4] / "restart.npz"
    assert sum(c[2] for c in calls) == 1


def test_a_leg_must_be_swept():
    with pytest.raises(ValueError, match="no leg"):
        Chic(stall=False, choke=False)


@pytest.mark.parametrize("speed", [1.0, 0.9])
@pytest.mark.parametrize("stall", [True, False])
def test_the_choke_leg_runs_the_open_duct(tmp_path, monkeypatch, stall, speed):
    """At the design speed from the design point's own pressure and field; at
    another, from the pressure its datum throat left at the machine outlet,
    from that throat's field, the first point soft started as the throat
    comes off. Either lowers the back pressure, with no nozzle, and nothing
    is throttled."""
    config = dataclasses.replace(
        build(),
        operating_point=OperatingPoint(DP_adjust=0.2),
        chic=Chic(
            nozzle=True,
            choke=True,
            stall=stall,
            step_coarse=0.05,
            step_fine=0.05,
            speed=speed,
        ),
    )
    stand_in = choking_runner(config)
    calls = []

    def solve(config_now, point_dir, restart_path=None, soft=False, **_):
        op = config_now.operating_point
        calls.append((op, restart_path, point_dir, soft))
        if op.nozzle_area is not None:
            result = stand_in(config_now, None)
            # Every throat closed from the sized one refuses
            area = loop.sized_throat(config, design_result(config))
            result.converged = op.nozzle_area >= area * (1 - 1e-9)
            return result
        # The open duct refuses its second step towards choke
        n_open = len([c for c in calls if c[0].nozzle_area is None])
        return SimpleNamespace(converged=n_open < 2)

    monkeypatch.setattr(loop.pipeline, "solve", solve)

    loop.characteristic(config, tmp_path, "design.npz", design=design_result(config))

    assert not [c for c in calls if c[0].mdot_adjust is not None]
    swept = [c for c in calls if c[0].nozzle_area is None]
    throats = [c for c in calls if c[0].nozzle_area is not None]
    if speed == 1.0:
        assert [c[0].DP_adjust for c in swept] == pytest.approx([0.15, 0.10])
        assert swept[0][1] == "design.npz" and swept[0][3] is False
        # Nothing run at the sized throat itself, only closed from it
        closed = throats
    else:
        datum = throats[0]
        datum_config = dataclasses.replace(config, operating_point=datum[0])
        start = loop._outlet_DP_adjust(config, stand_in(datum_config, None))
        assert [c[0].DP_adjust for c in swept] == pytest.approx(
            [start - 0.05, start - 0.10]
        )
        assert swept[0][1] == datum[2] / loop.RESTART_NAME
        assert swept[0][3] is True and swept[1][3] is False
        assert all(c[0].speed == 0.9 for c in calls)
        closed = throats[1:]

    # Towards stall only when asked, the first closure soft started at the
    # design speed as the nozzle goes on
    assert bool(closed) == stall
    if stall:
        assert closed[0][3] is (speed == 1.0)


def test_the_choke_leg_departs_on_the_scale_of_every_other_DP_adjust():
    """The pressure the datum reached, carried down the duct to the exit, and
    measured as bconds measures a held one: the design's own flow is the
    design point, and a flow reaching the exit at a held pressure is that
    pressure's DP_adjust whatever it was at the machine outlet."""
    config = build()
    machine = config.design()

    design = SimpleNamespace(actual=machine.mean_line)
    assert loop._outlet_DP_adjust(config, design) == pytest.approx(0.0, abs=1e-9)

    held = bconds.exit_pressure(machine, OperatingPoint(DP_adjust=-0.2))
    at_exit = SimpleNamespace(P=held)
    result = SimpleNamespace(
        actual=SimpleNamespace(
            outlet=SimpleNamespace(P=0.5 * held),
            interpolate_to=lambda annulus, m: [at_exit],
        )
    )
    assert loop._outlet_DP_adjust(config, result) == pytest.approx(-0.2)


@pytest.mark.parametrize("DP_adjust", [-0.3, 0.0, 0.15])
def test_DP_adjust_at_inverts_the_exit_pressure(DP_adjust):
    machine = build().design()
    P = bconds.exit_pressure(machine, OperatingPoint(DP_adjust=DP_adjust))

    assert bconds.DP_adjust_at(machine, P) == pytest.approx(DP_adjust)


#
# A DATUM TAKEN FROM A RESTART
#


@pytest.fixture
def beside(monkeypatch):
    """Stand in for the output.yaml and stamp beside a restart field: a
    converged design point at DP_adjust=0.1, stamped as `config`'s, unless
    told otherwise."""
    from turbigen import Result, restart

    config = chic_config(solve_datum=False)
    stored = dataclasses.replace(config, operating_point=OperatingPoint(DP_adjust=0.1))
    state = {"found": True, "converged": True, "stamp": None}

    def stored_at(path):
        if not state["found"]:
            return None
        return stored, Result(converged=state["converged"])

    def read_stamp(path):
        if state["stamp"] is not None:
            return state["stamp"]
        return restart.design_stamp(loop.iterate.resolve(stored))

    monkeypatch.setattr(loop, "_stored", stored_at)
    monkeypatch.setattr(loop.restart, "read_stamp", read_stamp)
    return config, state


def test_a_restart_beside_a_converged_design_point_is_the_datum(beside):
    config, _ = beside

    datum, result = loop.restarted_design(config, "run/restart.npz")

    assert result.converged
    assert datum.operating_point.DP_adjust == 0.1
    assert datum.chic == config.chic


@pytest.mark.parametrize(
    "field, change, match",
    [
        (None, {}, "names none"),
        ("run/restart.npz", {"found": False}, "no result"),
        ("run/restart.npz", {"stamp": "0" * 64}, "another design"),
        ("run/restart.npz", {"converged": False}, "did not converge"),
    ],
)
def test_a_restart_that_is_not_this_design_point_is_refused(
    beside, field, change, match
):
    """Loudly, rather than quietly solving the datum after all."""
    config, state = beside
    state.update(change)

    with pytest.raises(ValueError, match=match):
        loop.restarted_design(config, field)

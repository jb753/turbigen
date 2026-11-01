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
- test_sizing_solves_the_1d_throat_once: whatever it passes, it is the datum
- test_a_throat_that_does_not_converge_stops_the_sizing: no datum, no sweep
- test_a_nozzle_sweep_closes_the_throat: by step, from the sized area
- test_a_nozzle_sweep_needs_a_sized_throat: and says so
- test_at_takes_a_nozzle_off: a pressure sweep is a pressure at exit
- test_the_nozzle_table_names_the_closure: in the heading and the bracket
- test_a_nozzle_choke_leg_is_a_back_pressure: the throat comes off
- test_speeds_need_a_nozzle: and must be positive and distinct
- test_the_speed_scales_every_row: blocks turn at the fraction asked for
- test_a_nozzle_keeps_its_speed: placing a throat leaves the speed alone
- test_speedlines_depart_from_the_design_throat: soft started, nearest first
- test_a_leg_must_be_swept: stall and choke not both off
- test_the_choke_leg_runs_the_open_duct: throttled to the datum, then pressure
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
            mdot=mdot, Po=outlet.Po, To=outlet.To, cp=outlet.cp, gamma=outlet.gamma
        )
        actual = SimpleNamespace(outlet=state, PR_ts=1.0)
        return SimpleNamespace(converged=area > limit, actual=actual)

    return run


def test_sizing_solves_the_1d_throat_once():
    config = chic_config()
    design = design_result(config)
    seen = []

    sized = loop.size_nozzle(config, design, choking_runner(config, seen))

    one_d = nozzle.throat_fraction(design.actual.outlet, config.design().annulus)
    assert seen == pytest.approx([one_d])
    assert sized.operating_point.nozzle_area == pytest.approx(one_d)
    assert sized.operating_point.mdot_adjust is None


def test_a_throat_that_does_not_converge_stops_the_sizing():
    config = chic_config()
    with pytest.raises(ValueError, match="no datum"):
        loop.size_nozzle(
            config, design_result(config), choking_runner(config, limit=np.inf)
        )


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
    "speeds, match",
    [((0.9,), "needs chic.nozzle"), ((1.0, -0.5), "positive"), ((0.9, 0.9), "repeats")],
)
def test_speeds_need_a_nozzle(speeds, match):
    with pytest.raises(ValueError, match=match):
        Chic(speeds=speeds)


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


def test_speedlines_depart_from_the_design_throat(tmp_path, monkeypatch):
    """Sized at the design speed, then each other speed's datum is that
    throat, soft started from the nearest speed already run."""
    config = dataclasses.replace(
        build(),
        chic=Chic(
            nozzle=True, step_coarse=0.05, step_fine=0.05, speeds=(1.0, 0.8, 0.9)
        ),
    )
    stand_in = choking_runner(config, limit=0.0)
    calls = []

    def solve(config_now, point_dir, restart_path=None, soft=False, **_):
        op = config_now.operating_point
        calls.append((op.speed, op.nozzle_area, soft, restart_path, point_dir))
        result = stand_in(config_now, None)
        # Every speedline refuses its second step towards stall, closed by
        # 0.10 from the sized throat
        if len(calls) > 1:
            result.converged = op.nozzle_area > 0.93 * calls[0][1]
        return result

    monkeypatch.setattr(loop.pipeline, "solve", solve)

    loop.characteristic(config, tmp_path, design=design_result(config))

    speeds = [c[0] for c in calls]
    # Sizing at the design speed, then its speedline, then 0.9 and 0.8
    assert speeds[:2] == [1.0, 1.0]
    assert speeds.index(0.9) < speeds.index(0.8)
    assert all(c[4].parent.name == f"speed_{c[0]:.3f}" for c in calls)

    datum = calls[0][1]
    first = {speed: calls[speeds.index(speed)] for speed in (0.9, 0.8)}
    for speed, call in first.items():
        # The design throat, soft started; 0.8 from the 0.9 datum, nearer
        assert call[1] == pytest.approx(datum)
        assert call[2] is True
    assert first[0.9][3] == calls[0][4] / "restart.npz"
    assert first[0.8][3] == first[0.9][4] / "restart.npz"
    # Nothing else soft starts
    assert sum(c[2] for c in calls) == 2


def test_a_leg_must_be_swept():
    with pytest.raises(ValueError, match="no leg"):
        Chic(stall=False, choke=False)


@pytest.mark.parametrize("stall", [True, False])
def test_the_choke_leg_runs_the_open_duct(tmp_path, monkeypatch, stall):
    """At the design speed from the design point itself; at another, from
    the open duct throttled to the flow its datum throat passed. Each then
    lowers the back pressure, with no nozzle."""
    config = dataclasses.replace(
        build(),
        operating_point=OperatingPoint(DP_adjust=0.2),
        chic=Chic(
            nozzle=True,
            choke=True,
            stall=stall,
            step_coarse=0.05,
            step_fine=0.05,
            speeds=(1.0, 0.9),
        ),
    )
    stand_in = choking_runner(config)
    settled_at = -0.3
    calls = []

    def solve(config_now, point_dir, restart_path=None, soft=False, **_):
        op = config_now.operating_point
        calls.append((op, restart_path, point_dir))
        if op.nozzle_area is not None:
            result = stand_in(config_now, None)
            # Every throat closed from the sized one refuses
            if len(calls) > 1:
                result.converged = op.nozzle_area >= calls[0][0].nozzle_area
            result.operating_point = None
            return result
        if op.mdot_adjust is not None:
            return SimpleNamespace(
                converged=True,
                operating_point=dataclasses.replace(op, DP_adjust=settled_at),
            )
        # The open duct refuses its second step towards choke
        datum = 0.2 if op.speed == 1.0 else settled_at
        return SimpleNamespace(converged=op.DP_adjust > datum - 0.08)

    monkeypatch.setattr(loop.pipeline, "solve", solve)

    loop.characteristic(config, tmp_path, design=design_result(config))

    # Every open-duct point a pressure sweep, below its datum
    open_ = [c for c in calls if c[0].nozzle_area is None]
    throttled = [c for c in open_ if c[0].mdot_adjust is not None]
    swept = [c for c in open_ if c[0].mdot_adjust is None]
    by_speed = {
        speed: [c[0].DP_adjust for c in swept if c[0].speed == speed]
        for speed in (1.0, 0.9)
    }
    assert by_speed[1.0] == pytest.approx([0.15, 0.10])
    assert by_speed[0.9] == pytest.approx([settled_at - 0.05, settled_at - 0.10])

    # Only 0.9 is throttled, to the flow its datum throat passed, from its field
    (throttle,) = throttled
    assert throttle[0].speed == 0.9
    datum = next(c for c in calls if c[0].speed == 0.9 and c[0].nozzle_area is not None)
    mdot_design = float(config.design().mean_line.outlet.mdot)
    mdot = float(
        stand_in(
            dataclasses.replace(config, operating_point=datum[0]), None
        ).actual.outlet.mdot
    )
    assert throttle[0].mdot_adjust == pytest.approx(mdot / mdot_design - 1.0)
    assert throttle[1] == datum[2] / loop.RESTART_NAME

    # Towards stall only when asked
    closed = [c for c in calls if c[0].nozzle_area is not None][2:]
    assert bool(closed) == stall

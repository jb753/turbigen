"""Tests for the surface dissipation metric.

Two cases, for two different jobs. The fast single-row cascade is marched, so
it is the one with a real field to integrate and a measured loss to weigh the
answer against. The two-row tip-gap machine is not marched: it is here for
shape, for the frame each wall is measured in, and for the gapped blade, none
of which need the flow to have settled.

There is deliberately no equivalence test against `turbigen_ref`. Unlike the
mesher, this port changes four things on purpose --- the reference entropy is
per row rather than per machine, the integral is clipped at both ends rather
than one, each wall is measured in its own frame rather than by a test that is
inverted under this package's convention, and a clearance gap is not blade. A
bit-exact comparison would pin the faults rather than the behaviour.

Test cases:
- test_edge_velocity_is_the_speed_relative_to_the_wall: the reconstruction
- test_a_stationary_casing_is_measured_in_the_absolute_frame: and its frame
- test_nothing_to_measure_returns_nothing: the Metric contract
- test_keys_and_shape: (2, n_row), endwalls then blades
- test_entropy_rate_is_linear_in_the_coefficient: and area is not
- test_endwall_area_counts_every_passage: the Nb scaling, exactly
- test_the_clip_removes_exactly_the_ducts: against the geometry, not the code
- test_blades_stand_on_the_endwall_they_are_counted_over: and take room on it
- test_the_loss_is_the_right_size: a units-and-blunders guard
- test_values_survive_the_case_file: floats and NaN both
"""

import dataclasses

import numpy as np
import pytest
import yaml

from test_blade import build
from test_cli import RUN_CASE
from test_mesh import MESH, TIP
from turbigen import Config, Result, SurfaceDissipation, case, cli, metric, mixout, util

TIP_ROW = 1
"""The row of the two-row fixture that has a clearance gap."""


@pytest.fixture(scope="module")
def gapped():
    """The two-row tip-gap machine on its initial guess, never marched."""
    config = build(blades=TIP, mesh=MESH)
    _, machine, grid = cli.prepare(config)
    actual, Ds_mix = mixout.mean_line(grid, machine)
    result = Result(machine=machine, grid=grid, actual=actual, Ds_mix=Ds_mix)
    return config, result


@pytest.fixture(scope="module")
def solved():
    """The fast cascade, marched briefly for a real field and a real loss."""
    config = Config.from_dict(yaml.safe_load(RUN_CASE))
    _, machine, grid = cli.prepare(config)
    history = config.solver.solve(grid)
    actual, Ds_mix = mixout.mean_line(grid, machine)
    result = Result(
        machine=machine,
        grid=grid,
        actual=actual,
        Ds_mix=Ds_mix,
        converged=True,
        history=history,
    )
    return config, result


#
# THE RECONSTRUCTION
#
# The edge velocity is where the frame and the isentropic expansion both live,
# and it is the one part of this with an exactly known answer: expanded from a
# point's own entropy, a point must come back at its own speed.
#


def _wall_relative_speed(cut):
    """Speed relative to the wall the cut was taken from, from the components."""
    return np.sqrt(
        cut.Vx**2 + cut.Vr**2 + (cut.Vt - float(cut.Omega) * cut.r) ** 2
    )


def test_edge_velocity_is_the_speed_relative_to_the_wall(gapped):
    """Expanded from a node's own entropy, that node comes back at its speed.

    Nothing is idealised when the reference entropy is the local one, so the
    isentropic velocity collapses to the real one --- which pins the expansion,
    the frame and the algebra together against a value known in advance.
    """
    _, result = gapped
    cut = util.cut_blade_surfs(result.grid)[TIP_ROW][0]

    node = (cut.shape[0] // 3, cut.shape[1] // 3)
    Vs = metric.isentropic_velocity(cut, float(cut.s[node]))

    assert float(cut.Omega) > 0.0, "a rotor blade should turn"
    assert Vs[node] == pytest.approx(float(_wall_relative_speed(cut)[node]), rel=1e-4)


def test_a_stationary_casing_is_measured_in_the_absolute_frame(gapped):
    """The same, on the one wall whose frame is not its block's.

    The casing over a clearance gap stands still while the row turns under it,
    so the boundary layer on it grows in the absolute velocity.
    """
    _, result = gapped
    casing = util.cut_endwalls(result.grid)[TIP_ROW][1]

    node = (casing.shape[0] // 3, casing.shape[1] // 3)
    Vs = metric.isentropic_velocity(casing, float(casing.s[node]))

    assert float(casing.Omega) == 0.0
    V_abs = np.sqrt(casing.Vx**2 + casing.Vr**2 + casing.Vt**2)
    assert Vs[node] == pytest.approx(float(V_abs[node]), rel=1e-4)


#
# THE CONTRACT
#


@pytest.mark.parametrize(
    "missing", ["grid", "machine", "actual"], ids=["no_grid", "no_machine", "no_actual"]
)
def test_nothing_to_measure_returns_nothing(solved, missing):
    """An observation of a field that is not there is not an error."""
    config, result = solved
    result = dataclasses.replace(result, **{missing: None})

    assert SurfaceDissipation().evaluate(config, result) == {}


def test_a_diverged_march_is_not_measured(solved):
    """A blown-up field is NaN throughout and no state can be found in it."""
    config, result = solved
    history = result.history.copy()
    history.diverged = True

    values = SurfaceDissipation().evaluate(
        config, dataclasses.replace(result, history=history)
    )

    assert values == {}


def test_keys_and_shape(gapped):
    """Shaped like the mean line, but over surface type rather than station."""
    config, result = gapped
    values = SurfaceDissipation().evaluate(config, result)

    assert set(values) == {"Sdot_surf", "A_surf", "Vcu_surf"}
    for name, value in values.items():
        assert np.shape(value) == (2, result.grid.n_row), name
        assert np.all(np.isfinite(value)), name


#
# WHAT THE NUMBERS OBEY
#


def test_entropy_rate_is_linear_in_the_coefficient(solved):
    """Exactly linear, which is why a second instance would measure nothing new.

    Area and the velocity cube are properties of the surface and the flow over
    it, so the coefficient cannot touch them.
    """
    config, result = solved

    one = SurfaceDissipation(Cd=0.002).evaluate(config, result)
    two = SurfaceDissipation(Cd=0.004).evaluate(config, result)

    np.testing.assert_allclose(two["Sdot_surf"], 2.0 * np.asarray(one["Sdot_surf"]))
    np.testing.assert_allclose(two["A_surf"], one["A_surf"])
    np.testing.assert_allclose(two["Vcu_surf"], one["Vcu_surf"])


def test_endwall_area_counts_every_passage(solved):
    """One passage is meshed; the annulus holds `Nb` of them.

    A scaling law rather than a comparison with a geometric estimate, so it is
    exact: an integral that forgot the blade count would not move at all.

    Endwalls only, and not because the blades are treated differently --- the
    count is applied on one line, shared. It is that `cut_blade_sides` shifts
    one side of the blade by a pitch so the two meet at the leading edge, and a
    pitch is a blade count, so changing the count moves the surface as well as
    the number of them. There is no version of this test for the blades that
    holds their geometry still.
    """
    config, result = solved

    one = SurfaceDissipation().evaluate(config, result)

    doubled = result.grid.copy()
    for block in doubled:
        block.set_Nb(2 * float(block.Nb))
    two = SurfaceDissipation().evaluate(
        config, dataclasses.replace(result, grid=doubled)
    )

    for name in ("Sdot_surf", "A_surf", "Vcu_surf"):
        np.testing.assert_allclose(
            two[name][0], 2.0 * np.asarray(one[name][0]), rtol=1e-6
        )


def _swept_area(annulus, m0, m1):
    """Hub and casing area between two meridional positions, from geometry alone.

    An endwall is a surface of revolution, so its area is its meridional arc
    length carried through a full turn. No mesh, no patch and no cut are
    involved, which is the point of using it to check the ones that are.
    """
    m = np.linspace(m0, m1, 4001)
    total = 0.0
    for spf in (0.0, 1.0):
        xr = annulus.evaluate_xr(m, spf)
        ds = np.hypot(np.diff(xr[0]), np.diff(xr[1]))
        r_mid = 0.5 * (xr[1][:-1] + xr[1][1:])
        total += float(np.sum(2.0 * np.pi * r_mid * ds))
    return total


def _m_of(annulus, plane):
    """Return the meridional coordinate a cut plane sits at."""
    m = np.linspace(0.0, annulus.m_max, 20001)
    x_mid = annulus.evaluate_xr(m, 0.5)[0]
    return float(np.interp(float(plane.mean(axis=0)[0]), x_mid, m))


def _whole_endwall_area(grid, i_row=0):
    """Endwall area of a row with nothing clipped off it."""
    total = 0.0
    for cut in util.cut_endwalls(grid)[i_row]:
        dA = np.linalg.norm(cut.dA_quad, axis=-1, ord=2)
        total += float(np.sum(dA)) * float(cut.Nb)
    return total


def test_the_clip_removes_exactly_the_ducts(solved):
    """What the clip takes off is duct endwall, and the geometry says how much.

    A duct carries no blades, so its endwall is the whole annulus swept through
    a turn and the comparison is an equality rather than a bound. That makes
    this the tight check on area: a lost blade count, a mis-taken face area or
    a clip at the wrong place would each show up here at once.
    """
    config, result = solved
    annulus = result.machine.annulus
    planes = annulus.cut_planes()

    values = SurfaceDissipation().evaluate(config, result)
    removed = _whole_endwall_area(result.grid) - float(values["A_surf"][0][0])

    ducts = _swept_area(annulus, 0.0, _m_of(annulus, planes[0])) + _swept_area(
        annulus, _m_of(annulus, planes[-1]), annulus.m_max
    )

    assert removed == pytest.approx(ducts, rel=0.02)


def test_blades_stand_on_the_endwall_they_are_counted_over(solved):
    """Inside a row the endwall is the pitch less the blades' own footprint.

    So the area kept must fall short of a full sweep, and only by the room the
    blades take up: a bound in both directions rather than an equality, because
    what separates them is a blade thickness this does not try to predict.
    """
    config, result = solved
    annulus = result.machine.annulus
    planes = annulus.cut_planes()

    kept = float(SurfaceDissipation().evaluate(config, result)["A_surf"][0][0])
    row = _swept_area(annulus, _m_of(annulus, planes[0]), _m_of(annulus, planes[-1]))

    assert kept < row, "the blades take up no endwall at all"
    assert kept > 0.5 * row, "the blades take up most of the endwall"


def test_the_loss_is_the_right_size(solved):
    """A guard against blunders, not a validation of the correlation.

    Surface dissipation is one term of the profile loss, so it must be a real
    fraction of what the mean line measured across the same control volume and
    cannot exceed it. The band is wide on purpose: what it catches is a missing
    blade count, a velocity that was not cubed, a frame lost, or a temperature
    that turned out to be an enthalpy.
    """
    config, result = solved

    values = SurfaceDissipation().evaluate(config, result)
    mdot = float(np.atleast_1d(result.actual.mdot).ravel()[0])
    predicted = float(np.nansum(values["Sdot_surf"])) / mdot

    measured = float(result.actual.flat.s[-1] - result.actual.flat.s[0])

    assert 0.02 * measured < predicted < measured


#
# WHAT IS KEPT
#


def test_values_survive_the_case_file(solved, tmp_path):
    """Written as plain numbers, and read back as the same ones."""
    config, result = solved
    config = dataclasses.replace(config, metrics=(SurfaceDissipation(),))
    result = dataclasses.replace(result, metrics=metric.measure(config, result))

    assert result.metrics, "the metric has to have run for this to say anything"

    path = tmp_path / "case.yaml"
    case.write(path, config, result)
    _, read_back = case.read(path)

    for name, value in result.metrics.items():
        np.testing.assert_allclose(read_back.metrics[name], value)
    assert set(read_back.metrics) == {"Sdot_surf", "A_surf", "Vcu_surf"}

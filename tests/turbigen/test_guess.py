"""Tests for the initial flow field written into a fresh grid.

The guess is deliberately crude, so there is little to assert about its
accuracy. What is worth pinning down is that it lands on the design it came
from, that it depends on where a node is and not on how the grid lines run, the
two ways it can silently land somewhere else (the wrong equation of state, or
energy measured from a datum the grid does not share), and that each of its
optional parts does what it says.

Test cases:
- test_guess_reproduces_the_mean_line: at the stations, at mid-span
- test_relative_flow_leaves_each_row_at_its_design_angle: the vortex law
- test_guess_leaves_the_grid_ready_to_solve: finite, positive state everywhere
- test_guess_keeps_the_reference_scales_the_mesher_set
- test_guess_is_independent_of_the_datum_it_is_built_against
- test_guess_is_independent_of_the_grid_lines: a block reversed in i
- test_flow_runs_along_the_hub_and_casing
- test_swirl_varies_across_the_span
- test_blend_turns_the_flow_onto_the_blade_surfaces
- test_blend_fades_out_inside_a_thinned_blade: a tip pinch's knife edge
- test_loading_puts_the_low_pressure_on_the_suction_side
- test_loading_is_continuous_round_the_blade
- test_a_grid_without_a_fluid_is_refused
"""

import copy

import numpy as np
import pytest
from test_blade import build
from test_mesh import MESH

from turbigen import guess


@pytest.fixture(scope="module")
def machine():
    return build(mesh=MESH).design()


@pytest.fixture(scope="module")
def meshed(machine):
    return build(mesh=MESH).mesh.mesh(machine)


@pytest.fixture
def grid(meshed):
    """A fresh grid each time, since applying a guess writes into it."""
    return copy.deepcopy(meshed)


def test_guess_reproduces_the_mean_line(machine):
    """At each station, at mid-span, the design's state and velocity.

    Swirl to within the difference between the mid-span radius the vortex
    law is evaluated at and the root-mean-square one the mean line is.
    """
    field = guess._Field(machine)
    flat = machine.mean_line.flat
    m = np.arange(1, flat.size + 1, dtype=float)
    spf = np.full_like(m, 0.5)
    r = machine.annulus.evaluate_xr(m, spf)[1]

    P, T, Vxrt = field.evaluate(m, spf, r)

    np.testing.assert_allclose(P, flat.P, rtol=1e-6)
    np.testing.assert_allclose(T, flat.T, rtol=1e-6)
    np.testing.assert_allclose(np.hypot(Vxrt[:, 0], Vxrt[:, 1]), flat.Vm, rtol=1e-6)
    np.testing.assert_allclose(Vxrt[:, 2], flat.Vt, rtol=0.02, atol=1.0)


@pytest.mark.parametrize("spf", [0.1, 0.5, 0.9])
def test_relative_flow_leaves_each_row_at_its_design_angle(machine, spf):
    """At each row's edges, the relative angle its blade was designed for."""
    field = guess._Field(machine)
    for i, row in enumerate(machine.rows):
        m = np.array([2.0 * i + 1.0, 2.0 * i + 2.0])
        spf_now = np.full(2, spf)
        r = machine.annulus.evaluate_xr(m, spf_now)[1]
        _, _, Vxrt = field.evaluate(m, spf_now, r)
        Omega = machine.mean_line.Omega[0, i]
        Alpha = np.degrees(
            np.arctan((Vxrt[:, 2] - Omega * r) / np.hypot(Vxrt[:, 0], Vxrt[:, 1]))
        )
        design = machine.spanwise.Alpha_rel(
            machine.mean_line[:, i], i, row.blade.spf, [spf]
        )
        np.testing.assert_allclose(Alpha, design[0], atol=1.0)


def test_guess_leaves_the_grid_ready_to_solve(machine, grid):
    guess.apply(grid, machine)

    for block in grid:
        assert np.isfinite(block.T).all()
        assert (block.T > 0.0).all()
        assert (block.P > 0.0).all()
        assert (block.rho > 0.0).all()
        assert np.isfinite(block.mu_turb).all()


def test_guess_keeps_the_reference_scales_the_mesher_set(machine, grid):
    """The guess writes state into the grid's fluid and never replaces it.

    Replaced, the scales and datum the mesher chose for the design would be
    lost, and every flow state written afterwards stored against the wrong
    ones. Scalars snapshotted, not the fluid object, which would compare equal
    to itself even if changed in place.
    """
    fields = ("rho_ref", "V_ref", "Rgas_ref", "P_dtm", "T_dtm")
    before = {name: getattr(grid[0].fluid, name) for name in fields}

    guess.apply(grid, machine)

    after = {name: getattr(grid[0].fluid, name) for name in fields}
    assert after == before


def test_guess_is_independent_of_the_datum_it_is_built_against(machine, grid):
    """Two datums, one physical state.

    Conserved energy is measured from the datum where internal energy is zero,
    so a guess that carried conserved variables between fluids would
    reinterpret them, by a hundred kelvin for a datum moved this far. Setting
    pressure, temperature and velocity is datum-independent.
    """
    shifted = copy.deepcopy(grid)
    shifted.set_fluid(grid[0].fluid.change_datum(P_dtm=2.0e5, T_dtm=500.0))

    guess.apply(grid, machine)
    guess.apply(shifted, machine)

    assert shifted[0].fluid.T_dtm != grid[0].fluid.T_dtm
    for a, b in zip(grid, shifted):
        np.testing.assert_allclose(b.T, a.T, rtol=1e-5)
        np.testing.assert_allclose(b.P, a.P, rtol=1e-5)


def test_guess_is_independent_of_the_grid_lines(machine, grid):
    """A block with its `i` reversed gets the same field, reversed.

    Which a guess aligned with the grid lines would not: it would run the flow
    upstream.
    """
    reversed_grid = copy.deepcopy(grid)
    block = reversed_grid[0]
    block.set_x(np.ascontiguousarray(block.x[::-1]))
    block.set_r(np.ascontiguousarray(block.r[::-1]))
    block.set_t(np.ascontiguousarray(block.t[::-1]))

    guess.apply(grid, machine)
    guess.apply(reversed_grid, machine)

    for name in ("P", "T", "Vx", "Vr", "Vt"):
        np.testing.assert_allclose(
            getattr(reversed_grid[0], name)[::-1],
            getattr(grid[0], name),
            rtol=1e-5,
            atol=1e-3,
            err_msg=name,
        )


def test_flow_runs_along_the_hub_and_casing(machine, grid):
    """Meridional velocity along the lines of constant span fraction, so
    parallel to the endwalls wherever they slope."""
    guess.apply(grid, machine)
    annulus = machine.annulus

    for block in grid:
        for j in (0, -1):
            x = np.asarray(block.x[:, j, :], dtype=float)
            r = np.asarray(block.r[:, j, :], dtype=float)
            m, spf = annulus.evaluate_mspf(block.x[:, j, :], block.r[:, j, :])
            dm = 1e-4
            tangent = annulus.evaluate_xr(m + dm, spf) - annulus.evaluate_xr(
                m - dm, spf
            )
            cross = block.Vx[:, j, :] * tangent[1] - block.Vr[:, j, :] * tangent[0]
            Vm = np.hypot(block.Vx[:, j, :], block.Vr[:, j, :])
            assert np.all(np.abs(cross) / np.hypot(*tangent) < 1e-3 * Vm + 1e-6)
            del x, r


def test_swirl_varies_across_the_span(machine, grid):
    """Uniform round the annulus, but not up it: the vortex law each blade was
    designed with carries through to the guess."""
    guess.apply(grid, machine)
    block = grid[0]

    Vt_hub = np.abs(block.Vt[:, 0, :]).mean()
    Vt_cas = np.abs(block.Vt[:, -1, :]).mean()

    assert Vt_hub > Vt_cas


def _passages(machine, block, i_row):
    """Row `i_row`'s passage lookup over `block`, as the guess makes it."""
    field = guess._Field(machine)
    m, spf = field.mspf(block.x.reshape(-1), block.r.reshape(-1))
    r = np.asarray(block.r, dtype=float).reshape(-1)
    t = np.asarray(block.t, dtype=float).reshape(-1)
    return field.passage(i_row, m, spf, r, t)


def _Alpha_rel(block, Omega, index):
    W_t = block.Vt.reshape(-1)[index] - Omega * block.r.reshape(-1)[index]
    W_m = np.hypot(block.Vx.reshape(-1)[index], block.Vr.reshape(-1)[index])
    return np.arctan2(W_t, W_m)


def test_blend_turns_the_flow_onto_the_blade_surfaces(machine, meshed):
    """Next to a surface mid-chord, the relative flow runs at the surface's own
    angle, and the speed is what the design gives there before the blend."""
    blended = copy.deepcopy(meshed)
    guess.apply(blended, machine)
    field = guess._Field(machine)

    checked = 0
    for i_row, (blocks, Omega) in enumerate(
        zip(blended.rows, machine.mean_line.Omega[0])
    ):
        for block in blocks:
            passage = _passages(machine, block, i_row)
            if passage is None:
                continue
            mid = (np.abs(passage["m_row"] - 0.5) < 0.3) & (
                np.abs(passage["spf"] - 0.5) < 0.3
            )
            for side, near in enumerate(
                (passage["frac"] < 0.01, passage["frac"] > 0.99)
            ):
                pick = mid & near
                if not pick.any():
                    continue
                index = passage["index"][pick]
                surface = passage["Alpha_surface"][pick, side]
                after = _Alpha_rel(block, Omega, index)
                np.testing.assert_allclose(after, surface, atol=np.radians(2.0))
                checked += 1

            r = block.r.reshape(-1).astype(float)
            m, spf = field.mspf(block.x.reshape(-1), block.r.reshape(-1))
            _, _, W = field.evaluate(m, spf, r)
            W[:, 2] -= Omega * r
            b = block
            W_now = np.stack((b.Vx, b.Vr, b.Vt - Omega * b.r), axis=-1).reshape(-1, 3)
            np.testing.assert_allclose(
                np.linalg.norm(W_now, axis=-1), np.linalg.norm(W, axis=-1), rtol=1e-4
            )
    assert checked


def test_blend_fades_out_inside_a_thinned_blade(machine):
    """A node a mesh has put inside the designed blade, as a tip pinch does, is
    turned less the deeper it lies, and not at all at mid-thickness, so the
    two sides of a knife edge are not snapped to surfaces 7 degrees apart."""
    field = guess._Field(machine)
    row = field.rows[1]
    m_row, spf = np.full(4, 0.5), np.full(4, 0.5)
    m_LE = 3.0
    theta = row.blade._surface_theta(np.array([[0.5, 0.5]]))[0]
    upper, lower = theta.max(), theta.min()
    t = np.array(
        (upper, lower + row.pitch, 0.5 * (upper + lower), 0.75 * upper + 0.25 * lower)
    )
    # Radius only scales the surface angles, which are not checked here
    passage = field.passage(1, m_row + m_LE, spf, np.ones(4), t)

    np.testing.assert_allclose(passage["solid"], (1.0, 1.0, 0.0, 0.5), atol=1e-6)


def test_a_grid_without_a_fluid_is_refused(machine):
    """The guess reads the grid's equation of state, so the mesher must have
    set one first. It does, but the dependency is worth stating."""
    import ember.grid

    bare = ember.grid.Grid([ember.block.Block(shape=(3, 3, 3))])

    with pytest.raises(ValueError, match="fluid"):
        guess.apply(bare, machine)

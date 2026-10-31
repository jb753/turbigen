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
    from turbigen.blade import _Alpha_rel

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
        design = _Alpha_rel(machine.mean_line[:, i], [spf], row.blade.vortex_exponent)
        np.testing.assert_allclose(Alpha, design[0], atol=1.0)


def test_guess_leaves_the_grid_ready_to_solve(machine, grid):
    guess.apply(grid, machine, loading=True)

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
    angle, and the speed is what it was without the blend."""
    plain = copy.deepcopy(meshed)
    blended = copy.deepcopy(meshed)
    guess.apply(plain, machine, blend=False)
    guess.apply(blended, machine)

    checked = 0
    for i_row, (blocks, Omega) in enumerate(
        zip(blended.rows, machine.mean_line.Omega[0])
    ):
        for block in blocks:
            before = plain[[b is block for b in blended].index(True)]
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

            def speed(b):
                W = np.stack((b.Vx, b.Vr, b.Vt - Omega * b.r), axis=-1).reshape(-1, 3)
                return np.linalg.norm(W, axis=-1)

            np.testing.assert_allclose(speed(block), speed(before), rtol=1e-4)
    assert checked


def test_loading_puts_the_low_pressure_on_the_suction_side(machine, meshed):
    """Mid-chord, the suction side is at lower pressure than the pressure side
    across the passage, and the pitchwise mean barely moves."""
    plain = copy.deepcopy(meshed)
    loaded = copy.deepcopy(meshed)
    guess.apply(plain, machine, blend=False)
    guess.apply(loaded, machine, blend=False, loading=True)

    for i, row in enumerate(machine.rows):
        field = guess._Field(machine)
        for block, before in zip(loaded, plain):
            m, spf = field.mspf(block.x.reshape(-1), block.r.reshape(-1))
            m_row = m - (2.0 * i + 1.0)
            frac = row.evaluate_pitch_fraction(m_row, spf, block.t.reshape(-1))
            mid = (np.abs(m_row - 0.5) < 0.1) & (np.abs(spf - 0.5) < 0.2)
            P = block.P.reshape(-1)
            suction = mid & (frac < 0.2)
            pressure = mid & (frac > 0.8)
            if not (suction.any() and pressure.any()):
                continue
            assert P[suction].mean() < P[pressure].mean()
            mean = P[mid & np.isfinite(frac)].mean()
            mean0 = before.P.reshape(-1)[mid & np.isfinite(frac)].mean()
            assert mean == pytest.approx(mean0, rel=0.02)


def test_a_grid_without_a_fluid_is_refused(machine):
    """The guess reads the grid's equation of state, so the mesher must have
    set one first. It does, but the dependency is worth stating."""
    import ember.grid

    bare = ember.grid.Grid([ember.block.Block(shape=(3, 3, 3))])

    with pytest.raises(ValueError, match="fluid"):
        guess.apply(bare, machine)


def test_loading_is_continuous_round_the_blade(machine, meshed):
    """The loading tapers to nothing at both edges, where the two surfaces
    meet, so the pressure is continuous round them. A sawtooth with loading at
    the leading edge would jump by its whole height between the last suction
    surface node and the first pressure surface one."""
    field = guess._Field(machine)
    checked = 0
    for i_row in range(len(machine.rows)):
        for block in meshed:
            passage = _passages(machine, block, i_row)
            if passage is None:
                continue
            dP = np.abs(field.load(i_row, passage))
            m_row = passage["m_row"]
            edges = (m_row < 0.005) | (m_row > 0.995)
            if not edges.any():
                continue
            assert dP[edges].max() < 0.05 * dP.max()
            checked += 1
    assert checked

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
- test_no_relative_flow_through_the_blades: on the wall nodes
- test_flow_into_the_blades_fades_out_away_from_them
- test_flow_beyond_the_reach_of_the_blades_is_untouched
- test_a_blade_corner_is_clear_of_both_walls: blade and endwall at once
- test_the_wall_step_holds_pressure_and_temperature
- test_a_grid_without_a_fluid_is_refused
"""

import copy

import numpy as np
import pytest
from test_blade import build
from test_mesh import MESH

from turbigen import guess

F32 = float(np.finfo(np.float32).eps)
"""Single-precision machine epsilon, the grid's storage [--]."""


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
    upstream. The patches are reversed with the coordinates, so that the walls,
    which the guess reads off them, stay on the same nodes.
    """
    reversed_grid = copy.deepcopy(grid)
    block = reversed_grid[0]
    block.set_x(np.ascontiguousarray(block.x[::-1]))
    block.set_r(np.ascontiguousarray(block.r[::-1]))
    block.set_t(np.ascontiguousarray(block.t[::-1]))
    ni = block.shape[0]
    for patch in block.patches:
        ist, ien = (int(i) for i in patch.ijk_lim_abs[0])
        patch.set_i_lim((ni - 1 - ien, ni - 1 - ist))

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


def test_blend_turns_the_flow_onto_the_blade_surfaces(machine, meshed, monkeypatch):
    """Next to a surface mid-chord, the relative flow runs at the surface's own
    angle, and the speed is what the design gives there before the blend.

    With the wall step off, which slows the flow into a wall after the blend,
    so that this is the blend's own doing.
    """
    monkeypatch.setattr(guess, "_unblock_walls", lambda *args: None)
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


@pytest.fixture
def unblocked(machine, meshed, monkeypatch):
    """The same grid guessed with the wall step and without it, and the
    relative angular velocity and pitch of each block."""
    with_step = copy.deepcopy(meshed)
    guess.apply(with_step, machine)
    without = copy.deepcopy(meshed)
    monkeypatch.setattr(guess, "_unblock_walls", lambda *args: None)
    guess.apply(without, machine)
    rows = [
        (Omega, 2.0 * np.pi / float(row.n_blade))
        for Omega, row in zip(machine.mean_line.Omega[0], machine.rows)
    ]
    row_of = [rows[i] for i, blocks in enumerate(meshed.rows) for _ in blocks]
    return with_step, without, row_of


def _W(block, Omega):
    """Relative velocity in Cartesians, (ni, nj, nk, 3)."""
    r, t = (np.asarray(a, dtype=float) for a in (block.r, block.t))
    Vx, Vr, Vt = (np.asarray(a, dtype=float) for a in (block.Vx, block.Vr, block.Vt))
    Wt = Vt - Omega * r
    c, s = np.cos(t), np.sin(t)
    return np.stack((Vx, Vr * c - Wt * s, Vr * s + Wt * c), axis=-1)


def _xyz(block):
    x, r, t = (np.asarray(a, dtype=float) for a in (block.x, block.r, block.t))
    return np.stack((x, r * np.cos(t), r * np.sin(t)), axis=-1)


def _blade_faces(block, pitch_theta):
    """Each blade surface of an H block, as the pitchwise faces' wall nodes.

    Yields the face's `k`, the wall nodes as an (ni, nj) mask from the solver's
    own wall mask, the unit normal into the block at each, the distance from
    the face along the `k` lines (ni, nj, nk), counted from that face, and the
    fade weight there.
    """
    xyz = _xyz(block)
    r = np.asarray(block.r, dtype=float)
    for key, k in (("wallk1", 0), ("wallnk", -1)):
        wall = guess._wall_nodes(block.ijk_wall_conv[key][:, :, 0])
        if not wall.any():
            continue
        line = xyz if k == 0 else xyz[:, :, ::-1]
        r_line = r if k == 0 else r[:, :, ::-1]
        surface = line[:, :, 0]
        n = np.cross(guess._tangent(surface, 0), guess._tangent(surface, 1))
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        n *= np.sign(np.sum(n * (line[:, :, 1] - surface), axis=-1, keepdims=True))
        step = np.linalg.norm(np.diff(line, axis=2), axis=-1)
        dist = np.concatenate(
            (np.zeros(wall.shape + (1,)), np.cumsum(step, axis=2)), axis=2
        )
        w = np.clip(1.0 - dist / (guess.WALL_PITCH * pitch_theta * r_line), 0.0, 1.0)
        w *= wall[:, :, None]
        if k == -1:
            dist, w = dist[:, :, ::-1], w[:, :, ::-1]
        yield k, wall, n, dist, w


def test_no_relative_flow_through_the_blades(unblocked):
    """On a blade's wall nodes, none of the relative flow goes through it, where
    without the step it arrives round the nose at full speed."""
    with_step, without, row_of = unblocked

    worst_before = 0.0
    for block, plain, (Omega, pitch) in zip(with_step, without, row_of):
        W, W0 = _W(block, Omega), _W(plain, Omega)
        for k, wall, n, *_ in _blade_faces(block, pitch):
            through = np.sum(W[:, :, k] * n, axis=-1)[wall]
            speed = np.linalg.norm(W[:, :, k], axis=-1)[wall]
            assert np.all(np.abs(through) < 1e-4 * speed + 1e-3)
            worst_before = max(
                worst_before, np.abs(np.sum(W0[:, :, k] * n, axis=-1)[wall]).max()
            )
    assert worst_before > 1.0


def test_flow_into_the_blades_fades_out_away_from_them(unblocked):
    """Off the wall, a fraction of the flow into a blade is taken out, falling
    linearly from all of it at the wall to none at the reach; flow away from a
    blade is left alone."""
    with_step, without, row_of = unblocked

    checked = 0
    for block, plain, (Omega, pitch) in zip(with_step, without, row_of):
        W, W0 = _W(block, Omega), _W(plain, Omega)
        for k, wall, n, dist, w in _blade_faces(block, pitch):
            off = (w > 0.0) & (dist > 0.0)
            normal = np.broadcast_to(n[:, :, None], W.shape)
            before = np.sum(W0 * normal, axis=-1)[off]
            after = np.sum(W * normal, axis=-1)[off]
            expected = before - w[off] * np.minimum(before, 0.0)
            speed = np.linalg.norm(W0, axis=-1)[off]
            np.testing.assert_allclose(after, expected, atol=1e-4 * speed.max())
            checked += np.count_nonzero(off & (np.sum(W0 * normal, axis=-1) < -1.0))
    assert checked


def test_flow_beyond_the_reach_of_the_blades_is_untouched(unblocked):
    """Further than the reach from every blade, the guess is as without the
    step, and the step reaches only a few layers off the blades.

    As to single-precision rounding, not exactly: the block's state is stored
    as conserved variables, which setting it, on every node at once, rounds
    afresh.
    """
    with_step, without, row_of = unblocked

    for block, plain, (Omega, pitch) in zip(with_step, without, row_of):
        reached = np.zeros(block.shape, dtype=bool)
        for *_, w in _blade_faces(block, pitch):
            reached |= w > 0.0
        assert reached.any()
        assert reached.mean() < 0.2
        for name in ("P", "T", "Vx", "Vr", "Vt"):
            a, b = (np.asarray(getattr(g, name))[~reached] for g in (block, plain))
            np.testing.assert_allclose(
                a, b, rtol=4.0 * F32, atol=4.0 * F32 * np.abs(b).max(), err_msg=name
            )


def test_a_blade_corner_is_clear_of_both_walls(machine, unblocked):
    """Where a blade meets the hub or casing, the flow neither goes through the
    blade nor is pushed into the endwall: it runs along the annulus's line of
    constant span fraction, as everywhere else on an endwall."""
    with_step, _, row_of = unblocked
    annulus = machine.annulus

    for block, (Omega, pitch) in zip(with_step, row_of):
        W = _W(block, Omega)
        for k, wall, n, *_ in _blade_faces(block, pitch):
            for j in (0, -1):
                corner = wall[:, j]
                speed = np.linalg.norm(W[corner, j, k], axis=-1)
                through = np.sum(W[corner, j, k] * n[corner, j], axis=-1)
                assert np.all(np.abs(through) < 1e-4 * speed + 1e-3)

                x, r = block.x[corner, j, k], block.r[corner, j, k]
                m, spf = annulus.evaluate_mspf(x, r)
                dm = 1e-4
                tangent = annulus.evaluate_xr(m + dm, spf) - annulus.evaluate_xr(
                    m - dm, spf
                )
                tangent = tangent / np.hypot(*tangent)
                Vx, Vr = block.Vx[corner, j, k], block.Vr[corner, j, k]
                cross = Vx * tangent[1] - Vr * tangent[0]
                assert np.all(np.abs(cross) < 1e-3 * speed + 1e-3)


def test_the_wall_step_holds_pressure_and_temperature(unblocked):
    """Only the velocity changes: raised towards stagnation in a layer this
    thin, the pressure would have nothing to hold it up, and the solver would
    release it as a pulse round the nose."""
    with_step, without, row_of = unblocked

    for block, plain, (Omega, _) in zip(with_step, without, row_of):
        moved = np.any(_W(block, Omega) != _W(plain, Omega), axis=-1)
        assert moved.any()
        for name in ("P", "T"):
            a, b = (np.asarray(getattr(g, name), dtype=float) for g in (block, plain))
            np.testing.assert_allclose(a, b, rtol=4.0 * F32, err_msg=name)


def test_a_grid_without_a_fluid_is_refused(machine):
    """The guess reads the grid's equation of state, so the mesher must have
    set one first. It does, but the dependency is worth stating."""
    import ember.grid

    bare = ember.grid.Grid([ember.block.Block(shape=(3, 3, 3))])

    with pytest.raises(ValueError, match="fluid"):
        guess.apply(bare, machine)

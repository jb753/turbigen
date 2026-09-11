"""Tests for seeding a fresh grid from a converged neighbour's field.

`warm_field.Seed.apply` puts a neighbour's `restart.npz` onto the new grid,
measures what that field is once it is there, and adds the perturbation that
carries it onto the new design's nominal mean line.

The contract is measured the way the code measures it: mix the seeded grid out
and see where it lands. Asserting on block-end means, as an earlier version of
these tests did, checks the wrong planes -- the mean line is cut a blade-chord
offset into the gaps, not at the ends of the blocks.

Test cases:
- test_apply_is_the_identity_when_the_design_is_unchanged: every delta is zero
- test_apply_moves_the_levels_onto_the_target_mean_line: mix out and compare
- test_apply_beats_the_cold_guess: the seed is closer than what it replaced
- test_apply_leaves_the_grid_ready_to_solve: finite, positive state everywhere
- test_apply_survives_swirl_through_zero: the case a ratio could not do
"""

import dataclasses

import numpy as np
from test_blade import build
from test_mesh import MESH

from turbigen import guess, mixout, restart, warm_field


def _config(psi):
    cfg = build(mesh=MESH)
    return dataclasses.replace(
        cfg, mean_line=dataclasses.replace(cfg.mean_line, psi=psi)
    )


def _neighbour_restart(tmp_path, cfg, name="neighbour.npz"):
    """Solve nothing: write the meridional guess of `cfg` as a `restart.npz`.

    The guess reproduces the mean line station for station, so the saved field
    is a stand-in for a converged neighbour's -- uniform across span and pitch
    where a real one would not be, but sitting at the levels of a design that
    is not the one being seeded, which is what the correction has to move.
    """
    machine = cfg.design()
    grid = cfg.mesh.mesh(machine)
    guess.apply(grid, machine)
    path = tmp_path / name
    restart.save(path, grid)
    return path


def _seeded(tmp_path, neighbour_psi, target_psi):
    """Return the target machine and a grid seeded from the neighbour."""
    field = _neighbour_restart(tmp_path, _config(neighbour_psi))

    target_cfg = _config(target_psi)
    machine = target_cfg.design()
    grid = target_cfg.mesh.mesh(machine)
    guess.apply(grid, machine)

    warm_field.Seed(field).apply(grid, machine)

    return machine, grid


def _station_error(machine, grid):
    """Return the fractional error of each mixed-out station, per quantity."""
    achieved, _ = mixout.mean_line(grid, machine)
    nominal = machine.mean_line.flat
    actual = achieved.flat

    return {
        name: np.abs(
            (np.asarray(getattr(actual, name)) - np.asarray(getattr(nominal, name)))
            / np.asarray(getattr(nominal, name))
        )
        for name in ("P", "T", "Vx", "mdot")
    }


def test_apply_is_the_identity_when_the_design_is_unchanged(tmp_path):
    """Neighbour and target are the same design, so every delta is zero.

    Not quite bit-for-bit: the perturbation is measured rather than assumed, so
    what lands here is mixing-out's own residual on a field that already sits
    on the mean line -- small, and the assertion is that it stays small.
    """
    cfg = _config(1.6)
    field = _neighbour_restart(tmp_path, cfg)

    machine = cfg.design()
    grid = cfg.mesh.mesh(machine)
    guess.apply(grid, machine)
    before = [np.asarray(block.P).copy() for block in grid]

    warm_field.Seed(field).apply(grid, machine)

    for was, block in zip(before, grid):
        np.testing.assert_allclose(np.asarray(block.P), was, rtol=5e-3)


def test_apply_moves_the_levels_onto_the_target_mean_line(tmp_path):
    """Seeded from a different design, the grid mixes out onto the target."""
    machine, grid = _seeded(tmp_path, neighbour_psi=1.4, target_psi=1.8)

    error = _station_error(machine, grid)

    # Two orders of magnitude tighter than where an uncorrected interpolation
    # of the same field lands, which is a couple of percent on pressure and
    # over ten on mass flow.
    for name in ("P", "T", "Vx", "mdot"):
        assert error[name].max() < 5e-3


def test_apply_beats_the_cold_guess(tmp_path):
    """The correction is worth making: closer to the target than not making it.

    The neighbour's field is a whole design away, so a seed that did nothing
    but interpolate it across would land on the *neighbour's* mean line. This
    is the test that the perturbation is pointed the right way.
    """
    field = _neighbour_restart(tmp_path, _config(1.4))

    target_cfg = _config(1.8)
    machine = target_cfg.design()

    grid = target_cfg.mesh.mesh(machine)
    guess.apply(grid, machine)
    restart.apply(grid, field)
    uncorrected = _station_error(machine, grid)

    grid = target_cfg.mesh.mesh(machine)
    guess.apply(grid, machine)
    warm_field.Seed(field).apply(grid, machine)
    corrected = _station_error(machine, grid)

    for name in ("P", "T", "Vx"):
        assert corrected[name].max() < uncorrected[name].max()


def test_apply_leaves_the_grid_ready_to_solve(tmp_path):
    _, grid = _seeded(tmp_path, neighbour_psi=1.4, target_psi=1.9)

    for block in grid:
        assert np.isfinite(block.T).all() and (block.T > 0.0).all()
        assert np.isfinite(block.P).all() and (block.P > 0.0).all()
        assert (block.rho > 0.0).all()
        assert np.isfinite(block.Vt).all()


def test_apply_survives_swirl_through_zero(tmp_path):
    """The case that motivated perturbing rather than scaling.

    Somewhere between these two designs a station's tangential velocity passes
    close to zero. A ratio would be unbounded there; a difference is not, and
    the grid has to come back solvable either way.
    """
    machine, grid = _seeded(tmp_path, neighbour_psi=1.2, target_psi=2.0)

    for block in grid:
        assert np.isfinite(block.Vt).all()
        assert np.isfinite(block.P).all() and (block.P > 0.0).all()
        assert (block.T > 0.0).all()

    assert _station_error(machine, grid)["P"].max() < 5e-3

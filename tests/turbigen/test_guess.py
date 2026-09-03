"""Tests for the initial flow field written into a fresh grid.

The guess is deliberately crude -- circumferentially uniform, taken from the
mean line along the annulus mid-span -- so there is little to assert about its
accuracy. What is worth pinning down is that it lands at all, that it lands on
the design it came from, and the two ways it can silently land somewhere else:
by carrying the wrong equation of state onto the grid, and by transferring
energy that is measured from a datum the grid does not share.

Test cases:
- test_guess_reproduces_the_mean_line: the block is the design, station by station
- test_guess_leaves_the_grid_ready_to_solve: finite, positive state everywhere
- test_guess_keeps_the_reference_scales_the_mesher_set: apply_guess_meridional
  copies the guess block's fluid onto every block, so a guess built with the
  mean line's own fluid would undo them
- test_guess_is_independent_of_the_datum_it_is_built_against: conserved energy
  is measured from a datum, so it cannot be copied between fluids
- test_swirl_varies_across_the_span: what a conserved-variable transfer implies
- test_a_grid_without_a_fluid_is_refused: the ordering the mesher guarantees
"""

import numpy as np
import pytest

from test_blade import build
from test_mesh import MESH
from turbigen import guess


@pytest.fixture(scope="module")
def machine():
    return build(mesh=MESH).design()


@pytest.fixture
def grid(machine):
    """A fresh grid each time, since applying a guess writes into it."""
    return build(mesh=MESH).mesh.mesh(machine)


def test_guess_reproduces_the_mean_line(machine, grid):
    """The block handed to ember is the design, station for station."""
    block = guess.meridional(machine, grid[0].fluid)
    flat = machine.mean_line.flat

    assert block.shape == (2 * machine.mean_line.n_row,)
    for name in ("P", "T", "Vx", "Vr", "Vt"):
        np.testing.assert_allclose(
            np.asarray(getattr(block, name)),
            np.asarray(getattr(flat, name)),
            rtol=1e-6,
            err_msg=f"{name} does not match the mean line",
        )


def test_guess_leaves_the_grid_ready_to_solve(machine, grid):
    guess.apply(grid, machine)

    for block in grid:
        assert np.isfinite(block.T).all()
        assert (block.T > 0.0).all()
        assert (block.P > 0.0).all()
        assert (block.rho > 0.0).all()
        assert np.isfinite(block.mu_turb).all()


def test_guess_keeps_the_reference_scales_the_mesher_set(machine, grid):
    """`apply_guess_meridional` copies the guess block's fluid onto every block.

    So the guess has to be built with the fluid the grid already carries. Built
    with the mean line's own, it would quietly replace the scales and datum the
    mesher chose for the design, and every flow state written afterwards would
    be stored against the wrong ones.
    """
    # Snapshot the scalars, not the fluid object: a reference would compare
    # equal to itself even if `apply` swapped the scales in place. These must
    # survive bit-for-bit, so the comparison stays exact -- the failure this
    # guards against is `apply` rebuilding from the mean line, whose float64
    # scales land a rounding step off the grid's float32 ones.
    fields = ("rho_ref", "V_ref", "Rgas_ref", "P_dtm", "T_dtm")
    before = {name: getattr(grid[0].fluid, name) for name in fields}

    guess.apply(grid, machine)

    after = {name: getattr(grid[0].fluid, name) for name in fields}
    assert after == before


def test_guess_is_independent_of_the_datum_it_is_built_against(machine, grid):
    """Two datums, one physical state.

    Conserved energy is measured from the datum where internal energy is zero,
    so copying a mean line's conserved variables into a block whose fluid has
    a different datum reinterprets them -- silently, and by a hundred kelvin
    for a datum moved as far as `get_referenced_fluid` moves it. Transferring
    pressure, temperature and velocity instead is datum-independent.
    """
    shifted = grid[0].fluid.change_datum(P_dtm=2.0e5, T_dtm=500.0)

    as_meshed = guess.meridional(machine, grid[0].fluid)
    as_shifted = guess.meridional(machine, shifted)

    assert shifted.T_dtm != grid[0].fluid.T_dtm
    np.testing.assert_allclose(
        np.asarray(as_shifted.T), np.asarray(as_meshed.T), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(as_shifted.P), np.asarray(as_meshed.P), rtol=1e-6
    )


def test_swirl_varies_across_the_span(machine, grid):
    """The guess is uniform round the annulus, but not up it.

    It is applied by copying conserved variables, and one of them is angular
    momentum, so tangential velocity falls as 1/r from hub to casing. Static
    temperature therefore ranges a little wider than the mean line does, which
    is the guess being physical rather than being wrong.
    """
    guess.apply(grid, machine)
    block = grid[0]

    Vt_hub = np.abs(block.Vt[:, 0, :]).mean()
    Vt_cas = np.abs(block.Vt[:, -1, :]).mean()

    assert Vt_hub > Vt_cas


def test_a_grid_without_a_fluid_is_refused(machine):
    """The guess reads the grid's equation of state, so the mesher must have
    set one first. It does, but the dependency is worth stating."""
    import ember.grid  # noqa: PLC0415

    bare = ember.grid.Grid([ember.block.Block(shape=(3, 3, 3))])

    with pytest.raises(ValueError, match="fluid"):
        guess.apply(bare, machine)

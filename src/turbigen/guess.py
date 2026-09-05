"""Initial flow field for a freshly meshed grid.

A grid leaves the mesher as geometry with reference scales and an equation of
state, and no flow in it. :func:`apply` writes one, so that a solver has
somewhere to start.

It is a free function rather than part of the ``run`` verb, so that the grid a
solver would start from can be built and plotted without running anything. What
is inspected is the grid itself: both the guess and the thing it is applied to
are ember objects, so nothing of ours needs to exist in between and there is no
guess class to carry data around in.

The field is circumferentially uniform, taken from the mean line along the
annulus mid-span. It is deliberately crude --- it exists to give the solver a
sane starting point, not to be right.
"""

import logging

import ember.block
import numpy as np

logger = logging.getLogger("turbigen")

REFINE_FACTOR = 50
"""Points interpolated along the guess per mean-line station.

The application is a nearest-neighbour search in the meridional plane, so the
guess needs to be a dense point cloud rather than the handful of stations a
mean line actually has, or every cell in a row would take the value of the one
station nearest it.
"""


def apply(grid, machine):
    """Write an initial flow field into `grid`, in place.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.

    """
    guess = meridional(machine, grid[0].fluid)
    grid.apply_guess_meridional(guess, refine_factor=REFINE_FACTOR)
    logger.debug(f"Applied a meridional guess from {guess.shape[0]} stations")


def meridional(machine, fluid):
    """Return the mean line as a 1D block along the annulus mid-span.

    Parameters
    ----------
    machine : Machine
        The designed geometry and the flow it was designed for.
    fluid : Fluid
        Equation of state to build the block with. This must be the one the
        grid already carries: :meth:`ember.grid.Grid.apply_guess_meridional`
        copies the guess block's fluid onto every block it touches, so a guess
        built with the mean line's own fluid would silently undo the reference
        scales the mesher set.

    Returns
    -------
    ember.block.Block
        Shape ``(2 * n_row,)``, in streamwise station order.

    """
    ml = machine.mean_line.flat
    n = ml.size

    # Mid-span coordinates at each mean-line station. Stations occupy m = 1 to
    # 2 * n_row, which is the annulus's own convention for where a row begins
    # and ends.
    m = np.arange(1, n + 1, dtype=float)
    xr = machine.annulus.evaluate_xr(m, 0.5)

    block = ember.block.Block(shape=(n,))
    block.set_fluid(fluid)
    block.set_xrt(np.stack([xr[0], xr[1], np.zeros(n)], axis=-1))

    # Pressure, temperature and velocity, rather than the conserved variables
    # straight off the mean line. Conserved energy is measured from its fluid's
    # datum, and `fluid` is the grid's, whose datum was moved to suit the
    # design -- so copying conserved across would land the guess a hundred
    # kelvin out while looking perfectly well formed.
    block.set_P_T(ml.P, ml.T)
    block.set_Vx(ml.Vx)
    block.set_Vr(ml.Vr)
    block.set_Vt(ml.Vt)

    block.set_mu_turb(np.full(n, float(np.mean(machine.mean_line.mu))))

    return block

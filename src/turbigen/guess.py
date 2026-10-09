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
from scipy.spatial import cKDTree

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
    if any(_wraps(block) for block in grid):
        # ember divides by zero aligning the flow with an O block's grid
        # lines, and those values are about to be replaced.
        with np.errstate(divide="ignore", invalid="ignore"):
            grid.apply_guess_meridional(guess, refine_factor=REFINE_FACTOR)
        _fill_wrapped_blocks(grid)
    else:
        grid.apply_guess_meridional(guess, refine_factor=REFINE_FACTOR)
    logger.debug(f"Applied a meridional guess from {guess.shape[0]} stations")


def _wraps(block):
    """Whether a block closes on itself in `i`, as one round a blade does."""
    return np.allclose(block.xrt[0], block.xrt[-1])


def _fill_wrapped_blocks(grid):
    """Give a block wrapped round a blade the flow of the blocks round it.

    ember aligns the guessed velocity with the `i` grid lines wherever a block
    has a pressure difference across its `k` faces, which is right for a
    passage whose `i` runs downstream. An O block's `i` runs round the blade
    instead: upstream along one surface, and across the pitch at the edges,
    where the meridional length of a grid line is zero and the alignment
    divides by it. So a block that wraps on itself in `i` takes its flow from
    the nearest node of the other blocks in its row, which do run downstream.
    """
    for row in grid.rows:
        wrapped = [b for b in row if _wraps(b)]
        others = [b for b in row if not any(b is w for w in wrapped)]
        if not wrapped or not others:
            continue

        # Pitchwise as a length at a mean radius, and repeated a pitch either
        # side, so the nearest node may be across a periodic boundary.
        r_ref = np.mean([b.r.mean() for b in others])
        pitch = others[0].pitch
        pts, cons, mu = [], [], []
        for b in others:
            xrt = b.xrt.reshape(-1, 3)
            for shift in (-1, 0, 1):
                pts.append(
                    np.stack(
                        (xrt[:, 0], xrt[:, 1], (xrt[:, 2] + shift * pitch) * r_ref),
                        axis=1,
                    )
                )
                cons.append(b.conserved.reshape(-1, 5))
                mu.append(b.mu_turb.reshape(-1))
        tree = cKDTree(np.concatenate(pts))
        cons = np.concatenate(cons)
        mu = np.concatenate(mu)

        for b in wrapped:
            xrt = b.xrt.reshape(-1, 3)
            _, near = tree.query(
                np.stack((xrt[:, 0], xrt[:, 1], xrt[:, 2] * r_ref), axis=1)
            )
            b.set_conserved(cons[near].reshape(*b.shape, 5))
            b.set_mu_turb(mu[near].reshape(b.shape))


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

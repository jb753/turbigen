"""Small helpers with no home of their own.

Numeric odds and ends, a table formatter, and the blade-surface cuts a surface
plot needs. Nothing here knows about a config, a design or a machine: what
lands in this module is what several parts want and none of them owns.

It exists to end the package's last dependency on the one it replaces. Until
this, `annulus`, `blade`, `machine`, `meanline` and `post` all reached into
`turbigen.util` and `turbigen.util_post` --- which meant the rebuild could not
be finished while the thing it replaces was still installed. The functions are
carried over unchanged in behaviour; `_log_ram`, at three call sites in the
cuts below, is not, for the same reason the mesher dropped its seven.
"""

import logging

import numpy as np
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid

import ember.block_util
import ember.patch

logger = logging.getLogger("turbigen")


#
# ANGLES
#
# Degrees, because that is what a designer says and what a config file holds.
# The conversions live here so that no caller has to remember which way round
# `np.radians` goes.
#


def tand(x):
    """Return the tangent of an angle in degrees."""
    return np.tan(np.radians(x))


def cosd(x):
    """Return the cosine of an angle in degrees."""
    return np.cos(np.radians(x))


#
# CUMULATIVE QUANTITIES
#
# Both keep the input length by prepending the zero the bare numpy and scipy
# versions leave out, so a cumulative quantity indexes alongside what it was
# computed from rather than one short of it.
#


def cumsum0(x, axis=None):
    """Return the cumulative sum of `x`, starting from an explicit zero."""
    return np.insert(np.cumsum(x, axis=axis), 0, 0.0, axis=axis)


def cumtrapz0(x, *args, axis=-1):
    """Return the cumulative integral of `x`, starting from an explicit zero."""
    return np.insert(cumulative_trapezoid(x, *args, axis=axis), 0, 0.0, axis=axis)


#
# CURVES
#


def arc_length(xr, axis=1):
    """Return the total arc length of a curve.

    Parameters
    ----------
    xr : array, shape (2, npts)
        Coordinates, stacked on the first axis.
    axis : int
        Axis running along the curve.

    """
    dxr = np.diff(xr, n=1, axis=axis) ** 2.0
    return np.sum(np.sqrt(np.sum(dxr, axis=0, keepdims=True)), axis=axis).squeeze()


def cum_arc_length(xr, axis=1):
    """Return the arc length to each point of a curve, starting from zero.

    Parameters
    ----------
    xr : array, shape (2, npts)
        Coordinates, stacked on the first axis.
    axis : int
        Axis running along the curve.

    """
    dxr = np.diff(xr, n=1, axis=axis) ** 2.0
    ds = np.sqrt(np.sum(dxr, axis=0, keepdims=True))
    return cumsum0(ds, axis=axis)[0]


def vecnorm(x):
    """Return the length of each vector in `x`, stacked on the first axis."""
    return np.sqrt(np.einsum("i...,i...", x, x))


def resample(x, f, mult=None):
    """Return `x` with its point count scaled by `f`, keeping its spacing.

    The relative spacing is preserved, so a clustered distribution stays
    clustered and only its resolution changes. The ends are exactly the ends
    it was given.

    Parameters
    ----------
    x : array
        Monotonic coordinates to resample.
    f : float
        Factor on the number of intervals. One returns `x` untouched.
    mult : int, optional
        Round the new interval count up to a multiple of this, for a mesh
        direction that a multigrid level has to be able to halve.

    """
    if np.isclose(f, 1.0):
        return x

    xnorm = (x - x[0]) / np.ptp(x)
    npts = len(x)
    npts_new = np.round((npts - 1) * f).astype(int) + 1
    if mult:
        npts_new = int(mult * np.ceil((npts_new - 1) / mult)) + 1

    inorm = np.linspace(0.0, 1.0, npts)
    inorm_new = np.linspace(0.0, 1.0, npts_new)
    xnew = np.interp(inorm_new, inorm, xnorm) * np.ptp(x) + x[0]

    assert np.allclose(xnew[(0, -1),], x[(0, -1),])

    return xnew


#
# COORDINATES
#
# A tangential coordinate is an angle, so it is not a length and cannot be
# compared with one. Scaling by a reference radius makes the three components
# commensurate, which is what an interpolation or a distance needs.
#


def to_xrrt_ref(xrt, rref):
    """Return `xrt` with its angle scaled to a length by `rref`."""
    return np.stack((xrt[0], xrt[1], xrt[2] * rref)).copy()


def from_xrrt_ref(xrrt_ref, rref):
    """Return coordinates scaled by `to_xrrt_ref` back to an angle."""
    return np.stack((xrrt_ref[0], xrrt_ref[1], xrrt_ref[2] / rref)).copy()


def cluster_cosine(npts):
    """Return `npts` points on the unit interval, clustered at both ends.

    Cosinusoidal, so the spacing is finest where a boundary layer or a leading
    edge needs it and coarsest in the middle.
    """
    xc = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))
    xc -= xc[0]
    xc /= xc[-1]
    return xc


def interp1d_linear_extrap(x, y, axis=0):
    """Return a spline through `x, y` that extrapolates linearly.

    A cubic spline inside the data, and straight lines beyond it. scipy's own
    extrapolation continues the end cubics, which on a blade section can turn
    sharply away from anything the data implies; this pins the ends to their
    own slopes instead.

    Degenerate inputs are handled rather than refused, because a design may
    legitimately give one section or two: a single point is a constant, and two
    are a straight line.
    """
    n = len(x)

    if n == 1:
        # One section is a constant, not a curve.
        def spline(xq):
            return np.take(y.copy(), 0, axis=axis)

        return spline

    if n == 2:
        return scipy.interpolate.interp1d(
            x, y, fill_value="extrapolate", axis=axis, kind="linear"
        )

    spline = scipy.interpolate.CubicSpline(x, y, axis=axis, bc_type="natural")

    # Extend with one breakpoint just outside each end, carrying the slope the
    # spline already has there. A PPoly of zero curvature is a straight line,
    # so the two leading coefficients are zero.
    for end, direction in ((0, -1), (-1, 1)):
        xe = np.atleast_1d(spline.x[end])
        ye = spline(xe)
        slope = spline(xe, nu=1)

        xnext = np.nextafter(xe, xe + direction)
        ynext = ye + slope * (xnext - xe)
        zero = np.zeros_like(slope)
        coefficients = np.expand_dims(
            np.concatenate([zero, zero, slope, ynext], axis=0), 1
        )
        spline.extend(coefficients, xnext)

    return spline


#
# REPORTING
#


def format_table(title, nrow, properties, col_w=8, paired=True):
    """Return a table of per-row turbomachinery quantities.

    Parameters
    ----------
    title : str
        Label for the top-left cell.
    nrow : int
        Number of blade rows, which is the number of column groups.
    properties : list of (label, values, spec)
        A row label, its values, and a format spec. With `paired`, a value per
        station fills the inlet and outlet columns and a value per row is
        centred over its pair.
    col_w : int
        Width of each value column.
    paired : bool
        Whether each row gets an inlet and an outlet column, or just one.

    """
    label_w = 14
    rows = []

    if paired:
        pair_w = col_w * 2
        header = f"{title:<{label_w}}"
        for i in range(nrow):
            header += f"{'  Row ' + str(i):^{pair_w}}"
        subheader = " " * label_w + f"{'Inlet':>{col_w}}{'Outlet':>{col_w}}" * nrow
        rows = [header, subheader]

        for label, values, spec in properties:
            row = f"{label:<{label_w}}"
            if len(values) == 2 * nrow:
                for value in values:
                    row += f"{value:{col_w}{spec}}"
            else:
                for value in values:
                    row += f"{value:^{pair_w}{spec}}"
            rows.append(row)
    else:
        header = f"{title:<{label_w}}"
        for i in range(nrow):
            header += f"{'Row ' + str(i):>{col_w}}"
        rows = [header]

        for label, values, spec in properties:
            row = f"{label:<{label_w}}"
            for value in values:
                row += f"{value:{col_w}{spec}}"
            rows.append(row)

    return "\n".join(rows)


#
# BLADE SURFACES
#
# What a surface distribution is drawn from: the flow on the blade itself,
# pulled off a solved grid.
#


def get_zeta(block):
    """Return the arc length along each i-gridline of `block`.

    Zero at ``i = 0`` and increasing, with the same shape as the block. Worked
    in Cartesian coordinates, so that the tangential component is a distance
    and not an angle.
    """
    x = block.x
    y = block.r * np.sin(block.t)
    z = block.r * np.cos(block.t)

    xyz = np.stack((x, y, z))
    dxyz = np.diff(xyz, n=1, axis=1) ** 2.0
    ds = np.sqrt(np.sum(dxyz, axis=0, keepdims=True))

    return np.insert(np.cumsum(ds, axis=1), 0, 0.0, axis=1)[0]


def get_i_stag(block, xrt_LE=None):
    """Return the streamwise index of the stagnation point on each j-line.

    Found as a pressure maximum near the leading edge, in *rotary* static
    pressure so that the centrifugal gradient of a rotating frame does not
    stand in for one.

    Parameters
    ----------
    block : ember.block.Block
        A two-dimensional cut with a flow field on it.
    xrt_LE : array, shape (3,), optional
        Where the leading edge is. Given, the search window is centred on the
        node nearest it rather than on the midpoint of the cut, which is what
        makes the answer robust on a blade whose two sides are of very
        different length.

    """
    if block.ndim != 2:
        raise ValueError(
            f"Can only find a stagnation point on a 2D cut; this block has "
            f"shape {block.shape}."
        )

    P = block.P_rot

    # Normalised arc length, -1 to 1 along each j-line.
    zeta = get_zeta(block)
    z = zeta / np.ptp(zeta, axis=0) * 2.0 - 1.0

    _, nj = block.shape[:2]

    if xrt_LE is not None:
        dx = block.xrt[:, :, 0] - xrt_LE[0]
        dr = block.xrt[:, :, 1] - xrt_LE[1]
        dt = block.xrt[:, :, 2] - xrt_LE[2]
        # The third coordinate is an angle, so weight it by radius to compare
        # it with the other two as a distance.
        r_avg = 0.5 * (block.xrt[:, :, 1] + xrt_LE[1])
        d2 = dx**2 + dr**2 + (r_avg * dt) ** 2
        i_nose = np.argmin(d2, axis=0)
        z_nose = z[i_nose, np.arange(nj)]
    else:
        z_nose = np.zeros((nj,))

    half_window = 0.05
    i_stag = np.full((nj,), 0, dtype=int)

    for j in range(nj):
        z_centre = z_nose[j]
        dP = np.diff(P[:, j])

        # Downward zero crossings of the pressure gradient, which are maxima.
        crossings = np.where(np.diff(np.sign(dP[:-2])) < 0.0)[0] + 1
        crossings = crossings[np.abs(z[crossings, j] - z_centre) < half_window]

        if len(crossings):
            i_stag[j] = crossings[np.argmax(P[crossings, j])]
        elif xrt_LE is not None:
            # No maximum in the window: take the highest pressure in it rather
            # than the highest anywhere, which on an asymmetric blade would
            # find the far side.
            inside = np.where(np.abs(z[:, j] - z_centre) < half_window)[0]
            i_stag[j] = inside[np.argmax(P[inside, j])]
        else:
            i_stag[j] = np.argmax(P[:, j])

    return i_stag


def cut_blade_sides(grid, offset=0):
    """Return the pressure and suction side cuts of each row.

    H-mesh only: the blade is the pair of periodic-bounded k faces between the
    leading and trailing edges, so the edges are found by looking for where the
    upstream and downstream periodic patches stop.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    offset : int
        Cells away from the surface, for reading just off the wall.

    Returns
    -------
    list
        Two cuts per row, or None for a row whose edges were not found.

    """
    cuts = []

    for i in range(len(grid.rows)):
        ile = None
        ite = None

        for block in grid.rows[i]:
            for patch in block.patches.periodic:
                # A pitchwise periodic spans the span and sits on one k face,
                # and the pair of them stop at the two edges of the blade.
                lim = patch.ijk_lim_abs
                spans_j = np.allclose(lim[1], [0, block.shape[1] - 1])
                spans_i = np.allclose(lim[0], [0, block.shape[0] - 1])
                at_k_boundary = (lim[2, 0] == lim[2, 1]) and (
                    lim[2, 0] == 0 or lim[2, 0] == block.shape[2] - 1
                )

                if spans_j and at_k_boundary and not spans_i:
                    if lim[0, 0] == 0:
                        ile = lim[0, 1]
                    elif lim[0, 1] == block.shape[0] - 1:
                        ite = lim[0, 0]

            # A cusp or an inviscid patch on a k face marks the trailing edge
            # where the periodics do not.
            for patch in block.patches:
                if (
                    isinstance(
                        patch, (ember.patch.CuspPatch, ember.patch.InviscidPatch)
                    )
                    and patch.const_dim == 2
                ):
                    ite = patch.ijk_lim_abs[0, 0]

        if not ile or not ite:
            cuts.append(None)
            continue

        sides = [
            grid[i][ile : (ite + 1), :, None, 0 + offset].copy(keep_patches=False),
            grid[i][ile : (ite + 1), :, None, -1 - offset].copy(keep_patches=False),
        ]
        # The patches described the block these were sliced out of, not the
        # slices.
        for side in sides:
            side.patches.clear()

        # Bring the two sides into one pitch, so a surface made of them is
        # continuous rather than a pitch apart.
        upper = np.argmax([side.t.max() for side in sides])
        sides[upper].set_t(sides[upper].t - grid[i].pitch)

        cuts.append(sides)

    assert len(cuts) == len(grid.rows)
    return cuts


def cut_blade_surfs(grid, offset=0):
    """Return the blade surface of each row, as one cut running round it.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    offset : int
        Cells away from the surface, for reading just off the wall.

    Returns
    -------
    list
        One list of cuts per row, or None for a row that has none.

    """
    surfs = []

    # One block per row is an H-mesh, where the blade is two k faces to be
    # joined; anything else is an O-mesh, where a block already wraps the
    # blade and only has to be recognised.
    if len(grid) == len(grid.rows):
        for sides in cut_blade_sides(grid, offset):
            if sides is None:
                surfs.append(None)
            else:
                # Reversed and joined, so the surface runs from trailing edge
                # round the nose and back, as a distribution is read.
                surfs.append(
                    [
                        ember.block_util.concatenate(
                            sides[0].flip(axis=0), sides[1][1:, ...], axis=0
                        )
                    ]
                )
        return surfs

    for row_block in grid.rows:
        surfs.append([])

        # Full span is whatever most blocks in the row have; the others are
        # the tip gap and its neighbours.
        nj_values, nj_counts = np.unique(
            [block.shape[1] for block in row_block], return_counts=True
        )
        nj = nj_values[np.argmax(nj_counts)]

        for block in row_block:
            wraps = np.allclose(block[0, :, 0].xrt, block[-1, :, 0].xrt)
            if wraps and block.shape[1] == nj:
                surfs[-1].append(block[:, :, None, offset])

    return surfs

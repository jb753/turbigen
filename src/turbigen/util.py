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


def sind(x):
    """Return the sine of an angle in degrees."""
    return np.sin(np.radians(x))


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

    Returns
    -------
    i_stag : ndarray of int, shape (nj,)
        Streamwise index of the stagnation point on each j-line.
    found : ndarray of bool, shape (nj,)
        Whether that index is a pressure maximum inside the search window, as
        opposed to the best guess made when there was none. A guess is good
        enough to normalise a surface distance by, and not good enough to
        measure an incidence from, so the two answers are told apart here
        rather than left for each caller to decide it got a real one.

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
    found = np.full((nj,), False)

    for j in range(nj):
        z_centre = z_nose[j]
        dP = np.diff(P[:, j])

        # Downward zero crossings of the pressure gradient, which are maxima.
        crossings = np.where(np.diff(np.sign(dP[:-2])) < 0.0)[0] + 1
        crossings = crossings[np.abs(z[crossings, j] - z_centre) < half_window]

        if len(crossings):
            i_stag[j] = crossings[np.argmax(P[crossings, j])]
            found[j] = True
        elif xrt_LE is not None:
            # No maximum in the window: take the highest pressure in it rather
            # than the highest anywhere, which on an asymmetric blade would
            # find the far side.
            inside = np.where(np.abs(z[:, j] - z_centre) < half_window)[0]
            i_stag[j] = inside[np.argmax(P[inside, j])]
        else:
            i_stag[j] = np.argmax(P[:, j])

    return i_stag, found


def get_zeta_stag(block, i_stag):
    """Return the arc length of the stagnation point on each j-line.

    The index :func:`get_i_stag` returns, refined to somewhere between nodes by
    fitting a parabola through the rotary pressure at ``i - 1``, ``i`` and
    ``i + 1`` against arc length and taking its vertex.

    Worth the three lines because the integer index is a step function of the
    flow: a leading edge that moves by less than a cell returns the same node,
    and an incidence measured from it does not change at all until it jumps by
    a whole cell. Differencing that gives a slope of either zero or nonsense,
    which is exactly what a secant update cannot be fed.

    Parameters
    ----------
    block : ember.block.Block
        The two-dimensional cut `i_stag` was found on.
    i_stag : array_like of int, shape (nj,)
        Streamwise index of the stagnation point on each j-line.

    Returns
    -------
    ndarray, shape (nj,)
        Arc length of the stagnation point, in the units of :func:`get_zeta`.

    """
    P = block.P_rot
    zeta = get_zeta(block)
    ni, nj = block.shape[:2]

    # Clamped so that the neighbours either side exist, which costs the sub-cell
    # correction on a stagnation point sitting on the very end of the cut --- a
    # blade whose flow attaches at the trailing edge has larger problems.
    i = np.clip(np.asarray(i_stag), 1, ni - 2)
    j = np.arange(nj)

    z0, z1, z2 = zeta[i - 1, j], zeta[i, j], zeta[i + 1, j]
    p0, p1, p2 = P[i - 1, j], P[i, j], P[i + 1, j]

    # Divided differences, so that a mesh clustered towards the nose is fitted
    # on its own spacing rather than on an assumed uniform one.
    d01 = z1 - z0
    d12 = z2 - z1
    s01 = (p1 - p0) / d01
    s12 = (p2 - p1) / d12
    curvature = (s12 - s01) / (z2 - z0)
    slope = 0.5 * (s01 + s12)

    # A triple that is not concave down has no vertex to find, which happens
    # only where `get_i_stag` did not find a maximum either. Left on the node.
    delta = np.where(curvature >= 0.0, 0.0, -slope / (2.0 * curvature))

    # Kept inside the bracket the three points span: a nearly flat parabola
    # puts its vertex arbitrarily far away, and the answer is known to lie
    # between the neighbours of a maximum.
    return z1 + np.clip(delta, -d01, d12)


def surface_normal_yaw(cut, zeta, e_m, chi):
    """Return the yaw of the inward surface normal at arc length `zeta` [deg].

    Yaw in the ``(m, r * theta)`` frame a metal angle is quoted in, so it can be
    differenced against one directly. At a stagnation point this is the angle
    the flow arrives at: the dividing streamline meets the wall along its
    normal, which is what lets an incidence be read off the blade rather than
    off a plane somewhere upstream.

    Parameters
    ----------
    cut : ember.block.Block
        A two-dimensional cut of a blade surface.
    zeta : array_like, shape (nj,)
        Arc length along each j-line to take the normal at, as
        :func:`get_zeta_stag` returns.
    e_m : array_like, shape (2,)
        Unit vector in ``(x, r)`` pointing downstream along the meridional
        direction. What tells a coordinate difference along the surface from a
        signed meridional distance --- arc length alone cannot, being positive
        on both sides of a leading edge.
    chi : float
        Metal angle at the leading edge [deg]. Orients the normal: of the two
        normals to a surface, the one wanted here points into the blade, and at
        the nose that is the camber direction. Which of the two the arithmetic
        produces depends on the direction the cut runs in, and that is a
        property of the mesh --- `cut_blade_sides` joins the ``k = 0`` and
        ``k = -1`` faces in an order the H-mesh and O-mesh branches do not
        share --- so it is settled from the geometry rather than assumed.

    Returns
    -------
    ndarray, shape (nj,)
        Yaw of the inward normal on each j-line [deg].

    """
    zeta_line = get_zeta(cut)
    nj = cut.shape[1]

    # Central differences, so the tangent is centred on the node its arc length
    # is, and the interpolation below is between neighbours rather than across
    # a half-cell offset.
    dx = np.gradient(cut.x, axis=0)
    dr = np.gradient(cut.r, axis=0)
    dt = np.gradient(cut.t, axis=0)

    # Into the frame the angle is quoted in. The tangential step is an angle
    # until a radius makes it a distance.
    t_m = dx * e_m[0] + dr * e_m[1]
    t_rt = cut.r * dt

    zeta = np.atleast_1d(zeta)
    yaw = np.zeros((nj,))
    for j in range(nj):
        # The components are interpolated, not the angle they make: an angle
        # would have to be unwrapped first, and the point of a sub-cell arc
        # length is that what comes out of it moves smoothly.
        m_j = np.interp(zeta[j], zeta_line[:, j], t_m[:, j])
        rt_j = np.interp(zeta[j], zeta_line[:, j], t_rt[:, j])

        # The tangent turned a quarter turn is a normal; which of the two it is
        # follows from the sign.
        n_m, n_rt = -rt_j, m_j
        if n_m * cosd(chi) + n_rt * sind(chi) < 0.0:
            n_m, n_rt = -n_m, -n_rt

        yaw[j] = np.degrees(np.arctan2(n_rt, n_m))

    return yaw


def _wall_Omega(block, const_dim, at_end):
    """Return the angular velocity of one whole boundary face [rad/s].

    A wall turns with its block unless a `RotatingPatch` says otherwise, which
    on a turbigen mesh happens exactly once: `bconds.apply_rotation` puts one
    over a tip gap to hold the casing still while the row turns under it.

    Parameters
    ----------
    block : ember.block.Block
        Block the face belongs to.
    const_dim : int
        Axis the face is normal to, 0 for i, 1 for j, 2 for k.
    at_end : bool
        Whether the face is the high-index one.

    Returns
    -------
    float
        The single angular velocity that applies over the whole face.

    Raises
    ------
    ValueError
        If the face does not resolve to one speed --- two patches disagreeing,
        or one covering part of the face and leaving the block's speed on the
        rest. Both are the same fault, and reducing two speeds to one number
        would be wrong without being visible.

    """
    end = block.shape[const_dim] - 1 if at_end else 0
    face = f"{'ijk'[const_dim]}={'-1' if at_end else '0'}"

    covering = [
        patch
        for patch in block.patches.rotating
        if patch.const_dim == const_dim and patch.ijk_lim_abs[const_dim, 0] == end
    ]

    if not covering:
        return float(block.Omega)

    spanned = [
        patch
        for patch in covering
        if all(
            patch.ijk_lim_abs[d].tolist() == [0, block.shape[d] - 1]
            for d in range(3)
            if d != const_dim
        )
    ]

    speeds = {float(patch.Omega) for patch in covering}
    if len(spanned) != len(covering):
        # What the patches leave uncovered still turns with the block, so a
        # partial patch is two speeds over the face just as two patches are.
        speeds.add(float(block.Omega))

    if len(speeds) > 1:
        raise ValueError(
            f"Face {face} of block {block.label!r} has more than one wall "
            f"speed: {sorted(speeds)}. A cut of it has no single frame."
        )

    return speeds.pop()


def cut_blade_sides(grid, offset=0):
    """Return the pressure and suction side cuts of each row.

    H-mesh only: the blade is the pair of periodic-bounded k faces between the
    leading and trailing edges, so the edges are found by looking for where the
    upstream and downstream periodic patches stop.

    A clearance gap is trimmed off the span. Over the gap the same k faces are
    periodic rather than solid, so a cut that kept them would return the flow
    passing over the tip as though it were a surface distribution --- which is
    a wrong answer where no answer is the honest one.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    offset : int
        Cells away from the surface, for reading just off the wall.

    Each cut carries the speed its own wall turns at, so `ho_rel` on it is the
    stagnation enthalpy in the frame the boundary layer actually sees. That is
    the block's speed for a blade, which no mesh turbigen builds overrides.

    Returns
    -------
    list
        Two 2D ``(ni, nj)`` cuts per row, or None for a row whose edges were
        not found.

    """
    cuts = []

    for i in range(len(grid.rows)):
        ile = None
        ite = None
        j_gap = None

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

                # The one k-face periodic that does not span the span is the
                # clearance gap: it runs along the blade rather than upstream
                # or downstream of it. Both sides carry one with the same
                # limits, so finding it twice is finding the same gap.
                if at_k_boundary and not spans_j and not spans_i:
                    j_gap = lim[1]

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

        # Where the gap sits comes from its own patch rather than being assumed
        # to be at the casing, so a hub clearance would trim the other end.
        nj = grid[i].shape[1]
        if j_gap is None:
            jst, jen = 0, nj
        elif j_gap[1] == nj - 1:
            jst, jen = 0, j_gap[0] + 1
        else:
            jst, jen = j_gap[1], nj

        sides = [
            grid[i][ile : (ite + 1), jst:jen, 0 + offset].copy(keep_patches=False),
            grid[i][ile : (ite + 1), jst:jen, -1 - offset].copy(keep_patches=False),
        ]
        # The patches described the block these were sliced out of, not the
        # slices. Read the wall speed off the block first, the patches that say
        # it being among those about to go.
        speeds = [_wall_Omega(grid[i], 2, at_end) for at_end in (False, True)]
        for side, Omega in zip(sides, speeds):
            side.patches.clear()
            side.set_Omega(Omega)

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
        One list of 2D ``(ni, nj)`` cuts per row, streamwise by spanwise, or
        None for a row that has none, each carrying the speed its wall turns
        at. `ember.cut.structured_meridional` wants a third axis to interpolate
        along; the caller that needs one adds it, because it is the caller that
        takes it off the result again.

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
                # Copied, not sliced: a view shares its block's angular
                # velocity, so setting the wall speed on one would re-time the
                # grid itself.
                surf = block[:, :, offset].copy(keep_patches=False)
                surf.patches.clear()
                surf.set_Omega(_wall_Omega(block, 2, offset != 0))
                surfs[-1].append(surf)

    return surfs


def cut_endwalls(grid, offset=0):
    """Return the hub and casing surfaces of each row.

    The endwalls need none of the edge-hunting `cut_blade_sides` does. The
    mesher's convention is that ``j`` runs hub to casing, so an endwall is one
    whole ``j`` face of a block and there is nothing to search for. A tip gap
    does not change that: the gap is periodic patches on the ``k`` faces of the
    same block, and the casing over it is still a wall, only one turning at its
    own speed rather than the blade's.

    Each cut carries the speed its own wall turns at rather than the frame its
    block was solved in, so `ho_rel` on it is the stagnation enthalpy the
    boundary layer on that wall sees. The two differ for exactly one surface a
    turbigen mesh contains: the casing over a tip gap, which stands still while
    the row turns under it.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    offset : int
        Cells away from the surface, for reading just off the wall.

    Returns
    -------
    list
        One list of 2D ``(ni, nk)`` cuts per row, streamwise by pitchwise,
        hub then casing for each block in the row. Never `None`: unlike a
        blade surface, an endwall is always there.

    """
    walls = []

    for row_block in grid.rows:
        walls.append([])
        for block in row_block:
            for at_end, j in ((False, 0 + offset), (True, -1 - offset)):
                # Copied, not sliced: a view shares its block's angular
                # velocity, so setting the wall speed on one would re-time the
                # grid itself.
                wall = block[:, j, :].copy(keep_patches=False)
                wall.patches.clear()
                wall.set_Omega(_wall_Omega(block, 1, at_end))
                walls[-1].append(wall)

    return walls

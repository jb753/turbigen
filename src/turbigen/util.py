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

import ember.block_util
import ember.cut
import ember.patch
import numpy as np
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid

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
    x = np.asarray(x)
    if axis is None:
        x, axis = x.ravel(), 0
    c = np.cumsum(x, axis=axis)
    # Into a preallocated array, which `np.insert` takes several times as long
    shape = list(c.shape)
    shape[axis] += 1
    out = np.empty(shape, dtype=c.dtype)
    out[(slice(None),) * (axis % c.ndim) + (slice(0, 1),)] = 0
    out[(slice(None),) * (axis % c.ndim) + (slice(1, None),)] = c
    return out


def cumtrapz0(x, *args, axis=-1):
    """Return the cumulative integral of `x`, starting from an explicit zero."""
    x = np.asarray(x)
    if x.ndim == 1 and len(args) == 1 and np.ndim(args[0]) == 1:
        # Along one curve, summed as `cumulative_trapezoid` sums it, without
        # its checks, which on a mesher's thousands of calls cost the most.
        d = np.diff(args[0])
        return cumsum0(d * (x[1:] + x[:-1]) / 2.0)
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

    npts_new = np.round((len(x) - 1) * f).astype(int) + 1
    if mult:
        npts_new = int(mult * np.ceil((npts_new - 1) / mult)) + 1

    return resample_to(x, npts_new)


def resample_to(x, npts):
    """Return `x` with exactly `npts` points, keeping its relative spacing.

    What :func:`resample` does once it has decided on a count, and what a
    caller wants when the count is decided elsewhere --- a mesh direction
    assembled from two pieces, where one has to make up whatever the other
    leaves before the total can be halved by a multigrid level.
    """
    if npts == len(x):
        return x

    xnorm = (x - x[0]) / np.ptp(x)
    inorm = np.linspace(0.0, 1.0, len(x))
    inorm_new = np.linspace(0.0, 1.0, npts)
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


def interp1d_hold(x, y, axis=0):
    """Return a spline through `x, y` that holds its end values beyond them.

    The same curve as :func:`interp1d_linear_extrap` inside the data, and the
    nearest end value outside it. What a blade's sections want: carrying the
    end slopes on to the endwalls can drive a parameter somewhere no section
    put it, such as a nose radius below zero, while holding the outermost
    section claims nothing the design did not say.
    """
    spline = interp1d_linear_extrap(x, y, axis=axis)
    lo, hi = np.min(x), np.max(x)

    def held(xq):
        return spline(np.clip(xq, lo, hi))

    return held


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

    # How far either side of the nose to look, in normalised arc length (the
    # whole perimeter spans two). Pressure is nearly flat around a stagnation
    # point, so a window too narrow to hold the whole plateau picks a ripple on
    # its flank. Wide enough for the plateau, still well short of the far side.
    half_window = 0.15
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

    # Vertex of the Newton form `p0 + s01 (z - z0) + c (z - z0)(z - z1)`, at
    # `(z0 + z1) / 2 - s01 / (2 c)`, written relative to `z1`. This is exact
    # for unequal spacings, unlike the midpoint-slope shortcut. A triple that
    # is not concave down has no vertex, so the point stays on the node.
    delta = np.where(curvature >= 0.0, 0.0, -0.5 * d01 - s01 / (2.0 * curvature))

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


def row_blocks(grid, i_row):
    """Split one row into the block flow passes through and the rest.

    A row is a list of blocks and nothing says which is which, but the
    difference matters to everything that reads a surface off a mesh: the
    passage is where the blade, the endwalls and the edges are, and a tip
    block meshing a clearance is a piece hanging inside it.

    Told apart by where the row's flow enters and leaves, which is a property
    of the topology rather than of one mesher: a passage carries an inlet, a
    mixing plane or an outlet on each of its `i` faces, and a block filling a
    clearance carries none. Preferred to counting blocks, reading a label, or
    taking the one with the most spanwise points --- the first stops being
    true the moment a row has two blocks, and the other two are a convention
    of whichever mesher wrote them.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    i_row : int
        Which row, in the order `grid.rows` gives them.

    Returns
    -------
    tuple
        The passage block, and a list of the other blocks in the row.

    Raises
    ------
    ValueError
        If the row does not have exactly one passage block, which no mesh
        turbigen builds produces and which nothing downstream could read.

    """
    row = grid.rows[i_row]
    throughflow = (
        ember.patch.InletPatch,
        ember.patch.OutletPatch,
        ember.patch.MixingPatch,
    )
    passage = [
        block
        for block in row
        if any(
            isinstance(patch, throughflow) and patch.const_dim == 0
            for patch in block.patches
        )
    ]
    if len(passage) != 1:
        raise ValueError(
            f"Row {i_row} has {len(passage)} blocks the flow passes through, "
            f"and a row is read through exactly one. Blocks: "
            f"{[block.label for block in row]}."
        )
    return passage[0], [block for block in row if block is not passage[0]]


def span_blocks(grid, i_row):
    """Split one row into the blocks that span it and those in a clearance.

    The spanning blocks run from hub to casing; a block meshing a tip gap
    runs from the tip of the blade to the casing, and so has fewer spanwise
    points than the rest of its row. That holds whatever the blade-to-blade
    topology, where :func:`row_blocks` needs exactly one block the flow passes
    through and an O-H mesh has two in series, ahead of and behind the blade.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    i_row : int
        Which row, in the order `grid.rows` gives them.

    Returns
    -------
    tuple
        A list of the blocks spanning the row, and a list of the rest, each in
        the order the row holds them.

    """
    row = grid.rows[i_row]
    nj = max(block.shape[1] for block in row)
    full = [block for block in row if block.shape[1] == nj]
    clearance = [block for block in row if block.shape[1] != nj]
    return full, clearance


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

        # The passage alone: a block filling a clearance carries k-face
        # periodics of its own, spanning all of its own i and j, and they say
        # nothing about where the blade begins or ends.
        passage, _ = row_blocks(grid, i)

        for patch in passage.patches.periodic:
            # A pitchwise periodic spans the span and sits on one k face,
            # and the pair of them stop at the two edges of the blade.
            lim = patch.ijk_lim_abs
            spans_j = np.allclose(lim[1], [0, passage.shape[1] - 1])
            spans_i = np.allclose(lim[0], [0, passage.shape[0] - 1])
            at_k_boundary = (lim[2, 0] == lim[2, 1]) and (
                lim[2, 0] == 0 or lim[2, 0] == passage.shape[2] - 1
            )

            if spans_j and at_k_boundary and not spans_i:
                if lim[0, 0] == 0:
                    ile = lim[0, 1]
                elif lim[0, 1] == passage.shape[0] - 1:
                    ite = lim[0, 0]

            # The one k-face periodic that does not span the span is the
            # clearance gap: it runs along the blade rather than upstream
            # or downstream of it. Both sides carry one with the same
            # limits, so finding it twice is finding the same gap.
            if at_k_boundary and not spans_j and not spans_i:
                j_gap = lim[1]

        # A cusp or an inviscid patch on a k face marks the trailing edge
        # where the periodics do not.
        for patch in passage.patches:
            if (
                isinstance(patch, (ember.patch.CuspPatch, ember.patch.InviscidPatch))
                and patch.const_dim == 2
            ):
                ite = patch.ijk_lim_abs[0, 0]

        if not ile or not ite:
            cuts.append(None)
            continue

        # Where the gap sits comes from its own patch rather than being assumed
        # to be at the casing, so a hub clearance would trim the other end.
        nj = passage.shape[1]
        if j_gap is None:
            jst, jen = 0, nj
        elif j_gap[1] == nj - 1:
            jst, jen = 0, j_gap[0] + 1
        else:
            jst, jen = j_gap[1], nj

        sides = [
            passage[ile : (ite + 1), jst:jen, 0 + offset].copy(keep_patches=False),
            passage[ile : (ite + 1), jst:jen, -1 - offset].copy(keep_patches=False),
        ]
        # The patches described the block these were sliced out of, not the
        # slices. Read the wall speed off the block first, the patches that say
        # it being among those about to go.
        speeds = [_wall_Omega(passage, 2, at_end) for at_end in (False, True)]
        for side, Omega in zip(sides, speeds):
            side.patches.clear()
            side.set_Omega(Omega)

        # Bring the two sides into one pitch, so a surface made of them is
        # continuous rather than a pitch apart.
        upper = np.argmax([side.t.max() for side in sides])
        sides[upper].set_t(sides[upper].t - passage.pitch)

        cuts.append(sides)

    assert len(cuts) == len(grid.rows)
    return cuts


def _is_h_mesh(grid):
    """Whether a grid's rows are H passages rather than O blocks.

    Read off the first row's passage: an H passage carries the pitchwise
    periodics that bound it either side of the blade, on a k face and spanning
    the span. An O block has no such face, being wrapped around the blade
    instead, and an O-H row has several blocks spanning it where an H row has
    one.
    """
    full, _ = span_blocks(grid, 0)
    if len(full) != 1:
        return False
    passage = full[0]
    return any(
        patch.const_dim == 2
        and np.allclose(patch.ijk_lim_abs[1], [0, passage.shape[1] - 1])
        and not np.allclose(patch.ijk_lim_abs[0], [0, passage.shape[0] - 1])
        for patch in passage.patches.periodic
    )


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

    # An H-mesh passage is bounded pitchwise by the blade, so its blade is two
    # k faces to be joined; an O-mesh has a block already wrapping the blade,
    # which only has to be recognised. Told apart by asking the passage of the
    # first row which it is, rather than by counting blocks against rows --- a
    # count that says H-mesh only while no row has a second block, and a
    # gridded tip clearance is one.
    if _is_h_mesh(grid):
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
        for block, jen in _o_blade_blocks(row_block):
            # Copied, not sliced: a view shares its block's angular
            # velocity, so setting the wall speed on one would re-time the
            # grid itself.
            surf = block[:, :jen, offset].copy(keep_patches=False)
            surf.patches.clear()
            surf.set_Omega(_wall_Omega(block, 2, offset != 0))
            surfs[-1].append(surf)

    return surfs


def cut_endwalls(grid, offset=0):
    """Return the hub and casing surfaces of each row.

    The endwalls need none of the edge-hunting `cut_blade_sides` does. The
    mesher's convention is that ``j`` runs hub to casing, so an endwall is one
    whole ``j`` face of a block and there is nothing to search for.

    How many faces that is depends on how the row is built. A pinched tip gap
    leaves one block spanning the whole annulus, so the two endwalls are two
    faces; an O-H row has several blocks spanning it, and a hub and casing
    face for each, in pairs. A gridded one puts a block in the clearance, and the casing is then
    the passage's ``j = -1`` outside the blade and the tip block's inside it
    --- two pieces that tile it exactly once between them. So this returns a
    list per row of whatever length the mesh implies, and a caller that wants
    the casing wants all of the pieces rather than one of them. What is never
    returned is the tip block's ``j = 0``: that face is the tip of the blade,
    not an endwall, and :func:`cut_blade_tips` has it.

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
        One list of 2D ``(ni, nk)`` cuts per row, streamwise by pitchwise, of
        no fixed length. Never `None`: unlike a blade surface, an endwall is
        always there.

    """
    walls = []

    for i_row in range(len(grid.rows)):
        full, others = span_blocks(grid, i_row)
        walls.append([])

        faces = [(block, at_end) for block in full for at_end in (False, True)]
        # A block inside the row reaches the casing and not the hub, its lower
        # face being the tip of the blade it sits on.
        faces += [(block, True) for block in others]

        for block, at_end in faces:
            j = -1 - offset if at_end else offset
            # Copied, not sliced: a view shares its block's angular velocity,
            # so setting the wall speed on one would re-time the grid itself.
            wall = block[:, j, :].copy(keep_patches=False)
            wall.patches.clear()
            wall.set_Omega(_wall_Omega(block, 1, at_end))
            walls[-1].append(wall)

    return walls


def cut_blade_tips(grid, offset=0):
    """Return the tip face of each row's blade, where the mesh has one.

    The face closing the top of a blade against its clearance, which exists as
    a surface only where the clearance is gridded: pinched, the blade is
    brought to nothing over the gap and there is no tip to cut. It is blade
    rather than endwall --- it turns with the row, and the loss on it belongs
    with the blade's --- but it is not part of a surface distribution either,
    running chordwise by pitchwise where a blade surface runs chordwise by
    spanwise. So it is its own cut and its own bucket.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    offset : int
        Cells away from the surface, for reading just off the wall.

    Returns
    -------
    list
        One list of 2D ``(ni, nk)`` cuts per row, empty for a row whose
        clearance is pinched or which has none.

    """
    tips = []

    for i_row in range(len(grid.rows)):
        _, others = span_blocks(grid, i_row)
        tips.append([])
        for block in others:
            tip = block[:, offset, :].copy(keep_patches=False)
            tip.patches.clear()
            tip.set_Omega(_wall_Omega(block, 1, False))
            tips[-1].append(tip)

    return tips


def cut_periodic(grid):
    """Return the pitchwise-periodic surface of each row, joined where it runs on.

    Read off the periodic patches rather than found by geometry: of each pair
    a pitch apart, the side whose partner lies a pitch above, which is the
    lower bound of the passage. Pairs that join blocks in place, with no pitch
    between them, are interfaces and not this surface, so the joins of an O-H
    mesh's blocks and its tip gap are passed over without asking which they
    are. Pieces that meet end to end are joined into one surface, so an O-H
    mesh's runs from inlet to outlet unbroken, over the blade as the passage
    block's face against the next blade's O block. An H mesh's stops at the
    leading and trailing edges, where the blade takes over the same face, and
    comes back as the two pieces either side.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid.

    Returns
    -------
    list
        One list per row of 2D ``(n, nj)`` cuts, streamwise along the first
        axis and hub to casing along the second, upstream first.

    """
    pairs = grid.connectivity.periodic.pair()
    blocks = list(grid)

    surfs = []
    for row in grid.rows:
        in_row = {id(block) for block in row}
        faces = []
        for (i_block, i_patch), ((i_mate, i_mate_patch), _) in pairs.items():
            block, mate = blocks[i_block], blocks[i_mate]
            if id(block) not in in_row:
                continue
            patch = block.patches[i_patch]
            mate_patch = mate.patches[i_mate_patch]
            if _pitches_apart(block, patch, mate, mate_patch) == 1:
                faces.append(_patch_face(block, patch))
        surfs.append(_join_faces(faces))

    return surfs


def _pitches_apart(block, patch, mate, mate_patch):
    """Return how many pitches above the nodes of `patch` those of `mate_patch`
    lie, to the nearest whole pitch."""
    t = np.mean(block.t[_patch_slices(patch)])
    t_mate = np.mean(mate.t[_patch_slices(mate_patch)])
    return round((t_mate - t) * block.Nb / (2.0 * np.pi))


def _patch_slices(patch):
    """Return the index of the nodes of `patch` in its block."""
    lim = patch.ijk_lim_abs
    return tuple(
        int(start) if dim == patch.const_dim else slice(start, end + 1)
        for dim, (start, end) in enumerate(lim)
    )


def _patch_face(block, patch):
    """Return the nodes of `patch` as a 2D cut, hub to casing along axis 1.

    Copied, not sliced, so the cut owns its data and carries no patches, and
    given its block's frame, so a relative quantity read off it is in the
    frame of the flow it was cut from.
    """
    face = block[_patch_slices(patch)].copy(keep_patches=False)
    face.patches.clear()
    face.set_Omega(block.Omega)
    # Every block runs hub to casing in `j`, so a face of constant `i` has it
    # first and is turned round.
    if patch.const_dim == 0:
        face = face.transpose((1, 0))
    return face


def _join_faces(faces):
    """Return `faces` joined end to end where they share a line of nodes.

    Each runs downstream first, by its mean axial coordinate, and the pieces
    are taken in streamwise order, each joined to the surface whose last line
    is its first.
    """
    downstream = [
        face if np.mean(face.x[0]) <= np.mean(face.x[-1]) else face.flip(axis=0)
        for face in faces
    ]
    downstream.sort(key=lambda face: np.mean(face.x[0]))

    joined = []
    for face in downstream:
        for i, surf in enumerate(joined):
            if surf.shape[1] == face.shape[1] and _same_line(surf, -1, face, 0):
                joined[i] = ember.block_util.concatenate(surf, face[1:], axis=0)
                break
        else:
            joined.append(face)

    return joined


def _same_line(a, ia, b, ib):
    """Return whether line `ia` of `a` and line `ib` of `b` are the same nodes."""
    xrt_a, xrt_b = a.xrt[ia], b.xrt[ib]
    scale = np.max(np.abs(xrt_a[..., :2]))
    return np.allclose(xrt_a, xrt_b, rtol=0.0, atol=1e-6 * scale)


#
# SURFACE DISTRIBUTIONS
#
# Shared by the surface plot and `turbigen.loading`, so the two agree about
# what the distribution is.
#

N_SPAN_CUT = 101
"""Meridional points defining the span curve a blade surface is cut along.

Only the placement of the cut surface depends on this: `structured_meridional`
walks the grid's own gridlines, so the resolution of what comes back is the
mesh's, not the curve's.
"""


SCAN_STRIDE = 8
"""Stride the band scan subsamples a block by, in `i` and `k`.

Only to find which `j` the cut passes through, so it can be coarse: the band
is padded by :data:`BAND_PAD` afterwards, and a cut surface that moved by more
than that between neighbouring gridlines would be a mesh nobody could contour
anyway. Eight cuts the scan to a sixty-fourth of a full pass.
"""

BAND_PAD = 2
"""Extra `j` nodes kept either side of the band the scan found.

The scan reads every eighth gridline, so the sign change it misses is on one
of the seven between --- which can only be a node or two further out, the
surface being continuous. Two is that, doubled.
"""


def cut_band(block, xr_cut, stride=SCAN_STRIDE, pad=BAND_PAD):
    """Return the `j` slice of `block` a meridional cut can pass through.

    The whole of `j` where the scan cannot bracket it, so this only ever
    narrows the work when it is sure --- an empty cut where a block should
    have been drawn is a wrong picture, and being slow is not.

    A constant-span cut crosses a narrow band of `j` and nothing else, so
    everything outside that band can only ever be one sign of distance from
    it. Found by subsampling, which is what makes this worth doing: the scan
    costs a sixty-fourth of what it saves.
    """
    whole = slice(0, block.shape[1])

    d = ember.cut.signed_distance(
        xr_cut, np.asarray(block.xrt[::stride, :, ::stride, :2])
    )

    # Between neighbouring `j` rather than within one, the surface passing
    # between two gridlines. Non-positive, not negative: a cut that lands
    # exactly on a gridline leaves a distance of exactly zero there, whose
    # sign multiplies to zero and never to less than it -- and a mesh with an
    # odd number of spanwise nodes puts a gridline on mid-span every time.
    signs = np.sign(d)
    crossed = np.where((signs[:, :-1, :] * signs[:, 1:, :] <= 0.0).any(axis=(0, 2)))[0]
    if not len(crossed):
        return whole

    return slice(
        max(int(crossed.min()) - pad, 0),
        min(int(crossed.max()) + 1 + pad, block.shape[1]),
    )


N_CUT_SPAN = 113
"""Spanwise nodes a meridional cut is regridded onto by :func:`regrid_cut`.

Brute force: well beyond any practical mesh, so the regrid resolves whatever
the solution holds rather than setting a resolution of its own.
"""

N_CUT_PITCH = 137
"""Pitchwise nodes a meridional cut is regridded onto by :func:`regrid_cut`."""


def regrid_cut(cut):
    """Return the unstructured meridional `cut` regridded structured.

    :func:`ember.cut.interpolate_to_structured` onto ``(N_CUT_SPAN,
    N_CUT_PITCH)`` nodes: index 0 runs along the cut line, hub to casing, and
    index 1 in theta over one pitch. Every average over a regridded cut goes
    through here, so two of them taken on the same plane agree.

    Raises
    ------
    ValueError
        If the cut cannot be regridded, as for a line between two rows of
        different blade count, which has no single pitch to wrap theta by.
    """
    return ember.cut.interpolate_to_structured(cut, (N_CUT_SPAN, N_CUT_PITCH))


def cut_structured(grid, xr_cut):
    """Return the meridional cut of `grid` along `xr_cut`, regridded structured.

    :func:`ember.cut.unstructured` then :func:`regrid_cut`.

    Returns
    -------
    ember.block.Block or None
        The structured cut, or None if no block reaches the line.

    Raises
    ------
    ValueError
        If the cut cannot be regridded, as for a line between two rows of
        different blade count, which has no single pitch to wrap theta by.
    """
    cut = ember.cut.unstructured(grid, xr_cut)
    if cut is None:
        return None
    return regrid_cut(cut)


def cut_spanwise(grid, xr_cut):
    """Return the constant-span cut of `grid` along the curve `xr_cut`.

    :func:`ember.cut.structured_meridional`, with each block cropped to the
    band the cut actually crosses first. The distance it evaluates costs one
    multiply per grid node per curve segment, and a whole machine against a
    curve spanning it is hundreds of millions of them --- almost all on nodes
    nowhere near the cut.

    Exact, not an approximation: the cut surface lies inside the band by
    construction, so the nodes dropped could only have contributed distances
    of one sign. Measured on a two-row machine of 1.1M nodes it returns the
    same cut, node for node, in a thirteenth of the time.

    Blocks the cut misses are left whole and handed on, for
    `structured_meridional` to leave out as it always has: deciding that here
    would be a second opinion on what intersects, and the two could disagree.
    """
    slabs = [block[:, cut_band(block, xr_cut), :] for block in grid]

    return ember.cut.structured_meridional(slabs, xr_cut)


def cut_section(surface, annulus, i_row, spf, n=N_SPAN_CUT):
    """Return a constant-span cut of one blade surface.

    The three lines of index arithmetic that place a row on the annulus, and
    the padding `ember.cut.structured_meridional` insists on, written once.

    Parameters
    ----------
    surface : ember.block.Block
        One row's blade surface, as `cut_blade_surfs` returns it: a 2D
        ``(ni, nj)`` cut running trailing edge, round the nose, and back.
    annulus : Annulus
        The annulus the row sits in.
    i_row : int
        Index of the row, which is what places it on the meridional curve.
    spf : float
        Span fraction to cut at.
    n : int
        Meridional points along the cut.

    Returns
    -------
    cut : ember.block.Block or None
        A 2D cut of `surface` at `spf`, or None where the blade has no surface
        there --- above a clearance gap, the span being trimmed off as flow
        rather than wall.
    xr : ndarray
        The ``(2, n)`` meridional curve the cut was taken along, which is the
        only thing that knows which way is downstream: arc length around a
        nose is positive on both sides of it and cannot say.

    """
    # Rows occupy the odd meridional segments of the annulus, so row i runs
    # from the leading edge at 2i+1 to the trailing edge at 2i+2.
    m = np.linspace(2 * i_row + 1, 2 * i_row + 2, n)
    xr = annulus.evaluate_xr(m, spf)

    # `structured_meridional` walks the second axis of a three-axis block, so
    # the surface is padded to put its spanwise axis there and the cut comes
    # back one wide.
    cut = ember.cut.structured_meridional(surface[:, :, None], xr.T)

    return (cut[0] if len(cut) else None), xr


def isentropic_mach(cut, s_ref):
    """Return isentropic Mach number over `cut`, referred to entropy `s_ref`.

    Expanded isentropically from the row inlet entropy to the local static
    pressure, so the result reads as the Mach number the blade would see with
    no loss upstream of the point in question.
    """
    # Set in place on a copy, not chained off one: ember's setters return
    # nothing, whatever the idiom in the package this is ported from suggests.
    isen = cut.copy()
    isen.set_P_s(cut.P, s_ref)

    # Stagnation enthalpy and sound speed are taken as surface means so that
    # only local static pressure drives the distribution. Left local, radial
    # redistribution of ho_rel and variation in a split the two surfaces apart
    # at the trailing edge, where they must meet.
    ho = np.mean(cut.ho_rel)
    a_ref = np.mean(isen.a)

    # Shift so the lowest point sits exactly at rest rather than slightly
    # below it, which the discrete field can otherwise produce.
    hs = isen.h
    hs = hs + np.min(ho - hs)

    return np.sqrt(2.0 * np.maximum(ho - hs, 0.0)) / a_ref


def normalise_surface_distance(cut, mas, xrt_nose):
    """Return surface distance in [-1, 1], zero at the stagnation point.

    Each surface is normalised by its own length, so both reach one at the
    trailing edge however asymmetric the blade is. The sign says which surface
    a point is on, following the direction the cut loops in; a plot folds it
    away, but normalising the two sides has to happen while they are still
    told apart.
    """
    zeta = get_zeta(cut)[:, 0]

    # The geometric nose anchors the search window, which is more robust on
    # blades with a strongly asymmetric leading edge than the arc-length
    # midpoint the function falls back on. Whether it found a real maximum does
    # not matter here: the origin moves onto the lowest Mach number below in
    # any case, and this only has to land on the right side of the blade.
    i_stag = int(get_i_stag(cut, xrt_LE=xrt_nose)[0][0])
    zeta = zeta - zeta[i_stag]

    # Then move the origin onto the lowest Mach number, which is the
    # stagnation point of the flow rather than of the grid.
    zeta = zeta - zeta[np.argmin(mas)]

    upper = zeta.max()
    lower = np.abs(zeta.min())
    return zeta / np.where(zeta > 0.0, upper or 1.0, lower or 1.0)


def _blade_side_limits(passage):
    """Return where the blade lies on the k faces of an H passage.

    The blade is the stretch of the two k faces between the leading and
    trailing edges, found by where the upstream and downstream periodic
    patches stop, less any clearance gap off the span.

    Returns
    -------
    tuple or None
        Node indices ``(ile, ite, jst, jen)``: the blade runs over
        ``ile:ite + 1`` and ``jst:jen``. None if the edges were not found.

    """
    ile = None
    ite = None
    j_gap = None

    for patch in passage.patches.periodic:
        # A pitchwise periodic spans the span and sits on one k face,
        # and the pair of them stop at the two edges of the blade.
        lim = patch.ijk_lim_abs
        spans_j = np.allclose(lim[1], [0, passage.shape[1] - 1])
        spans_i = np.allclose(lim[0], [0, passage.shape[0] - 1])
        at_k_boundary = (lim[2, 0] == lim[2, 1]) and (
            lim[2, 0] == 0 or lim[2, 0] == passage.shape[2] - 1
        )

        if spans_j and at_k_boundary and not spans_i:
            if lim[0, 0] == 0:
                ile = lim[0, 1]
            elif lim[0, 1] == passage.shape[0] - 1:
                ite = lim[0, 0]

        # The one k-face periodic that does not span the span is the
        # clearance gap: it runs along the blade rather than upstream
        # or downstream of it. Both sides carry one with the same
        # limits, so finding it twice is finding the same gap.
        if at_k_boundary and not spans_j and not spans_i:
            j_gap = lim[1]

    # A cusp or an inviscid patch on a k face marks the trailing edge
    # where the periodics do not.
    for patch in passage.patches:
        if (
            isinstance(patch, (ember.patch.CuspPatch, ember.patch.InviscidPatch))
            and patch.const_dim == 2
        ):
            ite = patch.ijk_lim_abs[0, 0]

    if not ile or not ite:
        return None

    # Where the gap sits comes from its own patch rather than being assumed
    # to be at the casing, so a hub clearance would trim the other end.
    nj = passage.shape[1]
    if j_gap is None:
        jst, jen = 0, nj
    elif j_gap[1] == nj - 1:
        jst, jen = 0, j_gap[0] + 1
    else:
        jst, jen = j_gap[1], nj

    return ile, ite, jst, jen


def _o_blade_blocks(row):
    """Return the O blocks wrapping the blade in one row, and how far up each.

    Returns
    -------
    list
        ``(block, jen)`` pairs: the blade is the ``k = 0`` face of `block`
        over ``j < jen``, trimmed off where a gridded clearance meets it.

    """
    # Full span is whatever most blocks in the row have; the others are
    # the tip gap and its neighbours.
    nj_values, nj_counts = np.unique(
        [block.shape[1] for block in row], return_counts=True
    )
    nj = nj_values[np.argmax(nj_counts)]

    found = []
    for block in row:
        wraps = np.allclose(block[0, :, 0].xrt, block[-1, :, 0].xrt)
        if wraps and block.shape[1] == nj:
            # Over a gridded clearance the face is the outline the gap's
            # blocks meet it on, and a patch there says so: trimmed off,
            # as it is from an H passage's sides.
            jen = nj
            for patch in block.patches.permeable:
                lim = patch.ijk_lim_abs
                if (
                    patch.const_dim == 2
                    and lim[2, 0] == 0
                    and lim[1, 1] == nj - 1
                    and lim[1, 0] > 0
                ):
                    jen = lim[1, 0] + 1
            found.append((block, jen))

    return found


def _faces_to_nodes(faces, wrap=False):
    """Return a field over the faces of a surface averaged onto its nodes.

    Each node takes the mean of the up to four faces round it; along an edge
    that is the faces on the one side there are, and with `wrap` the first and
    last `i` faces are neighbours, as round an O block.
    """
    mode_i = "wrap" if wrap else "edge"
    padded = np.pad(faces, ((1, 1), (0, 0)), mode=mode_i)
    padded = np.pad(padded, ((0, 0), (1, 1)), mode="edge")
    return 0.25 * (
        padded[:-1, :-1] + padded[1:, :-1] + padded[:-1, 1:] + padded[1:, 1:]
    )


def blade_yplus(grid, wall_law="fit"):
    """Return y+ on the blade surface of each row, at its nodes.

    Evaluated by :func:`ember.block_util.wall_yplus` on whole blocks, then read
    off the same faces `cut_blade_surfs` cuts at no offset, so each array lines
    up node for node with the surface it belongs to and can be cut alongside it
    by `cut_field`. The wall function samples the first node off the wall, so
    this is y+ there, not at the centroid of the wall cell.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.
    wall_law : {"fit", "reichardt"}
        Wall law the solution was marched with.

    Returns
    -------
    list
        Shaped as `cut_blade_surfs` returns: one list of ``(ni, nj)`` arrays
        per row, or None for a row with no blade surface found.

    """
    out = []

    if _is_h_mesh(grid):
        for i_row in range(len(grid.rows)):
            passage, _ = row_blocks(grid, i_row)
            limits = _blade_side_limits(passage)
            if limits is None:
                out.append(None)
                continue
            ile, ite, jst, jen = limits
            yplus = ember.block_util.wall_yplus(passage, wall_law)
            sides = [
                _faces_to_nodes(yplus[key][ile:ite, jst : jen - 1])
                for key in ("yplus_k1", "yplus_nk")
            ]
            # Joined as `cut_blade_surfs` joins the sides.
            out.append([np.concatenate((sides[0][::-1], sides[1][1:]), axis=0)])
        return out

    for row in grid.rows:
        out.append([])
        for block, jen in _o_blade_blocks(row):
            yplus = ember.block_util.wall_yplus(block, wall_law)["yplus_k1"]
            out[-1].append(_faces_to_nodes(yplus[:, : jen - 1], wrap=True))

    return out


EDGE_RATIO = 0.99
"""Fraction of the isentropic velocity that marks a boundary layer's edge."""


OVERSHOOT = 1.01
"""Most the velocity may exceed the isentropic inside a boundary layer before
its free stream reference is taken to have failed."""


def blade_shape_factor(grid):
    r"""Return the kinematic shape factor of the blade boundary layer of each row.

    Integrated along each line of an O block off the blade, in distance
    normal to the wall, against the isentropic velocity at each point, from
    its static pressure and the free stream's entropy and rothalpy,

    .. math::
        V_I = \sqrt{2\left(I_e + \tfrac{1}{2}U^2 - h(p, s_e)\right)},

    so that the integrands vanish outside the layer however the velocity
    varies across the passage:

    .. math::
        \delta^* = \int_0^\delta \left(1 - \frac{V}{V_I}\right) dn,
        \quad
        \theta = \int_0^\delta \frac{V}{V_I} \left(1 - \frac{V}{V_I}\right) dn,
        \quad H = \delta^* / \theta.

    The free stream is taken at the edge `delta`, the first point out from
    the wall where `V / V_I` reaches `EDGE_RATIO`, found against the block's
    outer face and then against that first edge, so a shock leaning through
    the block sets only the first guess. A line whose reference still fails,
    `V / V_I` above `OVERSHOOT` inside its edge or `theta` not positive, is
    NaN: a shock reaching into the layer.

    Kinematic, the density taken constant across the layer, as the
    correlations and the familiar values (about 1.4 turbulent, 2.6 laminar)
    are: the compressible one rises with Mach number for the same profile
    (Drela & Giles 1986, eq. 15). The relative velocity's magnitude, so
    crossflow counts as streamwise. On an H mesh each side of the blade is
    integrated out to mid-pitch, along the pitchwise grid lines.

    Parameters
    ----------
    grid : ember.grid.Grid
        A solved grid.

    Returns
    -------
    list
        Shaped as `cut_blade_surfs` returns: one list of ``(ni, nj)`` arrays
        per row, at the nodes of its blade surface, or None for a row whose
        blade edges were not found.

    """
    return _blade_layer(grid, 0)


def _blade_layer(grid, which):
    """Return one of the integrals of :func:`_layer_integrals` on every blade."""
    if _is_h_mesh(grid):
        return _h_blade_layer(grid, which)

    out = []
    for row in grid.rows:
        out.append([])
        for block, jen in _o_blade_blocks(row):
            B = block[:, :jen, :].copy(keep_patches=False)
            out[-1].append(_layer_integrals(B, wrap=True)[which])

    return out


def _h_blade_layer(grid, which):
    """The H-mesh half of :func:`_blade_layer`.

    The blade is the two k faces between the edges, with pitchwise lines
    running off each into the passage, so each side is integrated over its own
    half of the pitch and the two are joined as :func:`cut_blade_surfs` joins
    the surfaces: trailing edge, round the nose, and back.
    """
    out = []
    for i in range(len(grid.rows)):
        passage, _ = row_blocks(grid, i)
        limits = _blade_side_limits(passage)
        if limits is None:
            out.append(None)
            continue
        ile, ite, jst, jen = limits
        kmid = passage.shape[2] // 2

        # Each side with its wall at k = 0 and the passage middle outermost
        near = passage[ile : (ite + 1), jst:jen, : kmid + 1].copy(keep_patches=False)
        far = passage[ile : (ite + 1), jst:jen, kmid:].copy(keep_patches=False)
        sides = [
            _layer_integrals(B, wrap=False)[which] for B in (near, far.flip(axis=2))
        ]
        out.append([np.concatenate((sides[0][::-1], sides[1][1:]), axis=0)])

    return out


def _layer_integrals(B, wrap):
    """Return the shape factor and displacement thickness of the layer on the
    ``k = 0`` face of `B`.

    `B` is a block whose ``k`` lines run out from a wall at ``k = 0`` far enough
    to leave the layer, and whose ``i`` runs along the wall: round it and back
    to the same node if `wrap`, as an O block's does, or from end to end.
    See :func:`blade_shape_factor` for the integrals. Each shaped ``(ni, nj)``,
    and NaN together.
    """
    x, r, t = (np.asarray(v, dtype=float) for v in (B.x, B.r, B.t))
    X = np.stack((x, r * np.cos(t), r * np.sin(t)))

    # The wall's normal from its tangents, round the O block's seam (its first
    # and last `i` the same node) if it wraps, turned into the flow
    wall = X[..., 0]
    if wrap:
        ring = wall[:, :-1]
        t_i = np.roll(ring, -1, axis=1) - np.roll(ring, 1, axis=1)
        t_i = np.concatenate((t_i, t_i[:, :1]), axis=1)
    else:
        t_i = np.gradient(wall, axis=1)
    t_j = np.gradient(wall, axis=2)
    n = np.cross(t_i, t_j, axis=0)
    n /= np.linalg.norm(n, axis=0)
    n *= np.sign(np.sum(n * (X[..., 1] - wall), axis=0))
    dist = np.sum((X - wall[..., None]) * n[..., None], axis=0)

    # The velocity against the isentropic velocity at each point, from the free
    # stream at the edge: the outer face, then twice where the profile reaches it
    s, I = np.asarray(B.s), np.asarray(B.I)
    U2, V = 0.5 * np.asarray(B.U) ** 2, np.asarray(B.V_rel)
    isen = B.copy(keep_patches=False)
    nk = B.shape[2]
    edge = np.full(B.shape[:2], nk - 1)
    for _ in range(3):
        at = edge[..., None]
        s_e = np.take_along_axis(s, at, axis=-1)
        I_e = np.take_along_axis(I, at, axis=-1)
        isen.set_P_s(B.P, np.broadcast_to(s_e, B.shape).copy())
        V_I = np.sqrt(np.maximum(2.0 * (I_e + U2 - isen.h), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = V / V_I
        reached = ratio >= EDGE_RATIO
        edge = np.where(reached.any(axis=-1), np.argmax(reached, axis=-1), nk - 1)

    inside = np.arange(nk) <= edge[..., None]
    with np.errstate(divide="ignore", invalid="ignore"):
        d_star = np.trapezoid(np.where(inside, 1.0 - ratio, 0.0), dist, axis=-1)
        theta = np.trapezoid(
            np.where(inside, ratio * (1.0 - ratio), 0.0), dist, axis=-1
        )
        H = d_star / theta
    bad = np.any(inside & (ratio > OVERSHOOT), axis=-1) | ~(theta > 0.0)
    return np.where(bad, np.nan, H), np.where(bad, np.nan, d_star)


def cut_field(surface, field, xr):
    """Return a nodal field on `surface` along the cut `cut_section` made.

    `structured_meridional` interpolates only what a block holds, so a field
    computed beside one, y+ say, is cut here by the same rule: linearly, at the
    first crossing of each streamwise line with the curve `xr`.

    Parameters
    ----------
    surface : ember.block.Block
        The 2D ``(ni, nj)`` surface that was cut.
    field : ndarray
        ``(ni, nj)`` values at the nodes of `surface`.
    xr : ndarray
        The ``(2, n)`` curve `cut_section` returned.

    Returns
    -------
    ndarray
        ``(ni,)`` values on the cut, NaN where a line does not cross it.

    """
    xr_nodes = surface.xrt_nd[..., :2] * surface.L_ref
    dist = ember.cut.signed_distance(xr.T, xr_nodes)
    return ember.cut._first_j_crossing(field[:, :, None, None], dist[:, :, None])[
        :, 0, 0
    ]


def suction_surface(grid, machine, i_row, spf, H=None, surfaces=None):
    """Return surface distance, isentropic Mach number and shape factor along a
    row's suction side.

    At span fraction `spf`, from the stagnation point to the trailing edge,
    ordered by normalised surface distance in [0, 1]. The suction side is the
    surface with the higher mean isentropic Mach number, where the layer grows
    and separates. The shape factor is the kinematic ``delta* / theta``. `H`
    and `surfaces`, from :func:`blade_shape_factor` and
    :func:`cut_blade_surfs`, may be passed in to share them between rows.

    Returns
    -------
    zeta, mas, h : ndarray or None
        None, None, None when the row or that span has no blade surface to cut.

    """
    if H is None:
        H = blade_shape_factor(grid)
    if surfaces is None:
        surfaces = cut_blade_surfs(grid)
    if surfaces[i_row] is None or H[i_row] is None:
        return None, None, None

    surface = surfaces[i_row][0]
    cut, xr = cut_section(surface, machine.annulus, i_row, spf)
    if cut is None:
        return None, None, None

    mas = isentropic_mach(cut, machine.mean_line[:, i_row].s[0])[:, 0]
    xrt_nose = machine.rows[i_row].blade.evaluate_section(spf)[0][:3, 0]
    zeta = normalise_surface_distance(cut, mas, xrt_nose)

    h = cut_field(surface, H[i_row][0], xr)
    side = np.sign(zeta)
    ss = 1.0 if np.nanmean(mas[side > 0]) > np.nanmean(mas[side < 0]) else -1.0
    on = side == ss
    order = np.argsort(np.abs(zeta[on]))
    return np.abs(zeta[on])[order], mas[on][order], h[on][order]


def shape_factor_line(zeta, mas, h, zeta_TE=0.97):
    """Return the ends of a linear shape factor target along the suction side.

    A straight line in shape factor against surface distance, from where the
    flow behind the shock has recovered to the trailing edge: it starts at the
    least shape factor behind the shock, searched from where the suction
    surface Mach number first falls back below one after its peak, or at the
    least shape factor anywhere if the surface never goes supersonic, and ends
    at `zeta_TE`, short of the trailing edge where the layer is not a layer. A
    boundary layer growing linearly between the two diffuses evenly, with no
    part of the surface working harder than the rest.

    Behind the spike the shock raises, not at the sonic point: from the sonic
    point, the line would run over the layer relaxing from the shock, which no
    camber removes.

    Returns
    -------
    (zeta_0, h_0), (zeta_1, h_1) : tuple of float
        The two ends of the line.

    """
    ok = np.isfinite(h) & np.isfinite(mas)
    zeta, mas, h = zeta[ok], mas[ok], h[ok]
    i_peak = int(np.argmax(mas))
    behind = np.flatnonzero((mas < 1.0) & (np.arange(len(mas)) > i_peak))
    if mas[i_peak] >= 1.0 and len(behind):
        search = (zeta >= zeta[behind[0]]) & (zeta < zeta_TE)
    else:
        search = zeta < zeta_TE
    zeta_0 = float(zeta[search][np.argmin(h[search])])
    h_0 = float(np.interp(zeta_0, zeta, h))
    h_1 = float(np.interp(zeta_TE, zeta, h))
    return (zeta_0, h_0), (float(zeta_TE), h_1)


def mach_smoothness(zeta, mas, zeta_TE=0.97):
    """Return the subsonic smoothness area of a suction surface Mach number.

    The area between the suction surface isentropic Mach number and a cubic
    fitted to it by least squares, from where the flow behind the shock is
    first subsonic, or from the peak if the surface never goes supersonic, to
    `zeta_TE`. Zero for a distribution with no bumps; it says how bumpy, not
    which way to move, so it is a monitor rather than something to step on.

    From the subsonic point rather than the peak, because no cubic follows a
    shock and the area would then measure the shock rather than the diffusion
    behind it.

    Returns
    -------
    area : float
        Unsigned area between the two curves [--], NaN when there are too few
        points behind the peak to fit.
    zeta, fit : ndarray
        Surface distance over the fit, and the cubic there, for drawing.

    """
    ok = np.isfinite(mas)
    zeta, mas = zeta[ok], mas[ok]
    i_peak = int(np.argmax(mas))
    behind = np.flatnonzero((mas < 1.0) & (np.arange(len(mas)) > i_peak))
    zeta_0 = zeta[behind[0]] if mas[i_peak] >= 1.0 and len(behind) else zeta[i_peak]
    on = (zeta >= zeta_0) & (zeta <= zeta_TE)
    if on.sum() < 4:
        # Peak at the trailing edge: nothing behind it to fit a cubic to
        return np.nan, zeta[on], mas[on]
    fit = np.polyval(np.polyfit(zeta[on], mas[on], 3), zeta[on])
    area = float(np.trapezoid(np.abs(mas[on] - fit), zeta[on]))
    return area, zeta[on], fit

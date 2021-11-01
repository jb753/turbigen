"""Functions to produce a H-mesh from stage design."""
import numpy as np
import design

# def

# Configure numbers of points
nxb = 97  # Blade chord
nr = 81  # Span
nr_casc = 4  # Radial points in cascade mode
nrt = 65  # Pitch
rate = 0.5  # Axial chords required to fully relax
dxsmth_c = 0.25  # Distance over which to fillet shroud corners


def _cluster(npts):
    """Return a cosinusoidal clustering function with a set number of points."""
    # Define a non-dimensional clustering function
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))


def streamwise_grid(dx_c):
    """Generate non-dimensional streamwise grid vector for a blade row.

    The first step in generating an H-mesh is to lay out a vector of axial
    coordinates --- all grid points at a fixed streamwise index are at the same
    axial coordinate.  Specify the number of points across the blade chord,
    clustered towards the leading and trailing edges. The clustering is then
    mirrored up- and downstream of the row. If the boundary of the row is
    within half a chord of the leading or trailing edges, the clustering is
    truncated. Otherwise, the grid is extendend with constant cell size the
    requested distance.

    The coordinate system origin is the row leading edge. The coordinates are
    normalised by the chord such that the trailing edge is at unity distance.

    Parameters
    ----------
    dx_c: array, length 2
        Distances to row inlet and exit planes, normalised by axial chord [--].

    Returns
    -------
    x_c: array
        Streamwise grid vector, normalised by axial chord [--].

    """

    clust = _cluster(nxb)
    dclust = np.diff(clust)
    dmax = dclust.max()

    # Stretch clustering outside of blade row
    nxb2 = nxb // 2  # Blade semi-chord
    x_c = clust + 0.0  # Make a copy of clustering function
    x_c = np.insert(x_c[1:], 0, clust[nxb2:] - 1.0)  # In front of LE
    x_c = np.append(x_c[:-1], x_c[-1] + clust[: nxb2 + 1])  # Behind TE

    # Numbers of points in inlet/outlet
    # Half a chord subtracted to allow for mesh stretching from LE/TE
    # N.B. Can be negative if we are going to truncate later
    nxu, nxd = [int((dx_ci - 0.5) / dmax) for dx_ci in dx_c]

    if nxu > 0:
        # Inlet extend inlet if needed
        x_c = np.insert(x_c[1:], 0, np.linspace(-dx_c[0], x_c[0], nxu))
    else:
        # Otherwise truncate and rescale so that inlet is in exact spot
        x_c = x_c[x_c > -dx_c[0]]
        x_c[x_c < 0.0] = x_c[x_c < 0.0] * -dx_c[0] / x_c[0]
    if nxd > 0:
        # Outlet extend if needed
        x_c = np.append(x_c[:-1], np.linspace(x_c[-1], dx_c[1] + 1.0, nxd))
    else:
        # Otherwise truncate and rescale so that outlet is in exact spot
        x_c = x_c[x_c < dx_c[1] + 1.0]
        x_c[x_c > 1.0] = (x_c[x_c > 1.0] - 1.0) * dx_c[1] / (x_c[-1] - 1.0) + 1.0

    # Get indices of leading and trailing edges
    # These are needed later for patching
    i_edge = [np.where(x_c == xloc)[0][0] for xloc in [0.0, 1.0]]
    # i_edge[1] = i_edge[1] + 1

    return x_c, i_edge


def merid_grid(x_c, rm, Dr):
    """Generate meridional grid for a blade row.

    Each spanwise grid index corresponds to a surface of revolution. So the
    gridlines have the same :math:`(x, r)` meridional locations across row
    pitch.
    """

    # Evaluate hub and casing lines on the streamwise grid vector
    # Linear between leading and trailing edges, defaults to constant outside
    rh = np.interp(x_c, [0.0, 1.0], rm - Dr / 2.0)
    rc = np.interp(x_c, [0.0, 1.0], rm + Dr / 2.0)

    # Smooth the corners over a prescribed distance
    design._fillet(x_c, rh, dxsmth_c)  # Leading edge around 0
    design._fillet(x_c - 1.0, rc, dxsmth_c)  # Trailing edge about 1

    # Check htr to decide if this is a cascade
    htr = rc[0] / rh[0]
    if htr > 0.95:
        # Define a uniform span fraction row vector
        spf = np.atleast_2d(np.linspace(0.0, 1.0, nr_casc))
    else:
        # Define a clustered span fraction row vector
        spf = np.atleast_2d(_cluster(nr))

    # Evaluate radial coordinates: dim 0 is streamwise, dim 1 is radial
    r = spf * np.atleast_2d(rc).T + (1.0 - spf) * np.atleast_2d(rh).T

    return r


def b2b_grid(x_c, r2, chi, s_c, c, a=0.0):
    """Generate circumferential coordiantes for a blade row."""

    ni = len(x_c)
    nj = r2.shape[1]
    nk = nrt

    # Dimensional axial coordinates
    x = np.atleast_3d(x_c).transpose(1, 2, 0) * c
    r = np.atleast_3d(r2)

    # Determine number of blades and angular pitch
    r_m = np.mean(r[0, (0, -1), 0])

    nblade = np.round(2.0 * np.pi * r_m / (s_c * c))  # Nearest whole number
    pitch_t = 2 * np.pi / nblade

    # Preallocate and loop over radial stations
    rtlim = np.nan * np.ones((ni, nj, 2))
    for j in range(nj):

        # Retrieve blade section
        sec_x, sec_rt0, sec_rt1 = design.blade_section(chi[:, j]) * c

        # Get centroid for stacking
        Area = np.trapz(sec_rt1 - sec_rt0, sec_x)
        rt_cent = (
            np.trapz(0.5 * (sec_rt1 - sec_rt0) * (sec_rt1 + sec_rt0), sec_x) / Area
        )

        # Stack with centroid at t=0
        sec_rt0 -= rt_cent
        sec_rt1 -= rt_cent

        rtlim[:, j, 0] = np.interp(x[:, 0, 0], sec_x, sec_rt0)
        rtlim[:, j, 1] = np.interp(x[:, 0, 0], sec_x, sec_rt1) + pitch_t * r[:, j, 0]

    # Define a pitchwise clustering function with correct dimensions
    clust = np.atleast_3d(_cluster(nk)).transpose(2, 0, 1)

    # Relax clustering towards a uniform distribution at inlet and exit
    # With a fixed ramp rate
    unif_rt = np.atleast_3d(np.linspace(0.0, 1.0, nk)).transpose(2, 0, 1)
    relax = np.ones_like(x_c)
    relax[x_c < 0.0] = 1.0 + x_c[x_c < 0.0] / rate
    relax[x_c > 1.0] = 1.0 - (x_c[x_c > 1.0] - 1.0) / rate
    relax[relax < 0.0] = 0.0
    clust = relax[:, None, None] * clust + (1.0 - relax[:, None, None]) * unif_rt

    # Fill in the intermediate pitchwise points using clustering function
    rt = rtlim[..., (0,)] + np.diff(rtlim, 1, 2) * clust

    return rt


def stage_grid(stg, rm, Dr, s, c, dev, dx_c):

    # Separate spacings for stator and rotor
    dx_c_sr = ((dx_c[0], dx_c[1] / 2.0), (dx_c[1] / 2.0, dx_c[2]))

    # Streamwise grids for stator and rotor require
    x_c, ilte = zip(*[streamwise_grid(dx_ci) for dx_ci in dx_c_sr])

    # Generate radial grid
    Dr_sr = (Dr[:2], Dr[1:])
    r = [merid_grid(x_ci, rm, Dri) for x_ci, Dri in zip(x_c, Dr_sr)]

    # Evaluate blade angles
    r_rm = np.concatenate([ri[iltei, :] / rm for ri, iltei in zip(r, ilte)])
    chi = design.free_vortex(stg, r_rm[(0, 1, 3), :], (0.0, 0.0))

    # Dimensionalise x
    x = [x_ci * c for x_ci in x_c]

    # Offset the rotor so it is downstream of stator
    x[1] = x[1] + x[0][-1] - x[1][0]

    # Now we can do b2b grids
    s_c = s / c
    rt = [b2b_grid(*argsi, c=c) for argsi in zip(x_c, r, chi, s_c)]

    return x, r, rt, ilte

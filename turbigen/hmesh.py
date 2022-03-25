"""Functions to produce a H-mesh from stage design."""
import numpy as np
from . import geometry

import matplotlib.pyplot as plt

# Configure numbers of points
nxb = 97  # Blade chord
nr = 81  # Span
nr_casc = 4  # Radial points in cascade mode
nrt = 65  # Pitch
rate = 0.5  # Axial chords required to fully relax
dxsmth_c = 0.1  # Distance over which to fillet shroud corners
tte = 0.04  # Trailing edge thickness

def streamwise_grid(dx_c):
    """Generate non-dimensional streamwise grid vector for a blade row.

    The first step in generating an H-mesh is to lay out a vector of axial
    coordinates --- all grid points at a given streamwise index are at the same
    axial coordinate.  Fix the number of points across the blade chord,
    clustered towards the leading and trailing edges. The clustering is then
    mirrored up- and downstream of the row. If the boundary of the row is
    within half a chord of the leading or trailing edges, the clustering is
    truncated. Otherwise, the grid is extendend with constant cell size the
    requested distance.

    The coordinate system origin is the row leading edge. The coordinates are
    normalised by the chord such that the trailing edge is at unity distance.

    Parameters
    ----------
    dx_c: (2,) or (nrow, 2) array [--]
        Distances to row inlet and exit planes, normalised by axial chord. If
        two dimensional, evaluate each row and stack the return values.

    Returns
    -------
    x_c: (nx,) or (nrow, nx) array [--]
        Streamwise grid vector or vectors, normalised by axial chord.
    ilete: (2,) or (nrow, 2) array [--]
        Indices for the blade leading and trailing edges.

    """

    # Call recursivly if multiple rows are input
    if np.ndim(dx_c) > 1:
        return zip(*[streamwise_grid(dx_ci) for dx_ci in dx_c])

    clust = geometry.cluster_cosine(nxb)
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
        x_c[x_c > 1.0] = (x_c[x_c > 1.0] - 1.0) * dx_c[1] / (
            x_c[-1] - 1.0
        ) + 1.0

    # Get indices of leading and trailing edges
    # These are needed later for patching
    i_edge = [np.where(x_c == xloc)[0][0] for xloc in [0.0, 1.0]]

    return x_c, i_edge


def merid_grid(x_c, rm, Dr):
    """Generate meridional grid for a blade row.

    Each spanwise grid index corresponds to a surface of revolution. So the
    gridlines have the same :math:`(x, r)` meridional locations across the
    entire row pitch.

    Parameters
    ----------
    x_c: (nx,) or (nrow, nx) array [--]
        Streamwise grid vector or vectors normalised by axial chord .
    rm: float [m]
        A constant mean radius for this blade row or rows.
    Dr: (2,) or (nrow, 2) array [m]
        Annulus spans at inlet and exit of blade row.

    Returns
    -------
    r : (nx, nr) or (nrow, nx, nr) array [m]
        Radial coordinates for each point in the meridional view.

    """

    # If multiple rows are input, call recursively and stack them
    if isinstance(x_c, tuple):
        return [merid_grid(x_ci, rm, Dri) for x_ci, Dri in zip(x_c, Dr)]

    # Evaluate hub and casing lines on the streamwise grid vector
    # Linear between leading and trailing edges, defaults to constant outside
    rh = np.interp(x_c, [0.0, 1.0], rm - Dr / 2.0)
    rc = np.interp(x_c, [0.0, 1.0], rm + Dr / 2.0)

    # Smooth the corners over a prescribed distance
    geometry.fillet(x_c, rh, dxsmth_c)  # Leading edge around 0
    geometry.fillet(x_c - 1.0, rc, dxsmth_c)  # Trailing edge about 1

    # Check htr to decide if this is a cascade
    htr = rh[0] / rc[0]
    if htr > 0.95:
        # Define a uniform span fraction row vector
        spf = np.atleast_2d(np.linspace(0.0, 1.0, nr_casc))
    else:
        # Define a clustered span fraction row vector
        spf = np.atleast_2d(geometry.cluster_cosine(nr))

    # Evaluate radial coordinates: dim 0 is streamwise, dim 1 is radial
    r = spf * np.atleast_2d(rc).T + (1.0 - spf) * np.atleast_2d(rh).T

    return r


def b2b_grid(x, r, s, c, sect):
    """Generate circumferential coordinates for a blade row."""

    ni = len(x)
    nj = r.shape[1]
    nk = nrt

    # Dimensional axial coordinates
    x = np.reshape(x, (-1, 1, 1))
    r = np.atleast_3d(r)

    x_c = x / c

    # Determine number of blades and angular pitch
    r_m = np.mean(r[0, (0, -1), 0])
    nblade = np.round(2.0 * np.pi * r_m / s)  # Nearest whole number
    pitch_t = 2 * np.pi / nblade

    # Preallocate and loop over radial stations
    rtlim = np.nan * np.ones((ni, nj, 2))
    for j in range(nj):

        # Retrieve blade section as [surf, x or y, index]
        loop_xrt = sect[j]

        # Offset so that LE at x=0
        loop_xrt[0] -= loop_xrt[0].min()

        # Rescale based on chord and max x
        loop_xrt *= c / loop_xrt[0].max()

        # Area and centroid of the loop
        terms_cross = (
            loop_xrt[0, :-1] * loop_xrt[1, 1:]
            - loop_xrt[0, 1:] * loop_xrt[1, :-1]
        )
        terms_rt = loop_xrt[1, :-1] + loop_xrt[1, 1:]
        Area = 0.5 * np.sum(terms_cross)
        rt_cent = np.sum(terms_rt * terms_cross) / 6.0 / Area

        # Now split the loop back up based on true LE/TE
        ile = np.argmin(loop_xrt[0])
        ite = np.argmax(loop_xrt[0])
        upper_xrt = loop_xrt[:, ile : (ite + 1)]
        lower_xrt = np.insert(
            np.flip(loop_xrt[:, ite:-1], -1), 0, loop_xrt[:, ile], -1
        )

        # fig, ax = plt.subplots()
        # ax.plot(*upper_xrt,'-x')
        # ax.plot(*lower_xrt,'-x')
        # ax.axis('equal')
        # # ax.plot(x_cent, rt_cent,'*k')
        # plt.savefig("test2.pdf")
        # quit()

        # Stack with centroid at t=0
        upper_xrt[1, :] -= rt_cent
        lower_xrt[1, :] -= rt_cent

        # print(np.interp(0., *upper_xrt)-np.interp(0., *lower_xrt))

        rtlim[:, j, 0] = np.interp(x[:, 0, 0], *upper_xrt)
        rtlim[:, j, 1] = (
            np.interp(x[:, 0, 0], *lower_xrt) + pitch_t * r[:, j, 0]
        )

    # Define a pitchwise clustering function with correct dimensions
    clust = geometry.cluster_cosine(nk).reshape(1, 1, -1)

    # Relax clustering towards a uniform distribution at inlet and exit
    # With a fixed ramp rate
    unif_rt = np.linspace(0.0, 1.0, nk).reshape(1, 1, -1)
    relax = np.ones_like(x_c)
    relax[x_c < 0.0] = 1.0 + x_c[x_c < 0.0] / rate
    relax[x_c > 1.0] = 1.0 - (x_c[x_c > 1.0] - 1.0) / rate
    relax[relax < 0.0] = 0.0
    clust = relax * clust + (1.0 - relax) * unif_rt

    # Fill in the intermediate pitchwise points using clustering function
    rt = rtlim[..., (0,)] + np.diff(rtlim, 1, 2) * clust

    return rt


def stage_grid(Dstg, A, dx_c, min_inscribed_radius=None):
    """Generate an H-mesh for a turbine stage."""

    # Distribute the spacings between stator and rotor
    dx_c = np.array([[dx_c[0], dx_c[1] / 2.0], [dx_c[1] / 2.0, dx_c[2]]])

    # Streamwise grids for stator and rotor
    x_c, ilte = streamwise_grid(dx_c)
    x = [ x_ci * Dstg.cx[0] for x_ci in x_c ]

    # Generate radial grid
    Dr = np.array([Dstg.Dr[:2], Dstg.Dr[1:]])
    r = merid_grid(x_c, Dstg.rm, Dr)

    # Evaluate radial blade angles
    r1 = r[0][ilte[0][0], :]
    spf = (r1 - r1.min()) / r1.ptp()
    chi = np.stack((Dstg.free_vortex_vane(spf), Dstg.free_vortex_blade(spf)))

    # Get sections (normalised by axial chord for now)
    sect = [
        geometry.radially_interpolate_section(spf, chii, spf, Ai)
        for chii, Ai in zip(chi, A)
    ]

    # fig, ax = plt.subplots()
    # ax.plot(*sect[0][2,...])
    # ax.plot(*sect[1][2,...])
    # ax.axis('equal')
    # plt.savefig('sect.pdf')
    # quit()

    # If we have asked for a minimum inscribed circle, confirm that the
    # constraint is not violated
    if min_inscribed_radius:
        for row_sect in sect:
            for rad_sect in row_sect:
                current_radius = geometry.largest_inscribed_circle(rad_sect.T)
                if current_radius < min_inscribed_radius:
                    raise geometry.ConstraintError(
                        (
                            "Thickness is too small for the constraint "
                            "inscribed circle: %.3f < %.3f" %
                            (current_radius, min_inscribed_radius)
                        )
                    )

    # Now we can do b2b grids
    rt = [b2b_grid(*args) for args in zip(x, r, Dstg.s, Dstg.cx, sect)]

    # Offset the rotor so it is downstream of stator
    x[1] = x[1] + x[0][-1] - x[1][0]

    # fig, ax = plt.subplots()
    # ax.plot(x[0],rt[0][:,0,(0,-1)])
    # ax.plot(x[1],rt[1][:,0,(0,-1)])
    # ax.axis('equal')
    # plt.savefig('sect.pdf')
    # quit()

    return x, r, rt, ilte

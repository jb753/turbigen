"""Functions to produce a H-mesh from stage design."""
import numpy as np
from . import design, geometry
from scipy.interpolate import interp1d

# import matplotlib.pyplot as plt

# Configure numbers of points
nxb = 97  # Blade chord
nr = 81  # Span
nr_casc = 4  # Radial points in cascade mode
nrt = 65  # Pitch
rate = 0.5  # Axial chords required to fully relax
dxsmth_c = 0.25  # Distance over which to fillet shroud corners
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
    dx_c: array, length 2
        Distances to row inlet and exit planes, normalised by axial chord [--].

    Returns
    -------
    x_c: array
        Streamwise grid vector, normalised by axial chord [--].

    """

    clust = geometry.cluster(nxb)
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
    # i_edge[1] = i_edge[1] + 1

    return x_c, i_edge


def merid_grid(x_c, rm, Dr):
    """Generate meridional grid for a blade row.

    Each spanwise grid index corresponds to a surface of revolution. So the
    gridlines have the same :math:`(x, r)` meridional locations across the
    entire row pitch.

    Parameters
    ----------
    x_c: array, length nx
        Streamwise grid vector normalised by axial chord [--].
    rm: float
        A constant mean radius for this blade row [m].
    Dr: array, length 2
        Annulus spans at inlet and exit of blade row [m].

    Returns
    -------
    r : array, nx by nr
        Radial coordinates for each point in the meridional view [m].

    """

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
        spf = np.atleast_2d(geometry.cluster(nr))

    # Evaluate radial coordinates: dim 0 is streamwise, dim 1 is radial
    r = spf * np.atleast_2d(rc).T + (1.0 - spf) * np.atleast_2d(rh).T

    return r


def section_coords(r, chi, A=None, spf_A=None):
    """Generate blade sections at all radii."""

    if A is None:
        # If no detailed thickness specified, use preliminary
        sec_xrt = np.array(
            [geometry.blade_section(chii, tte=tte) for chii in chi.T]
        )
    elif spf_A is None:
        # If no span fraction specified, use same A at all radii
        sec_xrt = [geometry.blade_section(chii, A=A, tte=tte) for chii in chi.T]
    else:
        # Interpolate thickness coeffs as function of span fraction
        spf = (r - r.min()) / (r.max() - r.min())
        A_func = interp1d(spf_A, A, axis=0)
        sec_xrt = [
            geometry.blade_section(chii, A=A_func(spfi), tte=tte)
            for spfi, chii in zip(spf, chi.T)
        ]

    return sec_xrt


def b2b_grid(x_c, r2, chi, s_c, c, A=None):
    """Generate circumferential coordinates for a blade row."""

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

        # Retrieve blade section as [surf, x or y, index]
        sec_xrt = geometry.blade_section(chi[:, j], A)

        # Join to a loop
        loop_xrt = np.concatenate(
            (sec_xrt[0], np.flip(sec_xrt[1, :, 1:-1], -1)), -1
        )

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
    clust = np.atleast_3d(geometry.cluster(nk)).transpose(2, 0, 1)

    # Relax clustering towards a uniform distribution at inlet and exit
    # With a fixed ramp rate
    unif_rt = np.atleast_3d(np.linspace(0.0, 1.0, nk)).transpose(2, 0, 1)
    relax = np.ones_like(x_c)
    relax[x_c < 0.0] = 1.0 + x_c[x_c < 0.0] / rate
    relax[x_c > 1.0] = 1.0 - (x_c[x_c > 1.0] - 1.0) / rate
    relax[relax < 0.0] = 0.0
    clust = (
        relax[:, None, None] * clust + (1.0 - relax[:, None, None]) * unif_rt
    )

    # Fill in the intermediate pitchwise points using clustering function
    rt = rtlim[..., (0,)] + np.diff(rtlim, 1, 2) * clust

    return rt


def stage_grid(stg, rm, Dr, s, c, dev, dx_c, Asta=None, Arot=None):

    # Distribute the spacings for stator and rotor
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
    A = (Asta, Arot)
    rt = [
        b2b_grid(*argsi[:-1], c=c, A=argsi[-1])
        for argsi in zip(x_c, r, chi, s_c, A)
    ]

    return x, r, rt, ilte

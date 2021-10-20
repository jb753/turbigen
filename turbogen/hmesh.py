"""Functions to produce a H-mesh from stage design."""
import numpy as np
from turbogen import make_design
import matplotlib.pyplot as plt

# def 

# Configure numbers of points
nxb = 97  # Blade chord
nr = 81  # Span
nrt = 65  # Pitch

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
    dmin, dmax = dclust.min(), dclust.max()

    # Stretch clustering outside of blade row
    nxb2 = nxb // 2  # Blade semi-chord
    x_c = clust + 0.  # Make a copy of clustering function
    x_c = np.insert(x_c[1:], 0, clust[nxb2:] - 1.0)  # In front of LE
    x_c = np.append(x_c[:-1], x_c[-1] + clust[:nxb2+1])  # Behind TE

    # Numbers of points in inlet/outlet
    # Half a chord subtracted to allow for mesh stretching from LE/TE
    # N.B. Can be negative if we are going to truncate later
    nxu, nxd = [int((dx_ci - 0.5) / dmax) for dx_ci in dx_c]

    if nxu > 0:
        # Inlet extend inlet if needed
        x_c = np.insert(x_c[1:], 0, np.linspace(-dx_c[0],x_c[0], nxu))
    else:
        # Otherwise truncate and rescale so that inlet is in exact spot 
        x_c = x_c[x_c > -dx_c[0]]
        x_c[x_c < 0.] = x_c[x_c < 0.] * -dx_c[0]/x_c[0]
    if nxd > 0:
        # Outlet extend if needed
        x_c = np.append(x_c[:-1], np.linspace(x_c[-1], dx_c[1] +1., nxd))
    else:
        # Otherwise truncate and rescale so that outlet is in exact spot 
        x_c = x_c[x_c < dx_c[1]+1.]
        x_c[x_c > 1.] = ((x_c[x_c > 1.] - 1.) * dx_c[1]/(x_c[-1] - 1.) + 1.)

    # Get indices of leading and trailing edges
    # These are needed later for patching
    i_edge = [np.where(x_c == xloc)[0][0] for xloc in [0., 1.]]
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
    rh = np.interp(x_c, [0., 1.], rm - Dr / 2.0)
    rc = np.interp(x_c, [0., 1.], rm + Dr / 2.0)

    # Smooth the corners over a prescribed distance
    dxsmth_c = 0.25
    make_design._fillet(x_c, rh, dxsmth_c)  # Leading edge around 0
    make_design._fillet(x_c - 1., rc, dxsmth_c)  # Trailing edge about 1

    # Define a clustered span fraction row vector 
    spf = np.atleast_2d(_cluster(nr))

    # Evaluate radial coordinates: dim 0 is streamwise, dim 1 is radial
    r = spf * np.atleast_2d(rc).T + (1.-spf)* np.atleast_2d(rh).T

    # x = np.atleast_2d(x_c).T*0.1
    # f,a = plt.subplots()
    # a.plot(x,r,'k-')
    # # a.axis('equal')
    # plt.show()

    return r


def b2b_grid(x_c, r2, iedge, chi, c, s_c, a=0.0):
    """Generate circumferential coordiantes for a blade row."""

    ni = len(x_c)
    nj = r2.shape[1]
    nk = nrt

    # Define a pitchwise clustering function with correct dimensions
    clust3 = np.tile((_cluster(nk)[..., None, None]).transpose((2, 1, 0)),(ni,nj,1))

    # Dimensional axial coordinates
    x = np.atleast_3d(x_c).transpose(1,2,0) * c
    r = np.atleast_3d(r2)
    print(x.shape)

    # Determine number of blades and angular pitch
    r_m = np.mean(r[0,(0,1)])
    nblade = np.round(2.*np.pi*r_m/(s_c*c))  # Nearest whole number
    pitch_t = 2*np.pi*r_m/nblade

    # Preallocate and loop over radial stations
    rtlim = np.nan * np.ones((ni,nj,2))
    for j in range(nj):

        # Retrieve blade section
        sec_x, sec_rt0, sec_rt1 = make_design.blade_section(chi[:, j], 0.) * c

        # Get centroid for stacking
        Area = np.trapz(sec_rt1 - sec_rt0, sec_x)
        rt_cent = (
                np.trapz( 0.5
                    * (sec_rt1 - sec_rt0)
                    * (sec_rt1 + sec_rt0),
                    sec_x)
                / Area
            )

        # Stack with centroid at t=0
        sec_rt0 -= rt_cent
        sec_rt1 -= rt_cent

        rtlim[:,j,0] = np.interp(x[:,0,0], sec_x, sec_rt0)
        rtlim[:,j,-1] = np.interp(x[:,0,0], sec_x, sec_rt1) + pitch_t * r[:,j,0]

    # Fill in the intermediate pitchwise points using clustering function
    rt = rtlim[...,(0,)] + np.diff(rtlim,1,2) * clust3

    # # Set endpoints to a uniform distribution
    # unif_rt = np.linspace(0.0, 1.0, nrt)
    # unif_rt3 = (unif_rt[..., None, None]).transpose((2, 1, 0))
    # rt[(0, -1), :, :] = (
    #     rt[(0, -1), :, 0][..., None]
    #     + (rt[(0, -1), :, -1] - rt[(0, -1), :, 0])[..., None] * unif_rt3
    # )

    # We need to map streamwise indices to fractions of cluster relaxation
    # If we have plenty of space, relax linearly over 1 chord, then unif
    # If we have less than 1 chord, relax linearly all the way
    # Relax clustering linearly

    # if (x[ii[0]] - x[0]) / c > 1.0:
    #     xnow = x[: ii[0]] - x[ii[0]]
    #     icl = np.where(xnow / c > -1.0)[0]
    #     lin_x_up = np.zeros((ii[0],))
    #     lin_x_up[icl] = np.linspace(0.0, 1.0, len(icl))
    # else:
    #     lin_x_up = np.linspace(0.0, 1.0, ii[0])

    # if (x[-1] - x[ii[1] - 1]) / c > 1.0:
    #     icl = np.where(np.abs((x - x[ii[1] - 1]) / c - 0.5) < 0.5)[0]
    #     lin_x_dn = np.zeros((len(x),))
    #     lin_x_dn[icl] = np.linspace(1.0, 0.0, len(icl))
    #     lin_x_dn = lin_x_dn[-(len(x) - ii[1]) :]
    # else:
    #     lin_x_dn = np.linspace(1.0, 0.0, len(x) - ii[1])

    # lin_x_up3 = lin_x_up[..., None, None]
    # lin_x_dn3 = lin_x_dn[..., None, None]

    # rt[: ii[0], :, :] = (
    #     rt[0, :, :][None, ...]
    #     + (rt[ii[0], :, :] - rt[0, :, :])[None, ...] * lin_x_up3
    # )
    # rt[ii[1] :, :, :] = (
    #     rt[-1, :, :][None, ...]
    #     + (rt[ii[1], :, :] - rt[-1, :, :])[None, ...] * lin_x_dn3
    # )

    return rt

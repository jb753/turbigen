"""Geometry functions for manipulating aerofoil sections and annulus lines."""
import numpy as np
from scipy.special import binom
from numpy.linalg import lstsq
from scipy.optimize import newton, fminbound
from scipy.interpolate import interp1d
from scipy.spatial import Voronoi
# import matplotlib.path as mplpath

nx = 201

## Private methods


def _prelim_thickness(x, tte=0.04, xtmax=0.2, tmax=0.15):
    """A rough cubic thickness distribution."""
    tle = tte / 2.0
    tlin = tle + x * (tte - tle)
    # Additional component to make up maximum thickness
    tmod = tmax - xtmax * (tte - tle) - tle
    # Total thickness
    thick = tlin + tmod * (
        1.0 - 4.0 * np.abs(x ** (np.log(0.5) / np.log(xtmax)) - 0.5) ** 2.0
    )
    return thick


## Public API


class GeometryConstraintError(Exception):
    """Throw this when a geometric constraint is violated."""

    pass


def fillet(x, r, dx):
    """Fillet over a join at |x|<dx."""

    # Get indices for the points at boundary of fillet
    ind = np.array(np.where(np.abs(x) <= dx)[0])
    ind1 = ind[
        (0, -1),
    ]

    dr = np.diff(r) / np.diff(x)

    # Assemble matrix problem
    rpts = r[ind1]
    xpts = x[ind1]
    drpts = dr[ind1]

    b = np.atleast_2d(np.concatenate((rpts, drpts))).T
    A = np.array(
        [
            [xpts[0] ** 3.0, xpts[0] ** 2.0, xpts[0], 1.0],
            [xpts[1] ** 3.0, xpts[1] ** 2.0, xpts[1], 1.0],
            [3.0 * xpts[0] ** 2.0, 2.0 * xpts[0], 1.0, 0.0],
            [3.0 * xpts[1] ** 2.0, 2.0 * xpts[1], 1.0, 0.0],
        ]
    )
    poly = np.matmul(np.linalg.inv(A), b).squeeze()

    r[ind] = np.polyval(poly, x[ind])


def cluster_cosine(npts):
    """Return a cosinusoidal clustering function with a set number of points."""
    # Define a non-dimensional clustering function
    return 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, npts)))


def cluster_hyperbola(npts, fac=3.0):
    """Return a hyperbolic tangent clustering function."""
    xhat = np.linspace(-1.0, 1.0, npts)
    return np.tanh(fac * xhat) / 2.0 / np.tanh(fac) + 0.5


def cluster_wall(npts, ER, dwall):
    """."""

    def iter_dist(ERi, nci):
        """Function to make a distribution with given num of const cells."""
        dy_half = dwall * ERi ** np.arange(0, (npts - nci) // 2)
        dy_const = np.ones((nci,)) * dy_half[-1]
        dy = np.concatenate((dy_half, dy_const, np.flip(dy_half)))
        return np.insert(np.cumsum(dy), 0, 0)

    # Initial guess with no constant cells
    y0 = iter_dist(ER, 0)
    dy0 = np.diff(y0).max()
    err = 1.0 - y0[-1]
    if err < 0.0:
        raise Exception("Not going to work.")

    # Now find number of constant cells to get just over correct distance
    nconst_target = np.ceil(err / dy0).astype(int)

    for nconst in [nconst_target, nconst_target - 1]:

        # if np.mod(npts,2):
        #     nconst-=1
        if nconst > npts:
            raise Exception("Not going to work.")

        # Reduce expansion ratio very slightly until exactly correct distance
        def iter_ER(ERi):
            erri = np.abs(iter_dist(ERi, nconst)[-1] - 1.0)
            return erri

        ER_tweak = fminbound(iter_ER, x1=1.001, x2=ER * 1.1)

        y = iter_dist(ER_tweak, nconst)
        if len(y) == npts:
            break

    # Force to symmetry
    tol = dwall / 10.0
    if np.abs(y[-1] - 1.0) > tol:
        raise Exception("Not going to work.")
    y[
        (0, 1, -2, -1),
    ] = (0.0, dwall, 1.0 - dwall, 1.0)
    y = 0.5 * y + 0.5 * (1.0 - np.flip(y))

    return y


def cluster_wall_solve_npts(ER, dwall):
    """Determine minimum points needed at a given dwall and ER."""
    max_iter = 100
    npts = 8
    for _ in range(max_iter):
        try:
            return cluster_wall(npts, ER, dwall)
        except:
            npts += 8


def cluster_wall_solve_ER(npts, dwall):
    """Determine correct ER at given npts."""
    max_iter = 10000
    ER = 1.001
    for _ in range(max_iter):
        try:
            y = cluster_wall(npts, ER, dwall)
            # print('Final ER = %.3f' % ER)
            return y
        except:
            ER += 0.0001


def A_from_Rle_thick_beta(Rle, thick, beta, tte):
    """Assemble shape-space coeffs from more physical quantities.

    Parameters
    ----------
    Rle : float [--]
        Leading-edge radius normalised by axial chord.
    thick : (2, n) array [--]
        `n` thickness coefficients for the pressure and suction sides.
    beta : float [deg]
        Trailing edge wedge angle.
    tte : float [--]
        Trailing edge thickness, normalised by axial chord.

    Returns
    -------
    A : (2, n+2) array [-]]
        The full set of thickness coefficients for upper and lower surfaces.
    """

    Ale = np.sqrt(2.0 * Rle)
    Ate = np.tan(np.radians(beta)) + tte
    thick = np.reshape(thick, (2, -1))
    n = thick.shape[1]

    A = np.empty((2, n + 2))
    A[:, 1:-1] = thick
    A[:, 0] = Ale
    A[:, -1] = Ate

    return A


def prelim_A():
    """Get values of A corresponding to preliminary thickness distribution."""

    xc = cluster_cosine(nx)

    # Camber line
    thick = _prelim_thickness(xc)

    # Choose arbitrary camber line (A independent of chi)
    chi = (-10.0, 20.0)

    # Assemble preliminary upper and lower coordiates
    xy_prelim = [_thickness_to_coord(xc, sgn * thick, chi) for sgn in [1, -1]]

    # Fit Bernstein polynomials to the prelim coords in shape space
    A, _ = _fit_aerofoil(xy_prelim, chi, order=4)

    return A


def _section_xy(chi, A, tte, stag, x=None):
    r"""Coordinates for blade section with specified camber and thickness."""

    # Choose some x coordinates if not provided
    if x is None:
        x = cluster_cosine(nx)

    # Convert from shape space to thickness
    s = _evaluate_coefficients(x, A)
    t = _from_shape_space(x, s, zte=tte)

    # Flip the lower thickness
    t[1] = -t[1]

    # Apply thickness to camber line and return
    xy = np.stack([_thickness_to_coord(x, ti, chi, stag) for ti in t])
    return xy


def _bernstein(x, n, i):
    """Evaluate ith Bernstein polynomial of degree n at some x-coordinates."""
    return binom(n, i) * x ** i * (1.0 - x) ** (n - i)


def _to_shape_space(x, z, zte):
    """Transform real thickness to shape space."""
    # Ignore singularities at leading and trailing edges
    eps = 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        ii = np.abs(x - 0.5) < (0.5 - eps)
    s = np.ones(x.shape) * np.nan
    s[ii] = (z[ii] - x[ii] * zte) / (np.sqrt(x[ii]) * (1.0 - x[ii]))
    # s[ii] = z[ii] / (np.sqrt(x[ii]) * np.sqrt(1.0 - x[ii]))
    return s


def _from_shape_space(x, s, zte):
    """Transform shape space to real coordinates."""
    return np.sqrt(x) * (1.0 - x) * s + x * zte
    # return np.sqrt(x) * np.sqrt(1.0 - x) * s


def _rotate_section(sect, gam):
    """Restagger a blade section with a solid-body rotation."""

    # Make rotation matrix
    gamr = np.radians(gam)
    rot = np.array(
        [[np.cos(gamr), -np.sin(gamr)], [np.sin(gamr), np.cos(gamr)]]
    )

    return np.matmul(rot, sect)


def _evaluate_coefficients(x, A):
    """Evaluate a set of Bernstein polynomial coefficients at some x-coords."""
    A = np.atleast_2d(A)
    nsurf, order = A.shape
    n = order - 1
    t = np.empty(
        (
            order,
            nsurf,
        )
        + x.shape
    )
    # Loop over surfaces and polynomials
    for i in range(order):
        for j in range(nsurf):
            t[i, j, ...] = A[j, i] * _bernstein(x, n, i)
    # Sum the polynomials
    return np.sum(t, axis=0)


def _fit_aerofoil(xy, chi, order):
    """Fit Bernstein polynomials to both aerofoil surfaces simultaneously."""
    n = order - 1
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    dx = 0.02
    xtrim_all = []
    strim_all = []
    X_all = []
    X_le_all = []
    for xyi in xy:
        xc, yc, t = _coord_to_thickness(xyi, chi)
        s = _to_shape_space(xc, t, 0.02)
        with np.errstate(invalid="ignore", divide="ignore"):
            itrim = np.abs(xc - 0.5) < (0.5 - dx)
        xtrim_all.append(xc[itrim])
        strim_all.append(s[itrim])
        X_all.append(
            np.stack([_bernstein(xc[itrim], n, i) for i in range(1, n + 1)]).T
        )
        X_le_all.append(_bernstein(xc[itrim], n, 0))

    strim = np.concatenate(strim_all)
    X_le = np.concatenate(X_le_all)
    X = np.block(
        [
            [X_all[0], np.zeros(X_all[1].shape)],
            [np.zeros(X_all[0].shape), X_all[1]],
        ]
    )
    X = np.insert(X, 0, X_le, 1)
    A_all, resid = lstsq(X, strim, rcond=None)[:2]
    Au = A_all[:order]
    Al = np.insert(A_all[order:], 0, A_all[0])
    return np.vstack((Au, Al)), resid


def evaluate_camber(x, chi, stag):
    """Camber line as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    tangam = np.tan(np.radians(stag))
    # return tanchi[0] * x + (tanchi[1] - tanchi[0]) * (
    #     (1.0 - aft) / 2.0 * x ** 2.0 + aft / 3.0 * x ** 3.0
    # )
    n = np.sum(tanchi) / tangam
    a = tanchi[1] / n
    b = -tanchi[0] / n
    y = a * x ** n + b * (1.0 - x) ** n
    y = y - y[0]
    return y


def evaluate_camber_slope(x, chi, stag):
    """Camber line slope as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    tangam = np.tan(np.radians(stag))
    # return tanchi[0] + (tanchi[1] - tanchi[0]) * x * (aft * x + (1.0 - aft))
    n = np.sum(tanchi) / tangam
    a = tanchi[1] / n
    b = -tanchi[0] / n
    y = a * n * x ** (n - 1.0) - b * n * (1.0 - x) ** (n - 1.0)
    return y


def _coord_to_thickness(xy, chi):
    """Perpendicular thickness distribution given camber line angles.

    Parameters
    ----------

    xy : array, 2-by-...
        Cartesian coordinates of a blade surface.
    chi : array, len 2
        Camber angles for inlet and outlet."""

    # Split into x and y
    x, y = xy
    # Find intersections of xu, yu with camber line perpendicular
    def iterate(xi):
        return (
            y
            - evaluate_camber(xi, chi)
            + (x - xi) / evaluate_camber_slope(xi, chi)
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        xc = newton(iterate, x)
    yc = evaluate_camber(xc, chi)
    # Now evaluate thickness
    t = np.sqrt(np.sum(np.stack((x - xc, y - yc), axis=0) ** 2.0, axis=0))

    return xc, yc, t


def _thickness_to_coord(xc, t, chi, stag):
    theta = np.arctan(evaluate_camber_slope(xc, chi, stag))
    yc = evaluate_camber(xc, chi, stag)
    xu = xc - t * np.sin(theta)
    yu = yc + t * np.cos(theta)
    return xu, yu


def _loop_section(xy):
    """Join a section with separate pressure and suction sides into a loop."""
    # Concatenate the upper side with a flipped version of the lower side,
    # discarding the first and last points to prevent repetition
    return np.concatenate((xy[0], np.flip(xy[1, :, 1:-1], axis=-1)), axis=-1)


def radially_interpolate_section(
    spf, chi, spf_q, tte, A=None, spf_A=None, stag=None
):
    """From radial angle distributions, interpolate aerofoil at query spans.

    Parameters
    ----------
    spf: (nr,) array [--]
        Span fractions at which metal angles are specified.
    chi: (2, nr) array [deg]
        Inlet and exit metal angles at the span fractions `spf`.
    spf_q : (nq,) array [--]
        Query span fractions to interpolate blade sections on.
    A : (2, order) or (nt, 2, order) array [--]
        Coefficients defining perpendicular thicknesses of upper and lower
        surfaces using a sum of `order` Bernstein polynomials. Either one set
        radially uniform, or `nt` sets of coefficients defined at `spf_A`.
    stag : float or (nt,) array [deg]
        Stagger angle, either uniform over span or defined at `nt` loc'ns.
    spf_A : (nt,) array, optional [--]
        If specifying thickness at multiple heights, the span fractions for
        each of the `nt` sets of thickness coefficients.

    Returns
    -------
    xrt : (nq, 2, 2, nx) array [--]
        Section coordinates normalised by axial chord. The indexes are:
            `xrt[span, upper/lower, x/rt, streamwise]`

    """

    # Check input shape
    spf = spf.reshape(-1)
    nr = len(spf)
    if not chi.shape == (2, nr):
        raise ValueError("Input metal angle data wrong shape.")

    # Interpolator for the radial angle distributions
    # Returns inlet and exit flow angle as rows
    func_chi = interp1d(spf, chi)

    # If the query span fraction is not an array, make it one
    if np.shape(spf_q) == ():
        nq = 1
        spf_q = (spf_q,)
    else:
        nq = len(spf_q)
    chi_q = func_chi(spf_q)

    # First, get thickness coefficients at query spans
    if np.ndim(A) == 2:
        # If we only have one set of thickness coefficients, just repeat them
        A_q = np.tile(np.expand_dims(A, 0), (nq, 1, 1))
        stag_q = np.tile(stag, (nq,))
    else:
        # Otherwise Interpolate thicknesses to desired spans
        A_q = interp1d(spf_A, A, axis=0)(spf_q)

    # Second, convert thickness in shape space to real coords
    sec_xrt = np.stack(
        [
            _loop_section(_section_xy(chi_i, A_i, tte, stag_i))
            for chi_i, A_i, stag_i in zip(chi_q.T, A_q, stag_q)
        ]
    )

    return np.squeeze(sec_xrt)

def _surface_length(xrt):
    # xrt : (nq, 2, 2, nx) array [--]
    nr = xrt.shape[0]
    S_cx = np.empty((nr,))
    for j in range(nr):
        x = xrt[j,0,:]
        ite = np.argmax(x)

        upper = np.diff(xrt[j,:,:ite])
        lower = np.diff(xrt[j,:,ite:])

        S_upper = np.sum(np.sqrt(np.sum(upper**2.,0)))
        S_lower = np.sum(np.sqrt(np.sum(lower**2.,0)))

        S_cx[j] = np.max((S_upper, S_lower))

    return np.mean(S_cx)


# Next job is to apply thickness constraints. Can a given xy section fit a
# circle some fraction of the chord? Scipy has voronoi builtin, then for each
# voronoi vertex find the point on the surface greatest distance away.


def largest_inscribed_circle(xy):
    """Radius of the largest circle contained within an xy polygon.

    This is useful to constrain the thickness of our blade sections. In a real
    engine, the sections must be large enough to pass, e.g. cooling channels or
    oil pipes to feed bearings.

    Parameters
    ----------
    xy: (npt,2) [--]
        Cartesian coordinates for `npt` locations forming a looped polygon.

    Returns
    -------
    max_radius: float [--]
        Radius of the largest inscribed circle that fits within polygon."""

    # Check input
    if not xy.ndim or not xy.shape[1] == 2:
        raise ValueError(
            "Shape should be (npts, 2), you input %s" % repr(xy.shape)
        )

    # Calculate Voronoi vertices (medial axis)
    vor = Voronoi(xy).vertices

    # Only include points within the section, sorted
    # TODO replace matplotlib dependency with something else
    path = mplpath.Path(xy)
    vor = vor[path.contains_points(vor)]
    vor = vor[vor[:, 0].argsort()]

    # vor is shape (m,2), xy is shape (n,2)
    # we assemble distances (n,m) from each vor point to each xy point
    vor3 = np.expand_dims(vor, 0)
    xy3 = np.expand_dims(xy, 1)
    dist = np.sqrt(np.sum((vor3 - xy3) ** 2.0, axis=-1))

    # At any point on the medial axis, we are interested in the closest point
    min_dist = dist.min(axis=0)

    # fig, ax = plt.subplots()
    # ax.axis("equal")
    # ax.plot(*xy.T)
    # ax.plot(*vor.T)
    # plt.savefig("debug_radius.pdf")

    # The largest inscribed circle fits at the point on the medial axis that is
    # furthest away from the surface
    return min_dist.max()

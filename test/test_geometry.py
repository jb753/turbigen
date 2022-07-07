"""Tests for geometry generation capabilities."""
import numpy as np
from turbigen import geometry
import scipy.optimize


# Prepare a matrix of blade section input parameters
chi1 = np.linspace(-30.0, 30.0, 6)
chi2 = np.linspace(-60.0, 60.0, 6)

tte = 0.04

geometries = []
inputs = []


def get_geometries():
    if not geometries:
        for chi1i in chi1:
            for chi2i in chi2:
                inputs.append((chi1i, chi2i))
                A = geometry.prelim_A(inputs[-1])
                geometries.append(geometry.evaluate_section_xy(inputs[-1], A))
    return geometries


def test_section_is_closed():
    """Verify the surfaces meet at leading and trailing edges."""
    for xyi in get_geometries():
        assert np.all(np.diff(xyi[..., (0, -1)], 1, 0) < 1e-9)


def test_section_x_direction():
    """Verify that the trailing edge is downstream of leading edge."""
    for xyi in geometries:
        assert np.all(np.diff(xyi[:, 0, (0, -1)], 1, -1) > 0.0)


def test_section_surface_orientation():
    """Verify that the upper surface is above lower surface."""
    for xyi in geometries:
        assert np.diff(np.mean(xyi[:, 1, :], -1), 1, 0) <= 0.0


def test_camber_angles():
    """Check that camber line slope is consistent with desired angles."""
    xc = np.array([0.0, 1.0])
    for chii in inputs:
        dyc_dx = geometry.evaluate_camber_slope(xc, chii)
        chi_out = np.degrees(np.arctan(dyc_dx))
        assert np.all(np.isclose(chii, chi_out))


def test_camber_slope():
    """Compare camber slope function to finite difference approximation."""
    xc = np.linspace(0.0, 1.0)
    for chii in inputs:
        dyc_dx = geometry.evaluate_camber_slope(xc, chii)
        yc = geometry.evaluate_camber(xc, chii)
        dyc_dx_out = np.gradient(yc, xc)
        # Exclude the ends because they use a first-order accurate difference
        assert np.all(np.isclose(dyc_dx, dyc_dx_out)[1:-1])


def test_bernstein():
    """If Bernstein polynomials implemented correctly, they sum to unity."""
    A = np.ones((6,))
    x = np.linspace(0.0, 1.0)
    t = geometry.evaluate_coefficients(x, A)
    assert np.all(np.isclose(t, 1.0))


# def test_bernstein_shape():
#     """Bernstein evaluated shape should be same as input x"""
#     A = np.ones((5,))
#     x1 = np.ones((10,))
#     x2 = np.ones((10, 7))
#     x3 = np.ones((10, 3, 6))
#     for x in (x1, x2, x3):
#         t = geometry.evaluate_coefficients(x, A)
#         assert np.shape(x) == t.shape


def test_shape_space_cycle():
    """Go from shape space to real space and back again."""
    x = np.linspace(0.0, 1)
    s = 1.0 + x ** 2.0  # Arbitrary thickness distribution
    t = geometry.from_shape_space(x, s, 0.0)
    s_out = geometry.to_shape_space(x, t, 0.0)
    # Remove the inevitable singularities from comparison
    ind = np.logical_not(np.isnan(s_out))
    assert np.all(np.isclose(s[ind], s_out[ind]))


def test_fit_shape_space():
    """Check that we can fit a low-order polynomial with negligible error."""
    x = np.linspace(0.0, 1)
    s = 1.0 + x ** 2.0  # Arbitrary thickness distribution
    params = 3
    A, resid = geometry.fit_coefficients(x, s, params)
    assert len(A) == params
    s_out = geometry.evaluate_coefficients(x, A)
    assert np.all(np.isclose(s, s_out))
    assert resid < 1e-9


# def test_fit_resample():
#     """Check when we resample coeffcients, still aprroximates original."""
#     x = np.linspace(0.0, 1)
#     s = 1.0 + x ** 2.0  # Arbitrary thickness distribution
#     A_10, _ = geometry.fit_coefficients(x, s, 10)
#     A_3, _ = geometry.resample_coefficients(A_10, 3)
#     s_out = geometry.evaluate_coefficients(x, A_3)
#     assert np.all(np.isclose(s, s_out))


def test_perpendicular_thickness_cycle():
    """We should be able to go from thickness to coords and back again."""
    x = np.linspace(0.0, 1)
    chi_ref = (-15.0, -15.0)
    t = -x * (x - 1.0)
    xy = geometry.thickness_to_coord(x, t, chi_ref)
    _, _, t_out = geometry.coord_to_thickness(xy, chi_ref)
    assert np.all(np.isclose(t, t_out))


def test_fit_aerofoil():
    """Verify that with a large number of coeffs, we can reduce error to zero."""
    order = 20
    xc = geometry.cluster(geometry.nx)
    for xyi, chii in zip(geometries, inputs):
        A, resid = geometry.fit_aerofoil(xyi, chii, order)
        # Check shape-space residual
        assert resid < 1e-9
        # Check coordinate residual
        print(A.shape)
        xy_out = np.stack(geometry.evaluate_section_xy(chii, A, xc))
        assert np.all(np.isclose(xy_out, xyi))


def test_Rle_beta():
    """Check that leading edge radius obeys Kulfan relation."""

    # Make some thickness coefficients with prescribed Rle
    thick = np.ones((2, 1)) * 0.1
    for Rle in (0.1, 0.2, 0.05):
        for beta in (8.0, 16.0, 24.0):
            A = geometry.A_from_Rle_thick_beta(Rle, thick, beta, tte)

            # Evaluate section coordinates
            chi = (0.0, 0.0)  # No camber so we can look at LE/TE easily
            xy = np.stack(geometry._loop_section(geometry._section_xy(chi, A, tte)))

            # Fit a circle to get centre point
            xfit = 0.01
            x, y = [xyi[xy[0, :] < xfit] for xyi in xy]

            def guess_center(c):
                xc, yc = c
                Rcalc = (x - xc) ** 2.0 + (y - yc) ** 2.0
                return Rcalc - Rcalc.mean()

            xc, yc = scipy.optimize.leastsq(guess_center, (xfit / 2.0, 0.0))[0]

            # Fitted radius should be close to input
            Rfit = np.sqrt(np.mean((x - xc) ** 2.0 + (y - yc) ** 2.0))
            assert np.abs(Rfit / Rle - 1.0) < 0.1

            # Now get gradient at TE for upper/lower surfs
            xy = np.stack(geometry._section_xy(chi, A, tte))
            for xyi in xy:
                x, y = [xyj[xyi[0] > 0.99] for xyj in xyi]
                dydx = np.gradient(y, x)
                beta_calc = np.mean(np.abs(np.degrees(np.arctan(dydx))))
                assert np.abs(beta_calc / beta - 1.0) < 0.1


def test_inscribed_circle():
    """Check radius of inscribed circle in a rectangle."""

    # Lay out a square polygon of side length l
    l = 1.0
    x = np.linspace(0.0, 1.0)
    ls = np.ones_like(x) * l
    zs = np.zeros_like(x)
    xf = np.flip(x)
    xall = np.concatenate((x, ls, xf, zs))
    yall = np.concatenate((zs, x, ls, xf))
    square = np.stack((xall, yall)).T

    # Calculate inscribed circle
    radius = geometry.largest_inscribed_circle(square)

    # If all is well, the radius is half side length
    # Not exact because the sides are discretised
    assert np.isclose(radius, l / 2.0, rtol=1e-3)

    # Now flip direction and try again
    square_flip = np.flip(square, axis=0)
    radius_flip = geometry.largest_inscribed_circle(square)
    assert np.isclose(radius_flip, l / 2.0, rtol=1e-3)


def test_cluster_wall():
    """Check that wall clustering distributions are working."""
    ER = 1.1
    dwall = 0.001
    for npts in [73, 74, 81, 82]:
        y = geometry.cluster_wall(npts, ER, dwall)
        dy = np.diff(y)
        ER_out = dy[1:] / dy[:-1]
        ER_out[ER_out < 1.0] = 1.0 / ER_out[ER_out < 1.0]
        ER_out = ER_out[np.abs(ER_out) > 1.0]
        # Correct range
        assert (y >= 0).all()
        assert (y <= 1.0).all()
        # Endpoints
        assert y[0] == 0.0
        assert y[-1] == 1.0
        assert np.isclose(dy[0], dwall)
        assert np.isclose(dy[-1], dwall)
        # Monotonically increasing
        assert (dy > 0.0).all()
        assert len(y) == npts
        # Symmetry
        assert np.isclose(y, 1.0 - np.flip(y)).all()
        # Target expansion ratio
        assert np.isclose(ER_out, ER, rtol=1e-1).all()


def test_cluster_wall_solve_npts():
    """Check that wall clustering distributions are working."""
    dwall = 0.001
    for ER in [1.05, 1.1, 1.2]:
        y = geometry.cluster_wall_solve_npts(ER, dwall)
        dy = np.diff(y)
        ER_out = dy[1:] / dy[:-1]
        ER_out[ER_out < 1.0] = 1.0 / ER_out[ER_out < 1.0]
        ER_out = ER_out[np.abs(ER_out) > 1.0]
        # Correct range
        assert (y >= 0).all()
        assert (y <= 1.0).all()
        # Endpoints
        assert y[0] == 0.0
        assert y[-1] == 1.0
        assert np.isclose(dy[0], dwall)
        assert np.isclose(dy[-1], dwall)
        # Monotonically increasing
        assert (dy > 0.0).all()
        # Symmetry
        assert np.isclose(y, 1.0 - np.flip(y)).all()
        # Target expansion ratio
        assert np.isclose(ER_out, ER, rtol=2e-1).all()


def test_cluster_wall_solve_ER():
    """Check that wall clustering distributions are working."""
    dwall = 0.001
    for npts in [65, 73, 81, 89, 97]:
        y = geometry.cluster_wall_solve_ER(npts, dwall)
        dy = np.diff(y)
        # Correct range
        assert (y >= 0).all()
        assert (y <= 1.0).all()
        # Endpoints
        assert y[0] == 0.0
        assert y[-1] == 1.0
        assert np.isclose(dy[0], dwall)
        assert np.isclose(dy[-1], dwall)
        # Monotonically increasing
        assert (dy > 0.0).all()
        assert len(y) == npts
        # Symmetry
        assert np.isclose(y, 1.0 - np.flip(y)).all()

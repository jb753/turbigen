"""Tests for geometry generation capabilities."""
import numpy as np
from turbigen import geometry
# import matplotlib.pyplot as plt


# Prepare a matrix of blade section input parameters
chi1 = np.linspace(-30.0, 30.0, 6)
chi2 = np.linspace(-60.0, 60.0, 6)
aft = np.linspace(-1.0, 1.0, 6)

tte = 0.04

geometries = []
inputs = []
def get_geometries():
    if not geometries:
        for chi1i in chi1:
            for chi2i in chi2:
                for afti in aft:
                    inputs.append(((chi1i, chi2i), afti))
                    geometries.append(geometry.blade_section(*inputs[-1],tte=tte))
    return geometries


def test_section_is_closed():
    """Verify the surfaces meet at leading and trailing edges."""
    for xyi in get_geometries():
        assert np.all(np.diff(xyi[..., (0, -1)],1,0)<1e-9)

def test_section_x_direction():
    """Verify that the trailing edge is downstream of leading edge."""
    for xyi in geometries:
        assert np.all(np.diff(xyi[:,0,(0, -1)],1,-1)>0.0)

def test_section_surface_orientation():
    """Verify that the upper surface is above lower surface."""
    for xyi in geometries:
        assert np.diff(np.mean(xyi[:,1,:],-1),1,0) <= 0.0

def test_camber_angles():
    """Check that camber line slope is consistent with desired angles."""
    xc = np.array([0., 1.])
    for chii, _ in inputs:
        dyc_dx = geometry.evaluate_camber_slope(xc,chii)
        chi_out = np.degrees(np.arctan(dyc_dx))
        assert np.all(np.isclose(chii, chi_out))

def test_camber_slope():
    """Compare camber slope function to finite difference approximation."""
    xc = np.linspace(0.,1.)
    for chii, _ in inputs:
        dyc_dx = geometry.evaluate_camber_slope(xc,chii)
        yc = geometry.evaluate_camber(xc,chii)
        dyc_dx_out = np.gradient(yc, xc)
        # Exclude the ends because they use a first-order accurate difference
        assert np.all(np.isclose(dyc_dx, dyc_dx_out)[1:-1])


def test_bernstein():
    """If Bernstein polynomials implemented correctly, they sum to unity."""
    A = np.ones((6,))
    x = np.linspace(0., 1.)
    t = geometry.evaluate_coefficients(x, A)
    assert np.all(np.isclose(t,1.))


def test_shape_space_cycle():
    """Go from shape space to real space and back again."""
    x = np.linspace(0., 1)
    s = 1. + x**2.  # Arbitrary thickness distribution
    t = geometry.from_shape_space(x, s, 0.)
    s_out = geometry.to_shape_space(x, t, 0.)
    # Remove the inevitable singularities from comparison
    ind = np.logical_not(np.isnan(s_out))
    assert np.all(np.isclose(s[ind], s_out[ind]))


def test_fit_shape_space():
    """Check that we can fit a low-order polynomial with negligible error."""
    x = np.linspace(0., 1)
    s = 1. + x**2.  # Arbitrary thickness distribution
    params = 3
    A, resid = geometry.fit_coefficients(x, s, params)
    assert len(A)==params
    s_out = geometry.evaluate_coefficients(x, A)
    assert np.all(np.isclose(s, s_out))
    assert resid < 1e-9

def test_fit_resample():
    """Check when we resample coeffcients, still aprroximates original."""
    x = np.linspace(0., 1)
    s = 1. + x**2.  # Arbitrary thickness distribution
    A_10, _ = geometry.fit_coefficients(x, s, 10)
    A_3, _ = geometry.resample_coefficients(A_10,3)
    s_out = geometry.evaluate_coefficients(x, A_3)
    assert np.all(np.isclose(s, s_out))

def test_perpendicular_thickness_cycle():
    """We should be able to go from thickness to coords and back again."""
    x = np.linspace(0., 1)
    chi_ref = (-15., -15.)
    t = -x*(x-1.)
    xy = geometry.thickness_to_coord(x, t, chi_ref)
    _, _, t_out = geometry.coord_to_thickness(xy, chi_ref)
    assert np.all(np.isclose(t, t_out))

def test_fit_aerofoil():
    """Verify that with a large number of coeffs, we can reduce error to zero."""
    order = 20
    xc = np.linspace(0.,1.)
    for xyi, (chii, _) in zip(geometries, inputs):
        A, resid = geometry.fit_aerofoil(xyi, chii, tte, order)
        # Check shape-space residual
        assert resid < 1e-9
        # Check coordinate residual 
        xy_out = np.array(np.stack(geometry.evaluate_aerofoil(A, xc, chii, tte)))
        assert np.all(np.isclose(xy_out,xyi))


if __name__=="__main__":
    test_shape_space_cycle()

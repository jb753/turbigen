"""Tests for geometry generation capabilities."""
import numpy as np
from turbigen import geometry


# Prepare a matrix of blade section input parameters
chi1 = np.linspace(-30.0, 30.0, 7)
chi2 = np.linspace(-60.0, 60.0, 7)
aft = np.linspace(-1.0, 1.0, 7)

geometries = []
inputs = []
def get_geometries():
    if not geometries:
        for chi1i in chi1:
            for chi2i in chi2:
                for afti in aft:
                    inputs.append(((chi1i, chi2i), afti))
                    geometries.append(geometry.blade_section(*inputs[-1]))
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



if __name__=="__main__":
    pass

"""Tests for the hmesh module"""
import numpy as np
import compflow as cf
from turbogen.hmesh import *
from turbogen import make_design

# Begin test functions

def test_streamwise():
    """Verify the properties of generated streamwise grid vectors."""
    for dx_c_in in [0.2, 0.5, 1.0, 10.]:
        for dx_c_out in [0.2, 0.5, 1.0, 10.]:

            dx_c = (dx_c_in,dx_c_out)

            # Generate grid
            x_c, (ile, ite) = streamwise_grid(dx_c)

            # Diff should always be > 0
            # i.e. x increases monotonically with index, no repeats
            assert np.all(np.diff(x_c,1)>0.)

            # Check outlet and inlet at requested points
            assert np.isclose(x_c[0],-dx_c[0])
            assert np.isclose(x_c[-1],1.+dx_c[1])

            # The leading and trailing edges should be at 0. and 1.
            assert np.isclose(x_c[ile] , 0.)
            assert np.isclose(x_c[ite] , 1.)

            # Tolerance on smoothness of expansion
            # This is sufficient for 101 points along chord
            tol = 1e-3
            assert np.all(np.abs(np.diff(x_c,2))<tol)


def test_merid():
    """Verify properties of the meridional grid."""

    dx_c = (1., 1.)
    x_c, (ile, ite) = streamwise_grid(dx_c)

    stg = make_design.nondim_stage_from_Lam(
        phi=0.8, psi=1.6, Lam=0.5, Al1=0., Ma=0.7, ga=1.4, eta=0.9
    )

    htr = 0.9
    cpTo1 = 1.e6
    Omega = 2.*np.pi*50.
    rm, Dr = make_design.annulus_line(stg, htr, cpTo1, Omega)

    merid_grid(x_c, rm, Dr[:2], 0.1)



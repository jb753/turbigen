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

    # Generate a streamwise grid first
    dx_c = (1., 1.)
    x_c, (ile, ite) = streamwise_grid(dx_c)

    # Turbine stage design
    stg = make_design.nondim_stage_from_Lam(
        phi=0.8, psi=1.6, Lam=0.5, Al1=0., Ma=0.7, ga=1.4, eta=0.9
    )

    # Annulus line
    htr = 0.6
    cpTo1 = 1.e6
    Omega = 2.*np.pi*50.
    rm, Dr = make_design.annulus_line(stg, htr, cpTo1, Omega)

    # Generate radial grid
    r = merid_grid(x_c, rm, Dr[:2])

    # Non-negative radius
    assert np.all(r>=0.)

    # Radial coordinate should monotonically increase with j index
    assert np.all(np.diff(r,1,1)>0.)

    # Check smoothness in radial direction
    tol = 1e-3
    assert np.all(np.abs(np.diff(r/rm,2,1))<tol)

    # Verify correct mean radius
    rm_calc = np.mean(r[:,(0,-1)],1)
    err_rm = np.abs(rm_calc/rm-1.)
    tol_rm = 2e-3
    assert np.all(err_rm<tol_rm)
    assert np.isclose(rm_calc[0],rm)
    assert np.isclose(rm_calc[-1],rm)

    # Verify the spans at inlet and exit
    rh_calc = r[(0,-1),0]
    rc_calc = r[(0,-1),-1]
    Dr_calc = rc_calc-rh_calc
    assert np.all(np.isclose(Dr_calc,Dr[:2]))


def test_b2b():
    """Verify properties of the blade-to-blade grid."""

    # Generate a streamwise grid first
    dx_c = (1., 1.)
    x_c, (ile, ite) = streamwise_grid(dx_c)

    # Turbine stage design
    stg = make_design.nondim_stage_from_Lam(
        phi=0.8, psi=1.6, Lam=0.5, Al1=0., Ma=0.7, ga=1.4, eta=0.9
    )

    # Annulus line
    htr = 0.9
    cpTo1 = 1.e6
    Omega = 2.*np.pi*50.
    rm, Dr = make_design.annulus_line(stg, htr, cpTo1, Omega)

    # Generate radial grid
    r_stator = merid_grid(x_c, rm, Dr[:2])
    r_rotor = merid_grid(x_c, rm, Dr[1:])

    r_rm = np.concatenate([r_stator[(ile,ite),:],r_rotor[(ite,),:]])/rm

    chi_vane, chi_blade = make_design.free_vortex(stg, r_rm, (0.,0.))

    s_c = make_design.pitch_Zweifel(stg, (0.8,0.8))

    c = 0.1
    rt = b2b_grid(x_c, r_stator, (ile, ite), chi_vane, c, s_c[0])

    jplot = -1
    f, a = plt.subplots()
    a.plot(x_c*c, rt[:,jplot,:],'k-')
    plt.show()


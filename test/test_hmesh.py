"""Tests for the hmesh module"""
import numpy as np
from turbigen import design, hmesh, geometry
from test_design import get_geometries

htr = 0.6
cpTo1 = 1.0e6
Omega = 2.0 * np.pi * 50.0
Re = 4e6
Po1 = 16e5
rgas = 287.14
ga = 1.33
Co = 0.6


cp = rgas * ga / (ga-1.)
To1 = cpTo1/cp


def test_streamwise():
    """Verify the properties of generated streamwise grid vectors."""
    for dx_c_in in [0.2, 0.5, 1.0, 10.0]:
        for dx_c_out in [0.2, 0.5, 1.0, 10.0]:

            dx_c = (dx_c_in, dx_c_out)

            # Generate grid
            x_c, (ile, ite) = hmesh.streamwise_grid(dx_c)

            # Diff should always be > 0
            # i.e. x increases monotonically with index, no repeats
            assert np.all(np.diff(x_c, 1) > 0.0)

            # Check outlet and inlet at requested points
            assert np.isclose(x_c[0], -dx_c[0])
            assert np.isclose(x_c[-1], 1.0 + dx_c[1])

            # The leading and trailing edges should be at 0. and 1.
            assert np.isclose(x_c[ile], 0.0)
            assert np.isclose(x_c[ite], 1.0)

            # Tolerance on smoothness of expansion
            # This is sufficient for 101 points along chord
            tol = 1e-3
            assert np.all(np.abs(np.diff(x_c, 2)) < tol)


def test_merid():
    """Verify properties of the meridional grid."""

    # Generate a streamwise grid first
    dx_c = (1.0, 1.0)
    x_c, (ile, ite) = hmesh.streamwise_grid(dx_c)

    # Loop over entire mean-line design space
    for stg in get_geometries("datum"):

        # Annulus line
        rm, Dr = design.annulus_line(stg, htr, cpTo1, Omega)

        # Generate radial grid
        r = hmesh.merid_grid(x_c, rm, Dr[:2])

        # Non-negative radius
        assert np.all(r >= 0.0)

        # Radial coordinate should monotonically increase with j index
        assert np.all(np.diff(r, 1, 1) > 0.0)

        # Check smoothness in radial direction
        tol = 1e-3
        assert np.all(np.abs(np.diff(r / rm, 2, 1)) < tol)

        # Verify correct mean radius
        rm_calc = np.mean(r[:, (0, -1)], 1)
        err_rm = np.abs(rm_calc / rm - 1.0)
        tol_rm = 2e-3
        assert np.all(err_rm < tol_rm)
        assert np.isclose(rm_calc[0], rm)
        assert np.isclose(rm_calc[-1], rm)

        # Verify the spans at inlet and exit
        rh_calc = r[(0, -1), 0]
        rc_calc = r[(0, -1), -1]
        Dr_calc = rc_calc - rh_calc
        assert np.all(np.isclose(Dr_calc, Dr[:2]))

def test_stage():
    """Verify properties of a stage grid."""

    # Use a rough thickness distribution
    A1 = geometry.prelim_A()*0.5
    A = np.stack((A1,A1))
    dx_c = (2.,1.,3.)

    # Loop over entire mean-line design space
    for stg in get_geometries("datum"):

        Dstg = design.scale_geometry(stg, htr, Omega, To1, Po1, rgas, Re, Co)

        x_all, r_all, rt_all, ilte_all = hmesh.stage_grid(Dstg, A, dx_c)

        for x, r, rt, ilte, s in zip(x_all, r_all, rt_all, ilte_all, Dstg.s):

            rm = np.mean(r[0, (0, -1)])

            # Check that theta increases as k increases
            assert np.all(np.diff(rt, 1, 2) > 0.)

            # Smoothness criteria in k direction
            tol = 1e-3
            assert np.all(np.diff(rt / rm, 2, 2) < tol)

            # Angular pitch should never be more than reference value
            # And equal to the reference value outside of the blade row
            nblade = np.round(2.0 * np.pi * rm / s)
            pitch_t = 2.0 * np.pi / nblade
            t = rt / np.expand_dims(r,2)
            dt = t[:, :, -1] - t[:, :, 0]
            tol = 1e-6
            assert np.all(dt - pitch_t < tol)  # Everywhere
            assert np.all(np.isclose(dt[: ilte[0] + 1, :], pitch_t))  # Inlet duct
            assert np.all(np.isclose(dt[ilte[1], :], pitch_t))  # Exit duct


"""Tests for the hmesh module"""
import numpy as np
from turbigen import design, hmesh, geometry
from test_design import get_designs

htr = 0.6
cpTo1 = 1.0e6
Omega = 2.0 * np.pi * 50.0
Re = 4e6
Po1 = 16e5
rgas = 287.14
ga = 1.33
Co = 0.6
tte = 0.02

# Override radial points for speed
hmesh.nr = 33


cp = rgas * ga / (ga - 1.0)
To1 = cpTo1 / cp


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
            d2x = np.abs(np.diff(x_c, 2))
            assert np.all(d2x < 1e-1)
            assert np.mean(d2x)< 1e-3


def test_merid():
    """Verify properties of the meridional grid."""

    # Generate a streamwise grid first
    dx_c = (1.0, 1.0)
    x_c, (ile, ite) = hmesh.streamwise_grid(dx_c)

    # Loop over entire mean-line design space
    for stg in get_designs("datum"):

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
    # A1 = geometry.prelim_A() * 0.5
    A1 = np.array([[.2,.1,.1,.2],[.2,.1,.1,.2]])
    A = np.stack((A1, A1))
    dx_c = (2.0, 1.0, 3.0)

    # Loop over entire mean-line design space
    for stg in get_designs("datum"):

        Dstg = design.scale_geometry(stg, htr, Omega, To1, Po1, rgas, Re, Co)

        x_all, r_all, rt_all, ilte_all = hmesh.stage_grid(Dstg, A, dx_c, tte)

        for x, r, rt, ilte, s, cx in zip(x_all, r_all, rt_all, ilte_all, Dstg.s, Dstg.cx):

            rm = np.mean(r[0, (0, -1)])

            # Check that theta increases as k increases
            assert np.all(np.diff(rt, 1, 2) > 0.0)

            # Smoothness criteria in k direction
            tol = 1e-2
            assert np.all(np.diff(rt / np.expand_dims(r,2), 2, 2) < tol)

            # Adjust pitches to account for surface length
            rtsect = rt[ilte[0]:ilte[1],:,(0,-1)]
            x1 = np.reshape(x[ilte[0]:ilte[1]],(-1,1,1))
            xsect = np.tile(x1,(1,rt.shape[1],2))
            sect = np.stack((xsect, rtsect)).transpose(2,3,0,1)
            So_cx = geometry._surface_length(sect) / cx
            s_adjusted = s*So_cx

            # Mesh angular pitch
            t = rt / np.expand_dims(r, 2)
            dt = t[:, :, -1] - t[:, :, 0]
            mesh_pitch = dt[1]

            # The true pitch
            nblade = np.round(2.0 * np.pi * rm / s_adjusted)
            pitch_t = 2.0 * np.pi / nblade

            # We cannot be too precise because integration of surface length
            # depends on discretisation level of the blade section
            assert np.isclose(mesh_pitch, pitch_t, rtol=4e-2).all()

            err = dt/mesh_pitch - 1.

            # Angular pitch should never be more than reference value
            # And equal to the reference value outside of the blade row
            tol = 1e-6
            assert np.all(err < tol)  # Everywhere
            assert np.all(np.isclose(err[:ilte[0] + 1], 0.,atol=tol))  # Inlet duct
            assert np.all(np.isclose(err[ilte[1]:], 0.,atol=tol))  # Exit duct

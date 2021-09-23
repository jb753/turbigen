"""Tests for the turbogen module"""
import numpy as np
import compflow as cf
from turbogen.make_design import *

# Set up test data

# Ranges of velocity triangle parameters covering the classic Smith chart
phi = np.linspace(0.4, 1.2, 7)
psi = np.linspace(0.8, 2.4, 7)

# "Reasonable" range of reaction (usually close to Lam = 0.5 in gas turbines)
# Known limitation: does not converge for very high reaction Lam > 0.8
Lam = np.linspace(0.3, 0.7, 5)

# Other parameters
Al1 = 10.0
Ma2 = 0.6
ga = 1.33
eta = 0.9
Lam_ref = 0.5
Al_range = np.linspace(-30.0, 30.0, 11)

Ma_low = 0.01
eta_ideal = 1.0

# Begin test functions


def test_Zweifel():
    """Verify Zweifel pitch-to-chord for low-speed lossless repeating stages."""
    for phii in phi:
        for psii in psi:
            for Al1i in Al_range:
                Alnow = (Al1i, Al1i)  # Same inlet and exit angle
                stg = nondim_stage_from_Al(
                    phii, psii, Alnow, Ma_low, ga, eta_ideal
                )

                # Evaluate Zweifel using built in function
                Z = 0.8
                s_c_out = np.array(pitch_Zweifel((Z, Z), stg))

                # Evaluate low-speed lossless approximation
                Alr = np.radians(stg.Al)
                s_c_stator = (
                    Z
                    / 2.0
                    / (np.cos(Alr[1]) ** 2.0)
                    / (np.tan(Alr[1]) - np.tan(Alr[0]))
                )
                Alrelr = np.radians(stg.Alrel)
                s_c_rotor = (
                    Z
                    / 2.0
                    / (np.cos(Alrelr[2]) ** 2.0)
                    / (np.tan(Alrelr[2]) - np.tan(Alrelr[1]))
                )

                # Check that the two are within a tolerance
                assert np.all(
                    np.abs(s_c_out - np.array((s_c_stator, s_c_rotor))) < 1e-4
                )


def test_repeating():
    """Verify analytically some repeating stage velocity triangles."""
    for phii in phi:
        for psii in psi:
            for Al1i in Al_range:
                Alnow = (Al1i, Al1i)  # Same inlet and exit angle
                stg = nondim_stage_from_Al(phii, psii, Alnow, Ma2, ga, eta)
                psi_out = 2.0 * (
                    1.0 - stg.Lam - phii * np.tan(np.radians(Al1i))
                )
                assert np.isclose(psii, psi_out)


def test_mass():
    """Check for mass conservation."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta, Vx_rat=(0.9, 1.2)
                )
                mdot_out = (
                    cf.mcpTo_APo_from_Ma(stg.Ma, ga)
                    * stg.Ax_Ax1
                    * stg.Po_Po1
                    * np.cos(np.radians(stg.Al))
                    / np.sqrt(stg.To_To1)
                )
                assert np.isclose(*mdot_out)


def test_Lam():
    """Check target reaction is achieved by the yaw angle iteration."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta, Vx_rat=(0.9, 1.2)
                )
                assert np.isclose(stg.Lam, Lami)


def test_Vx():
    """Verify that the axial velocity is as required."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta, Vx_rat=(0.9, 1.2)
                )
                V_cpTo = cf.V_cpTo_from_Ma(stg.Ma, ga) * np.sqrt(stg.To_To1)
                Vx_cpTo = V_cpTo * np.cos(np.radians(stg.Al))
                Vx_U = Vx_cpTo / stg.U_sqrt_cpTo1
                Vx_rat_out = Vx_U / phii
                assert np.all(np.isclose(Vx_rat_out, (0.9, 1.0, 1.2)))


def test_euler():
    """Verify that the Euler's work equation is satisfied."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta, Vx_rat=(0.9, 1.2)
                )
                V_cpTo = cf.V_cpTo_from_Ma(stg.Ma, ga) * np.sqrt(stg.To_To1)
                Vt_cpTo = V_cpTo * np.sin(np.radians(stg.Al))
                Vt_U = Vt_cpTo / stg.U_sqrt_cpTo1
                dVt_U = Vt_U[1] - Vt_U[2]
                assert np.all(np.isclose(dVt_U, psii))


def test_loss():
    """Check that polytropic efficiency, loss coeffs and Po are correct."""
    for phii in phi:
        for psii in psi:
            for etai in [0.8, 0.9, 1.0]:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lam_ref, Al1, Ma2, ga, etai
                )
                # Check efficiency
                eta_out = (
                    np.log(stg.To_To1[-1])
                    / np.log(stg.Po_Po1[-1])
                    * ga
                    / (ga - 1.0)
                )
                assert np.isclose(eta_out, etai)

                # Check loss coeffs
                # Note compressor definition using inlet dyn head
                Po2_Po1 = stg.Po_Po1[1]
                Po3_Po2_rel = (
                    cf.Po_P_from_Ma(stg.Marel[2], ga)
                    / cf.Po_P_from_Ma(stg.Marel[1], ga)
                    * cf.Po_P_from_Ma(stg.Ma[1], ga)
                    / cf.Po_P_from_Ma(stg.Ma[2], ga)
                    * stg.Po_Po1[2]
                    / stg.Po_Po1[1]
                )
                Po1_P1 = cf.Po_P_from_Ma(stg.Ma[0], ga)
                Po2_P2_rel = cf.Po_P_from_Ma(stg.Marel[1], ga)

                Yp_stator_out = (1.0 - Po2_Po1) / (1.0 - 1.0 / Po1_P1)
                assert np.isclose(Yp_stator_out, stg.Yp[0])
                Yp_rotor_out = (1.0 - Po3_Po2_rel) / (1.0 - 1.0 / Po2_P2_rel)
                assert np.isclose(Yp_rotor_out, stg.Yp[1])


def test_psi():
    """Check that stage loading coefficient is correct."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta
                )
                psi_out = (1.0 - stg.To_To1[2]) / stg.U_sqrt_cpTo1 ** 2.0
                assert np.isclose(psii, psi_out)


def test_Al():
    """Check that inlet and exit yaw angles are as specified."""
    for phii in phi:
        for psii in psi:
            for Al1i in Al_range:
                for Al3i in Al_range:
                    Alnow = (Al1i, Al3i)
                    stg = nondim_stage_from_Al(phii, psii, Alnow, Ma2, ga, eta)
                    assert np.all(
                        np.isclose(
                            np.array(stg.Al)[
                                (0, 2),
                            ],
                            Alnow,
                        )
                    )


def test_valid():
    """Check that output data is always physically sensible."""
    for phii in phi:
        for psii in psi:
            for Lami in Lam:
                stg = nondim_stage_from_Lam(
                    phii, psii, Lami, Al1, Ma2, ga, eta
                )
                # No nans or infinities
                for xi in stg:
                    assert np.all(np.isfinite(xi))
                # All variables excluding flow angles should be non-negative
                for vi, xi in stg._asdict().items():
                    if not "Al" in vi:
                        assert np.all(np.array(xi) >= 0.0)
                # Flow angles less than 90 degrees
                for vi in ["Al", "Alrel"]:
                    assert np.all(np.abs(getattr(stg, vi)) < 90.0)
                # No diverging annuli (for these designs Vx=const)
                assert np.all(np.array(stg.Ax_Ax1) >= 1.0)

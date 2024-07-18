"""Mean-line design of a radial impeller with vaneless diffuser"""
import numpy as np
import turbigen.flowfield


def forward(
    So1,
    PR_tt,
    eta_tt,
    mdot,
    phi1,
    Alpha1,
    Ma1_rel,
    htr1,
    Alpha2_rel,
    loss_split,
    DH_rotor,
    DH_diff,
):
    """Design the mean-line for a vaneless radial compressor.

    Parameters
    ----------
    So1: State
        Object specifing the working fluid and its state at inlet.
    PR_tt : float
        Stagnation pressure ratio
    eta_tt : float
        Estimate of total-to-total isentropic efficiency.
    mdot : float
        Mass flow rate.
    Alpha1 : float
        Impeller inlet yaw angle.
    Ma1_rel : float
        Impeller inlet relative Mach number.
    htr1 : float
        Impeller inlet hub-to-tip ratio.
    Alpha2_rel : float
        Impeller exit yaw angle.
    loss_split : float
        Guess of entropy rise split between impeller and diffuser.
    DHimp : float
        Impeller de Haller number.
    DHdiff : float
        Diffuser de Haller number.

    Returns
    -------
    ml: MeanLine
        An object specifying the flow along the mean line.


    """

    # Fully radial outlet pitch angle
    Beta2 = 90.0

    # Calculate outlet state using
    Po3 = So1.P * PR_tt
    So3s = So1.copy().set_P_s(Po3, So1.s)
    ho3 = So1.h + (So3s.h - So1.h) / eta_tt
    So3 = So1.copy().set_P_h(Po3, ho3)

    #
    # We shall use the following notation for states
    # (1) impeller inlet
    # (2) impeller outlet
    # (3) diffuser outlet

    # Precalculate some trig
    tanAlpha1 = np.tan(np.radians(Alpha1))
    cosAlpha1 = np.cos(np.radians(Alpha1))
    tanAlpha2_rel = np.tan(np.radians(Alpha2_rel))
    tanBeta2 = np.tan(np.radians(Beta2))

    #
    # (i) set rotor inlet state using, Ma1_rel, Al1, phi, Ypigv
    #

    # Solve for the axial Mach at rotor inlet
    Max1 = Ma1_rel * (1.0 + (tanAlpha1 - 1.0 / phi1) ** 2.0) ** -0.5

    # Resolve by IGV angle to get total Mach
    Ma1 = Max1 / cosAlpha1
    S1 = So1.to_static(Ma1)

    # # Guess lossless and iterate
    # So1 = Soin.copy()
    maxiter = 10
    # for _ in range(maxiter):
    #     Po1 = So1.P - Ypigv * (So1.P - S1.P)
    #     So1.set_P_h(Po1, Soin.h)

    #
    # (ii) evaluate rotor inlet velocities and geometry using mdot, htr
    #

    # Velocities
    Vx1 = Max1 * S1.a
    Vrel1 = Ma1_rel * S1.a
    Vt1 = Vx1 * tanAlpha1
    U1 = Vx1 / phi1

    # Geometry
    A1 = mdot / Vx1 / S1.rho
    # rmid1 = np.sqrt(A1 / 4.0 / np.pi * (1.0 + htr1) / (1.0 - htr1))
    # span1 = A1 / 2.0 / np.pi / rmid1
    rrms1 = np.sqrt(A1 / np.pi * 0.5 * (1.0 + htr1**2) / (1 - htr1**2))
    Omega = U1 / rrms1

    #
    # (iii) prescribe rotor DeHaller and angles to set rel frame velocities
    #

    # Velocities
    Vrel2 = DH_rotor * Vrel1
    # Vr2 = Vrel2 / np.sqrt(1.0 + tanAlpha2_rel**2.0)
    Vr2 = Vrel2 * np.cos(np.radians(Alpha2_rel))
    # print(Vr2, Vr2_check)
    Vtrel2 = Vr2 * tanAlpha2_rel
    Vm2 = Vr2 * np.sqrt(1.0 / tanBeta2**2.0 + 1.0)
    Vx2 = Vr2 / tanBeta2

    #
    # (iv) We need a general way of specifying rotor loss. Choose a prescribed
    # fraction of entropy rise between rotor inlet and machine outlet. This
    # choice can be updated in response to CFD results.
    #
    # Together with stagnation enthalpy, this fixes rotor exit stagntaion state
    #

    s2 = So1.s + loss_split * (So3.s - So1.s)
    So2 = So1.copy().set_h_s(So3.h, s2)

    # Inlet rothalpy
    I1 = S1.h + 0.5 * (Vrel1**2.0 - U1**2.0)

    # # Iterate on rotor exit static state
    # Ma2 = 0.01  # Initial guess
    # rf = 0.5  # Relaxation factor
    # for _ in range(maxiter):
    #     # New static state
    #     S2 = So2.to_static(Ma2)
    #     # New relative stagnation state
    #     Marel2 = Vrel2 / S2.a
    #     Sorel2 = S2.to_stagnation(Marel2)
    #     # Conserve rothalpy to get U2, V2
    #     U2 = np.sqrt(2.0 * (Sorel2.h - I1))
    #     Vt2 = Vtrel2 + U2
    #     V2 = np.sqrt(Vx2**2.0 + Vr2**2.0 + Vt2**2.0)
    #     # Updated guess of Mach with relaxation
    #     Ma2_new = V2 / S2.a
    #     Ma2 = rf * Ma2_new + (1.0 - rf) * Ma2
    # print(U2)

    h2 = So2.h
    for _ in range(maxiter):
        U2 = np.sqrt(2.0 * (h2 + 0.5 * Vrel2**2 - I1))
        h2 = So2.h - 0.5 * (Vr2**2.0 + (Vtrel2 + U2) ** 2.0)

    S2 = So2.copy().set_h_s(h2, So2.s)
    Vt2 = Vtrel2 + U2
    V2 = np.sqrt(Vx2**2 + Vr2**2 + Vt2**2)

    # Misc other results
    Alpha2 = np.degrees(np.arctan(Vt2 / Vm2))
    A2 = mdot / S2.rho / Vm2
    rrms2 = U2 / Omega

    V3 = DH_diff * V2

    # Iterate on diffuser exit static state
    Ma3 = 0.6  # Initial guess
    rf = 0.5  # Relaxation factor
    for _ in range(maxiter * 2):
        # New static state
        S3 = So3.to_static(Ma3)
        Ma3_new = V3 / S3.a
        Ma3 = rf * Ma3_new + (1.0 - rf) * Ma3

    #
    # (vi) select radius ratio that is consistent
    #

    # Conserve moment of momentum
    mom2 = rrms2 * S2.rho * Vt2

    V3 = Ma3 * S3.a
    RR3 = np.sqrt(((mdot / A2) ** 2.0 + (mom2 / rrms2) ** 2.0)) / S3.rho / V3
    A3 = A2 * RR3
    rrms3 = rrms2 * RR3

    Vr3 = mdot / S3.rho / A3
    Vt3 = mom2 / S3.rho / rrms3
    Alpha3 = np.degrees(np.arctan(Vt3 / Vr3))

    # Dummy state
    rrms2a = rrms2 + 0.7 * (rrms3 - rrms2)
    A2a = A2 * rrms2a / rrms2
    So2a = So3.copy()

    # Iterate on diffuser exit static state
    Ma2a = Ma3 + 0.0  # Initial guess
    rf = 0.5  # Relaxation factor
    for _ in range(maxiter):
        S2a = So2a.to_static(Ma2a)
        Vr2a = mdot / S2a.rho / A2a
        Vt2a = mom2 / S2a.rho / rrms2a
        V2a = np.sqrt(Vr2a**2.0 + Vt2a**2.0)
        Ma2a = V2a / S2a.a

    S_all = S1.stack((S1, S2, S2a, S3))

    rrms_all = np.array([rrms1, rrms2, rrms2a, rrms3])
    A_all = np.array([A1, A2, A2a, A3])
    Omega_all = np.array([Omega, Omega, 0.0, 0.0])

    Vx = (Vx1, 0.0, 0.0, 0.0)
    Vr = (0.0, Vr2, Vr2a, Vr3)
    Vt = (Vt1, Vt2, Vt2a, Vt3)
    Vxrt = np.stack((Vx, Vr, Vt))

    Al_all = np.array([Alpha1, Alpha2, Alpha2, Alpha3])
    Be_all = np.array([0.0, 90.0, 90.0, 90.0])

    ml = turbigen.flowfield.make_mean_line(rrms_all, A_all, Omega_all, Vxrt, S_all)

    assert np.allclose(ml.Alpha, Al_all, atol=0.1)
    assert np.allclose(ml.Beta, Be_all, atol=0.1)

    return ml


def inverse(ml):
    """Reverse a radial compressor mean-line to design variables.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying flow in a radial compressor.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields:
        `So1`, `PR_tt`, `eta_tt`, `mdot`, `phi1`, `Alpha1`, `Ma1_rel`, `htr1`,
        `Alpha2_rel`, `loss_split`, `DHimp`, `DHdiff`
        The fields have the same meanings as in :func:`forward`.
    """
    # Pull out states
    try:
        Sdot = ml.Sdot_wall
    except Exception:
        Sdot = np.nan * np.ones((2, 2))
    try:
        ske = 0.5 * np.array(ml.ske)
    except Exception:
        ske = np.full((1,), np.nan)
    try:
        Sdot_tip = ml.Sdot_tip
    except Exception:
        Sdot_tip = np.nan
    try:
        mdot_stall = ml.mdot_stall
    except Exception:
        mdot_stall = np.nan
    try:
        mdot_choke = ml.mdot_choke
    except Exception:
        mdot_choke = np.nan
    try:
        Co1 = ml.Co[0]
    except Exception:
        Co1 = np.nan
    try:
        Dsmix = np.array(ml.Ds_mix).tolist()
    except Exception:
        Dsmix = np.zeros_like(ml.s).tolist()
    try:
        blockage = ml.blockage[1]
    except Exception:
        blockage = np.nan
    Deta_s = ml.T[-1] / (ml.ho[-1] - ml.ho[0])
    Deta_h = 1.0 / (ml.ho[-1] - ml.ho[0])
    s_nomix = ml.s - Dsmix
    Dho = np.ptp(ml.ho)
    try:
        Asurf = ml.Asurf
    except Exception:
        Asurf = np.nan * np.ones((2, 2))
    Asurf_norm = np.sum(Asurf) / (Dho**-0.5) / ml.Q[0]
    Ds = ml.r[-1] * 2.0 * (Dho**0.25) * (ml.Q[0] ** -0.5)
    Os = ml.Omega[0] * (ml.Q[0] ** 0.5) * (Dho**-0.75)
    pc = 100.0
    Deta_surf = np.sum(Sdot) / ml.mdot[0] * Deta_s * pc

    Deta_surf_arr = np.array(Sdot) / ml.mdot[0] * Deta_s * pc
    Asurf_norm_arr = np.array(Asurf) / (Dho**-0.5) / ml.Q[0]
    with np.errstate(invalid="ignore"):
        V3av_arr = Deta_surf_arr / Asurf_norm_arr

    RR = ml.rrms[1] / ml.rrms[0]

    out = {
        "So1": ml.stagnation[0],
        "PR_tt": ml.PR_tt,
        "eta_tt_pc": ml.eta_tt * pc,
        "eta_ts_pc": ml.eta_ts * pc,
        "eta_tt": ml.eta_tt,
        "eta_poly": ml.eta_poly * pc,
        "eta_tt_nomix": (ml.eta_tt + Dsmix[-1] * Deta_s) * pc,
        "mdot": ml.mdot[0],
        "phi1": ml.phi[0],
        "Alpha1": ml.Alpha[0],
        "Ma1_rel": ml.Ma_rel[0],
        "htr1": ml.htr[0],
        "Alpha2_rel": ml.Alpha_rel[1],
        "DH_rotor": ml.V_rel[1] / ml.V_rel[0],
        "DH_diff": ml.V[-1] / ml.V[1],
        "loss_split": (s_nomix[1] - -s_nomix[0]) / (s_nomix[-1] - s_nomix[0]),
        "RR": ml.rrms[1] / ml.rrms[0],
        "DR": ml.rho[1] / ml.rho[0],
        "VmR": ml.Vm[1] / ml.Vm[0],
        "Vm1_Vrel1": ml.Vm[0] / ml.V_rel[0],
        "psi_turn": -0.5 * (ml.V_rel[1] ** 2.0 - ml.V_rel[0] ** 2.0) / ml.U[0] ** 2.0,
        "psi_ke": 0.5 * (ml.V[1] ** 2.0 - ml.V[0] ** 2.0) / ml.U[0] ** 2.0,
        "psi_rad": 0.5 * (ml.U[1] ** 2 - ml.U[0] ** 2) / ml.U[0] ** 2.0,
        "Nb1": ml.Nb[0],
        "Ma2": ml.Ma[1],
        "psi2": (ml.ho[-1] - ml.ho[0]) / ml.U[1] ** 2.0,
        "Phi": ml.mdot[0] / ml.stagnation.rho[0] / (ml.rrms[1] * 2.0) ** 2 / ml.U[1],
        "psi_tot": (ml.ho[-1] - ml.ho[0]) / ml.U[0] ** 2.0,
        "Deta_surf": Deta_surf,
        "blockage1": blockage,
        "Deta_mix": Dsmix[-1] * Deta_s * pc,
        "Deta_tt": (1.0 - ml.eta_tt) * pc,
        "Deta_ts": (1.0 - ml.eta_ts) * pc,
        "Deta_ke": (ml.eta_tt - ml.eta_ts) * pc,
        "Deta_ske": (ske / ml.mdot[0] * Deta_h).tolist()[-1] * pc,
        "Deta_tip": np.sum(Sdot_tip) / ml.mdot[0] * Deta_s * pc,
        "Deta_imp": (ml.s[1] - ml.s[0]) * Deta_s * pc,
        "Deta_diff": (ml.s[-1] - ml.s[1]) * Deta_s * pc,
        "rs1_r2": ml.rtip[0] / ml.rrms[1],
        "rs1_rm1": ml.rtip[0] / ml.rrms[0],
        "r3": ml.rrms[-1],
        "r2": ml.rrms[1],
        "r1": ml.rrms[0],
        "rs1": ml.rtip[0],
        "rh1": ml.rhub[0],
        "stall_margin": mdot_stall / ml.mdot[0] - 1.0,
        "choke_margin": mdot_choke / ml.mdot[0] - 1.0,
        "range": 1.0 - mdot_stall / mdot_choke,
        "Co1": Co1,
        "Omega1": ml.Omega[0],
        "U1": ml.U[0],
        "MU2": ml.U[1] / ml.ao[0],
        "Alpha3": ml.Alpha[2],
        "Asurf": Asurf_norm,
        "Vrel1": ml.V_rel[0],
        "Vrel2": ml.V_rel[1],
        "Vt2": ml.Vt[1],
        "V3": ml.V[2],
        "V3av": Deta_surf / Asurf_norm,
        "V3av00": V3av_arr[0, 0],
        "V3av01": V3av_arr[0, 1],
        "V3av10": V3av_arr[1, 0],
        "V3av11": V3av_arr[1, 1],
        "A00": Asurf_norm_arr[0, 0],
        "A01": Asurf_norm_arr[0, 1],
        "A10": Asurf_norm_arr[1, 0],
        "A11": Asurf_norm_arr[1, 1],
        "AR1": ml.ARflow[0],
        "Ds": Ds,
        "Os": Os,
        "logDs": np.log10(Ds),
        "logOs": np.log10(Os),
        "span0": ml.span[0],
        "Asurf_bld": Asurf[0][0],
        "Asurf_ann": Asurf[0][1],
        # "Lsurf1": ml.Lsurf[0],
        # "Lsurf1_recip": 1.0 / ml.Lsurf[0],
        "turning": np.diff(
            ml.Alpha_rel[
                (
                    0,
                    1,
                ),
            ]
        )[0],
        "pitch1": 2.0 * np.pi * ml.rrms[0] / ml.Nb[0],
        "V1": ml.V[0],
        "V2": ml.V[1],
        "Co_term1": (1 - RR**2.0) / ml.phi[0],
        "Co_term2": ml.tanAlpha_rel[0],
        "Co_term3": -RR * ml.tanAlpha_rel[1] * ml.Vm[1] / ml.Vm[0],
        "Co_term123": (
            ml.tanAlpha_rel[0]
            + (1 - RR**2.0) / ml.phi[0]
            - RR * ml.tanAlpha_rel[1] * ml.Vm[1] / ml.Vm[0]
        ),
    }
    return out

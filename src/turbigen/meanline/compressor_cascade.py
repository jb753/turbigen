import numpy as np
from turbigen import util
import turbigen.flowfield


def forward(So1, Alpha1, Ma1, span1, DH, AVDR, Ys, htr=0.99, rpm=0.0):

    # Evaluate losses using guess
    # The pseudo entropy loss coefficient is defined
    # Ys = To1*(s-s1)/(0.5*ao1**2)
    dhref = 0.5 * So1.a**2
    s = So1.s + np.array([0, Ys]) * dhref / So1.T

    # Geometry
    rm = span1 / 2.0 * (1.0 + htr) / (1.0 - htr)
    rh1 = rm - 0.5 * span1
    rt1 = rm + 0.5 * span1
    rrms1 = np.sqrt(0.5 * (rh1**2.0 + rt1**2.0))
    A1 = 2.0 * np.pi * rm * span1
    A2 = A1 / AVDR
    span2 = A2 / 2.0 / np.pi / rm
    rh2 = rm - 0.5 * span2
    rt2 = rm + 0.5 * span2
    rrms2 = np.sqrt(0.5 * (rh2**2.0 + rt2**2.0))
    rrms = np.array([rrms1, rrms2])

    # Evaluate velocities
    S1 = So1.to_static(Ma1)
    V1 = S1.a * Ma1
    Vx1 = V1 * util.cosd(Alpha1)
    V2 = DH * V1

    # Enthalpy and static states
    V = np.array([V1, V2])
    h = So1.h - 0.5 * V**2
    S = So1.empty(shape=(2,)).set_h_s(h, s)

    Vx2 = S.rho[0] * Vx1 / S.rho[1] * AVDR

    Vr = np.zeros((2,))
    Vx = np.array([Vx1, Vx2])
    Vt = np.sqrt(V**2 - Vx**2)
    A = np.array([A1, A2])

    # Make the reference frame rotate
    Omega = rpm / 60.0 * 2.0 * np.pi
    U = rm * Omega
    Vt += U
    Omega = np.full((2,), Omega)
    Vxrt = np.stack((Vx, Vr, Vt))

    ml = turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, S)

    return ml


def inverse(ml):
    So1 = ml.stagnation[0]
    dhref = 0.5 * So1.a**2
    out = {
        "So1": So1,
        "AVDR": ml.rhoVx[1] / ml.rhoVx[0],
        "Alpha1": ml.Alpha_rel[0],
        "span1": ml.span[0],
        "DH": ml.V_rel[1] / ml.V_rel[0],
        "Ma1": ml.Ma_rel[0],
        "Ys": So1.T * (ml.s[1] - ml.s[0]) / dhref,
        "Yp": (ml.Po[0] - ml.Po[1]) / (ml.Po[0] - ml.P[0]),
        "htr": ml.htr[0],
        "rpm": ml.rpm[0],
    }

    return out

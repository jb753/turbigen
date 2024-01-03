import numpy as np
from turbigen import util
import turbigen.flowfield


def forward(
    So1, span1, span2, Alpha1, Alpha2, Ma2, Yh=0.0, htr=0.99, RR=1.0, Beta=(0.0, 0.0)
):
    r"""Design the mean-line for a single-row stationary cascade.

    The minimal set of design choices are: spans, yaw angles, and exit Mach
    number. Optionally, loss can be acounted for using an energy loss
    coefficient. For annular cascades, a hub-to-tip ratio, radius change and
    pitch angles become additional free variables.

    Parameters
    ----------
    So1: State
        Object specifing the working fluid and its state at inlet.
    span1, span2: float
        Inlet and outlet spans, :math:`H`.
    Alpha1: float
    Alpha2: float
        Inlet and outlet yaw angles, :math:`\alpha/^\circ`.
    Ma2: float
        Exit Mach number, :math:`\Ma_2`.
    Yh: float
        Estimate of the row energy loss coefficient, :math:`Y_h`. Uses compressor
        definition for diffusing passages, and turbine definition for
        contracting passages.
    htr: float
        Inlet hub-to-tip radius ratio, :math:`\htr`, defaults to just less than
        unity to approximate a linear cascade.
    RR: float
        Outlet to inlet radius ration, :math:`r_2/r_1`.
    Beta: (2,) array
        Inlet and outlet pitch angles, :math:`\beta/^\circ`. Only makes sense
        to be non-zero if radius ratio is not unity.

    Returns
    -------
    ml: MeanLine
        An object specifying the flow along the mean line.

    """

    span = np.array((span1, span2))
    Alpha = np.array((Alpha1, Alpha2))
    span = np.reshape(span, 2)
    Alpha = np.reshape(Alpha, 2)
    Beta = np.reshape(Beta, 2)

    # Trig
    cosBeta = util.cosd(Beta)
    cosAlpha = util.cosd(Alpha)
    tanAlpha = util.tand(Alpha)

    # Evaluate geometry first
    span_rm1 = (1.0 - htr) / (1.0 + htr) * 2.0 / cosBeta[0]
    rm1 = span[0] / span_rm1
    rm = np.array([1.0, RR]) * rm1
    rh = rm - 0.5 * span * cosBeta
    rt = rm + 0.5 * span * cosBeta
    rrms = np.sqrt(0.5 * (rh**2.0 + rt**2.0))
    A = 2.0 * np.pi * rm * span
    Aflow = A * cosAlpha
    AR = Aflow[1] / Aflow[0]

    # We will have to guess an entropy rise, then update it according to the
    # loss coefficients and Mach number
    ds = 0.0
    err = np.inf
    atol_Ma = 1e-7
    Ma1 = 0.0

    for i in range(10):
        # Conserve energy to get exit stagnation state
        So2 = So1.copy().set_h_s(So1.h, So1.s + ds)

        # Static states
        S2 = So2.to_static(Ma2)
        S1 = So1.to_static(Ma1)

        # Velocities from Mach number
        V2 = S2.a * Ma2
        Vt2 = V2 * np.sqrt(tanAlpha[1] ** 2.0 / (1.0 + tanAlpha[1] ** 2.0))
        Vm2 = np.sqrt(V2**2.0 - Vt2**2.0)

        # Mass flow and inlet static state
        mdot = S2.rho * Vm2 * A[-1]
        Vm1 = mdot / S1.rho / A[0]
        Vt1 = tanAlpha[0] * Vm1
        V1 = np.sqrt(Vm1**2.0 + Vt1**2.0)

        # Update inlet Mach
        Ma1_new = V1 / S1.a
        err = Ma1 - Ma1_new
        Ma1 = Ma1_new

        if np.abs(err) < atol_Ma:
            break

        # Update loss using appropriate definition
        horef = So1.h
        if AR >= 1.0:
            # Compressor
            href = S1.h
        else:
            # Turbine
            href = S2.h

        # Ideal state is isentropic to the exit static pressure
        S2s = S2.copy().set_P_s(S2.P, So1.s)
        h2_new = S2s.h + Yh * (horef - href)
        S2_new = S2.copy().set_P_h(S2.P, h2_new)
        ds = S2_new.s - So1.s

    # Verify the loop has converged
    Yh_out = (S2.h - S2s.h) / (horef - href)
    assert np.isclose(Yh_out, Yh, atol=1e-3)

    # Assemble the data
    S = S1.stack((S1, S2))
    Ma = np.array((Ma1, Ma2))
    V = S.a * Ma
    Vxrt = np.stack(util.angles_to_velocities(V, Alpha, Beta))
    Omega = np.zeros_like(Vxrt[0])

    ml = turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, S)

    # Check mass, energy conserved
    assert np.ptp(ml.mdot) < ml.mdot[0] * 1e-3
    assert np.ptp(ml.ho) / np.mean(ml.cp) < 0.1

    return ml


def inverse(ml):
    """Reverse a cascade mean-line to design variables.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying the flow in a cascade.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields: `So1`, `span1`, `span2`, `Alpha1`, `Alpha2`, `Ma2`, `Yh`, `htr`, `RR`, `Beta`. The fields have the same meanings as in :func:`forward`.
    """
    # Pull out states
    S2s = ml.empty().set_P_s(ml.P[-1], ml.s[0])

    # Loss coefficient
    horef = ml.ho[0]
    if ml.ARflow[0] >= 1.0:
        # Compressor
        href = ml.h[0]
    else:
        # Turbine
        href = ml.h[1]
    Yh_out = (ml.h[1] - S2s.h) / (horef - href)

    out = {
        "So1": ml.stagnation[0],
        "span1": ml.span[0],
        "span2": ml.span[1],
        "Alpha1": ml.Alpha[0],
        "Alpha2": ml.Alpha[1],
        "Ma2": ml.Ma[1],
        "Yh": Yh_out,
        "htr": ml.htr[0],
        "RR": ml.RR[0],
        "Beta": ml.Beta.tolist(),
    }

    return out

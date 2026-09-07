"""Clark (2019) aerodynamic-style loading curve.

    suction (z, Ma_peak, z_peak, Ma_LE, Ma_PS) -> Ma_SS
    pressure(z, Ma_peak, z_peak, Ma_LE, Ma_PS) -> Ma_PS
    loading (z, Ma_peak, z_peak, Ma_LE, Ma_PS) -> (Ma_SS, Ma_PS)

All three take the full four-variable style vector; each surface uses only the
variables that affect it (suction ignores Ma_PS, pressure ignores the rest).

Reverse-engineered from Fig. 3 of

    C. J. Clark, "A Step Towards an Intelligent Aerodynamic Blade Design
    Process", ASME GT2019-91637.

The isentropic Mach-fraction distribution  Ma / Ma_te  is given, separately on
each blade surface, as a piecewise "simple polynomial curve" driven by four
control variables:

    Ma_peak   M_p  / M_te      peak Mach fraction                        (Fig 3a)
    z_peak    L_p  / L_surf    surface fraction at which the peak occurs  (Fig 3c)
    Ma_LE     suction-side Mach fraction at the leading-edge station Z_LE (Fig 3b)
    Ma_PS     pressure-side Mach fraction on the pre-acceleration plateau (Fig 3c)

Independent variable: surface fraction  z = l / L_surf  in [0, 1], from the
leading edge (z = 0, stagnation, Ma/Ma_te = 0) to the trailing edge
(z = 1, Ma/Ma_te = 1).  z may be a scalar or an array; the result matches.

    >>> import numpy as np, clark
    >>> z = np.linspace(0, 1, 201)
    >>> Ma_ss, Ma_ps = clark.loading(z, Ma_peak=1.3, z_peak=0.55,
    ...                              Ma_LE=0.62, Ma_PS=0.20)

--------------------------------------------------------------------------
Suction surface -- four pieces: a leading-edge curl, a straight loading ramp,
a roll-over onto the peak, and the diffusion to the trailing edge.

    sec   = (Ma_peak - Ma_LE) / (z_peak - Z_LE)         LE-to-peak secant slope
    sigma = C_RAMP * sec                                ramp slope (~1.15 sec)
    z_c   = 3 (Ma_LE - sigma Z_LE) / (G_LE - sigma)     curl -> ramp blend point
    z2    = Z_LE + RHO (z_peak - Z_LE)                  roll-over start

  [0, z_c]   curl   cubic, P(0)=0, P'(0)=G_LE, tangent to the ramp line at z_c
                    (P(z_c)=ramp, P'(z_c)=sigma, P''(z_c)=0).  z_c moves
                    downstream as Ma_LE rises.
  [z_c, z2]  ramp   straight line, Ma = Ma_LE + sigma (z - Z_LE)
  [z2, zp]   roll   cubic Hermite  (z2, ., sigma) -> (z_peak, Ma_peak, 0)
  [zp, 1]    diff   Ma = Ma_peak - (Ma_peak - 1)(1.5 y^2 - 0.5 y^3),
                    y = (z - z_peak)/(1 - z_peak)

A polynomial in z cannot give the true (near-vertical) LE tangent, so the
first ~2 % of surface -- physically the stagnation point, Ma ~ 0 -- is only
approximate.

--------------------------------------------------------------------------
Pressure surface -- three pieces: a leading-edge rise, a flat plateau, and a
cubic-Bezier rear.

  [0, Z_RISE]      rise      R = Ma_PS u (2 - u),  u = z / Z_RISE
  [Z_RISE, Z_ACC] plateau   Ma = Ma_PS
  [Z_ACC, 1]      rear      cubic Bezier (Z_ACC, Ma_PS) -> (1, 1) with
                            B1 = (Z_ACC + PS_A, Ma_PS)   (leaves the plateau flat)
                            B2 = (1 - PS_B, 1 - TE_K (1 - Ma_PS) PS_B)
                            -> dMa/dz|_1 = TE_K (1 - Ma_PS),  TE_K ~ 4.8

--------------------------------------------------------------------------
Only the module-level constants below are fixed shape parameters, calibrated
once against all nine digitised Fig. 3 curves; overall RMS ~0.024 in Mach
fraction (~0.015 excluding the leading-edge stagnation region).
"""

import numpy as np

__all__ = ["suction", "pressure", "loading"]

# ---- fixed shape constants (calibrated against digitised Fig. 3) ----------
Z_LE = 0.10  # suction: ramp reference station (ramp line has Ma = Ma_LE here)
C_RAMP = 1.15  # suction: ramp slope / LE-to-peak secant slope
G_LE = 13.0  # suction: leading-edge gradient dMa/dz at z = 0
RHO = 0.60  # suction: roll-over start, as a fraction of (z_peak - Z_LE)

Z_RISE = 0.10  # pressure: leading-edge rise completes here
Z_ACC = 0.20  # pressure: rear-Bezier start (plateau runs on past it via B1)
TE_K = 4.8  # pressure: trailing-edge gradient / (1 - Ma_PS)
PS_A = 0.11  # pressure: rear-Bezier B1 offset past the plateau end
PS_B = 0.22  # pressure: rear-Bezier B2 offset back from the trailing edge


def _hermite(z, z0, z1, y0, y1, d0, d1):
    """Cubic Hermite interpolant on [z0, z1], evaluated at z (assumed in range)."""
    h = z1 - z0
    t = (z - z0) / h
    return (
        (1 + 2 * t) * (1 - t) ** 2 * y0
        + t * (1 - t) ** 2 * h * d0
        + t**2 * (3 - 2 * t) * y1
        + t**2 * (t - 1) * h * d1
    )


def _bezier1d(t, p0, p1, p2, p3):
    mt = 1.0 - t
    return mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3


def _invert_bezier_x(z, x0, x1, x2, x3, iters=60):
    """t such that the cubic Bezier x(t) with control abscissae (x0..x3) equals
    z.  x(t) is monotone here, so plain Newton from t = (z - x0)/(x3 - x0)."""
    t = np.clip((z - x0) / (x3 - x0), 1e-9, 1.0 - 1e-9)
    for _ in range(iters):
        x = _bezier1d(t, x0, x1, x2, x3)
        dx = (
            3 * (1 - t) ** 2 * (x1 - x0)
            + 6 * (1 - t) * t * (x2 - x1)
            + 3 * t**2 * (x3 - x2)
        )
        t = np.clip(t - (x - z) / np.where(np.abs(dx) < 1e-12, 1e-12, dx), 0.0, 1.0)
    return t


def suction(z, Ma_peak, z_peak, Ma_LE, Ma_PS):
    """Suction-surface isentropic Mach fraction Ma/Ma_te at surface fraction z.
    Ma_PS is accepted for a uniform signature but not used."""
    z = np.asarray(z, float)

    sec = (Ma_peak - Ma_LE) / (z_peak - Z_LE)
    sigma = C_RAMP * sec
    z2 = Z_LE + RHO * (z_peak - Z_LE)
    m2 = Ma_LE + sigma * (z2 - Z_LE)

    # leading-edge curl: cubic tangent to the ramp line at z_c
    z_c = 3.0 * (Ma_LE - sigma * Z_LE) / (G_LE - sigma)
    a3 = (G_LE - sigma) / (3.0 * z_c**2)
    a2 = -3.0 * a3 * z_c
    curl = G_LE * z + a2 * z**2 + a3 * z**3

    ramp = Ma_LE + sigma * (z - Z_LE)

    roll = _hermite(np.clip(z, z2, z_peak), z2, z_peak, m2, Ma_peak, sigma, 0.0)

    y = np.clip((z - z_peak) / (1.0 - z_peak), 0.0, 1.0)
    diff = Ma_peak - (Ma_peak - 1.0) * (1.5 * y**2 - 0.5 * y**3)

    return np.where(
        z <= z_c, curl, np.where(z <= z2, ramp, np.where(z <= z_peak, roll, diff))
    )


def pressure(z, Ma_peak, z_peak, Ma_LE, Ma_PS):
    """Pressure-surface isentropic Mach fraction Ma/Ma_te at surface fraction z.
    Ma_peak, z_peak, Ma_LE are accepted for a uniform signature but not used."""
    z = np.asarray(z, float)

    u = np.clip(z / Z_RISE, 0.0, 1.0)
    rise = Ma_PS * u * (2.0 - u)

    b1x = Z_ACC + PS_A
    b2x = 1.0 - PS_B
    b2y = 1.0 - TE_K * (1.0 - Ma_PS) * PS_B
    t = _invert_bezier_x(z, Z_ACC, b1x, b2x, 1.0)
    rear = _bezier1d(t, Ma_PS, Ma_PS, b2y, 1.0)

    return np.where(z <= Z_RISE, rise, np.where(z <= Z_ACC, Ma_PS, rear))


def loading(z, Ma_peak, z_peak, Ma_LE, Ma_PS):
    """Both surfaces at once: returns (Ma_SS, Ma_PS)."""
    return (
        suction(z, Ma_peak, z_peak, Ma_LE, Ma_PS),
        pressure(z, Ma_peak, z_peak, Ma_LE, Ma_PS),
    )

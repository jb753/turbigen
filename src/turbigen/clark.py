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
    Ma_LE     value the ramp line takes at Z_LE, so the height of the      (Fig 3b)
              leading-edge acceleration rather than a Mach fraction the
              curve is required to reach there --- see Z_LE below
    Ma_PS     pressure-side Mach fraction on the pre-acceleration plateau (Fig 3c)

Independent variable: surface fraction  z = l / L_surf  in [0, 1], from the
leading edge (z = 0, stagnation, Ma/Ma_te = 0) to the trailing edge
(z = 1, Ma/Ma_te = 1).  z may be a scalar or an array; the result matches.

    >>> import numpy as np, clark
    >>> z = np.linspace(0, 1, 201)
    >>> Ma_ss, Ma_ps = clark.loading(z, Ma_peak=1.3, z_peak=0.55,
    ...                              Ma_LE=0.62, Ma_PS=0.20)

--------------------------------------------------------------------------
Stagnation point -- both surfaces. Each surface is an outer curve that knows
nothing of the nose, multiplied by the surface speed ratio of potential flow
over a parabolic nose (Lighthill 1951, "A new approach to thin aerofoil
theory"):

    Ma(z) = outer(z) * N(z / R_NOSE) / N(1 / R_NOSE),   N(eta) = eta / sqrt(1 + eta^2)

R_NOSE is the nose radius as a fraction of that surface's length. A parabola
is the nose a shape-space thickness has, t ~ sqrt(2 R_LE x), and the factor
is exact for it. N rises from zero at stagnation, with gradient 1 / R_NOSE,
and approaches one within a few R_NOSE, so the outer curve is recovered
downstream without a blend point to place. Dividing by N(1 / R_NOSE) keeps
Ma / Ma_te exactly one at the trailing edge.

--------------------------------------------------------------------------
Suction surface -- the outer curve has three pieces: a straight loading ramp,
a roll-over onto the peak, and the diffusion to the trailing edge.

    sec   = (Ma_peak - Ma_LE) / (z_peak - Z_LE)         LE-to-peak secant slope
    sigma = C_RAMP * sec                                ramp slope (~1.15 sec)
    z2    = Z_LE + RHO (z_peak - Z_LE)                  roll-over start

  [0, z2]    ramp   straight line, Ma = Ma_LE + sigma (z - Z_LE)
  [z2, zp]   roll   cubic Hermite  (z2, ., sigma) -> (z_peak, Ma_peak, 0)
  [zp, 1]    diff   Ma = Ma_peak - (Ma_peak - 1)(1.5 y^2 - 0.5 y^3),
                    y = (z - z_peak)/(1 - z_peak)

The ramp line must be positive at z = 0, Ma_LE > sigma Z_LE, or the nose
factor would carry a negative Mach number out of stagnation.

--------------------------------------------------------------------------
Pressure surface -- the outer curve has two pieces: a flat plateau and a
cubic-Bezier rear.

  [0, Z_ACC]      plateau   Ma = Ma_PS
  [Z_ACC, 1]      rear      cubic Bezier (Z_ACC, Ma_PS) -> (1, 1) with
                            B1 = (Z_ACC + PS_A, Ma_PS)   (leaves the plateau flat)
                            B2 = (1 - PS_B, 1 - TE_K (1 - Ma_PS) PS_B)
                            -> dMa/dz|_1 = TE_K (1 - Ma_PS),  TE_K ~ 4.8

--------------------------------------------------------------------------
Only the module-level constants below are fixed shape parameters, calibrated
against the nine Fig. 3 curves, which are read exactly from the vector paths
of the paper's PDF. With the style variables fitted per curve, RMS error in
Mach fraction is 0.023 on the suction surface and 0.011 on the pressure
surface; within z < 0.1 it is 0.035 and 0.008.
"""

import numpy as np

__all__ = ["suction", "pressure", "loading"]

# ---- fixed shape constants (calibrated against Fig. 3) -----------------------
# The ramp *line* passes through (Z_LE, Ma_LE); the curve sits below it by the
# nose factor, which is 0.98 at Z_LE, so Ma_LE anchors the ramp rather than
# naming a value the suction surface attains at z = Z_LE.
Z_LE = 0.10  # suction: ramp reference station (ramp line has Ma = Ma_LE here)
C_RAMP = 1.15  # suction: ramp slope / LE-to-peak secant slope
RHO = 0.60  # suction: roll-over start, as a fraction of (z_peak - Z_LE)
R_NOSE_SS = 0.0193  # suction: nose radius / suction surface length

R_NOSE_PS = 0.0283  # pressure: nose radius / pressure surface length
Z_ACC = 0.20  # pressure: rear-Bezier start (plateau runs on past it via B1)
TE_K = 4.8  # pressure: trailing-edge gradient / (1 - Ma_PS)
PS_A = 0.11  # pressure: rear-Bezier B1 offset past the plateau end
PS_B = 0.22  # pressure: rear-Bezier B2 offset back from the trailing edge


def _nose(z, r_nose):
    """Parabolic-nose speed ratio at z, scaled to be one at z = 1."""

    def ratio(eta):
        return eta / np.sqrt(1.0 + eta**2)

    return ratio(z / r_nose) / ratio(1.0 / r_nose)


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
    if not Ma_LE - sigma * Z_LE > 0.0:
        raise ValueError(
            f"The suction ramp line must be positive at the nose, but "
            f"Ma_LE - sigma * Z_LE = {Ma_LE - sigma * Z_LE:.3f} with "
            f"Ma_LE={Ma_LE}, Ma_peak={Ma_peak}, z_peak={z_peak}."
        )
    z2 = Z_LE + RHO * (z_peak - Z_LE)
    m2 = Ma_LE + sigma * (z2 - Z_LE)

    ramp = Ma_LE + sigma * (z - Z_LE)

    roll = _hermite(np.clip(z, z2, z_peak), z2, z_peak, m2, Ma_peak, sigma, 0.0)

    y = np.clip((z - z_peak) / (1.0 - z_peak), 0.0, 1.0)
    diff = Ma_peak - (Ma_peak - 1.0) * (1.5 * y**2 - 0.5 * y**3)

    outer = np.where(z <= z2, ramp, np.where(z <= z_peak, roll, diff))
    return outer * _nose(z, R_NOSE_SS)


def pressure(z, Ma_peak, z_peak, Ma_LE, Ma_PS):
    """Pressure-surface isentropic Mach fraction Ma/Ma_te at surface fraction z.
    Ma_peak, z_peak, Ma_LE are accepted for a uniform signature but not used."""
    z = np.asarray(z, float)

    b1x = Z_ACC + PS_A
    b2x = 1.0 - PS_B
    b2y = 1.0 - TE_K * (1.0 - Ma_PS) * PS_B
    t = _invert_bezier_x(z, Z_ACC, b1x, b2x, 1.0)
    rear = _bezier1d(t, Ma_PS, Ma_PS, b2y, 1.0)

    outer = np.where(z <= Z_ACC, Ma_PS, rear)
    return outer * _nose(z, R_NOSE_PS)


def loading(z, Ma_peak, z_peak, Ma_LE, Ma_PS):
    """Both surfaces at once: returns (Ma_SS, Ma_PS)."""
    return (
        suction(z, Ma_peak, z_peak, Ma_LE, Ma_PS),
        pressure(z, Ma_peak, z_peak, Ma_LE, Ma_PS),
    )

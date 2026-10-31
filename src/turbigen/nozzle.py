"""A choked exit nozzle, made by morphing a meshed grid.

A prescribed exit pressure cannot hold a compressor near the peak of its
characteristic: there ``dPR/dmdot`` tends to zero, so one pressure admits a
wide range of flows, and past the peak more than one. A choked throat
downstream fixes ``mdot sqrt(To) / Po`` instead, which through the machine is
a throttle line rising with the pressure ratio and crosses the characteristic
once, either side of the peak. Closing the throat walks the operating point
up the characteristic, as a rig's throttle valve does.

**Morphed, not re-meshed.** The mesh a design was solved on is moved node by
node, so a nozzle costs no mesher, the topology is untouched and a field from
any other point of the characteristic restarts onto it by index. Only the
exit duct moves: within `nozzle_spans` outlet spans of the outlet plane,
measured along its normal in the meridional plane, each node is moved along
the outlet line about the local mid-span, scaled so the annulus area follows

- the duct's own, at the upstream end of the fade,
- down to the throat area at `THROAT` of the fade upstream of the outlet,
- and up again to the area that expands sonic flow to `nozzle_Ma` at exit.

The blends are smoothsteps, so the walls carry no kink in slope. The local
area is the duct's own, read off the annulus walls, so the throat is the
minimum whatever contraction the duct already had there.

Being part of :class:`~turbigen.bconds.OperatingPoint` rather than of the
design, the nozzle is applied by :func:`turbigen.pipeline.prepare` after the
cached mesh is read, so a report of a nozzle point draws the grid it solved.
"""

import logging

import numpy as np

logger = logging.getLogger("turbigen")

THROAT = 0.3
"""Where the throat sits, as a fraction of the fade upstream of the outlet [--]."""

N_WALL = 4001
"""Samples along each annulus wall to read the local duct area from."""


def p_ratio(Ma, gamma):
    """Return the isentropic static-to-stagnation pressure ratio at `Ma`."""
    return (1.0 + 0.5 * (gamma - 1.0) * Ma**2) ** (-gamma / (gamma - 1.0))


def area_ratio(Ma, gamma):
    """Return the isentropic area ratio ``A/A*`` at `Ma`."""
    g1 = gamma + 1.0
    return (2.0 / g1 * (1.0 + 0.5 * (gamma - 1.0) * Ma**2)) ** (
        0.5 * g1 / (gamma - 1.0)
    ) / Ma


def critical_flux(Po, To, cp, gamma):
    """Return the mass flux through a sonic throat [kg/s/m^2], perfect gas."""
    R = cp * (gamma - 1.0) / gamma
    g1 = gamma + 1.0
    return (
        Po / np.sqrt(R * To) * np.sqrt(gamma) * (2.0 / g1) ** (0.5 * g1 / (gamma - 1.0))
    )


def outlet_line(annulus):
    """Return the outlet plane in the meridional plane.

    Returns
    -------
    centre : (2,) array
        Mid point of the outlet line, (x, r) [m].
    t : (2,) array
        Unit vector along the outlet line, hub to casing.
    n : (2,) array
        Unit normal to it, pointing upstream into the duct.
    span : float
        Length of the outlet line [m].
    area : float
        Annulus area of the outlet [m^2].
    """
    m_max = annulus.m_max
    hub, cas = np.asarray(annulus.evaluate_xr(m_max, [0.0, 1.0]), dtype=float).T
    centre = 0.5 * (hub + cas)
    span = float(np.hypot(*(cas - hub)))
    t = (cas - hub) / span
    n = np.array([t[1], -t[0]])

    # Whichever way the duct lies from the outlet
    inside = np.asarray(annulus.evaluate_xr(0.99 * m_max, 0.5), dtype=float)
    if np.dot(inside - centre, n) < 0.0:
        n = -n

    area = float(np.pi * (hub[1] + cas[1]) * span)
    return centre, t, n, span, area


def throat_fraction(station, annulus):
    """Return the throat area that chokes at `station`'s flow, as a fraction
    of the outlet area.

    One-dimensional and isentropic from the station's mixed-out stagnation
    state, so a throat with boundary layers and a spanwise profile passes a
    little less than this says. Not corrected: a choked throat is a point on
    the speedline whatever flow it passes.
    """
    flux = critical_flux(
        float(station.Po), float(station.To), float(station.cp), float(station.gamma)
    )
    *_, area = outlet_line(annulus)
    return float(station.mdot) / flux / area


def _smooth(u):
    """Smoothstep on [0, 1], flat at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _walls(annulus, centre, t, n, length):
    """Return the duct walls over the fade as functions of distance upstream.

    Returns ``(s, eta_hub, eta_cas, r_hub, r_cas)`` sampled at increasing `s`,
    `eta` being the coordinate along the outlet line from its centre.
    """
    m = np.linspace(0.0, annulus.m_max, N_WALL)
    out = []
    for spf in (0.0, 1.0):
        xr = np.asarray(annulus.evaluate_xr(m, spf), dtype=float).T - centre
        s = xr @ n
        # The duct from where the fade begins to the outlet, which must move
        # steadily towards the outlet to be read as a function of `s`
        start = np.flatnonzero(s > 1.1 * length)
        start = start[-1] + 1 if start.size else 0
        s, eta = s[start:], (xr @ t)[start:]
        if np.any(np.diff(s) >= 0.0):
            raise ValueError(
                "The exit duct doubles back on itself within the nozzle fade, "
                "so its walls cannot be read as a function of distance from "
                "the outlet. Shorten nozzle_spans."
            )
        out.append((s[::-1], eta[::-1]))

    (s_hub, eta_hub), (s_cas, eta_cas) = out
    s = np.linspace(0.0, length, 501)
    eta_h = np.interp(s, s_hub, eta_hub)
    eta_c = np.interp(s, s_cas, eta_cas)
    r_h = centre[1] + s * n[1] + eta_h * t[1]
    r_c = centre[1] + s * n[1] + eta_c * t[1]
    return s, eta_h, eta_c, r_h, r_c


def area_profile(s, area_duct, length, area_throat, area_exit):
    """Return the nozzle's annulus area at distances `s` upstream of the outlet.

    The duct's own area `area_duct` upstream of `length`, blended down to
    `area_throat` at ``THROAT * length`` and up again to `area_exit` at the
    outlet.
    """
    s_t = THROAT * length
    converging = area_duct + (area_throat - area_duct) * _smooth(
        (length - s) / (length - s_t)
    )
    diverging = area_throat + (area_exit - area_throat) * _smooth((s_t - s) / s_t)
    return np.where(s >= s_t, converging, diverging)


def _duct(annulus, nozzle_spans):
    """Return the outlet line, the fade length, and the duct's own walls and
    area over the fade, as `morph` reads them."""
    centre, t, n, span, area_out = outlet_line(annulus)
    length = nozzle_spans * span
    s, eta_h, eta_c, r_h, r_c = _walls(annulus, centre, t, n, length)
    area_duct = np.pi * (r_h + r_c) * (eta_c - eta_h)
    return centre, t, n, length, area_out, s, eta_h, eta_c, area_duct


def morph(grid, machine, operating_point):
    """Squeeze the exit duct of `grid` into a choked nozzle, in place.

    Does nothing unless `operating_point` states a `nozzle_area`.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid of `machine`, which is modified.
    machine : Machine
        The design it was meshed from, whose annulus walls give the duct's
        own area.
    operating_point : OperatingPoint or None

    Returns
    -------
    float or None
        The throat area as a fraction of the outlet area, or None if nothing
        was morphed.
    """
    if operating_point is None or operating_point.nozzle_area is None:
        return None

    gamma = float(machine.mean_line.outlet.gamma)
    centre, t, n, length, area_out, s, eta_h, eta_c, area_duct = _duct(
        machine.annulus, operating_point.nozzle_spans
    )

    area_throat = operating_point.nozzle_area * area_out
    area_exit = area_throat * area_ratio(operating_point.nozzle_Ma, gamma)

    # The throat is the narrowest section only if the duct is wider everywhere
    # the area is blended down to it
    widest = area_duct[s >= THROAT * length].min() / area_out
    if operating_point.nozzle_area >= widest:
        raise ValueError(
            f"nozzle_area={operating_point.nozzle_area:.4g} is no smaller than "
            f"the exit duct, {widest:.4g} of the outlet area at its narrowest, "
            f"so the nozzle would have no throat."
        )

    scale = area_profile(s, area_duct, length, area_throat, area_exit) / area_duct
    mid = 0.5 * (eta_h + eta_c)

    for block in grid:
        xrt = np.array(block.xrt)
        p = xrt[..., :2] - centre
        s_node = p @ n
        inside = s_node < length
        if not inside.any():
            continue
        eta = p @ t
        s_in = np.clip(s_node[inside], 0.0, length)
        mid_node = np.interp(s_in, s, mid)
        eta_new = mid_node + (eta[inside] - mid_node) * np.interp(s_in, s, scale)
        xrt[..., :2][inside] += np.outer(eta_new - eta[inside], t)
        block.set_xrt(xrt)

    logger.info(
        f"Exit nozzle: throat {operating_point.nozzle_area:.5g} of the outlet "
        f"area {area_out:.5g} m^2, at {THROAT * length * 1e3:.1f} mm upstream "
        f"of it, opening to Ma={operating_point.nozzle_Ma:.3g} at exit."
    )
    return operating_point.nozzle_area

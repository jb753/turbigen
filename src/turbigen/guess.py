"""Initial flow field for a freshly meshed grid.

A grid leaves the mesher as geometry with reference scales and an equation of
state, and no flow in it. :func:`apply` writes one, so that a solver has
somewhere to start.

It is a free function rather than part of the ``run`` verb, so that the grid a
solver would start from can be built and plotted without running anything.

The field is a function of position and the design alone, evaluated at each
node from its coordinates, so it is the same field whatever the mesh topology
and however the grid lines run:

* **Thermodynamic state and meridional speed** from the mean line, linear in
  the annulus meridional coordinate between its stations, with the velocity
  along the line of constant span fraction through the node, so parallel to
  hub and casing.
* **Swirl** from the vortex distribution each blade was designed with. Through
  a row, the relative flow angle turns from its inlet to its exit value in the
  shape of the camber line at that span; between rows, angular momentum is
  carried across unchanged at each span fraction.
* **Flow along the blades**: within a fifth of a pitch of a blade surface,
  the relative flow angle is blended towards the surface's own, keeping the
  speed, so the flow runs along the blade rather than into it. From the blade
  geometry, not the grid, like everything else here.
* **Blade loading**, optionally: the pitchwise pressure gradient that turning
  the flow takes, from the tangential momentum equation, laid linearly across
  the passage from the suction surface to the neighbouring pressure surface,
  and tapered to nothing at both edges, where the pressure on the two surfaces
  meets. The relative speed follows at constant rothalpy and entropy.

It is deliberately crude --- it exists to give the solver a sane start, not to
be right.
"""

import logging

import numpy as np
import scipy.interpolate

import turbigen.util
from turbigen.blade import FLAT_TURNING, N_M_TABLE, N_SPF_TABLE

logger = logging.getLogger("turbigen")

DM_DERIVATIVE = 1e-3
"""Step in normalised meridional coordinate for derivatives along a span
fraction line [--]."""

SNAP_PITCH = 0.2
"""Distance from a blade surface, as a fraction of the pitch, over which the
flow is turned onto the surface's angle [--]. Under half a pitch, so the two
sides of a passage never pull on the same node."""

M_EDGE = 0.05
"""Fraction of chord over which turning onto the surfaces fades in from the
leading edge and out to the trailing edge [--]. A surface's angle there runs
round the edge, which is no direction for the flow next to it."""

M_TAPER = (0.1, 0.2)
"""Fractions of chord over which the loading tapers to nothing at the leading
and trailing edges [--]. The two surfaces meet round each edge, so the pressure
difference between them has to go: at the leading edge through the stagnation
point, at the trailing edge as the Kutta condition has it."""

LOADING_MAX = 0.4
"""Largest pressure change the loading may make, as a fraction of the relative
dynamic pressure ``rho W^2`` [--], so the speed it implies stays real."""


def apply(grid, machine, blend=True, loading=False):
    """Write an initial flow field into `grid`, in place.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.
    blend : bool
        Whether to turn the flow onto the blade surfaces near them.
    loading : bool
        Whether to lay the blade loading across each passage.

    """
    for block in grid:
        if block.fluid is None:
            raise ValueError(
                "The grid has no fluid; the mesher sets one before the guess."
            )

    field = _Field(machine)
    Omega_row = np.asarray(machine.mean_line.Omega, dtype=float)[0]
    Omega_block = {
        id(block): Omega
        for blocks, Omega in zip(grid.rows, Omega_row)
        for block in blocks
    }

    for block in grid:
        Omega = Omega_block.get(id(block), 0.0)
        r, t = (np.asarray(v, dtype=float).reshape(-1) for v in (block.r, block.t))
        # Placed in the grid's own precision, so that a node on an end of the
        # annulus, rounded to it, is still inside
        m, spf = field.mspf(block.x.reshape(-1), block.r.reshape(-1))

        P, T, Vxrt = field.evaluate(m, spf, r)
        W = Vxrt.copy()
        W[:, 2] -= Omega * r
        dP = np.zeros_like(P)
        for i in range(len(field.rows) if blend or loading else 0):
            passage = field.passage(i, m, spf, r, t)
            if passage is None:
                continue
            if blend:
                _blend(W, passage)
            if loading:
                dP[passage["index"]] += field.load(i, passage)

        block.set_P_T(P.reshape(block.shape), T.reshape(block.shape))
        if loading:
            # Faster where the pressure is lower, at constant rothalpy and
            # entropy, and never by more than the flow has to give
            rho = np.asarray(block.rho, dtype=float).reshape(-1)
            W2 = np.sum(W**2, axis=-1)
            dP = np.clip(dP, -LOADING_MAX * rho * W2, LOADING_MAX * rho * W2)
            W *= np.sqrt(1.0 - 2.0 * dP / (rho * W2))[:, None]
            s = np.asarray(block.s, dtype=float)
            block.set_P_s((P + dP).reshape(block.shape), s)
        W[:, 2] += Omega * r
        block.set_Vxrt(W.reshape(*block.shape, 3))
        block.set_mu_turb(np.full(block.shape, field.mu))

    logger.debug(
        "Applied a guess from the design"
        + (", along the blades" if blend else "")
        + (", loaded" if loading else "")
    )


class _Field:
    """The design's flow as a function of annulus coordinates."""

    def __init__(self, machine):
        self.annulus = machine.annulus
        self.ml = machine.mean_line
        self.rows = machine.rows
        flat = self.ml.flat
        self.m_station = np.arange(1, flat.size + 1, dtype=float)
        self.P = np.asarray(flat.P, dtype=float)
        self.T = np.asarray(flat.T, dtype=float)
        self.Vm = np.asarray(flat.Vm, dtype=float)
        self.rho = np.asarray(flat.rho, dtype=float)
        self.mu = float(np.mean(self.ml.mu))
        self.turning = [_turning(row.blade) for row in self.rows]

    def mspf(self, x, r):
        """Annulus coordinates of meridional points, held inside the annulus."""
        m, spf = self.annulus.evaluate_mspf(x, r)
        if np.isnan(m).any():
            raise ValueError(
                f"{np.isnan(m).sum()} grid node(s) lie outside the annulus the "
                "design meshed; there is no flow defined there."
            )
        m, spf = m.astype(float), spf.astype(float)
        return np.clip(m, 0.0, self.annulus.m_max), spf

    def _station(self, values, m):
        return np.interp(m, self.m_station, values)

    def direction(self, m, spf):
        """Unit meridional tangent along the line of constant span fraction,
        and the length of that line per unit `m` [m]."""
        lo = np.clip(m - DM_DERIVATIVE, 0.0, self.annulus.m_max)
        hi = np.clip(m + DM_DERIVATIVE, 0.0, self.annulus.m_max)
        dxr = (
            self.annulus.evaluate_xr(hi, spf) - self.annulus.evaluate_xr(lo, spf)
        ) / (hi - lo)
        dl_dm = np.hypot(*dxr)
        return dxr / dl_dm, dl_dm

    def rVt(self, m, spf):
        """Angular momentum per unit mass [m^2/s].

        Through a row, the relative flow angle at each span fraction turns from
        its design inlet to exit value in the shape of the camber line there.
        Upstream of the first row, downstream of the last, and in each gap, it
        is the edge values carried across, linearly in `m` between two rows.
        """
        r = self.annulus.evaluate_xr(m, spf)[1]
        Vm = self._station(self.Vm, m)
        edges = [self._edges(i, spf) for i in range(len(self.rows))]

        # Gaps and the two ends first, then each row over the top of them
        m_LE = np.array([2.0 * i + 1.0 for i in range(len(self.rows))])
        rVt = np.empty_like(m)
        first, last = edges[0], edges[-1]
        rVt[:] = np.where(m <= m_LE[0], first["rVt"][0], last["rVt"][1])
        for i in range(len(self.rows) - 1):
            a, b = m_LE[i] + 1.0, m_LE[i + 1]
            gap = (m > a) & (m < b)
            w = (m[gap] - a) / (b - a)
            rVt[gap] = (1.0 - w) * edges[i]["rVt"][1][gap] + w * edges[i + 1]["rVt"][0][
                gap
            ]

        for i, turning in enumerate(self.turning):
            m_row = m - m_LE[i]
            inside = (m_row >= 0.0) & (m_row <= 1.0)
            Alpha_in, Alpha_out = (a[inside] for a in edges[i]["Alpha_rel"])
            f = turning(np.stack((spf[inside], m_row[inside]), axis=-1))
            Alpha = Alpha_in + (Alpha_out - Alpha_in) * f
            Omega = float(self.ml.Omega[0, i])
            r_in = r[inside]
            rVt[inside] = r_in * (Vm[inside] * np.tan(Alpha) + Omega * r_in)

        return rVt

    def _edges(self, i_row, spf):
        """Design swirl at the edges of a row, at each span fraction.

        The vortex distribution the blade was designed with, so that the guess
        and the blade's metal angles come from the same flow angles.
        """
        row = self.rows[i_row]
        ml = self.ml[:, i_row]
        spf_col = np.asarray(spf).reshape(-1, 1)
        r = ml.r_hub * (1.0 - spf_col) + ml.r_cas * spf_col
        Vt = ml.Vt * (r / ml.r) ** row.blade.vortex_exponent
        Alpha_rel = np.arctan((Vt - ml.Omega * r) / ml.Vm)
        return {"rVt": (r * Vt).T, "Alpha_rel": Alpha_rel.T}

    def evaluate(self, m, spf, r):
        """Pressure, temperature and polar velocity at points."""
        P = self._station(self.P, m)
        T = self._station(self.T, m)
        Vm = self._station(self.Vm, m)
        (dx, dr), _ = self.direction(m, spf)
        Vt = self.rVt(m, spf) / r
        return P, T, np.stack((Vm * dx, Vm * dr, Vt), axis=-1)

    def passage(self, i_row, m, spf, r, t):
        """Where points lie relative to the blades of row `i_row`.

        Points between the row's leading and trailing edges; None if there are
        none. Returned as a dict of arrays over those points, with `index`
        placing them in the inputs.
        """
        row = self.rows[i_row]
        m_LE = 2.0 * i_row + 1.0
        m_row = m - m_LE
        near = (m_row >= 0.0) & (m_row <= 1.0)
        if not near.any():
            return None

        index = np.flatnonzero(near)
        m_row, spf, r, t = (v[index] for v in (m_row, spf, r, t))
        _, dl_dm = self.direction(m_row + m_LE, spf)
        m_clip = np.clip(m_row, 0.0, 1.0)
        spf_clip = np.clip(spf, 0.0, 1.0)
        table = row.blade._surface_theta

        def theta_at(m_now):
            return table(np.stack((spf_clip, np.clip(m_now, 0.0, 1.0)), axis=-1))

        theta = theta_at(m_clip)
        lo, hi = m_clip - DM_DERIVATIVE, m_clip + DM_DERIVATIVE
        lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
        dtheta_dl = (theta_at(hi) - theta_at(lo)) / (hi - lo)[:, None] / dl_dm[:, None]

        frac = row.evaluate_pitch_fraction(m_clip, spf, t, snap=True)

        return {
            "index": index,
            "m_row": m_row,
            "spf": spf,
            "r": r,
            "dl_dm": dl_dm,
            "frac": frac,
            "width": row.pitch - np.abs(theta[:, 0] - theta[:, 1]),
            "Alpha_surface": np.arctan(r[:, None] * dtheta_dl),
            "row": row,
            "i_row": i_row,
        }

    def load(self, i_row, passage):
        """Pressure change the loading of row `i_row` makes at the points of
        `passage` [Pa].

        In a passage the tangential momentum equation is, to first order,
        ``dp/dtheta = -rho Vm d(r Vt)/dl``, so turning the flow takes a
        pitchwise pressure difference ``Delta = rho Vm s d(r Vt)/dl``, ``s``
        the angular pitch. It is laid linearly across the passage, averaging
        to the mean-line pressure, low on the suction side, and tapered to
        nothing at both edges over :data:`M_TAPER`.
        """
        row = passage["row"]
        m_LE = 2.0 * i_row + 1.0
        m_row = passage["m_row"]
        spf = passage["spf"]
        m = m_LE + m_row
        lo = np.clip(m - DM_DERIVATIVE, m_LE, m_LE + 1.0)
        hi = np.clip(m + DM_DERIVATIVE, m_LE, m_LE + 1.0)
        drVt_dl = (self.rVt(hi, spf) - self.rVt(lo, spf)) / (hi - lo) / passage["dl_dm"]
        rho_Vm = self._station(self.rho * self.Vm, m)
        Delta = np.abs(rho_Vm * row.pitch * drVt_dl)
        taper_LE, taper_TE = M_TAPER
        Delta *= np.clip(m_row / taper_LE, 0.0, 1.0)
        Delta *= np.clip((1.0 - m_row) / taper_TE, 0.0, 1.0)
        return Delta * (passage["frac"] - 0.5)


def _turning(blade):
    """Fraction of a row's turning reached along its camber line, as a
    function of ``(spf, m)``, tabulated once.

    The camber angle's progress from its leading to trailing edge value, so
    the flow turns where the blade does. A section with no turning to speak of
    has no shape to follow, and turns linearly.
    """
    spf = np.linspace(0.0, 1.0, N_SPF_TABLE)
    m = turbigen.util.cluster_cosine(N_M_TABLE)
    f = np.empty((N_SPF_TABLE, N_M_TABLE))
    for i, spf_now in enumerate(spf):
        camber = blade._at_spf(spf_now)[0]
        chi = camber.chi(m)
        dchi = chi[-1] - chi[0]
        f[i] = m if abs(dchi) < FLAT_TURNING else (chi - chi[0]) / dchi
    return scipy.interpolate.RegularGridInterpolator(
        (spf, m), f, bounds_error=False, fill_value=None
    )


def _blend(W, passage):
    """Turn relative velocity `W` onto the blade surfaces near them, in place.

    The relative flow angle is blended towards the nearer surface's, with a
    weight of one on the surface falling to nothing :data:`SNAP_PITCH` of a
    pitch away, faded in from the leading edge and out to the trailing edge
    over :data:`M_EDGE` of chord. The speed is kept, so turning the flow
    neither adds nor removes energy from it. Relative, because a rotor's
    blades move with it.
    """
    m_row = passage["m_row"]
    inside = np.isfinite(passage["frac"])
    if not inside.any():
        return
    index = passage["index"][inside]
    frac = passage["frac"][inside]
    across = passage["width"][inside] / passage["row"].pitch
    edge = np.clip(m_row[inside] / M_EDGE, 0.0, 1.0) * np.clip(
        (1.0 - m_row[inside]) / M_EDGE, 0.0, 1.0
    )
    # Suction surface at a fraction of zero, pressure surface at one
    w = (
        np.stack(
            (
                np.clip(1.0 - frac * across / SNAP_PITCH, 0.0, 1.0) ** 2,
                np.clip(1.0 - (1.0 - frac) * across / SNAP_PITCH, 0.0, 1.0) ** 2,
            ),
            axis=-1,
        )
        * edge[:, None]
    )
    if not w.any():
        return

    W_now = W[index]
    W_m = np.hypot(W_now[:, 0], W_now[:, 1])
    speed = np.linalg.norm(W_now, axis=-1)
    Alpha = np.arctan2(W_now[:, 2], W_m)
    Alpha_surface = passage["Alpha_surface"][inside]
    Alpha = Alpha + np.sum(w * (Alpha_surface - Alpha[:, None]), axis=-1)

    W_m_new = speed * np.cos(Alpha)
    scale = W_m_new / np.maximum(W_m, 1e-12 * speed)
    W[index, 0] = W_now[:, 0] * scale
    W[index, 1] = W_now[:, 1] * scale
    W[index, 2] = speed * np.sin(Alpha)

"""Initial flow field for a freshly meshed grid.

A grid leaves the mesher as geometry with reference scales and an equation of
state, and no flow in it. :func:`apply` writes one, so that a solver has
somewhere to start.

It is a free function rather than part of the ``run`` verb, so that the grid a
solver would start from can be built and plotted without running anything.

The field is a function of position and the design alone, evaluated at each
node from its coordinates, so it is the same field whatever the mesh topology
and however the grid lines run:

* **Thermodynamic state** from the mean line, linear in the annulus
  meridional coordinate between its stations through a row. In the ducts
  ahead of, between and behind the rows it is the mean line carried along the
  annulus conserving mass, stagnation enthalpy, angular momentum and entropy,
  :meth:`~turbigen.meanline.MeanLine.interpolate_to`, so that a duct whose
  area or radius changes, such as a vaneless diffuser, starts with the
  pressure rise and the mass flow it should have.
* **Meridional speed** from the :mod:`spanwise design <turbigen.spanwise>`
  at each station, linear in the same coordinate between them, along the line
  of constant span fraction through the node, so parallel to hub and casing.
  In a duct the profile is scaled to the level continuity gives there.
* **Swirl** from the spanwise design each blade was designed with. Through
  a row, between its edges at each span, swept or not, the relative flow
  angle turns from its inlet to its exit value in the shape of the camber line
  at that span; between rows, angular momentum is carried across unchanged at
  each span fraction.
* **Flow along the blades**: within a fifth of a pitch of a blade surface,
  the relative flow angle is blended towards the surface's own, keeping the
  speed, so the flow runs along the blade rather than into it. From the blade
  geometry, not the grid, like everything else here.
* **No flow into the walls**: the one part that reads the grid, because it is
  about the walls the solver will see. Round a leading edge the blend above
  fades out --- a surface's angle there runs round the nose --- so the flow
  arrives at the design angle and full speed, pointed into the blade. Left
  there, the solver stops it dead at the first step, which sends a pressure
  pulse round the nose; on a sharp, fast inducer nose it expands to vacuum. So
  near every wall the component of relative velocity into it is taken out,
  fading to nothing :data:`WALL_PITCH` of the local pitch away, holding
  rothalpy and entropy so the slowed flow is nearer its stagnation state.

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

N_DUCT_TABLE = 401
"""Points along the whole annulus at which the mean line is carried through
the ducts, interpolated linearly in between [--]."""

WALL_PITCH = 0.03
"""Distance from a wall over which the flow into it is taken out, as a
fraction of the local pitch, ``r`` times the pitch angle [--]."""

WALL_REGULARISE = 1e-9
"""Added to the diagonal of each node's wall normals' Gram matrix, so that a
node between two walls facing each other still has an answer [--]."""

SURFACE_OF_REVOLUTION = 0.05
"""Largest circumferential component of a wall's unit normal for it to count
as a hub or casing rather than a blade [--]. Discretisation leaves an
endwall's at a few thousandths on a coarse, clustered pitchwise grid in
single precision, and a blade's is most of one, so anything between does."""

M_EDGE = 0.05
"""Fraction of chord over which turning onto the surfaces fades in from the
leading edge and out to the trailing edge [--]. A surface's angle there runs
round the edge, which is no direction for the flow next to it."""


def apply(grid, machine):
    """Write an initial flow field into `grid`, in place.

    Parameters
    ----------
    grid : ember.grid.Grid
        A meshed grid, which is modified.
    machine : Machine
        The design the grid was meshed from.

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

        # The design's field depends on the meridional position alone, so
        # where the grid lines in one direction stay at one, as an H mesh's
        # do pitchwise, it is evaluated on one plane and repeated along it
        plane = _meridional_plane(block)
        x_mer, r_mer = block.x[plane], block.r[plane]
        # Placed in the grid's own precision, so that a node on an end of the
        # annulus, rounded to it, is still inside
        m, spf = field.mspf(x_mer.reshape(-1), r_mer.reshape(-1))
        P, T, Vxrt = field.evaluate(m, spf, np.asarray(r_mer, dtype=float).reshape(-1))

        m, spf, P, T = (_spread(a, x_mer.shape, block.shape) for a in (m, spf, P, T))
        W = _spread(Vxrt, x_mer.shape, block.shape).copy()
        W[:, 2] -= Omega * r
        for i in range(len(field.rows)):
            passage = field.passage(i, m, spf, r, t)
            if passage is not None:
                _blend(W, passage)

        block.set_P_T(P.reshape(block.shape), T.reshape(block.shape))
        W[:, 2] += Omega * r
        block.set_Vxrt(W.reshape(*block.shape, 3))
        block.set_mu_turb(np.full(block.shape, field.mu))

    # The walls are read off the patches, which are final by now: the guess
    # follows the boundary conditions in `pipeline.prepare`. It has to, as
    # ember caches the wall mask on first reading and a patch added after
    # that would not be seen as one.
    pitch_block = {
        id(block): 2.0 * np.pi / float(row.n_blade)
        for blocks, row in zip(grid.rows, machine.rows)
        for block in blocks
    }
    for block in grid:
        if id(block) in pitch_block:
            _unblock_walls(block, Omega_block[id(block)], pitch_block[id(block)], field)

    logger.debug("Applied a guess from the design")


def _meridional_plane(block):
    """Index of one plane of `block` holding all its meridional positions.

    The first plane along an axis on which `x` and `r` do not change, kept as
    a plane of length one, else the whole block.
    """
    x, r = block.x, block.r
    for axis in (2, 0, 1):
        first = (slice(None),) * axis + (slice(0, 1),)
        if np.array_equal(x, np.broadcast_to(x[first], x.shape)) and np.array_equal(
            r, np.broadcast_to(r[first], r.shape)
        ):
            return first
    return (slice(None),)


def _spread(a, plane_shape, shape):
    """Values at the nodes of a plane, `a` flat over them, repeated over a
    block of `shape` and flattened again, any trailing axes of `a` kept."""
    tail = a.shape[1:]
    a = a.reshape(*plane_shape, *tail)
    return np.broadcast_to(a, (*shape, *tail)).reshape(-1, *tail)


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
        self.spanwise = machine.spanwise
        self.mu = float(np.mean(self.ml.mu))
        self.turning = [_turning(row.blade) for row in self.rows]
        self.Vm_station = np.asarray(flat.Vm, dtype=float)

        # The mean line carried along the annulus, for the ducts
        self.m_duct = np.linspace(0.0, self.annulus.m_max, N_DUCT_TABLE)
        duct = self.ml.interpolate_to(self.annulus, self.m_duct)
        self.P_duct = np.asarray(duct.P, dtype=float)
        self.T_duct = np.asarray(duct.T, dtype=float)
        self.Vm_duct = np.asarray(duct.Vm, dtype=float)

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

    def in_duct(self, m):
        """Whether points are outside every row, edges belonging to rows."""
        duct = np.ones(np.shape(m), dtype=bool)
        for i in range(len(self.rows)):
            duct &= ~((m >= 2.0 * i + 1.0) & (m <= 2.0 * i + 2.0))
        return duct

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

        Through a row, between its edges at each span fraction, swept or not,
        the relative flow angle turns from its design inlet to exit value in
        the shape of the camber line there. Upstream of the first row,
        downstream of the last, and in each gap, it is the edge values carried
        across, linearly in `m` between two rows' edges.
        """
        r = self.annulus.evaluate_xr(m, spf)[1]
        Vm = self.Vm(m, spf)
        edges = [self._edges(i, spf) for i in range(len(self.rows))]

        # Where along each row's sections the points lie, which alone says
        # which row or gap each is in, so every point is in exactly one; the
        # edges themselves, in annulus `m` at each point's span, only weight
        # the gaps
        m_LE = np.array([2.0 * i + 1.0 for i in range(len(self.rows))])
        m_sec = [
            row.blade.interpolate_unswept_m(m - m_LE[i], spf)
            for i, row in enumerate(self.rows)
        ]

        # Gaps and the two ends first, then each row over the top of them
        rVt = np.empty_like(m)
        first, last = edges[0], edges[-1]
        rVt[:] = np.where(m_sec[0] <= 0.0, first["rVt"][0], last["rVt"][1])
        for i in range(len(self.rows) - 1):
            gap = (m_sec[i] > 1.0) & (m_sec[i + 1] < 0.0)
            a = m_LE[i] + self.rows[i].blade.interpolate_swept_m(1.0, spf[gap])
            b = m_LE[i + 1] + self.rows[i + 1].blade.interpolate_swept_m(0.0, spf[gap])
            w = np.clip((m[gap] - a) / (b - a), 0.0, 1.0)
            rVt[gap] = (1.0 - w) * edges[i]["rVt"][1][gap] + w * edges[i + 1]["rVt"][0][
                gap
            ]

        for i, turning in enumerate(self.turning):
            inside = (m_sec[i] >= 0.0) & (m_sec[i] <= 1.0)
            Alpha_in, Alpha_out = (a[inside] for a in edges[i]["Alpha_rel"])
            f = turning(np.stack((spf[inside], m_sec[i][inside]), axis=-1))
            Alpha = Alpha_in + (Alpha_out - Alpha_in) * f
            Omega = float(self.ml.Omega[0, i])
            r_in = r[inside]
            rVt[inside] = r_in * (Vm[inside] * np.tan(Alpha) + Omega * r_in)

        return rVt

    def _edges(self, i_row, spf):
        """Design swirl at the edges of a row, at each span fraction.

        The spanwise flow the blade was designed with, so that the guess and
        the blade's metal angles come from the same flow angles.
        """
        ml = self.ml[:, i_row]
        knots = self.rows[i_row].blade.spf
        spf = np.asarray(spf, dtype=float)
        rVt = np.stack(
            [self.spanwise.rVt(ml[j], 2 * i_row + j, knots, spf) for j in range(2)]
        )
        Alpha_rel = np.radians(self.spanwise.Alpha_rel(ml, i_row, knots, spf)).T
        return {"rVt": rVt, "Alpha_rel": Alpha_rel}

    def Vm(self, m, spf):
        """Meridional speed [m/s]: the spanwise profile at each station,
        linear in `m` between them and held beyond the first and last."""
        spf = np.asarray(spf, dtype=float)
        flat = self.ml.flat
        profiles = np.stack(
            [
                self.spanwise.Vm(flat[k], k, self.rows[k // 2].blade.spf, spf)
                for k in range(flat.size)
            ]
        )
        x = np.clip(m - self.m_station[0], 0.0, flat.size - 1.0)
        i0 = np.minimum(np.floor(x).astype(int), flat.size - 2)
        w = x - i0
        points = np.arange(spf.size)
        return (1.0 - w) * profiles[i0, points] + w * profiles[i0 + 1, points]

    def evaluate(self, m, spf, r):
        """Pressure, temperature and polar velocity at points."""
        P = self._station(self.P, m)
        T = self._station(self.T, m)
        Vm = self.Vm(m, spf)

        duct = self.in_duct(m)
        if duct.any():
            m_duct = m[duct]
            P[duct] = np.interp(m_duct, self.m_duct, self.P_duct)
            T[duct] = np.interp(m_duct, self.m_duct, self.T_duct)
            Vm[duct] *= np.interp(m_duct, self.m_duct, self.Vm_duct) / self._station(
                self.Vm_station, m_duct
            )
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
        blade = row.blade
        m_LE = 2.0 * i_row + 1.0
        m_row = m - m_LE
        # Along the section, 0 and 1 at its edges, swept or not
        m_sec = blade.interpolate_unswept_m(m_row, spf)
        near = (m_sec >= 0.0) & (m_sec <= 1.0)
        if not near.any():
            return None

        index = np.flatnonzero(near)
        m_row, m_sec, spf, r, t = (v[index] for v in (m_row, m_sec, spf, r, t))
        _, dl_dm = self.direction(m_row + m_LE, spf)
        m_clip = np.clip(m_sec, 0.0, 1.0)
        spf_clip = np.clip(spf, 0.0, 1.0)
        table = blade._surface_theta

        def theta_at(m_now):
            return table(np.stack((spf_clip, np.clip(m_now, 0.0, 1.0)), axis=-1))

        theta = theta_at(m_clip)
        lo, hi = m_clip - DM_DERIVATIVE, m_clip + DM_DERIVATIVE
        lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
        # Distance along the line per step along the section, which a swept
        # section stretches or shrinks
        if blade.swept:
            dm_row = blade.interpolate_swept_m(np.stack((lo, hi)), spf)
            dl_dm = dl_dm * (dm_row[1] - dm_row[0]) / (hi - lo)
        dtheta_dl = (theta_at(hi) - theta_at(lo)) / (hi - lo)[:, None] / dl_dm[:, None]

        frac = row.evaluate_pitch_fraction(m_row, spf, t, snap=True)

        # How far into the designed blade a node lies, as a fraction of half
        # its thickness. A mesh that thins the blade, as a tip pinch does, puts
        # nodes there; snapped to opposite surfaces, two nodes a whisker apart
        # would be turned the full difference between the surfaces' angles.
        upper, lower = theta.max(axis=-1), theta.min(axis=-1)
        thick = upper - lower
        across = np.mod(t - upper, row.pitch)
        depth = np.where(
            across > row.pitch - thick,
            np.minimum(across - (row.pitch - thick), row.pitch - across),
            0.0,
        )
        solid = np.clip(1.0 - 2.0 * depth / np.maximum(thick, 1e-12), 0.0, 1.0)

        return {
            "index": index,
            "m_row": m_row,
            "m_sec": m_sec,
            "spf": spf,
            "r": r,
            "frac": frac,
            "width": row.pitch - np.abs(theta[:, 0] - theta[:, 1]),
            "solid": solid,
            "Alpha_surface": np.arctan(r[:, None] * dtheta_dl),
            "row": row,
        }


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
    over :data:`M_EDGE` of chord, and out to nothing at the middle of a blade
    the mesh has thinned. The speed is kept, so turning the flow
    neither adds nor removes energy from it. Relative, because a rotor's
    blades move with it.
    """
    m_sec = passage["m_sec"]
    inside = np.isfinite(passage["frac"])
    if not inside.any():
        return
    index = passage["index"][inside]
    frac = passage["frac"][inside]
    across = passage["width"][inside] / passage["row"].pitch
    edge = np.clip(m_sec[inside] / M_EDGE, 0.0, 1.0) * np.clip(
        (1.0 - m_sec[inside]) / M_EDGE, 0.0, 1.0
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
        * (edge * passage["solid"][inside])[:, None]
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


# The six faces of a block: the key of its wall mask in
# :attr:`ember.block.Block.ijk_wall_conv`, the index axis normal to it, and
# whether it is at the start or the end of that axis.
_FACES = (
    ("walli1", 0, 0),
    ("wallni", 0, -1),
    ("wallj1", 1, 0),
    ("wallnj", 1, -1),
    ("wallk1", 2, 0),
    ("wallnk", 2, -1),
)


def _wall_nodes(face_free):
    """Nodes of a face touching any wall cell, from its cells' mask.

    `face_free` is the face's cells, one fewer each way than its nodes, 0.0 on
    a wall and 1.0 off one.
    """
    wall = face_free < 0.5
    out = np.zeros((wall.shape[0] + 1, wall.shape[1] + 1), dtype=bool)
    for di in (0, 1):
        for dj in (0, 1):
            out[di : di + wall.shape[0], dj : dj + wall.shape[1]] |= wall
    return out


def _tangent(surface, axis):
    """Return the unit tangent to a gridded surface along one of its axes.

    `surface` is (n1, n2, 3). Differenced over the actual chord lengths
    between nodes, second order however unevenly they are spaced, rather than
    over the node count as :func:`numpy.gradient` does: on a curved hub with
    clustered nodes the latter tilts the normal by most of a per cent, which
    taking the flow through a wall out along it turns into radial velocity.
    """
    P = np.moveaxis(surface, axis, 0)
    d = np.diff(P, axis=0)
    h = np.linalg.norm(d, axis=-1, keepdims=True)
    h = np.maximum(h, 1e-300)
    out = np.empty_like(P)
    out[0] = d[0] / h[0]
    out[-1] = d[-1] / h[-1]
    if len(P) > 2:
        h1, h2 = h[:-1], h[1:]
        # Weighted so that a quadratic through the three nodes is exact
        out[1:-1] = (d[1:] * h1 / h2 + d[:-1] * h2 / h1) / (h1 + h2)
    out /= np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-300)
    return np.moveaxis(out, 0, axis)


def _meridional_normal(field, x, r, t, inward):
    """Return the unit normal to a hub or casing at nodes `x`, `r`, `t`.

    In the meridional plane, square to the annulus's own line of constant span
    fraction through each node, the direction the guess runs the flow along
    there: from the design, not the grid, whose piecewise straight lines
    through clustered nodes miss the annulus by a quarter of a degree round a
    leading edge. Turned to Cartesians at each node's angle `t` and pointed
    along `inward`.
    """
    m, spf = field.mspf(np.ravel(x), np.ravel(r))
    (dx, dr), _ = field.direction(m, spf)
    n_x, n_r = -dr.reshape(x.shape), dx.reshape(x.shape)
    n = np.stack((n_x, n_r * np.cos(t), n_r * np.sin(t)), axis=-1)
    return n * np.sign(np.sum(n * inward, axis=-1, keepdims=True))


def _unblock_walls(block, Omega, pitch_theta, field):
    """Take the relative velocity into the walls of `block` out, in place.

    Walking out from each wall face along the grid lines that leave it, every
    node loses the component of its relative velocity into that wall, all of
    it at the wall and none :data:`WALL_PITCH` of the local pitch away, by the
    distance along the line. The direction is the wall's own normal, from the
    surface, not the line's: an H mesh leaves a blade circumferentially, well
    off normal. A node near two walls, in a corner, is corrected against each.

    In the row's frame, which is the blade's; an endwall's normal has no
    circumferential part, so the frame makes no difference there. Rothalpy and
    entropy are held, so what speed is taken out goes into static enthalpy.
    """
    walls = block.ijk_wall_conv
    x, r, t = (np.asarray(a, dtype=float) for a in (block.x, block.r, block.t))
    flat_index = np.arange(x.size).reshape(x.shape)

    def cartesian(x, r, t):
        return np.stack((x, r * np.cos(t), r * np.sin(t)), axis=-1)

    # Each wall face's nodes within reach of it, as flat indices into the
    # block, with the wall's normal and the fade weight at each. Only the
    # layers next to the wall are ever made into arrays: the fade reaches a
    # few of them, and the rest of the block is left alone.
    faces = []
    for key, axis, end in _FACES:
        wall = _wall_nodes(np.squeeze(walls[key], axis=axis))
        if not wall.any():
            continue

        # The block with the face's axis first, running away from the wall
        flip = slice(None, None, -1 if end == -1 else 1)
        xf, rf, tf = (np.moveaxis(a, axis, 0)[flip] for a in (x, r, t))
        surface = cartesian(xf[0], rf[0], tf[0])
        next_layer = cartesian(xf[1], rf[1], tf[1])

        # Normal off the wall, from the surface, pointing into the block
        n = np.cross(_tangent(surface, axis=0), _tangent(surface, axis=1))
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-300)
        n *= np.sign(np.sum(n * (next_layer - surface), axis=-1, keepdims=True))

        # Distance from the wall along each line, layer by layer out from it
        # until every line is out of reach
        dist = np.zeros(wall.shape)
        w = [wall.astype(float)]
        last = surface
        for layer in range(1, xf.shape[0]):
            here = cartesian(xf[layer], rf[layer], tf[layer])
            dist = dist + np.linalg.norm(here - last, axis=-1)
            last = here
            fade = WALL_PITCH * pitch_theta * rf[layer]
            w_layer = np.clip(1.0 - dist / fade, 0.0, 1.0) * wall
            if not w_layer.any():
                break
            w.append(w_layer)
        w = np.stack(w)

        # A surface of revolution, a hub or casing, has no circumferential
        # part to its normal. Its normal is then taken in the meridional
        # plane, from x and r alone: chords between nodes at different angles
        # sag inside the circle, which tilts a normal from the surface's own
        # chords radially wherever the grid lines turn in theta.
        e_t = np.stack((np.zeros_like(tf[0]), -np.sin(tf[0]), np.cos(tf[0])), axis=-1)
        revolution = (
            np.abs(np.sum(n * e_t, axis=-1))[wall].max() < SURFACE_OF_REVOLUTION
        )
        if revolution:
            # In the grid's own precision, in which a node on the hub or
            # casing is on the annulus, as for the guess itself
            x_own, r_own = (
                np.moveaxis(np.asarray(a), axis, 0)[flip][0] for a in (block.x, block.r)
            )
            n = _meridional_normal(field, x_own, r_own, tf[0], next_layer - surface)

        reach = w > 0.0
        index = np.moveaxis(flat_index, axis, 0)[flip][: len(w)][reach]
        normal = np.broadcast_to(n, (*w.shape, 3))[reach]
        on_wall = np.broadcast_to(np.arange(len(w))[:, None, None] == 0, w.shape)[reach]
        faces.append((index, normal, w[reach], on_wall, revolution))

    # The guess runs the flow along surfaces of constant span fraction, which
    # at a hub or casing are the wall itself, so there is nothing into one to
    # take out --- except in a corner, where taking it out of the blade along
    # the blade's normal puts some into the endwall. So an endwall acts only
    # on nodes another wall reaches too, which also keeps the work to a corner
    # rather than every layer along a whole hub and casing.
    # Masks over the whole block rather than sorting and searching index
    # lists, which on a radial row's thousands of near-wall nodes is slower
    if all(revolution for *_, revolution in faces):
        return
    blade_reach = np.zeros(x.size, dtype=bool)
    for index, *_, revolution in faces:
        if not revolution:
            blade_reach[index] = True
    trimmed = []
    for index, normal, w, on_wall, revolution in faces:
        if revolution:
            keep = blade_reach[index]
            index, normal, w, on_wall = (
                index[keep],
                normal[keep],
                w[keep],
                on_wall[keep],
            )
        if len(index):
            trimmed.append((index, normal, w, on_wall))

    # The nodes any wall reaches, and each face's entries among them
    reached = np.zeros(x.size, dtype=bool)
    for index, *_ in trimmed:
        reached[index] = True
    near = np.flatnonzero(reached)
    position = np.empty(x.size, dtype=np.int64)
    position[near] = np.arange(len(near))
    ijk = np.unravel_index(near, x.shape)
    faces = [(position[index], *rest) for index, *rest in trimmed]

    # Relative velocity at those nodes, in Cartesians
    r_near, t_near = r[ijk], t[ijk]
    c, s = np.cos(t_near), np.sin(t_near)
    Vx, Vr, Vt = (
        np.asarray(a, dtype=float)[ijk] for a in (block.Vx, block.Vr, block.Vt)
    )
    Wt = Vt - Omega * r_near
    W = np.stack((Vx, Vr * c - Wt * s, Vr * s + Wt * c), axis=-1)
    Wsq_before = np.sum(W**2, axis=-1)

    # What each wall in reach asks of the normal component through it: on the
    # wall, all of it gone; off it, a fraction `w` of the part into it gone.
    # Each node's walls, up to `n_most`, stacked with zero padding.
    count = np.zeros(len(near), dtype=int)
    for index, *_ in faces:
        count[index] += 1
    n_most = int(count.max())
    N = np.zeros((len(near), n_most, 3))
    target = np.zeros((len(near), n_most))
    slot = np.zeros(len(near), dtype=int)
    for index, normal, w, on_wall in faces:
        through = np.sum(W[index] * normal, axis=-1)
        # A node is in a face's reach once, so `slot[index]` has no repeats
        k = slot[index]
        N[index, k] = normal
        target[index, k] = -np.where(on_wall, through, w * np.minimum(through, 0.0))
        slot[index] += 1

    # The smallest change in velocity that makes all of them at once,
    # dW = N^T (N N^T)^-1 target: in a corner it moves the flow off the
    # blade without putting any into the endwall, which taking the walls in
    # turn would, and it keeps the fade, which repeating them would not.
    # Padding rows are zero, so the identity there leaves them out.
    G = N @ np.swapaxes(N, 1, 2)
    G += np.eye(n_most) * (np.arange(n_most) >= count[:, None])[:, None, :]
    G += WALL_REGULARISE * np.eye(n_most)
    W += np.einsum("nk,nkd->nd", np.linalg.solve(G, target[..., None])[..., 0], N)

    # Velocity read before the state is reset, which may move it with density
    Vxrt = np.stack(
        [np.array(a, dtype=float) for a in (block.Vx, block.Vr, block.Vt)], axis=-1
    )

    # Rothalpy and entropy held: h + W^2/2 is unchanged at fixed radius
    h = np.array(block.h, dtype=float)
    h[ijk] += 0.5 * (Wsq_before - np.sum(W**2, axis=-1))
    block.set_h_s(h, np.asarray(block.s, dtype=float))

    Vxrt[ijk] = np.stack(
        (
            W[:, 0],
            W[:, 1] * c + W[:, 2] * s,
            -W[:, 1] * s + W[:, 2] * c + Omega * r_near,
        ),
        axis=-1,
    )
    block.set_Vxrt(Vxrt)

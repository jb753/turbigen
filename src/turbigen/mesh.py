"""Meshing.

A :class:`Mesher` is a config node that turns a designed
:class:`~turbigen.machine.Machine` into an ember :class:`~ember.grid.Grid`.
The result is ember's, not ours, so there is no ``Mesh`` class to pair with a
``MeshDesign`` and the family keeps the name it already had.

The stage interface is the same as everywhere else, with the framework method
doing rather more than it does for a pure design::

    mesher.mesh(machine) -> Grid      # framework: wall spacings, then finishing
    mesher.forward(machine, spacing)  # the author writes this

Everything shared sits here: the wall spacings implied by ``yplus``, and the
four steps that turn a freshly generated grid into a usable one. The package
this replaces leaves all of it to the caller --- ``config.setup_mesh`` computes
the spacings, assembles a six-argument call, then remembers to set the
reference length, check the cell volumes and compute wall distance. A second
mesher would have to be given all of that again, and any caller can forget it.
"""

import dataclasses
import logging
from typing import ClassVar

import ember.patch
import numpy as np
from scipy.spatial import cKDTree

from turbigen.node import Node

logger = logging.getLogger("turbigen")

WDIST_LIMIT_PITCH = 0.03
"""Pitchwise fraction beyond which wall distance is not searched."""

NOT_FLAT_PLATE = 2.0
"""Factor on the viscous length scale, for a blade not being a flat plate."""


@dataclasses.dataclass(frozen=True)
class WallSpacing:
    """Wall-normal cell sizes for a mesh [m]."""

    surface: np.ndarray
    """Spacing on each blade surface, shape (n_row,)."""

    hub: float
    """Spacing at the hub."""

    casing: float
    """Spacing at the casing."""


class Mesher(Node):
    """Base for meshers."""

    meshes_sweep: ClassVar[bool] = False
    """Whether this mesher puts blade edges where a swept section has them,
    off its row's stations; a swept blade is refused otherwise."""

    yplus: float = 30.0
    """Target wall distance in viscous units, which sets the near-wall cell
    size [--]."""

    inviscid_endwalls: bool = False
    """Make the hub and casing inviscid slip walls in every block.

    For a quasi-2D cascade, where the endwalls are there only to bound a thin
    slab of span and their boundary layers are not wanted. A slip wall has no
    boundary layer to resolve, so the spanwise grid is uniform at the mesher's
    spacing at midspan rather than clustered to the spacing `yplus` asks for
    at the hub and casing; the blade surfaces still get theirs. Incompatible
    with a tip gap, whose casing is a real wall.
    """

    m_relax_out: float = 0.0
    """Where the spanwise grid starts to relax towards even spacing on its way
    to the outlet, in normalised meridional coordinate `m`. Must lie between
    the last row's trailing edge, at `m` of twice the number of rows, and the
    outlet, one further on. See :attr:`frac_relax_out`."""

    frac_relax_out: float = 0.0
    """How far span fractions are relaxed towards even spacing at the outlet:
    0 leaves the spanwise clustering as it is, 1 makes it even there. The
    blend eases in from :attr:`m_relax_out` as a smoothstep."""

    #
    # TO BE IMPLEMENTED BY A MESHER
    #

    def forward(self, machine, spacing):
        """Return a grid for `machine`, with the given wall spacings.

        Parameters
        ----------
        machine : Machine
            The designed geometry to mesh.
        spacing : WallSpacing
            Wall-normal cell sizes [m].

        Returns
        -------
        ember.grid.Grid

        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(self, machine, spacing)"
        )

    #
    # THE FRAMEWORK
    #

    def mesh(self, machine):
        """Return a finished grid for `machine`."""
        if not machine.rows:
            raise ValueError("A machine needs blades before it can be meshed.")

        for i_row, row in enumerate(machine.rows):
            if row.blade.swept and not self.meshes_sweep:
                raise NotImplementedError(
                    f"Row {i_row} has swept edges (non-zero dm_LE or dm_TE), "
                    f"which the {type(self).__name__} mesher cannot place."
                )

        spacing = self.wall_spacing(machine)
        grid = self.forward(machine, spacing)

        # A grid straight out of a mesher is not yet usable: it is bare
        # geometry, with no scales, no equation of state, no check and no wall
        # distance. Doing it here means a mesher cannot forget, and a caller
        # need not know.
        #
        # The scales are set before any flow state exists, so the initial guess
        # and everything the solver writes afterwards are stored against them.
        # Set them later and the whole field would have to be rescaled.
        grid.set_L_ref(self.L_ref(machine))
        # The mean line already carries the fluid the solve wants: `design`
        # referenced it before freezing, so it is the design's own scales and
        # datum. Adopted rather than derived again, so that the grid and the
        # mean line hold one equation of state and not two that agree to
        # rounding -- a difference too small to matter physically and quite
        # large enough for something downstream to compare and disbelieve.
        grid.set_fluid(machine.mean_line.fluid)
        self.check_volumes(grid)
        grid.calculate_wdist(limit_pitch=WDIST_LIMIT_PITCH)

        return grid

    def L_ref(self, machine):
        """Return the reference length for the grid [m].

        The longest row chord at mid-span, so that the largest blade in the
        machine is order one. Taken from the annulus rather than from the mean
        line, which carries no meaningful length: a chord is a property of the
        geometry, and the mean line is a flow field.
        """
        return float(machine.annulus.evaluate_chords(0.5)[1::2].max())

    #
    # ENDWALLS
    #

    def _uniform_span(self, dspf, tip):
        """Return span fractions evenly spaced no coarser than `dspf`, for
        :attr:`inviscid_endwalls`, in a count of cells a multigrid level can
        halve; raising if there is a `tip` gap."""
        if tip:
            raise ValueError(
                "inviscid_endwalls makes the casing a slip wall, but a tip "
                "gap needs it to be a real one; set one or the other."
            )
        # No boundary layer on a slip wall, so nothing to cluster to
        n = 8 * int(np.ceil(np.ceil(1.0 / dspf - 1e-9) / 8))
        return np.linspace(0.0, 1.0, n + 1)

    def _endwall_patches(self):
        """Return the patches every block takes on its hub and casing faces,
        `j` of 0 and -1: slip walls with :attr:`inviscid_endwalls`, else
        none, an unpatched face being a wall."""
        if not self.inviscid_endwalls:
            return []
        return [
            ember.patch.InviscidPatch(j=0, label="inviscid_hub"),
            ember.patch.InviscidPatch(j=-1, label="inviscid_casing"),
        ]

    def _check_inviscid_ducts(self, annulus):
        """Refuse `annulus.m_inviscid` together with :attr:`inviscid_endwalls`."""
        if self.inviscid_endwalls and any(annulus.m_inviscid):
            raise ValueError(
                "inviscid_endwalls already makes every endwall a slip wall, "
                "so annulus.m_inviscid would overlap it; leave that unset."
            )

    def _inviscid_ducts(self, machine, i_row, up, dn, i_le=None, i_te=None):
        """Add inviscid patches to the endwalls of block `up`, ahead of the
        first row, and `dn`, behind the last, from the inlet to the grid line
        nearest `annulus.m_inviscid`'s first two positions, on the hub and the
        casing, and from those nearest its last two to the outlet. The lines
        must lie inside the blocks, and no further in than `i_le` and `i_te`.

        Returns the range of `i` the casing of each block, by `id`, is not
        inviscid over, for its rotating patch.
        """
        annulus = machine.annulus
        last = len(machine.rows) - 1
        hub_in, cas_in, hub_out, cas_out = annulus.m_inviscid
        rotating = {}
        for blk, m, j, inlet, here in (
            (up, hub_in, 0, True, i_row == 0),
            (up, cas_in, -1, True, i_row == 0),
            (dn, hub_out, 0, False, i_row == last),
            (dn, cas_out, -1, False, i_row == last),
        ):
            if not m or not here:
                continue
            i = _i_at_m(blk, annulus, j, m)
            ni = blk.shape[0]
            lo = 1 if inlet or i_te is None else i_te
            hi = ni - 2 if not inlet or i_le is None else i_le
            if not lo <= i <= hi:
                raise ValueError(
                    f"annulus.m_inviscid {m} is not inside the duct "
                    f"{'ahead of the first row' if inlet else 'behind the last row'}."
                )
            wall = "hub" if j == 0 else "casing"
            end = "in" if inlet else "out"
            i_range = (0, i) if inlet else (i, -1)
            blk.patches.append(
                ember.patch.InviscidPatch(
                    i=i_range, j=j, label=f"inviscid_{wall}_{end}"
                )
            )
            if j == -1:
                # One block can hold both ends, in a machine of one H row
                ist, ien = rotating.get(id(blk), (0, ni - 1))
                rotating[id(blk)] = (i, ien) if inlet else (ist, i)
            logger.debug(f"Inviscid {wall} at the {end}let to grid line {i} of {ni}")
        return rotating

    #
    # OUTLET
    #

    def _relax_outlet(self, blocks, annulus, spf):
        """Relax, in place, the span fractions of the nodes of `blocks`
        downstream of :attr:`m_relax_out` towards even spacing, by
        :attr:`frac_relax_out` at the outlet.

        `spf` is the mesher's spanwise grid, whose nodes are relaxed towards
        as many evenly spaced. Each node is placed by its own position alone,
        through :meth:`~turbigen.annulus.Annulus.evaluate_mspf`, keeping its
        `m` and theta and moving only across the span, so nodes that coincide
        move alike and every interface between blocks still matches. Nodes
        upstream are left exactly as they are. Called by a mesher on its blocks
        before they are assembled into a grid.
        """
        if not self.frac_relax_out:
            return
        if not 0.0 < self.frac_relax_out <= 1.0:
            raise ValueError(
                f"frac_relax_out={self.frac_relax_out} must be from zero to one."
            )
        m_TE = annulus.m_max - 1.0
        if not m_TE <= self.m_relax_out < annulus.m_max:
            raise ValueError(
                f"m_relax_out={self.m_relax_out} must lie between the last "
                f"trailing edge at m={m_TE:g} and the outlet at "
                f"m={annulus.m_max:g}, or the blade would be moved."
            )

        spf = np.asarray(spf, dtype=float)
        even = np.linspace(spf[0], spf[-1], len(spf))
        for block in blocks:
            m, s = annulus.evaluate_mspf(block.x, block.r)
            u = (m - self.m_relax_out) / (annulus.m_max - self.m_relax_out)
            # NaN only beyond the ends, which no node is; left alone if so
            move = u > 0.0
            if not move.any():
                continue
            u = np.clip(u[move], 0.0, 1.0)
            w = self.frac_relax_out * u * u * (3.0 - 2.0 * u)
            s_new = s[move] + w * (np.interp(s[move], spf, even) - s[move])
            x, r = np.array(block.x), np.array(block.r)
            x[move], r[move] = annulus.evaluate_xr(m[move], s_new)
            block.set_x(x)
            block.set_r(r)

    def check_volumes(self, grid):
        """Raise if any cell in `grid` has a non-positive volume."""
        for i_block, block in enumerate(grid):
            if (block.vol_nd <= 0.0).any():
                raise ValueError(
                    f"Block {i_block} has {int((block.vol_nd <= 0.0).sum())} "
                    f"cells of zero or negative volume."
                )

    #
    # WALL SPACING
    #

    def wall_spacing(self, machine):
        """Return the wall-normal cell sizes implied by `yplus` [m].

        The boundary layer is idealised as a flat plate at the surface
        Reynolds number of each row, which gives a skin friction and hence a
        friction velocity and a viscous length scale.
        """
        Re_surf = machine.Re_surf()
        logger.debug(f"Surface Reynolds numbers: {Re_surf}")

        stations = [
            machine.mean_line.get_characteristic_station(i)
            for i in range(len(machine.rows))
        ]
        rho = np.array([st.rho for st in stations])
        mu = np.array([st.mu for st in stations])
        V_rel = np.array([st.V_rel for st in stations])

        # Flat plate skin friction correlation
        Cf = (2.0 * np.log10(Re_surf) - 0.65) ** -2.3
        logger.debug(f"Skin friction coefficients: {Cf}")

        tau_wall = Cf * 0.5 * rho * V_rel**2.0
        V_tau = np.sqrt(tau_wall / rho)
        L_visc = mu / rho / V_tau * NOT_FLAT_PLATE
        logger.debug(f"Viscous length scales: {L_visc}")

        d_surf = self.yplus * L_visc
        logger.debug(f"yplus={self.yplus}, wall cell spacings: {d_surf}")

        # The annulus lines are not attached to any one row, so they take the
        # mean of the rows they pass through.
        d_annulus = float(np.mean(d_surf))

        return WallSpacing(surface=d_surf, hub=d_annulus, casing=d_annulus)


def _i_at_m(block, annulus, j, m):
    """Return the `i` of `block` whose grid line along endwall `j`, 0 the hub
    or -1 the casing, lies nearest normalised meridional position `m`, its
    nodes' `m` averaged across the pitch."""
    spf = 0.0 if j == 0 else 1.0
    m_wall = np.linspace(0.0, annulus.m_max, 20001)
    xr_wall = np.asarray(annulus.evaluate_xr(m_wall, spf)).reshape(2, -1)
    x = np.asarray(block.x[:, j, :], dtype=float)
    r = np.asarray(block.r[:, j, :], dtype=float)
    _, idx = cKDTree(xr_wall.T).query(np.stack((x.ravel(), r.ravel()), axis=1))
    m_line = m_wall[idx].reshape(x.shape).mean(axis=1)
    return int(np.argmin(np.abs(m_line - m)))

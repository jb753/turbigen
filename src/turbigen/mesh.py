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

import numpy as np

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

    yplus: float = 30.0
    """Target wall distance in viscous units, which sets the near-wall cell
    size [--]."""

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
        grid.set_fluid(machine.mean_line.referenced_fluid())
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
        return float(machine.annulus.chords(0.5)[1::2].max())

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

        ref = [machine.mean_line.ref(i) for i in range(len(machine.rows))]
        rho = np.array([st.rho for st in ref])
        mu = np.array([st.mu for st in ref])
        V_rel = np.array([st.V_rel for st in ref])

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

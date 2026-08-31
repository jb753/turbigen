"""Flow solvers.

A :class:`Solver` is a config node that marches a grid. The family exists so a
config can name one by ``type:``; a member holds the settings and runs them.

The ember member is the whole integration::

    class Ember(Solver, ember.solver.Solver):
        type: ClassVar[str] = "ember"

Nothing is restated and nothing is delegated. :class:`ember.solver.Solver` is
already a frozen dataclass of plain scalars that carries ``run(grid)``, so
inheriting it gives the fields, the defaults and the march at once, and the
config protocol supplies the rest --- ``type:`` dispatch, unknown keys rejected
by name, values converted against ember's own annotations, and every resolved
default written into the archived config.

That matters more than it sounds. The wrapper in the package this replaces
restates the field list by hand, and it has rotted: of its 27 fields only 9
still exist in ember, 11 of ember's real settings cannot be reached from a
config file at all, and the module no longer imports. A schema copied by hand
is a schema that drifts, and nothing was checking.
"""

import logging
from typing import ClassVar

import ember.solver

from turbigen.node import Node

logger = logging.getLogger("turbigen")


class Solver(Node):
    """Base for flow solvers.

    Deliberately fieldless. Everything a solver needs is its own, and a
    convergence verdict belongs to the history a run produces rather than to
    the settings that produced it.
    """

    def solve(self, grid):
        """March `grid` in place and return its convergence history."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement solve(self, grid)"
        )

    def converged(self, history):
        """Return whether a run that produced `history` converged.

        Divergence only, for now. :meth:`ember.convergence_history.
        ConvergenceHistory.check_convergence` disables its residual-decay and
        residual-slope criteria at their defaults and always checks divergence,
        so calling it bare is exactly that -- and is the call that grows
        thresholds later without changing this signature.
        """
        return bool(history.check_convergence())


class Ember(Solver, ember.solver.Solver):
    """The ember explicit time-marching solver.

    Every field and ``run`` itself come from :class:`ember.solver.Solver`, so
    the settings this accepts are whatever ember accepts, always, and
    :meth:`Solver.options` lists them.

    Five defaults are restated below. ember tunes its own for large machine
    meshes; turbigen also meshes small cascades with thin boundary cells, where
    ember's ``cfl = 5`` is far past the stability limit its own docstring gives
    for the default ``n_stage = 0`` scheme (``cfl ~ 0.6``). These pick the bare,
    single-grid scree march that holds together on those: no multigrid, no
    implicit residual smoothing, a Courant number inside the documented limit.
    A config that knows its case can raise any of them.
    """

    type: ClassVar[str] = "ember"

    # Restated to carry a description, which ember does not give it.
    n_step: int
    """Number of time steps to march [--]."""

    cfl: float = 0.4
    n_stage: int = 0
    n_levels: int = 0
    fac_mgrid: float = 0.0
    sf_resid: float = 0.0

    def solve(self, grid):
        return self.run(grid)

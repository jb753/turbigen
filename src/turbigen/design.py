"""Mean-line designers.

A designer turns aerodynamic design variables into a mean-line flow field and
back again. It is a :class:`~turbigen.node.Node`, so the design variables are
its dataclass fields: they are the schema, they carry their own defaults, and
they serialise themselves. Writing one is a single class::

    class MyStage(MeanLineDesign):
        type: ClassVar[str] = "my_stage"
        n_row: ClassVar[int] = 2

        psi: float
        phi: float
        Po1: float = 1e5

        def forward(self, fluid: ember.fluid._Fluid):
            ml = self.allocate(fluid)
            ...
            return ml

        def backward(self, ml): ...

``backward`` is the single definition of what each design variable *means*. It
reports the design a CFD solution actually achieved, and it also supplies the
residual that :meth:`MeanLineDesign.solve_for` drives when ``forward`` cannot
hit a target directly. Writing a formula once and calling it from both
directions is what stops them drifting apart.

A mean line stores its state as float32 against the entropy and internal-energy
datum of its fluid, which defaults to 1 bar and 300 K. That is fine for air near
ambient, but a machine running hot enough or high enough pressure will store a
large internal energy with the kinetic energy as a small correction on top of
it, and lose the latter to rounding. A design that expects such conditions
should move the datum, before it solves, from the inlet conditions it already
knows::

    ml = self.allocate(fluid.change_datum(P_dtm=self.Po1, T_dtm=self.To1))

Note that the *reference scales* are a separate matter and not worth setting
here: floating-point precision is invariant under scaling, so dividing the
stored variables through by a density and a velocity changes their exponents
and nothing else. Scales matter to the grid, which a solver iterates on, and
:meth:`turbigen.meanline.MeanLine.referenced_fluid` supplies them there.
"""

import logging
from typing import ClassVar

import ember.fluid
import numpy as np
import scipy.optimize

from turbigen.meanline import MeanLine
from turbigen.node import Node

logger = logging.getLogger("turbigen")

_INFEASIBLE = 1.0e6
"""Residual returned for a trial the designer could not evaluate."""

_DIFF_STEP = float(np.sqrt(np.finfo(np.float32).eps))
"""Relative finite-difference step, sized for the float32 mean-line storage."""

NARROW_FRACTION = 0.1
"""How close to `rtol` a converged residual may sit before it is remarked on.

A tenth: the residuals a healthy solve reaches are three orders below rtol, so
this is quiet in normal use and speaks only for a design near the edge.
"""


class DesignError(Exception):
    """A mean-line design could not be produced."""


class MeanLineDesign(Node):
    """Base for mean-line designers."""

    n_row: ClassVar[int | None] = None
    """Number of blade rows this design describes."""

    #
    # TO BE IMPLEMENTED BY A DESIGN
    #

    def forward(self, fluid: ember.fluid._Fluid):
        """Return a mean line built from this design's variables.

        Use :meth:`allocate` for the empty mean line, fill it in, and return
        it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(self, fluid)"
        )

    def backward(self, ml):
        """Return the design variables represented by mean line `ml`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement backward(self, ml)"
        )

    #
    # PROVIDED
    #

    def allocate(self, fluid: ember.fluid._Fluid) -> MeanLine:
        """Return an empty mean line of the right size, ready to fill in.

        Parameters
        ----------
        fluid : ember.fluid._Fluid
            The equation of state, already built from the config. A design
            never sees the config node, only the fluid object it describes.

        """
        if not isinstance(self.n_row, int) or self.n_row < 1:
            raise DesignError(
                f"{type(self).__name__} must set n_row to a positive integer, "
                f"got {self.n_row!r}."
            )

        ml = MeanLine(self.n_row)
        ml.set_fluid(fluid)
        return ml

    def design(self, fluid: ember.fluid._Fluid) -> MeanLine:
        """Return a mean line built from this design.

        Checks that the result inverts back to the design variables that asked
        for it, then freezes it at the earliest opportunity, so that every
        stage which follows -- annulus, blades, mesher, post-processing --
        reads the mean line and cannot write to it.

        Because the check runs here, a nominal mean line that exists *is* the
        requested design, and there is no third state between what was asked
        for and what was achieved.
        """
        ml = self.forward(fluid)
        check_round_trip(self, ml)
        return ml.freeze()

    def solve_for(
        self, ml, build, unknowns, targets, *, rtol=1e-4, max_iter=100, name=""
    ):
        """Adjust `unknowns` until :meth:`backward` reports `targets`.

        Use this for the parts of a design that cannot be constructed
        explicitly, where some quantity has to be guessed and then corrected.
        Everything else should be built directly in ``forward``.

        Parameters
        ----------
        ml : MeanLine
            The mean line being designed. `build` writes into it, and
            :meth:`backward` is read from it.
        build : callable
            ``build(**unknowns)``, constructing `ml` for one trial set of
            values. Its return value is ignored.
        unknowns : dict
            Quantities to solve for, mapped to their initial guesses. Values
            may be scalars or arrays.
        targets : dict
            Maps a key of :meth:`backward`'s output to the value it must take.
            A number means that key must equal it. A string names another key
            of :meth:`backward`'s output that it must equal, which is how
            conditions like a repeating stage are written::

                targets={"psi": psi, "Alpha1": "Alpha3"}

        rtol : float
            Largest acceptable scaled residual. A mean line stores its state as
            float32, and targets are derived quantities, so residuals below
            about 1e-6 are not reachable however many iterations are spent:
            do not tighten this towards float64 tolerances.
        max_iter : int
            Iteration limit, in units of Jacobian evaluations.
        name : str
            Label for this solve, used in error messages.

        Returns
        -------
        dict
            The solved values, in the same shapes as `unknowns`.

        Raises
        ------
        DesignError
            If the system is underdetermined, or the solve does not converge.

        """
        label = name or "design"

        keys = list(unknowns)
        if not keys:
            raise DesignError(f"{self._who(label)}: no unknowns to solve for.")

        shapes = [np.shape(unknowns[k]) for k in keys]
        flat = [
            np.atleast_1d(np.asarray(unknowns[k], dtype=float)).ravel() for k in keys
        ]
        sizes = [f.size for f in flat]
        x0 = np.concatenate(flat)

        def unpack(x):
            out = {}
            i = 0
            for key, size, shape in zip(keys, sizes, shapes):
                chunk = x[i : i + size]
                out[key] = chunk.reshape(shape) if shape else chunk[0]
                i += size
            return out

        target_keys = list(targets)
        history = []

        def evaluate(x):
            """Residual at `x`. May raise if the trial state is unphysical."""
            build(**unpack(x))
            actual = self.backward(ml)

            parts = []
            for key in target_keys:
                want = targets[key]
                if isinstance(want, str):
                    if want not in actual:
                        raise DesignError(
                            f"{self._who(label)}: target {key!r} refers to "
                            f"{want!r}, which backward() does not return."
                        )
                    want = actual[want]
                if key not in actual:
                    raise DesignError(
                        f"{self._who(label)}: target {key!r} is not returned by "
                        f"backward()."
                    )
                got = np.atleast_1d(np.asarray(actual[key], dtype=float))
                want = np.atleast_1d(np.asarray(want, dtype=float))
                # Scale so that targets of very different magnitude (psi ~ 1,
                # Po1 ~ 1e5) contribute comparably to the least-squares norm.
                parts.append((got - want) / np.maximum(np.abs(want), 1.0))

            out = np.concatenate(parts)
            if not np.all(np.isfinite(out)):
                raise ValueError(f"non-finite residual {out}")
            return out

        # Evaluate once up front, both to size the residual and to fail early
        # on an underdetermined system rather than returning a compromise fit.
        # The initial guess must be feasible, so let any failure here surface.
        try:
            r0 = evaluate(x0)
        except DesignError:
            raise
        except Exception as err:
            raise DesignError(
                f"{self._who(label)}: the initial guess {unpack(x0)} does not "
                f"give a valid mean line: {err}"
            ) from err

        if r0.size < x0.size:
            raise DesignError(
                f"{self._who(label)}: underdetermined, {x0.size} unknown(s) "
                f"{keys} but only {r0.size} residual(s) from targets "
                f"{target_keys}. Add targets or remove unknowns."
            )
        n_res = r0.size

        def residual(x):
            """Residual for the solver, penalising infeasible trials.

            A trust-region solver routinely probes states that are not
            physically realisable -- negative static pressure, say -- and a
            designer's ``backward`` is entitled to raise on those. Returning a
            large finite residual makes the solver reject the step and shrink
            its trust radius, rather than aborting the whole design.
            """
            try:
                out = evaluate(x)
            except DesignError:
                raise
            except Exception as err:
                logger.debug(f"{self._who(label)}: infeasible trial {unpack(x)}: {err}")
                out = np.full(n_res, _INFEASIBLE)
            history.append((x.copy(), float(np.max(np.abs(out)))))
            return out

        solution = scipy.optimize.least_squares(
            residual,
            x0,
            # Unknowns routinely span orders of magnitude (a blade speed of
            # ~200 alongside a swirl velocity near 0), so let the solver take
            # its variable scaling from the Jacobian rather than assuming unity.
            x_scale="jac",
            # A mean line stores its state as float32. The default
            # finite-difference step is sized for float64 (~1e-8 relative) and
            # is smaller than float32 can resolve, so the Jacobian comes back
            # as rounding noise and the solve stalls short of the answer.
            diff_step=_DIFF_STEP,
            max_nfev=max_iter * (x0.size + 1),
        )

        err = float(np.max(np.abs(solution.fun)))
        if err > rtol:
            raise DesignError(
                f"{self._who(label)}: did not converge. Scaled residual "
                f"{err:.3e} exceeds rtol {rtol:.3e} after {len(history)} "
                f"evaluation(s).\n"
                f"  unknowns: {unpack(solution.x)}\n"
                f"  targets:  {target_keys}\n"
                f"  history:  {_format_history(history)}"
            )

        # Rebuild at the solution so that the mean line left behind is
        # consistent with the values returned, rather than with whatever trial
        # the solver happened to evaluate last.
        solved = unpack(solution.x)
        build(**solved)

        logger.debug(
            f"{self._who(label)} converged: {solved}, residual {err:.3e}, "
            f"{len(history)} evaluation(s)"
        )

        # A solve that only just converged is one that may not converge on
        # another machine. The mean line holds its state as float32, whose
        # epsilon is 1.2e-07, so a healthy residual here sits three orders
        # below rtol and a narrow one is a genuine signal rather than noise:
        # the same design has flipped between converging and not across
        # machines, and this is what says so before it does.
        if err > rtol * NARROW_FRACTION:
            logger.warning(
                f"{self._who(label)} converged narrowly: residual {err:.3e} "
                f"is within {1.0 / NARROW_FRACTION:.0f}x of rtol {rtol:.3e}, "
                f"after {len(history)} evaluation(s). This design may not "
                f"converge on another machine."
            )

        return solved

    def _who(self, label):
        return f"{self.type or type(self).__name__} [{label}]"


def _format_history(history, max_show=6):
    """Condense a solve history for an error message."""
    if len(history) <= max_show:
        shown = history
    else:
        head = history[: max_show // 2]
        tail = history[-max_show // 2 :]
        shown = head + [None] + tail
    parts = []
    for item in shown:
        if item is None:
            parts.append("...")
        else:
            x, err = item
            parts.append(f"{np.array2string(x, precision=4)}->{err:.2e}")
    return " ".join(parts)


def check_round_trip(design, ml, rtol=0.5e-2):
    """Verify a mean line reproduces the design variables that built it.

    Parameters
    ----------
    design : MeanLineDesign
        The design that produced `ml`.
    ml : MeanLine
        The mean line to check.
    rtol : float
        Relative tolerance for the comparison.

    Raises
    ------
    DesignError
        If an inverted variable disagrees with its nominal value, or mass is
        not conserved.

    """
    import dataclasses  # noqa: PLC0415 - only needed on this path

    who = design.type or type(design).__name__
    actual = design.backward(ml)

    for field in dataclasses.fields(design):
        key = field.name
        nominal = getattr(design, key)

        if key not in actual:
            # Not an error: a design may legitimately not invert every
            # variable. Say so once, clearly, rather than refusing to run.
            logger.warning(
                f"Mean-line type {who!r}: backward() does not return design "
                f"variable {key!r}, so it cannot be checked or reported."
            )
            continue

        inverted = actual[key]
        if inverted is None:
            # The design explicitly declares this one as not invertible.
            continue

        if np.all(np.asarray(nominal) == 0.0):
            if np.allclose(nominal, inverted, atol=0.1):
                continue
        elif np.allclose(nominal, inverted, rtol=rtol):
            continue

        raise DesignError(
            f"Mean-line type {who!r}: inverted {key}={inverted} does not match "
            f"the nominal value {nominal}."
        )

    # Mass must be conserved through the machine, in streamwise order.
    mdot = ml.flat.mdot
    if np.isnan(mdot).any():
        raise DesignError(f"Mean-line type {who!r}: NaN mass flow rate {mdot}.")

    if np.ptp(mdot) > np.abs(mdot[0]) * rtol:
        raise DesignError(
            f"Mean-line type {who!r}: mass flow rate not conserved, {mdot}."
        )

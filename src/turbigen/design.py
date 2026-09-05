"""Classes linking mean-line flow fields to design variables.

A :class:`MeanLineDesign` turns aerodynamic variables into a :class:`~turbigen.meanline.MeanLine`
flow field and back again. Writing the mean-line design for a new machine
is a single subclass::

    class MyStage(MeanLineDesign):
        type: ClassVar[str] = "my_stage"
        n_row: ClassVar[int] = 2

        psi: float
        phi: float
        Po1: float = 1e5

        def forward(self, fluid: ember.fluid.Fluid):
            ml = self.allocate(fluid)
            ...
            return ml

        def backward(self, ml): ...

The :doc:`/tutorial` works through such a class in full, from an empty file to
a designed fan; this page document the class in more detail. The
data structure storing the flow field is documented at :doc:`/meanline`, and
the annulus stage that follows at :doc:`/annulus`.

.. _design-contract:

The design contract
^^^^^^^^^^^^^^^^^^^

A design declares two class variables, writes two methods, and inherits the
rest:

.. list-table::
   :widths: 40 60

   * - ``type``
     - The name an input file asks for, under :ref:`mean_line:
       <config-mean_line>`.
   * - ``n_row``
     - Number of blade rows the design describes. The mean line it builds has
       shape ``(2, n_row)``.
   * - :meth:`MeanLineDesign.forward`
     - Written by the design: build a mean line from the design variables.
   * - :meth:`MeanLineDesign.backward`
     - Written by the design: recover the design variables from a mean line.
   * - :meth:`MeanLineDesign.allocate`
     - Provided: an empty mean line of the right size and fluid.
   * - :meth:`MeanLineDesign.design`
     - Provided: run ``forward``, check the round trip, freeze the result.
   * - :meth:`MeanLineDesign.solve_for`
     - Provided: drive unknowns until ``backward`` reports the targets asked
       for.

A built-in design defined in the :program:`turbigen` package is registered
automatically. A new user-created design need not be installed: it is picked up
from any ``turbigen_plugins`` directory beside the input file, or in any
directory above it.

In addition to the class variables, `type` and `n_row`, a design declares its
design variables as :class:`~dataclasses.dataclass` fields. Values for the
design variables are taken from under the :ref:`mean_line: <config-mean_line>`
key in the input file, converted to the annotated type and rejected if that
fails. A field with
no default is required, and omitting it from the input file is an error.
A field with a default is optional, and the defaulted value is recorded in
``output.yaml`` for future reproducibility.

.. _design-process:

Design process
^^^^^^^^^^^^^^

Loading an input file converts its :ref:`mean_line: <config-mean_line>` mapping
into an instance of the class named by ``type``, with defaults filled in.
:program:`turbigen` then calls :meth:`~MeanLineDesign.design` on that instance,
passing the working fluid specified by :ref:`fluid: <config-fluid>`, and that
method runs the design:

#. :meth:`~MeanLineDesign.forward` builds a
   :class:`~turbigen.meanline.MeanLine`;
#. :meth:`~MeanLineDesign.backward` inverts it, and the result is compared
   against the nominal design variables;
#. mass is checked to be conserved through the machine;
#. the :class:`~turbigen.meanline.MeanLine` is frozen, so that every stage
   which follows --- annulus, blades, mesher, post-processing --- reads it and
   cannot write to it.

Because the check runs there, a nominal :class:`~turbigen.meanline.MeanLine`
that exists *is* the requested design, and there is no third state between what
was asked for and what was achieved.

:meth:`~MeanLineDesign.forward` should start with
:meth:`~MeanLineDesign.allocate`, which returns an empty
:class:`~turbigen.meanline.MeanLine` of the right shape and working fluid,
fills it in using the setters documented at :doc:`/meanline`, and returns it.
It should make no assumption about the equation of state: a design written in
terms of enthalpy and entropy works for a perfect gas and a real one alike.
:ref:`tut-forward` builds one line by line from the design equations.

The `fluid` it is passed is the equation of state named by :ref:`fluid:
<config-fluid>`, whose interface is documented in :mod:`ember.fluid`.
Thermodynamic properties come
from its two method families: a `set_X_Y` returns the density and internal
energy pair for the two properties named ---
:meth:`~ember.fluid.Fluid.set_P_T`, :meth:`~ember.fluid.Fluid.set_P_s`,
:meth:`~ember.fluid.Fluid.set_P_h`, :meth:`~ember.fluid.Fluid.set_h_s` and the
rest --- and a `get_Z` evaluates one property from that pair, so
:meth:`~ember.fluid.Fluid.get_h`, :meth:`~ember.fluid.Fluid.get_s`,
:meth:`~ember.fluid.Fluid.get_T`, :meth:`~ember.fluid.Fluid.get_a` and so on.
The whole interface is documented in :mod:`ember.fluid`.

:meth:`~MeanLineDesign.backward` goes the other way, returning a plain dict
keyed by field name. A key that is not a field is reported for information but
never checked, so a design is free to return whatever else is worth printing
next to a CFD solution; a field mapped to ``None`` declares itself deliberately
not invertible, and is skipped; a field with no key at all warns once, naming
the variable that can no longer be checked or reported.
:meth:`~MeanLineDesign.backward` can run on a nominal design, or a mixed-out CFD
solution --- it is the single definition of what each design variable means.
:ref:`tut-backward` writes one for the design variables of a fan.

.. _design-implicit:

Implicit design problems
^^^^^^^^^^^^^^^^^^^^^^^^

Where possible, :meth:`~MeanLineDesign.forward` should build the mean line
explicitly from the design variables, but there are often situations where
the mean line cannot be built directly from the natural choice of design variables. :meth:`~MeanLineDesign.solve_for` adjusts unknowns until
the residual calculated  through :meth:`~MeanLineDesign.backward` meets the targets asked for, thus solving implicit design problems.

For example, a turbine stage at given stator exit Mach number cannot be built
in one pass: that Mach number depends on the temperature, which depends on the
static state and loss. So the
design puts the whole construction in a closure over the quantities it does not
yet know --- here the blade speed and the three swirl velocities --- and asks
for the values of those which make :meth:`~MeanLineDesign.backward` report the
design variables asked for::

    def build(U, Vt1, Vt2, Vt3_rel):
        \"\"\"Fill in `ml` for one trial set of unknowns.\"\"\"
        ...

    self.solve_for(
        ml,
        build,
        unknowns={"U": U0, "Vt1": Vt1_0, "Vt2": Vt2_0, "Vt3_rel": Vt3_rel_0},
        targets={
            "psi": self.psi,
            "Ma2": self.Ma2,
            # Repeating stage: the flow leaves as it entered
            "Alpha1": "Alpha3",
        },
        name="stage",
    )

``build`` is called as ``build(**unknowns)`` and writes into the same
:class:`~turbigen.meanline.MeanLine` every time.
The values in ``unknowns`` are initial guesses, which may be scalars or arrays,
and the guess must itself give a valid mean line (but not necessarily one that
meets the targets). During iteration, any calls to :meth:`~MeanLineDesign.backward` that error have a penalty residual applied.

There must be at least as many targets as unknowns, or the solve is refused as
underdetermined. On success the mean line is left rebuilt at the solution, so
``forward`` can return it directly, and the solved unknowns are returned as a
dict for a design that wants to keep them.

A numeric ``target`` is a value that key must take; a string ``target`` names
another key of :meth:`~MeanLineDesign.backward`'s output that it must equal.

.. _design-datum:

Thermodynamic datum
^^^^^^^^^^^^^^^^^^^

A design that expects high temperatures and pressures should
move the fluid dynamic datum before allocating the mean line.
For example, if the inlet stagnation conditions are specified as design variable fields::

    ml = self.allocate(fluid.change_datum(P_dtm=self.Po1, T_dtm=self.To1))

That is a hint for the design arithmetic, and it is not the datum the finished
mean line carries. :meth:`MeanLineDesign.design` moves it once more, onto
:meth:`~turbigen.meanline.MeanLine.get_referenced_fluid`, before freezing --- so
a mean line and the grid meshed from it always measure entropy and internal
energy from the same zero, and nothing downstream has to convert between them.
Only the datum moves; the state does not.

See :ref:`ember:datum-state` for more detail on this part of the fluid API.

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


class DesignError(Exception):
    """A mean-line design could not be produced."""


class MeanLineDesign(Node):
    """Choose the mean-line design algorithm.

    The :doc:`/design` page covers what a subclass declares and how
    :program:`turbigen` runs it.
    """

    n_row: ClassVar[int | None] = None
    """Number of blade rows this design describes."""

    #
    # TO BE IMPLEMENTED BY A DESIGN
    #

    def forward(self, fluid: ember.fluid.Fluid):
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

    def allocate(self, fluid: ember.fluid.Fluid) -> MeanLine:
        """Return an empty mean line of the right size, ready to fill in.

        Parameters
        ----------
        fluid : ember.fluid.Fluid
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

    def design(self, fluid: ember.fluid.Fluid) -> MeanLine:
        """Return a mean line built from this design.

        Checks that the result inverts back to the design variables that asked
        for it, moves it onto its own referenced fluid, then freezes it at the
        earliest opportunity, so that every stage which follows -- annulus,
        blades, mesher, post-processing -- reads the mean line and cannot write
        to it.
        """
        ml = self.forward(fluid)
        _check_round_trip(self, ml)

        # The scales and datum the CFD wants are derived from the design, so
        # this is the first moment they can be known and the last moment the
        # mean line is writable. Doing it here rather than leaving the mesher
        # to do it to the grid alone is what stops the two drifting apart: an
        # entropy measured from one zero and compared against another raises
        # nothing and reads perfectly plausibly, all the way through to an
        # isentropic velocity or a loss that is quietly wrong.
        ml.set_fluid(ml.get_referenced_fluid())

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
        if err > 0.1 * rtol:
            logger.warning(
                f"{self._who(label)} converged narrowly: residual {err:.3e} "
                f"is within 10x of rtol {rtol:.3e}, "
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


def _check_round_trip(design, ml, rtol=0.5e-2):
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
    import dataclasses

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

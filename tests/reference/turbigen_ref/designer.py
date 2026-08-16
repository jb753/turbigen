"""Base class and solver for mean-line designers.

A designer converts aerodynamic design variables into a mean-line flow field and
back again. Subclass :class:`Designer`, declare how many blade rows the design
has, and write two methods:

* ``forward(ml, **design_vars)`` builds the mean line from the design variables,
  mutating `ml` in place;
* ``backward(ml)`` reads a mean line and returns the design variables it
  represents, as a dict.

``backward`` is the single definition of what each design variable *means*. It
is used to report the design that a CFD solution actually achieved, and it also
supplies the residual that :meth:`Designer.solve_for` drives when ``forward``
cannot hit a target directly. Writing a formula once, in ``backward``, and
calling it from ``forward`` is what keeps the two directions consistent.

Design variables are ordinary keyword arguments. Their names, and which of them
are optional, are recovered from the signature of ``forward``: there is no
schema to declare.

The class deliberately holds only what a designer author writes or calls.
Operations the framework performs *on* a designer --- reading its parameters,
filling in defaults, checking a round trip --- are module-level functions
below.
"""

import inspect
import logging

import numpy as np
import scipy.optimize

logger = logging.getLogger("turbigen")

REQUIRED = inspect.Parameter.empty
"""Marker for a design variable with no default, i.e. one the user must set."""

_INFEASIBLE = 1.0e6
"""Residual returned for a trial the designer could not evaluate."""

_DIFF_STEP = float(np.sqrt(np.finfo(np.float32).eps))
"""Relative finite-difference step, sized for the float32 mean-line storage."""


class DesignError(Exception):
    """A mean-line design could not be produced."""


class Designer:
    """Base class for mean-line designers.

    Subclasses set :attr:`n_row` and implement :meth:`forward` and
    :meth:`backward`.
    """

    n_row = None
    """Number of blade rows this design describes."""

    name = None
    """Registered name, set by :func:`turbigen_ref.plugins.register_designer`."""

    def forward(self, ml, **design_vars):
        """Build the mean line `ml` from design variables, in place."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward(self, ml, ...)"
        )

    def backward(self, ml):
        """Return the design variables represented by mean line `ml`."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement backward(self, ml)"
        )

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
        shapes = [np.shape(unknowns[k]) for k in keys]
        flat = [
            np.atleast_1d(np.asarray(unknowns[k], dtype=float)).ravel() for k in keys
        ]
        sizes = [f.size for f in flat]
        x0 = np.concatenate(flat) if flat else np.zeros(0)

        if not keys:
            raise DesignError(f"{self._who(label)}: no unknowns to solve for.")

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
                            f"{self._who(label)}: target '{key}' refers to "
                            f"'{want}', which backward() does not return."
                        )
                    want = actual[want]
                if key not in actual:
                    raise DesignError(
                        f"{self._who(label)}: target '{key}' is not returned by "
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

        if r0.size < x0.size:
            raise DesignError(
                f"{self._who(label)}: underdetermined, {x0.size} unknown(s) "
                f"{keys} but only {r0.size} residual(s) from targets "
                f"{target_keys}. Add targets or remove unknowns."
            )

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

        return solved

    def _who(self, label):
        return f"{self.name or type(self).__name__} [{label}]"


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


def design_params(designer):
    """Return the design variables of `designer` as ``{name: default}``.

    A default of :data:`REQUIRED` means the user must supply the variable. The
    mean line argument of ``forward`` is not included.
    """
    sig = inspect.signature(designer.forward)
    params = list(sig.parameters.values())

    # Skip the mean line argument. On a bound method self is already gone.
    if params and params[0].name in ("ml", "mean_line", "self"):
        params = params[1:]

    return {p.name: p.default for p in params}


def resolve_defaults(designer, user_vars):
    """Merge `user_vars` over the designer's defaults, returning a full set.

    Storing the resolved set is what makes a written config reproducible: the
    archived file then records every value the design used, not only the ones
    the user typed, so changing a default later cannot silently change what an
    old config rebuilds.

    Raises
    ------
    ValueError
        If a required variable is missing, or an unknown one is supplied.

    """
    params = design_params(designer)
    who = designer.name or type(designer).__name__

    unexpected = set(user_vars) - set(params)
    if unexpected:
        raise ValueError(
            f"Unexpected design variables for mean_line type '{who}': "
            f"{sorted(unexpected)}. Valid names are {sorted(params)}."
        )

    missing = {k for k, v in params.items() if v is REQUIRED} - set(user_vars)
    if missing:
        raise ValueError(
            f"Missing required design variables for mean_line type '{who}': "
            f"{sorted(missing)}."
        )

    resolved = {k: v for k, v in params.items() if v is not REQUIRED}
    resolved.update(user_vars)
    return resolved


def check_round_trip(designer, ml, design_vars, rtol=0.5e-2):
    """Verify a mean line reproduces the design variables that built it.

    Parameters
    ----------
    designer : Designer
        The designer that produced `ml`.
    ml : MeanLine
        The mean line to check.
    design_vars : dict
        The resolved design variables passed to ``forward``.
    rtol : float
        Relative tolerance for the comparison.

    Raises
    ------
    DesignError
        If an inverted variable disagrees with its nominal value, or mass is
        not conserved.

    """
    who = designer.name or type(designer).__name__
    actual = designer.backward(ml)

    for key, nominal in design_vars.items():
        if key not in actual:
            # Not an error: a designer may legitimately not invert every
            # variable. Say so once, clearly, rather than refusing to run.
            logger.warning(
                f"Mean-line type '{who}': backward() does not return design "
                f"variable '{key}', so it cannot be checked or reported."
            )
            continue

        inverted = actual[key]
        if inverted is None:
            # The designer explicitly declares this one as not invertible.
            continue

        if np.all(nominal == 0.0):
            if np.allclose(nominal, inverted, atol=0.1):
                continue
        elif np.allclose(nominal, inverted, rtol=rtol):
            continue

        raise DesignError(
            f"Mean-line type '{who}': inverted {key}={inverted} does not match "
            f"the nominal value {nominal}."
        )

    # Mass must be conserved through the machine, in streamwise order.
    mdot = ml.flat.mdot
    if np.isnan(mdot).any():
        raise DesignError(f"Mean-line type '{who}': NaN mass flow rate {mdot}.")

    if np.ptp(mdot) > np.abs(mdot[0]) * rtol:
        raise DesignError(
            f"Mean-line type '{who}': mass flow rate not conserved, {mdot}."
        )

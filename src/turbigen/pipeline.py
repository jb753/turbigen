"""One case, from a config to a solved and measured result.

What every command that touches a grid shares, and what a script or a notebook
wants when it has a config and a directory rather than a command line:
:func:`prepare` designs, meshes and initialises a grid, :func:`solve` marches
it and writes everything a run leaves behind, and :func:`reconstruct` recovers
the answer a stored field records. Nothing here reads command-line arguments or
decides where a run's directory is; that is :mod:`turbigen.cli`.
"""

import dataclasses
import logging
from pathlib import Path

import ember.convergence_history
import numpy as np

from turbigen import bconds, case, guess, iterate, metric, mixout, post, restart
from turbigen.result import Result

logger = logging.getLogger("turbigen")

RESTART_NAME = restart.RESTART_NAME
"""Re-exported from :mod:`turbigen.restart`, where the name lives so that
:func:`turbigen.database.nearest_field` can find a neighbour's field without
importing this module."""

run_log = logging.getLogger("turbigen.run")
"""What one run produced: its tables, its verdict, the files it wrote.

Separated from the session's own messages by name rather than by level, so that
`iterate` can quieten the console for a hundred runs while the log file still
records every one of them in full. The package this replaces gets the same
effect by raising the level and emitting its results as *warnings*, which is
why a genuine warning there is indistinguishable from a startup banner.
"""


OUTPUT_NAME = "output.yaml"
"""What a run calls the resolved config and the answer it reached.

Deliberately not the name of anything anyone hands us, and now enforced rather
than merely conventional: :func:`turbigen.cli.check_not_output` refuses it as a target when the
run would write back over it --- the default, where output lands beside the
input. With `-o` sending the run to another directory the file is readable,
because the write goes somewhere else and nothing is lost.

The check earns its keep now that `report` writes here too. While only the
solving verbs did, the rule held by construction --- nothing read an
`output.yaml` and wrote one back. `report` reads a config and writes one
beside it, so without the refusal it would write over its own input.
"""


GRID_NAME = "grid.emb"
"""What a run calls the whole grid, written only when something went wrong.

Written by ember's own `write_emb`, gzipped, rather than by a pickle of our
own: a `Grid` carries weakrefs --- a patch back to its block, the connectivity
manager to its grid --- so a plain `pickle.dumps` refuses it, and a reducer
written here to dodge them would be ours to keep in step with a class that is
not. `Grid.read_emb` brings it back.
"""


HISTORY_NAME = "conv.cnv"
"""What a run calls its convergence history, written beside the flow field.

ember's own CNV format, which is a pickle: its `to_json` writes three files of
plotting points, drops the residuals and the divergence flag, and has no
reader, so it cannot bring a history back. The consequence of the format is
handled where it is read rather than avoided here --- see `read_history`.
"""


def prepare(config, restart_path=None, warm=None):
    """Return the resolved config, the machine, and a grid ready to solve.

    Shared by every verb that needs a grid, so there is one definition of
    "ready to solve" rather than one per verb. `report` stops here and `run`
    carries on, which is what makes the grid a report draws the one `run`
    actually solves. Written out twice instead, the two would drift -- which is
    what happened to `turbigen.main`, where the pipeline appears in both
    branches of one `if` and again in ninety-three unreachable lines that no
    longer match either.

    The grid is None when the config has no `mesh:` section. Whether that is an
    error belongs to the verb: `run` cannot proceed without one, while a report
    of a mean-line design is a perfectly good thing to want.

    The config comes back because it may not be the one that went in: knobs
    whose target is a property of the design alone are converged here, so the
    caller has the viscosity that was actually used rather than the guess it
    started from. `solve` archives what this returns, which is what makes an
    `output.yaml` record the design that ran.
    """
    config = iterate.resolve(config)

    # Each stage reports as it finishes rather than the verb reporting them all
    # at the end, so what is read on the way past is in the order it happened:
    # the mean line, then the grid it was meshed onto, then where the flow
    # field in that grid came from.
    machine = config.design()
    run_log.info(machine.to_string())

    if config.mesh is None:
        return config, machine, None

    grid = config.mesh.mesh(machine)
    run_log.info(grid_string(grid))

    bconds.apply(grid, machine, config.operating_point, config.inlet_profile)
    guess.apply(grid, machine)

    # A stored field supersedes the meridional guess. Applied after it rather
    # than instead of it, so that a block the field cannot fill is still left
    # with something sane in it. An explicit or chained `restart_path` wins; a
    # `warm` seed (a database neighbour's field, perturbed onto this design's
    # mean line) is the fallback for the first solve of a loop when nothing has
    # been chained yet.
    if restart_path is not None:
        restart.apply(grid, restart_path)
    elif warm is not None:
        warm.apply(grid, machine)

    return config, machine, grid


def stored_field(config_path):
    """Return the flow field a previous run left beside `config_path`.

    Found rather than named, because a report of a run that has one always
    wants it, and there is no second thing it could sensibly mean. That is what
    lets the re-plot be the same command as the plot, with no flag between
    them. Nothing raises when there is none: a case that has not been solved
    still has geometry worth drawing.
    """
    field = Path(config_path).resolve().parent / RESTART_NAME

    if not field.is_file():
        return None

    logger.info(f"Using the flow field at {field}")
    return field


def save_history(path, history):
    """Write `history` beside the flow field it belongs to."""
    history.write_cnv(path)
    logger.debug(f"Wrote the convergence history to {path}")


def read_history(path):
    """Return the convergence history at `path`, or None if there is not one.

    A history is a bonus rather than a requirement, so nothing here raises: a
    re-plot without one is the report minus its convergence page, which beats a
    re-plot that refuses to run because a file an older version wrote will not
    unpickle. That is the price of ember's CNV format, paid here rather than by
    the caller.
    """
    if not path.is_file():
        return None

    try:
        return ember.convergence_history.ConvergenceHistory.read_cnv(path)
    except Exception as err:
        logger.warning(f"Could not read the convergence history at {path}: {err}")
        return None


def reconstruct(config, machine, grid, history, field):
    """Return the answer `field` records for `config`, or None if it cannot.

    A report has the same ingredients a run has once its march is over -- the
    solved grid, the history beside it -- so it can measure the same three
    things `solve` does and archive an identical `result:`. What it does not
    have is the run itself, so it must first establish that the field it picked
    up is the solution to the config in front of it and not merely a useful
    starting guess for it. That is the stamp's whole purpose.

    None is returned wherever any part of that fails. It means "this report has
    no answer to record", never "the answer is that it did not converge": the
    two are very different to `turbigen.database`, which drops any sample whose
    `converged` is false, so guessing here would silently delete designs from a
    later fit.
    """
    if field is None or grid is None:
        return None

    stamp = restart.read_stamp(field)
    if stamp is None:
        logger.info(
            f"The field at {field} carries no design stamp, so this report "
            "cannot tell which design it solves and will not record an answer."
        )
        return None

    if stamp != restart.design_stamp(config):
        logger.warning(
            f"The field at {field} was written for a different design than "
            "this config describes, so it is drawn but not recorded as its "
            "answer."
        )
        return None

    # Beyond here the field is this config's solution. The remaining guards are
    # about whether the answer can be *described*, not whether it is the right
    # one.
    if config.solver is None or history is None:
        logger.info(
            "Without a solver: section and a convergence history there is no "
            "saying whether this field converged, so no answer is recorded."
        )
        return None

    try:
        actual, Ds_mix = mixout.mean_line(grid, machine)
    except Exception as err:
        logger.warning(f"Could not mix out the stored field: {err}")
        return None

    result = Result(
        machine=machine,
        actual=actual,
        Ds_mix=Ds_mix,
        grid=grid,
        converged=config.solver.converged(history),
        history=history,
    )

    # Tolerant, because this is describing a stored field rather than steering
    # a design off it: a march that diverged has nothing to measure and is
    # exactly the run whose report someone needs to read.
    return observe(config, result, strict=False)


def observe(config, result, strict):
    """Return `result` with its iterator errors and metrics measured.

    Everything a solved grid is asked before it goes out of scope, shared by a
    run that has just marched and a report of a field a run left behind, so
    the two archive the same `result:`. `strict` is passed to
    :func:`turbigen.iterate.errors`; failures are the caller's to handle.
    """
    result = dataclasses.replace(
        result, error=iterate.errors(config, result, strict=strict)
    )

    # Anything the config asked to measure from the field, for the same reason
    # and against the same deadline.
    return dataclasses.replace(result, metrics=metric.measure(config, result))


def save_grid(path, grid):
    """Write the whole grid to `path`, for a run that is not going to finish.

    Never allowed to be the thing that ends a run: this is called on paths that
    are already failing, and a diagnostic that raises on its way out would
    replace the error someone needs to read with one about writing a file.

    Catching the exception is not enough for that, because writing a grid moves
    it. `Grid.write_emb` detaches every patch before pickling and re-attaches
    them afterwards, so a write that fails partway can leave a perfectly good
    grid with patches that no longer know their block --- and the next thing to
    read a surface off it dies on that instead, far from here and looking like
    a fault of its own. So the grid is put back before the warning goes out,
    and the run carries on with what it was given.
    """
    try:
        grid.write_emb(str(path), compress=True)
        logger.info(f"Wrote the grid to {path}")
    except Exception as err:
        logger.warning(f"Could not write the grid to {path}: {err}")
        _reattach_patches(grid)


def _reattach_patches(grid):
    """Re-establish every patch's link to its block, quietly.

    What :meth:`ember.grid.Grid.write_emb` does on its way out, done again for
    a write that did not get that far. Quiet because it is already the second
    thing to go wrong: the first is on its way to the log, and this one has
    nothing to add that the grid being unreadable will not say later.
    """
    try:
        for block in grid:
            for patch in block.patches:
                patch.attach_to_block(block)
    except Exception as err:
        logger.debug(f"Could not re-attach the grid's patches: {err}")


def soft_start(solver, grid, n_step=None):
    """March `grid` with a detuned copy of `solver`, and return what it did.

    Nothing of it is kept. The grid is marched in place, so the production
    march that follows simply continues from the field this leaves, and the
    history comes back only so the caller can see whether it survived.

    The step count is the config's, replacing the one `soft()` names: how long
    to spend on a robust start depends on the case and the guess it is starting
    from, which the solver has no way to know. `n_step` defaults to
    `solver.n_step_soft`; a retry passes `solver.n_step_retry` instead.
    """
    if n_step is None:
        n_step = solver.n_step_soft

    soft = dataclasses.replace(solver.soft(), n_step=n_step)

    run_log.info(f"Soft start: {n_step} steps")
    history = soft.solve(grid)
    run_log.info(convergence_string(solver.converged(history)))

    return history


def _march(config, restart_path, warm, soft, retry):
    """Prepare `config` and march it, soft starting or retrying as asked.

    Returns ``(config, machine, grid, history)``. All four, because a retry
    rebuilds the start and so replaces everything `prepare` made the first time.
    See :func:`solve` for what `soft` and `retry` mean.
    """
    config, machine, grid = prepare(config, restart_path, warm=warm)

    if grid is None:
        raise ValueError("The 'run' command needs a mesh: section in the config file.")

    history = None
    soft_ran = bool(soft and config.solver.n_step_soft)
    if soft_ran:
        history = soft_start(config.solver, grid)

        # A soft pass that blew up leaves a field of NaNs, and marching the
        # production settings from those is CFD paid for and certain to reach
        # no answer. Its history stands as the run's own, so everything after
        # writes the failure and the field it failed in, which is the only
        # record there is of what happened here.
        if config.solver.converged(history):
            history = None

    if history is None:
        history = config.solver.solve(grid)

        # A hard start that diverged gets one more go. Only one that had no
        # soft start: a soft pass that ran and still led here is not helped by
        # running it again.
        if (
            retry
            and not soft_ran
            and config.solver.n_step_retry
            and getattr(history, "diverged", False)
        ):
            run_log.info(
                "Diverged without a soft start; retrying from the same field "
                f"with {config.solver.n_step_retry} soft steps"
            )

            # The march left NaNs in the grid, and the patches carry state of
            # their own -- mixing-plane exchange, throttle, relaxation -- so the
            # start is rebuilt rather than restored.
            del grid
            config, machine, grid = prepare(config, restart_path, warm=warm)

            history = soft_start(config.solver, grid, config.solver.n_step_retry)
            if config.solver.converged(history):
                history = config.solver.solve(grid)

    return config, machine, grid, history


def _save_field(out_dir, config, grid, history):
    """Write the flow field and its history, and the whole grid if it diverged.

    Written whatever happened, and written first --- before the mix-out, the
    measurements and the tables, every one of which can raise. A march that did
    not converge is the one most likely to be picked up and continued, so
    withholding its field would be exactly backwards; and nothing downstream
    may be able to discard a solution the CFD has already been paid for. The
    field is also the whole of what a failure needs to be diagnosed from: the
    mesh is not written because `input.yaml` beside it rebuilds one, and the
    two together are what `prepare` reads back.
    """
    restart_path = out_dir / RESTART_NAME
    restart.save(restart_path, grid, config)
    run_log.info(f"Wrote the flow field to {restart_path}")

    # Beside the field, and for the same reason: it is what a re-plot needs to
    # draw the convergence page, and it costs a few kilobytes.
    save_history(out_dir / HISTORY_NAME, history)

    # A march that fell over is the one nobody can rebuild their way back to.
    # `restart.npz` holds the flow and nothing else, on the reasoning that
    # `input.yaml` beside it rebuilds the mesh -- which is true only while the
    # mesher is the code that wrote it, and a failure worth keeping is often
    # one being chased across a change to that code. It is also the case that
    # needs the geometry most: a divergence reports where it happened in index
    # space, and turning `i[144:144]` into a trailing edge takes coordinates.
    # So the whole grid goes down, coordinates, patches and all.
    if getattr(history, "diverged", False):
        save_grid(out_dir / GRID_NAME, grid)


def _measure(config, machine, grid, history, out_dir):
    """Reduce a marched grid to a `Result`, while the grid is still in memory.

    Returns ``(config, result)``: the config too, because a throttled exit
    folds the operating point it reached back into it.

    Measured whether or not anything is iterating: the exit angle a row
    achieved and the incidence its leading edge saw are observations of the
    flow, and they can only be taken while the grid is in memory.

    A measurement that refuses is the other half of the case the divergence
    flag covers in :func:`_save_field`, and the harder one: the march
    converged, so nothing says the geometry is suspect, and yet a leading edge
    had no stagnation point to find or a section had no surface to cut. That is
    a question about the grid, asked of a grid about to go out of scope. Written
    once, whichever measurement raises, and the error goes on to be raised.
    """
    converged = config.solver.converged(history)

    # Reduce the solution to a mean line. A diverged grid has nothing to mix
    # out, and even a converged one can refuse, so this must not cost the run
    # the output it has already earned.
    actual = None
    Ds_mix = None
    try:
        actual, Ds_mix = mixout.mean_line(grid, machine)
    except Exception as err:
        logger.warning(f"Could not mix out the solution: {err}")

    result = Result(
        machine=machine,
        grid=grid,
        actual=actual,
        Ds_mix=Ds_mix,
        converged=converged,
        history=history,
    )

    # Strict only when the march reached an answer, because that is the only
    # time an unmeasurable knob means something is wrong. A diverged march
    # measures nothing by construction -- there is no mixed-out mean line to
    # read an exit angle off a field of NaNs -- and raising about it here
    # aborted the whole loop with a traceback, from inside the call
    # `iterate.converge` makes, before it could reach its own divergence check
    # and stop with the design that produced it. Describing the failure is what
    # is wanted; steering on it is what is refused.
    try:
        result = observe(config, result, strict=converged)
    except Exception:
        if not (out_dir / GRID_NAME).is_file():
            save_grid(out_dir / GRID_NAME, grid)
        raise

    # Where a throttled exit turned out to sit, for the same reason and with
    # the same deadline: the pressure the controller chose is on the patch, and
    # the patch goes out of scope with the grid. Folded into the config as well
    # as the result, so that `output.yaml` records the operating point the run
    # reached rather than the guess it was given -- which is what makes a
    # throttled run reproducible, and what a characteristic sweep measures from.
    achieved = bconds.achieved(grid, machine, config.operating_point)
    if achieved is not None:
        config = dataclasses.replace(config, operating_point=achieved)
        result = dataclasses.replace(result, operating_point=achieved)

    return config, result


def _log_answer(config, result):
    """Log the verdict, the mixed-out mean line and the design comparison."""
    run_log.info(convergence_string(result.converged))
    if result.actual is None:
        return

    run_log.info(result.actual.to_string())

    # Last, because it is the answer to the question the config asked, and what
    # someone reads first when scrolling back. Guarded for the same reason the
    # mix-out is: a table is a report of a solution the CFD has already been
    # paid for, and it must not be able to cost the run the output written
    # after it.
    try:
        run_log.info(design_variable_string(config, result))
    except Exception as err:
        logger.warning(f"Could not compare the design against its solution: {err}")


def solve(
    config,
    out_dir,
    restart_path=None,
    svg=False,
    trajectory=(),
    warm=None,
    soft=False,
    retry=False,
):
    """Design, mesh and solve `config`, writing everything into `out_dir`.

    The whole of a run, so that `iterate` composes runs rather than writing a
    second copy of one -- which is how `turbigen.main` came to hold the same
    pipeline three times over, two of them unreachable and already drifted.

    `trajectory` is what the design loop has been through so far, if anything,
    which the report draws the loop's own convergence from. Passed in rather
    than reachable from here, because a run knows nothing of the loop that may
    be repeating it -- and passed *through* rather than kept, because the pair
    this call is about does not exist until the solve is over.

    `warm` is a :class:`turbigen.warm_field.Seed` for the first solve of a loop,
    used only when nothing has been chained into `restart_path` yet.

    `soft` says this is the first solve of the invocation, which is the one
    entitled to a soft start if the config asked for one. Passed in rather than
    inferred from `restart_path`, because a restarted run is still a first
    solve: the field it was handed may be a neighbour's, or its own from before
    a change to the design, and both are exactly what a robust pass is for.
    What disqualifies a solve is having a loop's previous iteration behind it,
    which only the caller knows.

    `retry` says a divergence here may be tried again, once, from the same
    start behind `solver.n_step_retry` soft steps, if the config asked for that
    and this solve had no soft start of its own. The caller decides because a
    `chic` point that diverges is an answer rather than a failure.
    """
    config, machine, grid, history = _march(config, restart_path, warm, soft, retry)
    _save_field(out_dir, config, grid, history)
    config, result = _measure(config, machine, grid, history, out_dir)
    _log_answer(config, result)
    _write_output(config, result, out_dir, svg=svg, trajectory=trajectory)
    return result


def _write_output(config, result, out_dir, svg=False, trajectory=()):
    """Write what a run achieved, and draw it.

    Only the verbs that solve call this, and that is what makes it safe:
    `output.yaml` is written by whoever has a real answer to put in it, so no
    verb can replace a converged run's `result:` with an empty one of its own.
    An earlier arrangement had `mesh --restart --write` writing back over the
    config it had just read, guarded by comparing the two -- a guard that only
    existed because the wrong verb was writing.
    """
    config_path = out_dir / OUTPUT_NAME
    case.write(config_path, config, result)
    run_log.info(f"Wrote resolved configuration to {config_path}")

    post.write_report([*trajectory, (config, result)], out_dir, svg=svg)


def grid_string(grid):
    """One-line summary of the size of a grid."""
    return f"Mesh: n_block={len(grid)}, n_cell/1e6={grid.size / 1e6:.2f}"


def convergence_string(converged):
    """Report how a march ended, as a verdict alone.

    ember has already logged every record as the march went, so repeating the
    last one here would print it twice. Nor is a step count quoted: records
    are written every `n_step_log` steps, so the last record is not in general
    the last step marched, and reporting it as one would be wrong.
    """
    verdict = "converged" if converged else "NOT converged"
    return f"Solver: {verdict}"


def _design_variable_rows(config, result):
    """Yield ``(name, nominal, actual, is_variable)`` for every inverted key.

    **Both columns come from the same `backward()`**, one applied to the
    nominal mean line and one to the mixed-out actual, rather than reading the
    nominal off the config's own fields. Three things follow.

    The comparison is like for like: whatever definition of loss or loading the
    design uses, both sides are measured through it, so a difference is the
    flow differing and never the two sides being computed differently.

    Diagnostics get a nominal column for free. `backward` returns reaction,
    pressure ratio and efficiency alongside the design variables, and those are
    where a mismatch usually shows first. The package this replaces reached
    them through a second loop over "additional vars not in nominal" and left
    the nominal column blank, so the one comparison worth making was the one it
    could not print.

    And it is sound, because there are two states and not three: `solve_for`
    raises if it cannot hit its targets and `_check_round_trip` raises if the
    inverted variables disagree with the fields that asked for them, so a
    nominal mean line that exists *is* the requested design.

    Design variables are still marked, because a variable you set and a number
    you read are different kinds of thing even when they are printed the same
    way. That is field membership rather than the order `backward` happens to
    return its keys in, which is only the author's convention.
    """
    variables = {field.name for field in dataclasses.fields(config.mean_line)}

    nominal = config.mean_line.backward(result.nominal)

    # `design` builds a result with no CFD behind it at all, so there is no
    # actual to invert -- every row is nominal-only, not "neither can be
    # compared" (that verdict is for a field a run's actual declines to
    # invert, not for a run that never had one).
    has_actual = result.actual is not None
    actual = config.mean_line.backward(result.actual) if has_actual else {}

    for name, value in nominal.items():
        # A design may declare a variable as not invertible, and it may return
        # one the other call did not; neither is an error, and neither can be
        # compared.
        if value is None or (has_actual and actual.get(name) is None):
            continue

        was = np.atleast_1d(value)
        now = np.atleast_1d(actual[name]) if has_actual else None
        if now is not None and was.shape != now.shape:
            continue

        for i in range(was.size):
            label = name if was.size == 1 else f"{name}[{i}]"
            yield (
                label,
                float(was[i]),
                float(now[i]) if now is not None else None,
                name in variables,
            )


def design_variable_string(config, result):
    """Return a table of what the design asked for against what it achieved.

    The most valuable few lines a run prints: a mean line states an intent, and
    this is the only place that intent and the CFD are put side by side in the
    same units.

    Errors are `nominal - actual`, which is the sign
    :meth:`turbigen.iterate.MeanLine.error` already uses, so a row here and a
    row of the iteration table describe one number the same way round.
    """
    rows = list(_design_variable_rows(config, result))
    if not rows:
        return "Design variables: nothing that backward() returns can be compared."

    width = max(len(name) for name, _, _, _ in rows)

    # `design` never has a CFD actual to set against the nominal, so it gets
    # the narrower table this degrades to: a value, not a comparison.
    has_actual = result.actual is not None
    if has_actual:
        header = f"{'name':<{width}}  {'nominal':>10}  {'actual':>10}  {'err':>10}  {'err/%':>8}"
    else:
        header = f"{'name':<{width}}  {'nominal':>10}"
    lines = ["Design variables:", header, "-" * len(header)]

    # Set variables first, then what was read off the answer, with a rule
    # between. Within each, the order the design returned them in, which is the
    # order its author thought about them.
    for wanted in (True, False):
        block = [row for row in rows if row[3] is wanted]
        if not block:
            continue
        if not wanted:
            lines.append("-" * len(header))

        for name, was, now, _ in block:
            if not has_actual:
                lines.append(f"{name:<{width}}  {was:10.4g}")
                continue

            error = was - now
            # A nominal of zero has nothing to be relative to. Recamber and
            # swirl angles are routinely zero by design, so this is the common
            # case rather than a guard against the impossible.
            relative = f"{error / was * 100.0:8.2f}" if was else f"{'--':>8}"
            lines.append(
                f"{name:<{width}}  {was:10.4g}  {now:10.4g}  {error:10.3g}  {relative}"
            )

    return "\n".join(lines)

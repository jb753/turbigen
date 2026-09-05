"""Tests for starting an iteration from designs already run.

No CFD anywhere here. A sample is a config file with a fabricated `result:`,
and what is under test is which files count, which of their leaves count as
design variables, and what the blend does with them. A march would add minutes
and a noise floor while proving nothing extra.

The samples are laid out along one axis: three finished runs at `psi` of 1.4,
1.6 and 1.8, whose trailing-edge recamber ended at 5, 6 and 7 degrees. That is
enough for every property worth asserting, and the arithmetic can be done by
hand.

Test cases:
- test_the_database_section_round_trips: it is an ordinary config node
- test_x_is_deduced_from_what_varies: a leaf every run agrees on is not an axis
- test_an_iterated_knob_is_not_a_variable: an output cannot be its own input
- test_solver_settings_are_not_variables: a finer mesh is the same design
- test_a_sample_is_reproduced_exactly: querying a run already done returns it
- test_a_prediction_is_bounded_by_the_samples: it cannot leave the hull
- test_the_nearest_sample_dominates: which is what inverse distance means
- test_a_single_sample_is_copied: the graceful end of the decay
- test_repeat_runs_are_averaged: and neither of two on-top samples wins
- test_no_samples_leaves_the_config_alone: the only refusal
- test_no_database_leaves_the_config_alone: the key is optional
- test_unconverged_runs_are_not_samples: both senses of converged
- test_a_run_with_no_result_is_not_a_sample: never run is not converged
- test_the_run_being_started_is_excluded: or it predicts its own answer
- test_mismatched_iterators_are_skipped: no correspondence between the knobs
- test_an_unreadable_file_is_skipped: one bad file does not stop a run
- test_declared_variables_override_the_deduction: the escape hatch
- test_a_warm_start_is_a_new_config: frozen, and the original is untouched
"""

import dataclasses

import pytest
from test_blade import build

from turbigen import Config, Database, Result, case, database, iterate

ITERATORS = (iterate.Deviation(), iterate.Incidence())
"""What every config here iterates, query and sample alike."""

SPREAD = ((1.4, 5.0), (1.6, 6.0), (1.8, 7.0))
"""The finished runs: a stage loading, and the recamber it ended on."""


def make(psi, dchi_TE, **kwargs):
    """Return a two-row config at `psi` whose rows are recambered `dchi_TE`."""
    config = dataclasses.replace(
        build(), iterate=iterate.Iteration(correct=ITERATORS), **kwargs
    )
    config = dataclasses.replace(
        config, mean_line=dataclasses.replace(config.mean_line, psi=psi)
    )
    # Through the iterator rather than by hand, so the sections carry whatever
    # spanwise distribution `build` gave them, shifted to this mean.
    return iterate.Deviation().with_unknowns(
        config, {f"dchi_TE[{i_row}]": dchi_TE for i_row in range(2)}
    )


def finished(config, converged=True, error=None):
    """Return the result of a run that converged, and settled, at `config`."""
    if error is None:
        error = {name: 0.0 for name in iterate.unknowns(config)}
    return Result(converged=converged, error=error)


def write(directory, config, **kwargs):
    """Write `config` and its result as a finished run under `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.yaml"
    case.write(path, config, finished(config, **kwargs))
    return path


@pytest.fixture
def runs(tmp_path):
    """Three finished runs, under `runs/`, spread along `mean_line.psi`."""
    for i_run, (psi, dchi_TE) in enumerate(SPREAD):
        write(tmp_path / "runs" / f"{i_run:03d}", make(psi, dchi_TE))
    return tmp_path


def query(psi, **kwargs):
    """Return a config to be warm-started, at `psi`."""
    return make(psi, 0.0, database=Database(path="runs/*/config.yaml"), **kwargs)


def recamber(config, i_row=0):
    """Return the mean trailing-edge recamber of a row."""
    return iterate.unknowns(config)[f"dchi_TE[{i_row}]"]


#
# THE CONFIG NODE
#


def test_the_database_section_round_trips():
    config = query(1.5)

    assert Config.from_dict(config.to_dict()) == config


def test_no_database_leaves_the_config_alone(tmp_path):
    config = dataclasses.replace(build(), iterate=iterate.Iteration(correct=ITERATORS))

    assert database.warm_start(config, tmp_path) is config


#
# WHICH LEAVES ARE DESIGN VARIABLES
#


def test_x_is_deduced_from_what_varies(runs):
    """Only `psi` differs between the runs, so only `psi` is an axis."""
    config = query(1.5)
    samples = config.database.load(config, runs)

    assert config.database.candidates(config, samples) == ("mean_line.psi",)


def test_an_iterated_knob_is_not_a_variable(runs):
    """The recamber varies across the runs, and is still not an input.

    It is what is being predicted. Left in, the blend would place the query by
    the very number it is trying to supply.
    """
    config = query(1.5)
    samples = config.database.load(config, runs)

    variables = config.database.candidates(config, samples)

    assert not any("dchi_TE" in path for path in variables)


def test_solver_settings_are_not_variables(tmp_path):
    """A machine run for longer is the same machine."""
    from turbigen import Ember

    for i_run, (psi, dchi_TE) in enumerate(SPREAD):
        config = make(psi, dchi_TE, solver=Ember(n_step=100 * (i_run + 1)))
        write(tmp_path / "runs" / f"{i_run:03d}", config)

    config = query(1.5, solver=Ember(n_step=100))
    samples = config.database.load(config, tmp_path)

    assert config.database.candidates(config, samples) == ("mean_line.psi",)


def test_declared_variables_override_the_deduction(runs):
    config = query(1.5)
    config = dataclasses.replace(
        config,
        database=dataclasses.replace(config.database, variables=("mean_line.Ma2",)),
    )
    samples = config.database.load(config, runs)

    assert config.database.candidates(config, samples) == ("mean_line.Ma2",)


#
# THE PREDICTION
#


def test_a_sample_is_reproduced_exactly(runs):
    """A design already run starts where that run finished."""
    started = database.warm_start(query(1.6), runs)

    assert recamber(started) == pytest.approx(6.0)


def test_a_prediction_is_bounded_by_the_samples(runs):
    """Far outside the hull, and still a blend of things that converged.

    This is the property that removes the need for a clip: a warm start cannot
    ask for a recamber no finished design ever used, so it cannot ask for a
    blade that will not mesh.
    """
    started = database.warm_start(query(20.0), runs)

    assert 5.0 <= recamber(started) <= 7.0


def test_the_nearest_sample_dominates(runs):
    """Just off a sample, the answer is nearly that sample's."""
    started = database.warm_start(query(1.61), runs)

    assert recamber(started) == pytest.approx(6.0, abs=0.1)


def test_a_single_sample_is_copied(tmp_path):
    """One run varies in nothing, so every query sits on top of it."""
    write(tmp_path / "runs" / "000", make(1.4, 5.0))

    started = database.warm_start(query(1.9), tmp_path)

    assert recamber(started) == pytest.approx(5.0)


def test_repeat_runs_are_averaged(tmp_path):
    """Two runs of one design are equally authoritative, so neither wins."""
    write(tmp_path / "runs" / "000", make(1.4, 5.0))
    write(tmp_path / "runs" / "001", make(1.4, 7.0))

    started = database.warm_start(query(1.9), tmp_path)

    assert recamber(started) == pytest.approx(6.0)


def test_a_warm_start_is_a_new_config(runs):
    """Frozen, so the config handed in is the config still held."""
    config = query(1.5)

    started = database.warm_start(config, runs)

    # Something actually moved, or the rest of this asserts nothing.
    assert recamber(started) == pytest.approx(5.9, abs=0.5)
    assert recamber(config) == pytest.approx(0.0)


#
# WHAT COUNTS AS A SAMPLE
#


def test_no_samples_leaves_the_config_alone(tmp_path):
    config = query(1.5)

    assert database.warm_start(config, tmp_path) is config


def test_unconverged_runs_are_not_samples(tmp_path):
    """A march that blew up, and a design that had not settled, both refused.

    The second is what the old package got wrong by filtering on directory
    depth: an intermediate iteration is a converged march whose recambers are
    still on their way somewhere.
    """
    write(tmp_path / "runs" / "000", make(1.4, 5.0), converged=False)
    write(tmp_path / "runs" / "001", make(1.6, 6.0), error={"dchi_TE[0]": 99.0})

    config = query(1.5)

    assert config.database.load(config, tmp_path) == []


def test_a_run_with_no_result_is_not_a_sample(tmp_path):
    directory = tmp_path / "runs" / "000"
    directory.mkdir(parents=True)
    make(1.4, 5.0).to_file(directory / "config.yaml")

    config = query(1.5)

    assert config.database.load(config, tmp_path) == []


def test_the_run_being_started_is_excluded(runs):
    """Or a re-run predicts its own answer and learns nothing."""
    config = query(1.5)
    excluded = runs / "runs" / "001"

    samples = config.database.load(config, runs, exclude=(excluded,))

    assert len(samples) == 2
    assert all(sample.mean_line.psi != 1.6 for sample in samples)


def test_mismatched_iterators_are_skipped(runs, caplog):
    """A design with a different row count has no comparable knobs."""
    one_row = dataclasses.replace(
        make(1.5, 5.0),
        blades=make(1.5, 5.0).blades[:1],
        iterate=iterate.Iteration(correct=ITERATORS),
    )
    write(runs / "runs" / "003", one_row)

    config = query(1.5)
    samples = config.database.load(config, runs)

    assert len(samples) == len(SPREAD)
    # Dropped for having the wrong knobs, not for failing to read: those are
    # different bugs and only one of them is the behaviour under test.
    assert "did not read as a case" not in caplog.text


def test_an_unreadable_file_is_skipped(runs, caplog):
    """One bad file among a hundred is a warning, not the end of the run."""
    broken = runs / "runs" / "bad"
    broken.mkdir()
    (broken / "config.yaml").write_text("mean_line: [this is not a config]\n")

    config = query(1.5)
    samples = config.database.load(config, runs)

    assert len(samples) == len(SPREAD)
    assert "did not read as a case" in caplog.text

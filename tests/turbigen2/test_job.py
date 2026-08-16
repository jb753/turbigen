"""Tests for queue submission.

Nothing here shells out: `job.run_process` is the one place this module leaves
the process, so it is replaced and the calls it would have made are asserted
on. What matters is that a submitted invocation says exactly what the local one
would have said, which is the only way a queued run can silently differ from a
run you watched.
"""

import pytest

from turbigen2 import cli, job

CASE = """
fluid:
  type: perfect
  cp: 1005.0
  gamma: 1.4
  mu: 1.8e-5
mean_line:
  type: axial_turbine
  psi: 1.6
  phi2: 0.8
  Ma2: 0.9
  fac_Ma3_rel: 0.8
  mdot: 10.0
  Ys: [0.05, 0.05]
  r_rms: 0.3
solver: {type: ember, n_step: 10}
job:
  type: slurm
  hours: 1.5
  partition: icelake
"""


@pytest.fixture
def calls(monkeypatch):
    """Record what would have been run, and answer as the queue would."""
    seen = []

    def fake(argv, cwd):
        seen.append((list(argv), str(cwd)))
        if argv[0].endswith("sbatch"):
            return "Submitted batch job 4812291"
        return "17"

    monkeypatch.setattr(job, "run_process", fake)
    return seen


@pytest.fixture
def batch(tmp_path):
    """Three members of a batch, laid out as the batch verb writes them."""
    paths = []
    for index in range(3):
        directory = tmp_path / "batch_0000" / f"{index:04d}"
        directory.mkdir(parents=True)
        path = directory / "input.yaml"
        path.write_text(CASE)
        paths.append(path)
    return paths


def tasks_for(paths):
    return [job.Task(config=path, name=path.parent.name) for path in paths]


#
# THE FAMILY
#


def test_a_job_round_trips_like_any_other_node():
    spec = {"type": "slurm", "hours": 4.0, "partition": "ampere", "gres": "gpu:1"}

    assert job.Job.from_dict(spec).to_dict() == job.Slurm.from_dict(spec).to_dict()
    assert job.Job.from_dict(spec).partition == "ampere"


def test_a_config_carries_a_job(tmp_path):
    """It is a config node, so it is loaded and written like everything else."""
    from turbigen2 import Config

    path = tmp_path / "input.yaml"
    path.write_text(CASE)

    config = Config.from_file(path)

    assert config.job.type == "slurm"
    assert config.job.hours == pytest.approx(1.5)


def test_a_config_without_a_job_has_none(tmp_path):
    from turbigen2 import Config

    path = tmp_path / "input.yaml"
    path.write_text(CASE.split("job:")[0])

    assert Config.from_file(path).job is None


def test_a_job_is_not_a_design_variable():
    """Which queue a run used says nothing about the machine it designed."""
    from turbigen2 import database

    assert "job" not in database.SUBTREE


def test_a_backend_must_implement_forward():
    with pytest.raises(NotImplementedError):
        job.Job().submit([job.Task(config="x.yaml", name="x")], "run")


def test_submitting_nothing_is_refused():
    with pytest.raises(ValueError, match="nothing to submit"):
        job.Slurm().submit([], "run")


#
# SLURM
#


def test_slurm_omits_what_was_not_stated():
    """Zero or empty means unstated, so sbatch's own environment still applies.

    A cluster that sets SBATCH_ACCOUNT in your profile therefore needs nothing
    in the config file but `type: slurm`.
    """
    script = job.Slurm(partition="icelake").script("b", "run", ())

    assert "--partition=icelake" in script
    assert "--account" not in script
    assert "--time" not in script
    assert "--gres" not in script


def test_slurm_states_what_it_was_given():
    script = job.Slurm(
        hours=2.5, account="FOO-BAR", gres="gpu:1", cpus=4, qos="high"
    ).script("b", "run", ())

    assert "#SBATCH --time=02:30:00" in script
    assert "#SBATCH --account=FOO-BAR" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --qos=high" in script


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(1.0, "01:00:00"), (1.5, "01:30:00"), (0.25, "00:15:00"), (36.0, "36:00:00")],
)
def test_hours_become_a_time_limit(hours, expected):
    assert job._time_string(hours) == expected


def test_slurm_expands_the_config_and_not_the_options():
    """$CONFIG must reach the shell unquoted, or every task runs on a file of
    that name; the options must be quoted, or a space in one splits it."""
    script = job.Slurm().script("b", "run", ("-s", "mean_line.psi=1.8"))

    assert '"$CONFIG"' in script
    assert "-s mean_line.psi=1.8" in script
    assert f'sed -n "${{SLURM_ARRAY_TASK_ID}}p" {job.TASKS_NAME}' in script


def test_slurm_submits_one_array_over_a_file_of_tasks(batch, calls):
    """One submission, indexing lines rather than directory names.

    The package this replaces indexes numbered directories and so refuses
    anything but a consecutive range, which the batches `batch` writes are
    not: a point that will not design is skipped and never retried.
    """
    base = batch[0].parent.parent

    ids = job.Slurm().submit(tasks_for(batch), "run", ("-v",))

    ((argv, cwd),) = calls
    assert argv[0].endswith("sbatch")
    assert argv[1] == "--array=1-3"
    assert argv[2] == job.SCRIPT_NAME
    assert cwd == str(base)

    written = (base / job.TASKS_NAME).read_text().splitlines()
    assert written == [str(path.resolve()) for path in batch]

    assert (base / job.SCRIPT_NAME).is_file()
    assert "4812291" in ids[0]


def test_gaps_in_a_batch_do_not_disturb_the_array(tmp_path, calls):
    """The array indexes lines, so a skipped member index is not a hole."""
    paths = []
    for index in (0, 3, 11):
        directory = tmp_path / "batch_0000" / f"{index:04d}"
        directory.mkdir(parents=True)
        path = directory / "input.yaml"
        path.write_text(CASE)
        paths.append(path)

    job.Slurm().submit(tasks_for(paths), "run")

    ((argv, _),) = calls
    assert argv[1] == "--array=1-3"


def test_max_concurrent_throttles_the_array(batch, calls):
    job.Slurm(max_concurrent=2).submit(tasks_for(batch), "run")

    ((argv, _),) = calls
    assert argv[1] == "--array=1-3%2"


def test_one_target_submits_into_its_own_directory(tmp_path, calls):
    directory = tmp_path / "hiload"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(CASE)

    job.Slurm().submit(tasks_for([path]), "run")

    ((_, cwd),) = calls
    assert cwd == str(directory)


#
# TASK SPOOLER
#


def test_tsp_queues_one_job_per_task(batch, calls, monkeypatch):
    monkeypatch.setattr(job.shutil, "which", lambda name: f"/usr/bin/{name}")

    ids = job.Tsp(slots=2).submit(tasks_for(batch), "run", ("-v",))

    # The slot count first, then one job each.
    slots, *queued = calls
    assert slots[0] == ["/usr/bin/tsp", "-S", "2"]
    assert len(queued) == 3

    argv, cwd = queued[0]
    assert argv[:3] == ["/usr/bin/tsp", "-L", "0000"]
    assert argv[-3:] == ["run", str(batch[0].resolve()), "-v"]
    assert cwd == str(batch[0].parent)
    assert "17" in ids[0]


def test_tsp_asks_for_more_slots_when_a_job_needs_them(batch, calls, monkeypatch):
    monkeypatch.setattr(job.shutil, "which", lambda name: f"/usr/bin/{name}")

    job.Tsp(cpus=4).submit(tasks_for(batch[:1]), "run")

    _, (argv, _) = calls
    assert "-N" in argv
    assert argv[argv.index("-N") + 1] == "4"


def test_a_missing_tsp_names_the_package(batch, monkeypatch):
    monkeypatch.setattr(job.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="task-spooler"):
        job.Tsp().submit(tasks_for(batch), "run")


#
# THROUGH THE CLI
#


def test_queue_submits_and_runs_nothing(batch, calls):
    argv = ["run", *[str(path) for path in batch], "--queue"]

    assert cli.main(argv) == 0

    ((called, _),) = calls
    assert called[0].endswith("sbatch")

    # Nothing was solved, so nothing was written beside any of them.
    for path in batch:
        assert not (path.parent / cli.OUTPUT_NAME).exists()


def test_queue_without_a_job_section_says_so(tmp_path, capsys):
    path = tmp_path / "input.yaml"
    path.write_text(CASE.split("job:")[0])

    assert cli.main(["run", str(path), "--queue"]) == 1

    assert "job: section" in capsys.readouterr().err


def test_a_submitted_command_line_re_parses_to_the_same_config(batch):
    """The one way a queued run can silently differ from a local one.

    The options a submitted job carries are rebuilt from the parsed arguments,
    so this asserts the rebuild is faithful: parse it back, load it, and it is
    the config the parent would have run.
    """
    from turbigen2 import Config

    parser = cli._make_parser()
    original = parser.parse_args(
        ["run", str(batch[0]), "-s", "mean_line.psi=1.9", "--queue"]
    )

    options = cli.task_options(original)
    resubmitted = parser.parse_args(["run", str(batch[0]), *options])

    assert Config.from_dict(
        cli.load_config(batch[0], resubmitted).to_dict()
    ) == cli.load_config(batch[0], original)

    # And the queue flag is consumed rather than passed on, or the job would
    # submit itself again.
    assert "--queue" not in options
    assert "-Q" not in options


def test_a_submitted_invocation_carries_every_override(batch):
    """A value lives in the config, so `-s` carries it and nothing else must.

    `--max-iter` used to be a flag, and `task_options` a special case for it.
    Keeping values out of the argv is what stops that function growing a branch
    per verb -- which is the one place a submitted run can differ from a local
    one without anyone noticing.
    """
    parser = cli._make_parser()
    args = parser.parse_args(
        ["iterate", str(batch[0]), "-s", "max_iter=3", "-s", "mean_line.psi=1.9", "-Q"]
    )

    assert cli.task_options(args) == [
        "-s",
        "max_iter=3",
        "-s",
        "mean_line.psi=1.9",
    ]


#
# BATCH
#


BATCH_CASE = CASE + """
batch:
  seed: 0
  bounds:
    mean_line.psi: [1.4, 1.8]
"""


@pytest.fixture
def datum(tmp_path):
    """A datum config in a directory of its own, for its batches to sit in."""

    def make(iterating):
        directory = tmp_path / ("iter" if iterating else "plain")
        directory.mkdir()
        path = directory / "input.yaml"
        text = BATCH_CASE
        if iterating:
            text += "iterate:\n  - type: deviation\n"
        path.write_text(text)
        return path

    return make


def test_batch_writes_members_and_submits_them(datum, calls):
    path = datum(iterating=False)

    assert cli.main(["batch", str(path), "-n", "2", "--queue"]) == 0

    # Written first, so a submission that fails still leaves the batch.
    members = sorted((path.parent / "batch_0000").glob("*/input.yaml"))
    assert len(members) == 2

    ((argv, cwd),) = calls
    assert argv[0].endswith("sbatch")
    assert argv[1] == "--array=1-2"

    written = (path.parent / "batch_0000" / job.TASKS_NAME).read_text().splitlines()
    assert written == [str(member.resolve()) for member in members]


def test_a_batch_is_submitted_as_run_without_an_iterate_section(datum, calls):
    path = datum(iterating=False)

    cli.main(["batch", str(path), "-n", "2", "--queue"])

    assert " run " in (path.parent / "batch_0000" / job.SCRIPT_NAME).read_text()


def test_a_batch_is_submitted_as_iterate_when_the_datum_iterates(datum, calls):
    """Inferred from the section, and it matters which way it goes.

    A batch submitted as `run` builds an archive `database` reads back as
    empty: a sample must have converged *and* have its errors inside their
    tolerances, and closing that gap is what iterating is for.
    """
    path = datum(iterating=True)

    cli.main(["batch", str(path), "-n", "2", "--queue"])

    assert " iterate " in (path.parent / "batch_0000" / job.SCRIPT_NAME).read_text()


def test_a_submitted_batch_carries_no_options(datum, calls):
    """The members already hold the overrides, having been written with them.

    Unlike `run --queue`, where the submitted job re-reads the file the parent
    read and would otherwise lose them.
    """
    path = datum(iterating=False)

    cli.main(["batch", str(path), "-n", "2", "-s", "mean_line.mdot=17.0", "--queue"])

    script = (path.parent / "batch_0000" / job.SCRIPT_NAME).read_text()
    assert "-s" not in script

    member = next((path.parent / "batch_0000").glob("*/input.yaml"))
    assert "mdot: 17.0" in member.read_text()


def test_sampling_without_a_job_section_says_so(tmp_path, capsys):
    directory = tmp_path / "datum"
    directory.mkdir()
    path = directory / "input.yaml"
    # Everything but the job: section, which sits between the two halves.
    path.write_text(CASE.split("job:")[0] + BATCH_CASE.split(CASE)[1])

    assert cli.main(["batch", str(path), "-n", "2", "--queue"]) == 1

    assert "job: section" in capsys.readouterr().err

    # The batch is still there: writing happens before submitting, so a queue
    # that will not take it has not cost the designs.
    assert list((path.parent / "batch_0000").glob("*/input.yaml"))


#
# CLOBBERING
#


def test_several_targets_refuse_to_overwrite_an_answer(batch, capsys):
    """A batch is cluster hours whose loss is discovered a day later."""
    (batch[1].parent / cli.OUTPUT_NAME).write_text("{}\n")

    assert cli.main(["run", *[str(path) for path in batch]]) == 1

    assert "already been run" in capsys.readouterr().err


def test_force_overwrites_several_targets(batch, calls):
    (batch[1].parent / cli.OUTPUT_NAME).write_text("{}\n")

    assert cli.main(["run", *[str(p) for p in batch], "--queue", "--force"]) == 0


def test_one_target_overwrites_without_asking(batch, calls):
    """You named that directory by naming the file in it."""
    (batch[0].parent / cli.OUTPUT_NAME).write_text("{}\n")

    cli.check_clobber(cli._make_parser().parse_args(["run", str(batch[0])]), [batch[0]])

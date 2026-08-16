"""Tests for plugin discovery and the command line interface.

The work underneath the CLI is covered elsewhere, so these target what is
unique to the command line: that an ephemeral run really writes nothing, that
overrides reach the design, that a user's design is found without being told
where it is, and that a bad config reads as a message rather than a traceback.
"""

import textwrap

import pytest

from turbigen import cli, iterate, plugins

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
"""

PLUGIN_CASE = """
fluid:
  type: perfect
  cp: 1005.0
  gamma: 1.4
  mu: 1.8e-5
mean_line:
  type: {name}
  Ma: 0.6
"""


def plugin_source(name, class_name):
    """A minimal one-row design, as a user would write it in a scratch file."""
    return textwrap.dedent(f"""
        from typing import ClassVar
        from turbigen.design import MeanLineDesign

        class {class_name}(MeanLineDesign):
            type: ClassVar[str] = "{name}"
            n_row: ClassVar[int] = 1

            Ma: float
            P1: float = 1e5
            T1: float = 300.0

            def forward(self, fluid):
                ml = self.allocate(fluid)

                def build(Vx):
                    ml.set_r(0.5)
                    ml.set_Am(1.0)
                    ml.set_P_T(self.P1, self.T1)
                    ml.set_Vx(Vx)
                    ml.set_Vr(0.0)
                    ml.set_Vt(0.0)

                self.solve_for(
                    ml, build, unknowns={{"Vx": 50.0}}, targets={{"Ma": self.Ma}}
                )
                return ml

            def backward(self, ml):
                return {{"Ma": ml.outlet.Ma, "P1": ml.inlet.P, "T1": ml.inlet.T}}
    """)


@pytest.fixture
def case(tmp_path):
    """A config file in a directory of its own, one directory being one run."""
    directory = tmp_path / "case"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(CASE)
    return path


@pytest.fixture
def clean_registry():
    """Restore the design registry after a test loads a plugin into it."""
    from turbigen.design import MeanLineDesign
    from turbigen.node import _REGISTRY

    before = dict(_REGISTRY.get(MeanLineDesign, {}))
    yield
    _REGISTRY[MeanLineDesign].clear()
    _REGISTRY[MeanLineDesign].update(before)


#
# DISCOVERY
#


def test_finds_plugin_dir_beside_the_config(tmp_path):
    (tmp_path / plugins.PLUGIN_DIR_NAME).mkdir()

    assert plugins.find_plugin_dir(tmp_path) == tmp_path / plugins.PLUGIN_DIR_NAME


def test_walks_up_to_find_plugin_dir(tmp_path):
    """Discovery searches ancestors, so one directory can serve many cases."""
    wanted = tmp_path / plugins.PLUGIN_DIR_NAME
    wanted.mkdir()
    deep = tmp_path / "cases" / "family" / "one"
    deep.mkdir(parents=True)

    assert plugins.find_plugin_dir(deep) == wanted


def test_nearest_plugin_dir_wins(tmp_path):
    (tmp_path / plugins.PLUGIN_DIR_NAME).mkdir()
    near = tmp_path / "cases"
    near.mkdir()
    nearest = near / plugins.PLUGIN_DIR_NAME
    nearest.mkdir()

    assert plugins.find_plugin_dir(near) == nearest


def test_no_plugin_dir_is_not_an_error(tmp_path):
    assert plugins.find_plugin_dir(tmp_path) is None
    assert plugins.discover(tmp_path) is None


def test_load_plugins_skips_private_and_hidden_files(tmp_path, clean_registry):
    from turbigen.design import MeanLineDesign

    plug_dir = tmp_path / plugins.PLUGIN_DIR_NAME
    (plug_dir / ".hidden").mkdir(parents=True)
    (plug_dir / "good.py").write_text(plugin_source("_t_good", "Good"))
    (plug_dir / "_private.py").write_text(plugin_source("_t_private", "Private"))
    (plug_dir / ".hidden" / "nested.py").write_text(
        plugin_source("_t_hidden", "Hidden")
    )

    plugins.load_plugins(plug_dir)

    registered = MeanLineDesign.options()
    assert "_t_good" in registered
    assert "_t_private" not in registered
    assert "_t_hidden" not in registered


def test_load_plugins_reports_a_broken_plugin(tmp_path):
    plug_dir = tmp_path / plugins.PLUGIN_DIR_NAME
    plug_dir.mkdir()
    (plug_dir / "broken.py").write_text("this is not python\n")

    with pytest.raises(RuntimeError, match="Failed to import plugin"):
        plugins.load_plugins(plug_dir)


#
# THE design VERB
#


def test_design_writes_nothing_at_all(case, capsys):
    before = set(case.parent.iterdir())

    assert cli.main(["design", str(case)]) == 0

    assert set(case.parent.iterdir()) == before
    assert "Mean line:" in capsys.readouterr().err


def test_report_of_a_mean_line_design_is_not_an_error(case):
    """Every standard plot needs geometry this case does not have.

    Each processor degrades to no figures when what it needs is missing, which
    is what lets one verb cover every depth of case without a mode to select --
    and an empty document is not a report, so no file appears.
    """
    assert cli.main(["report", str(case)]) == 0

    assert (case.parent / cli.LOG_NAME).is_file()
    assert not (case.parent / "post.pdf").exists()

    # The answer belongs to whoever has one: a report has nothing to put under
    # `result:`, so it does not write a config at all.
    assert not (case.parent / cli.OUTPUT_NAME).exists()


def test_a_batch_is_written_beside_its_datum(batch_case):
    """No directory to name: a batch goes where the design it came from is.

    Which also makes the layout record the provenance that nothing else does --
    `--continue` cannot tell that the datum or its bounds changed between
    batches, and no file at the batch root says what generated it.
    """
    cli.main(["batch", str(batch_case), "-n", "2"])
    cli.main(["batch", str(batch_case), "-n", "2"])

    assert (batch_case.parent / "batch_0000").is_dir()
    assert (batch_case.parent / "batch_0001").is_dir()


def test_batch_numbering_carries_on_past_a_deleted_batch(tmp_path):
    """Counted from the highest that exists, not from how many there are."""
    (tmp_path / "batch_0000").mkdir()
    (tmp_path / "batch_0007").mkdir()

    assert cli.next_batch_dir(tmp_path) == tmp_path / "batch_0008"


def test_batch_numbering_ignores_what_is_not_a_batch(tmp_path):
    (tmp_path / "batch_0000").mkdir()
    (tmp_path / "batch_notes").mkdir()
    (tmp_path / "batch_0001.txt").write_text("")

    assert cli.next_batch_dir(tmp_path) == tmp_path / "batch_0001"


def test_the_first_batch_is_zero(tmp_path):
    assert cli.next_batch_dir(tmp_path) == tmp_path / "batch_0000"


def test_set_override_reaches_the_design(case, capsys):
    cli.main(["design", str(case)])
    baseline = capsys.readouterr().err

    cli.main(["design", str(case), "-s", "mean_line.psi=1.2"])
    changed = capsys.readouterr().err

    assert baseline != changed


def test_mistyped_override_key_is_rejected(case, capsys):
    assert cli.main(["design", str(case), "-s", "mean_line.psii=1.2"]) == 1

    assert "psii" in capsys.readouterr().err


def test_malformed_override_is_rejected(case, capsys):
    assert cli.main(["design", str(case), "-s", "no_equals_sign"]) == 1

    assert "KEY=VALUE" in capsys.readouterr().err


def test_nothing_but_batch_writes_to_stdout(case, capsys):
    """One stream for everything a run says, and it is stderr.

    stdout is reserved for the one machine-readable thing there is: the batch
    directory `batch` cannot name in advance.
    """
    assert cli.main(["design", str(case)]) == 0

    assert capsys.readouterr().out == ""


def test_missing_config_file_is_a_message_not_a_traceback(tmp_path, capsys):
    assert cli.main(["design", str(tmp_path / "nope.yaml")]) == 1

    captured = capsys.readouterr()
    assert "FileNotFoundError" in captured.err
    assert "Traceback" not in captured.err


def test_verbose_shows_the_traceback(tmp_path, capsys):
    assert cli.main(["design", str(tmp_path / "nope.yaml"), "-v"]) == 1

    assert "Traceback" in capsys.readouterr().err


@pytest.mark.parametrize("verb", ["design", "report", "run", "iterate", "batch"])
def test_every_verb_can_print_its_help(verb, capsys):
    """argparse percent-formats help strings, so a bare `%` in one raises.

    The placeholder is a `%`, and describing it in the `--out` help without
    doubling it made `batch --help` fail with a ValueError from inside
    argparse rather than print anything.
    """
    with pytest.raises(SystemExit) as excinfo:
        cli.main([verb, "--help"])

    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_unknown_verb_exits_two(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["frobnicate", "case.yaml"])

    assert excinfo.value.code == 2


#
# DISCOVERY THROUGH THE CLI
#


def test_design_finds_a_plugin_without_being_told(tmp_path, clean_registry, capsys):
    """The point of discovery: a user's design works with no flag and no key."""
    plug_dir = tmp_path / plugins.PLUGIN_DIR_NAME
    plug_dir.mkdir()
    (plug_dir / "mine.py").write_text(plugin_source("_t_cli", "CliStage"))

    cases = tmp_path / "cases"
    cases.mkdir()
    case = cases / "stage.yaml"
    case.write_text(PLUGIN_CASE.format(name="_t_cli"))

    assert cli.main(["design", str(case)]) == 0
    assert "Mean line:" in capsys.readouterr().err


def test_written_config_re_runs_from_its_output_directory(tmp_path, clean_registry):
    """An archived config still finds the plugins the original run used.

    This is why no plugin path is recorded anywhere: the output sits beside the
    case, so walking up from it reaches the same place.
    """
    plug_dir = tmp_path / plugins.PLUGIN_DIR_NAME
    plug_dir.mkdir()
    (plug_dir / "mine.py").write_text(plugin_source("_t_rerun", "RerunStage"))

    directory = tmp_path / "stage"
    directory.mkdir()
    case = directory / "input.yaml"
    case.write_text(PLUGIN_CASE.format(name="_t_rerun"))

    assert cli.main(["design", str(case)]) == 0

    # As a run would leave it: the resolved config, in the case's directory.
    from turbigen import Config  # noqa: PLC0415

    archived = directory / cli.OUTPUT_NAME
    Config.from_file(case).to_file(archived)

    assert cli.main(["design", str(archived)]) == 0


def test_run_without_a_mesh_section_is_a_message_not_a_traceback(run_case, capsys):
    """`Config` has no make_grid to hold this check, so the verb holds it.

    A missing `mesh:` section is a fact about the command the user typed, not
    about the config in the abstract -- `design` and `report` are both happy
    with the same file. Without the check `config.mesh.mesh(...)` would raise
    AttributeError on None, which is the unhelpful failure the strict config
    validation exists to avoid.
    """
    assert cli.main(["run", str(run_case), "-s", "mesh=null"]) == 1

    assert "mesh: section" in capsys.readouterr().err


#
# THE RUN VERB
#
# An integration test in the literal sense: the only one that takes a config
# file all the way to a solved grid, through design, meshing, boundary
# conditions, the initial guess and the solver. Everything below it is covered
# in isolation elsewhere, so what these check is that the stages compose.
#

RUN_CASE = """
fluid: {type: perfect, cp: 1005.0, gamma: 1.4, mu: 1.8e-5}
mean_line:
  type: turbine_cascade
  span: [0.01, 0.011]
  Alpha: [40.0, -65.0]
  Ma2: 0.6
  Ys: 0.029
  htr: 0.99
annulus:
  type: fixed_axial_chord
  cx_row: [0.00525]
  cx_gap: [0.0105, 0.0105]
blades:
  - count: {type: Co, Co: 0.7}
    sections:
      - spf: 0.5
        dchi_LE: 10.0
        dchi_TE: -2.0
        camber: {type: quadratic}
        thickness:
          type: taylor
          R_LE: 0.05
          t_max: 0.12
          m_tmax: 0.3
          t_TE: 0.03
          tanwedge: 0.18
mesh:
  type: h
  resolution_factor: 0.25
  dm_TE: 0.0
  AR_cusp: 2.0
  ni_cusp: 5
solver: {type: ember, n_step: 10, n_step_log: 10, n_stage: 4}
"""
"""A single stationary row, adapted from examples/turbine_cascade.yaml.

One row means no mixing planes, and the coarsest mesh that still passes the
multigrid divisibility check, so the whole thing runs in about a second.
"""


@pytest.fixture
def run_case(tmp_path):
    directory = tmp_path / "cascade"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(RUN_CASE)
    return path


def test_run_solves_a_case_end_to_end(run_case, capsys):
    assert cli.main(["run", str(run_case)]) == 0

    printed = capsys.readouterr().err
    assert "Mean line:" in printed
    assert "Mesh:" in printed
    assert "Solver: converged" in printed
    # The answer to the question the config asked, and the last thing said.
    assert "Design variables:" in printed


def test_a_run_never_overwrites_its_input(run_case):
    """The whole reason the written name is not a name we were given.

    What a run writes is the resolved config, every default expanded, so
    writing it over a hand-kept file would lose the file's comments to the safe
    loader. Guaranteed by the naming rather than by a check.
    """
    before = run_case.read_text()

    assert cli.main(["run", str(run_case)]) == 0

    assert run_case.read_text() == before


def test_run_writes_a_config_that_reads_back(run_case):
    """The archived config is the run, defaults and all."""
    from turbigen import Config  # noqa: PLC0415

    cli.main(["run", str(run_case)])

    written = run_case.parent / cli.OUTPUT_NAME
    assert written.exists()
    assert (run_case.parent / cli.LOG_NAME).exists()
    assert Config.from_file(written) == Config.from_file(run_case)


# The march is driven unstable on purpose, so ember's warning that the outlet
# has gone supersonic is the expected behaviour rather than a problem. Without
# this the suite's `filterwarnings = error` turns it into an exception, and the
# verb reports a config error (1) instead of a failed solve (2).
@pytest.mark.filterwarnings("ignore::ember.nonreflecting.UnsupportedMeanStateWarning")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_run_reports_a_failed_solve_in_its_exit_code(run_case):
    """Exit 2, and the output is still written.

    A diverged run is exactly the one whose output someone needs to look at, so
    failing must not also throw away the evidence. A distinct code from 1 keeps
    "the solver did not converge" apart from "the config was wrong", which a
    script driving a sweep has to tell apart without parsing the log.
    """
    # A CFL far past the stability limit, so it diverges within a few steps.
    code = cli.main(["run", str(run_case), "-s", "solver.cfl=50.0"])

    assert code == 2
    assert (run_case.parent / cli.OUTPUT_NAME).exists()


def test_run_without_a_solver_section_is_a_message(run_case, capsys):
    text = run_case.read_text()
    trimmed = "\n".join(
        line for line in text.splitlines() if not line.startswith("solver:")
    )
    run_case.write_text(trimmed)

    assert cli.main(["run", str(run_case)]) == 1

    assert "solver: section" in capsys.readouterr().err


def test_run_writes_its_answer_beside_the_config(run_case):
    """The point of the whole arrangement: one file, loaded once.

    A run's mixed-out mean line goes into the same file under `result:`, so
    comparing what was achieved against what was asked for needs no second
    artefact and no repeat of the CFD.
    """
    from turbigen import case  # noqa: PLC0415

    assert cli.main(["run", str(run_case)]) == 0

    config, result = case.read(run_case.parent / cli.OUTPUT_NAME)

    assert result.converged is True
    assert result.actual is not None
    assert result.actual.shape == result.nominal.shape

    # The achieved design variables come back in the design's own vocabulary,
    # recomputed from the stored state rather than stored themselves.
    achieved = config.mean_line.backward(result.actual)
    assert 0.0 < float(achieved["Ma2"]) < 1.0


def test_report_picks_up_the_field_a_run_left(run_case):
    """Re-plotting needs no more than the config, and no flag at all.

    The grid is not serialised, so a report re-designs and re-meshes to put the
    stored field back -- seconds against the minutes of the march it stands in
    for, which is why serialising it would not be worth the trouble.
    """
    out = run_case.parent
    assert cli.main(["run", str(run_case)]) == 0

    before = (out / "post.pdf").read_bytes()
    (out / "post.pdf").unlink()

    assert cli.main(["report", str(out / cli.OUTPUT_NAME)]) == 0

    # The same pages as the run drew, from the field it left behind.
    assert (out / "post.pdf").is_file()
    assert len((out / "post.pdf").read_bytes()) == pytest.approx(len(before), rel=0.05)


#
# THE ITERATE VERB
#


ITERATE_CASE = RUN_CASE + """
iterate:
  - type: deviation
  - type: incidence
"""


@pytest.fixture
def iterate_case(tmp_path):
    directory = tmp_path / "iterate"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(ITERATE_CASE)
    return path


def test_every_run_records_what_the_iterators_measured(iterate_case):
    """Iterating or not: these are observations of the flow, and only a solved
    grid holds them."""
    from turbigen import case  # noqa: PLC0415

    assert cli.main(["run", str(iterate_case)]) == 0

    _, result = case.read(iterate_case.parent / cli.OUTPUT_NAME, design=False)

    assert set(result.error) == {"dchi_TE[0]", "dchi_LE[0]"}
    assert all(isinstance(value, float) for value in result.error.values())


def test_iterate_keeps_every_iteration(iterate_case):
    """The directories are the archive, so none of them is deleted."""
    out = iterate_case.parent

    cli.main(["iterate", str(iterate_case), "-s", "max_iter=2"])

    for i_iter in range(2):
        iter_dir = out / f"iter_{i_iter:04d}"
        assert (iter_dir / cli.OUTPUT_NAME).is_file()
        assert (iter_dir / "restart.npz").is_file()
        assert (iter_dir / "post.pdf").is_file()

    assert (out / "final").is_symlink()
    assert (out / "final").resolve() == (out / "iter_0001").resolve()


def test_iterate_links_its_answer_where_a_run_would_have_put_it(iterate_case):
    """`output.yaml` means what this run achieved, whichever verb produced it."""
    from turbigen import case  # noqa: PLC0415

    cli.main(["iterate", str(iterate_case), "-s", "max_iter=1"])

    answer = iterate_case.parent / cli.OUTPUT_NAME
    assert answer.is_symlink()

    _, result = case.read(answer, design=False)
    assert result is not None


def test_iterate_moves_the_design_and_records_why(iterate_case):
    """Each iteration archives the config it ran and the error it measured.

    Together those are one sample of "this design gave that mismatch", which is
    what any later fit over an archive of runs would be built from.
    """
    from turbigen import case  # noqa: PLC0415

    out = iterate_case.parent
    cli.main(["iterate", str(iterate_case), "-s", "max_iter=2"])

    first, first_result = case.read(out / "iter_0000" / cli.OUTPUT_NAME, design=False)
    second, _ = case.read(out / "iter_0001" / cli.OUTPUT_NAME, design=False)

    # The case recambers its leading edge 10 degrees off the flow, so the
    # incidence is measured well below the target and the knob has to come down
    # to meet it.
    error = first_result.error["dchi_LE[0]"]
    assert error < -1.0

    # By the rule the stepper states, from the error this run recorded: the
    # first iteration has no history to improve on the declared gain.
    gain = iterate.Incidence().gain
    before = first.blades[0].sections[0].dchi_LE
    after = second.blades[0].sections[0].dchi_LE
    assert after == pytest.approx(before - gain * error)


def test_iterate_starts_from_the_database(iterate_case, tmp_path):
    """A design near one already solved starts where that one finished.

    The archive is fabricated rather than solved: what is under test is that
    the verb reads the key, anchors the glob on the config file and applies the
    prediction before the first run, none of which needs a march to prove. One
    sample, so every query sits on top of it and the answer is exact.
    """
    from turbigen import Config, Result, case  # noqa: PLC0415

    archived = Config.from_file(iterate_case)
    archived = iterate.Deviation().with_unknowns(archived, {"dchi_TE[0]": 7.5})
    archived = iterate.Incidence().with_unknowns(archived, {"dchi_LE[0]": -3.25})

    directory = tmp_path / "archive" / "000"
    directory.mkdir(parents=True)
    case.write(
        directory / cli.OUTPUT_NAME,
        archived,
        Result(converged=True, error={"dchi_TE[0]": 0.0, "dchi_LE[0]": 0.0}),
    )

    out = tmp_path / "with_database"
    out.mkdir()
    with_database = out / "input.yaml"
    with_database.write_text(
        ITERATE_CASE + f'\ndatabase:\n  path: "../archive/*/{cli.OUTPUT_NAME}"\n'
    )

    cli.main(["iterate", str(with_database), "-s", "max_iter=1"])

    started, _ = case.read(out / "iter_0000" / cli.OUTPUT_NAME, design=False)

    assert iterate.unknowns(started)["dchi_TE[0]"] == pytest.approx(7.5)
    assert iterate.unknowns(started)["dchi_LE[0]"] == pytest.approx(-3.25)


#
# THE BATCH VERB
#


BATCH_CASE = CASE + """
batch:
  seed: 0
  bounds:
    mean_line.psi: [1.4, 1.8]
"""


@pytest.fixture
def batch_case(tmp_path):
    """A datum config, in the directory its batches will be written into."""
    directory = tmp_path / "datum"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(BATCH_CASE)
    return path


def test_batch_writes_members_named_by_sequence_index(batch_case):
    assert cli.main(["batch", str(batch_case), "-n", "4"]) == 0

    batch = batch_case.parent / "batch_0000"

    # A directory each, because one directory is one run: it is what gives
    # every member an output.yaml of its own to be run into.
    assert sorted(p.name for p in batch.glob("*/")) == [
        "0000",
        "0001",
        "0002",
        "0003",
    ]
    assert (batch / "0000" / "input.yaml").is_file()


def test_batch_prints_the_batch_directory(batch_case, capsys):
    """The one thing on stdout: a numbered batch cannot be named in advance."""
    cli.main(["batch", str(batch_case), "-n", "2"])

    assert capsys.readouterr().out.strip() == str(batch_case.parent / "batch_0000")


def test_batch_members_are_runnable_designs(batch_case):
    from turbigen import Config  # noqa: PLC0415

    cli.main(["batch", str(batch_case), "-n", "2"])

    for member in (batch_case.parent / "batch_0000").glob("*/input.yaml"):
        config = Config.from_file(member)
        assert config.batch is None
        assert 1.4 <= config.mean_line.psi <= 1.8
        config.design()


def test_batch_continue_carries_on_into_a_new_batch(batch_case):
    """Nothing is written into an existing batch, so nothing can be lost."""
    cli.main(["batch", str(batch_case), "-n", "2"])
    cli.main(["batch", str(batch_case), "-n", "2", "--continue"])

    datum = batch_case.parent
    assert sorted(p.name for p in (datum / "batch_0000").glob("*/")) == [
        "0000",
        "0001",
    ]
    assert sorted(p.name for p in (datum / "batch_0001").glob("*/")) == [
        "0002",
        "0003",
    ]


def test_batch_continue_is_the_tail_of_one_batch(tmp_path):
    """Two batches of two hold what one batch of four would have."""
    from turbigen import Config  # noqa: PLC0415

    def datum(name):
        directory = tmp_path / name
        directory.mkdir()
        path = directory / "input.yaml"
        path.write_text(BATCH_CASE)
        return path

    split = datum("split")
    cli.main(["batch", str(split), "-n", "2"])
    cli.main(["batch", str(split), "-n", "2", "--continue"])

    whole = datum("whole")
    cli.main(["batch", str(whole), "-n", "4"])

    def psi(root):
        return [
            Config.from_file(p).mean_line.psi
            for p in sorted(
                root.glob("batch_*/*/input.yaml"), key=lambda p: p.parent.name
            )
        ]

    assert psi(split.parent) == psi(whole.parent)


def test_batch_without_a_batch_section_says_so(case, capsys):
    assert cli.main(["batch", str(case)]) == 1

    assert "needs a batch: section" in capsys.readouterr().err


def test_batch_takes_one_datum(batch_case, capsys):
    """One config describes one space, so a second would be a second space."""
    assert cli.main(["batch", str(batch_case), str(batch_case)]) == 1

    assert "one config file as its datum" in capsys.readouterr().err


#
# THE GRID
#


GRID_CASE = CASE + """
batch:
  values:
    mean_line.psi: [1.4, 1.6, 1.8]
"""


@pytest.fixture
def grid_case(tmp_path):
    """A datum naming its points outright, rather than a box to fill."""
    directory = tmp_path / "datum"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(GRID_CASE)
    return path


def test_a_grid_writes_one_member_per_point(grid_case):
    """No -n: the count is the product of what the section names."""
    assert cli.main(["batch", str(grid_case)]) == 0

    members = sorted(p.name for p in (grid_case.parent / "batch_0000").glob("*/"))
    assert members == ["0000", "0001", "0002"]


def test_a_grid_member_carries_its_own_point(grid_case):
    """The value is in the file, so the study is not only in shell history."""
    from turbigen import Config  # noqa: PLC0415

    cli.main(["batch", str(grid_case)])

    psi = [
        Config.from_file(member).mean_line.psi
        for member in sorted((grid_case.parent / "batch_0000").glob("*/input.yaml"))
    ]

    assert psi == [1.4, 1.6, 1.8]


def test_a_grid_refuses_a_number(grid_case, capsys):
    """The count is the product, so there is nothing for -n to choose."""
    assert cli.main(["batch", str(grid_case), "-n", "8"]) == 1

    assert "no -n to choose" in capsys.readouterr().err
    # Refused before the batch is opened, so no number is burned.
    assert not list(grid_case.parent.glob("batch_*"))


def test_a_grid_refuses_to_continue(grid_case, capsys):
    """A finite product has no tail to carry on from."""
    assert cli.main(["batch", str(grid_case), "--continue"]) == 1

    assert "nothing to --continue" in capsys.readouterr().err
    assert not list(grid_case.parent.glob("batch_*"))


def test_a_grid_is_reachable_from_the_command_line(batch_case):
    """The whole mapping is replaced, its keys having dots of their own."""
    assert (
        cli.main(
            [
                "batch",
                str(batch_case),
                "-s",
                "batch.bounds={}",
                "-s",
                "batch.values={mean_line.psi: [1.4, 1.8]}",
            ]
        )
        == 0
    )

    members = sorted(p.name for p in (batch_case.parent / "batch_0000").glob("*/"))
    assert members == ["0000", "0001"]


def test_set_takes_a_bracketed_path(case):
    """One spelling of a path, whether it comes from a file or the shell."""
    args = cli._make_parser().parse_args(
        ["design", str(case), "-s", "mean_line.Ys[1]=0.07"]
    )

    assert cli.load_config(case, args).mean_line.Ys[1] == pytest.approx(0.07)


def test_iterate_without_iterators_says_to_use_run(run_case, capsys):
    assert cli.main(["iterate", str(run_case)]) == 1

    assert "use 'run'" in capsys.readouterr().err


def test_report_on_an_unsolved_case_is_not_an_error(run_case):
    """No field beside it yet, so the geometry pages and nothing more."""
    assert cli.main(["report", str(run_case)]) == 0

    assert (run_case.parent / "post.pdf").is_file()
    assert not (run_case.parent / cli.RESTART_NAME).exists()


def test_reporting_in_place_keeps_the_recorded_answer(run_case):
    """A verb that computed no answer cannot erase the one already there.

    True by construction rather than by a guard: only the verbs that solve
    write `output.yaml`, so a report has no way to replace a converged run's
    `result:` with an empty one. An earlier arrangement had the report verb
    writing back over the config it had just read, and needed the two compared
    to notice.
    """
    from turbigen import case  # noqa: PLC0415

    out = run_case.parent
    assert cli.main(["run", str(run_case)]) == 0
    _, before = case.read(out / cli.OUTPUT_NAME, design=False)

    # Even with an override, which makes the config differ from the archived
    # one and so defeats any comparison-based guard.
    cli.main(["report", str(out / cli.OUTPUT_NAME), "-s", "mesh.yplus=25.0"])

    _, after = case.read(out / cli.OUTPUT_NAME, design=False)
    assert after.converged is before.converged is True
    assert after.actual is not None


def test_run_writes_its_convergence_history(run_case):
    """Beside the field, so a re-plot can draw the page the run had."""
    assert cli.main(["run", str(run_case)]) == 0

    assert (run_case.parent / cli.HISTORY_NAME).is_file()


def test_a_replot_recovers_the_convergence_history(run_case):
    cli.main(["run", str(run_case)])

    history = cli.read_history(run_case.parent / cli.HISTORY_NAME)

    assert history is not None
    assert history.i_log >= 0
    # The residuals are what the convergence plot needs and what the JSON
    # export ember offers instead would have dropped.
    assert float(history.residual.min()) > 0.0


def test_a_history_that_will_not_load_is_not_fatal(tmp_path):
    """A re-plot minus its convergence page beats a re-plot that refuses.

    CNV is a pickle, so a file written by another version may not come back.
    """
    path = tmp_path / cli.HISTORY_NAME
    path.write_text("not a pickle")

    assert cli.read_history(path) is None


def test_no_history_beside_a_restart_is_not_fatal(tmp_path):
    assert cli.read_history(tmp_path / cli.HISTORY_NAME) is None


def test_bare_restart_says_when_there_is_nothing_to_read(run_case, capsys):
    """`run --restart` still names a field explicitly, so it can still miss."""
    assert cli.main(["run", str(run_case), "--restart"]) == 1

    assert "No restart.npz" in capsys.readouterr().err


def test_a_failing_plot_cannot_lose_the_solution(run_case, tmp_path, monkeypatch):
    """The flow field is written before the report, not after.

    Post-processing raises by design, and the standard plots run whether or not
    a config asks for them -- so with the two the other way round, a plot that
    fell over would throw away a march that had already been paid for.
    """
    from turbigen import SectionsPlot  # noqa: PLC0415

    def boom(self, config, result):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(SectionsPlot, "report", boom)

    assert cli.main(["run", str(run_case)]) == 1

    assert (run_case.parent / "restart.npz").is_file()


#
# THE DESIGN VARIABLE TABLE
#
# What a run says at the end: the intent the config stated, beside what the CFD
# achieved, in the same units. Both columns come from one `backward()`, so a
# difference is always the flow differing rather than the two sides being
# computed differently.
#

TURBINE = {
    "fluid": {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5},
    "mean_line": {
        "type": "axial_turbine",
        "psi": 1.6,
        "phi2": 0.8,
        "Ma2": 0.9,
        "fac_Ma3_rel": 0.8,
        "mdot": 10.0,
        "Ys": [0.05, 0.05],
        "r_rms": 0.3,
    },
}


def _table(nominal_config, actual_config=None):
    """Format a table comparing one design against another's mean line."""
    from turbigen import Config, Result  # noqa: PLC0415

    config = Config.from_dict(nominal_config)
    machine = config.design()
    other = (
        machine if actual_config is None else Config.from_dict(actual_config).design()
    )
    return cli.design_variable_string(
        config, Result(machine=machine, actual=other.mean_line)
    )


def test_design_variables_are_zero_against_their_own_design():
    """The datum. A machine compared with itself has no error anywhere, which
    is what says the two columns are measured through the same definitions."""
    lines = _table(TURBINE).splitlines()[3:]

    for line in lines:
        if line.startswith("-"):
            continue
        assert float(line.split()[3]) == 0.0, line


def test_design_variables_are_separated_from_diagnostics():
    """A variable you set and a number you read are different kinds of thing.

    The split is field membership, not the order `backward` returns its keys
    in, which is only the author's convention.
    """
    lines = _table(TURBINE).splitlines()
    rule = [i for i, line in enumerate(lines) if line.startswith("-")][-1]

    names = [line.split()[0] for line in lines[3:] if not line.startswith("-")]
    above = [line.split()[0] for line in lines[3:rule]]

    assert "psi" in above
    assert "Ys[0]" in above
    # Reaction and efficiency are read off the answer, not asked for.
    assert "eta_tt" in names and "eta_tt" not in above
    assert "Lam" in names and "Lam" not in above


def test_design_variables_expand_element_by_element():
    """A per-row loss coefficient is two numbers, and a mismatch is usually in
    one of them."""
    names = [line.split()[0] for line in _table(TURBINE).splitlines()[3:]]

    assert "Ys[0]" in names and "Ys[1]" in names
    assert "Ys" not in names


def test_a_solution_that_differs_shows_the_error():
    """Errors are nominal - actual, the sign the mean_line iterator already
    uses, so a row here and a row of the iteration table read the same way."""
    hotter = {**TURBINE, "mean_line": {**TURBINE["mean_line"], "psi": 1.8}}

    row = next(
        line
        for line in _table(TURBINE, hotter).splitlines()
        if line.split()[0] == "psi"
    )

    _, nominal, actual, error, relative = row.split()
    assert float(nominal) == pytest.approx(1.6)
    assert float(actual) == pytest.approx(1.8)
    assert float(error) == pytest.approx(-0.2, abs=1e-3)
    assert float(relative) == pytest.approx(-12.5, abs=0.1)


def test_a_zero_nominal_has_no_relative_error():
    """Angles and efficiencies are routinely zero by design, so this is the
    common case rather than a guard against the impossible."""
    cascade = {
        "fluid": {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5},
        "mean_line": {
            "type": "turbine_cascade",
            "span": [0.01, 0.011],
            "Alpha": [40.0, -65.0],
            "Ma2": 0.6,
            "Ys": 0.029,
            "htr": 0.99,
        },
    }

    zeroed = [
        line
        for line in _table(cascade).splitlines()[3:]
        if not line.startswith("-") and float(line.split()[1]) == 0.0
    ]

    assert zeroed, "expected at least one design variable with a zero nominal"
    for line in zeroed:
        assert line.split()[-1] == "--", line


def test_a_failed_comparison_does_not_cost_the_run_its_output(run_case, monkeypatch):
    """A table is a report of a solution the CFD has already been paid for."""

    def boom(config, result):
        raise RuntimeError("table exploded")

    monkeypatch.setattr(cli, "design_variable_string", boom)

    assert cli.main(["run", str(run_case)]) == 0

    assert (run_case.parent / "restart.npz").is_file()
    assert (run_case.parent / "output.yaml").is_file()


#
# INCLUDES THROUGH THE CLI
#
# The merging rules are covered in test_include.py. These check the two things
# only the CLI can show: that an assembled document really designs, and that an
# override still has the last word over one.
#


@pytest.fixture
def split_case(tmp_path):
    """The same case as `case`, but with its mean line in a second file."""
    directory = tmp_path / "split"
    directory.mkdir()

    fluid, _, mean_line = CASE.partition("mean_line:")
    (directory / "mean_line.yaml").write_text(f"mean_line:{mean_line}")

    path = directory / "input.yaml"
    path.write_text(f"include: [mean_line.yaml]\n{fluid}")
    return path


def _tables_from_log(text):
    """The design tables out of a run's log, without its timing line."""
    return text.split("Mean line:")[1].split("Total time")[0]


def test_a_config_assembled_from_includes_designs(split_case, case, capsys):
    """Splitting a file across two must not change what it designs."""
    assert cli.main(["design", str(split_case)]) == 0
    split = capsys.readouterr().err

    assert cli.main(["design", str(case)]) == 0
    whole = capsys.readouterr().err

    assert "Mean line:" in split
    assert _tables_from_log(split) == _tables_from_log(whole)


def test_an_override_beats_an_included_value(split_case, capsys):
    """Includes are resolved before `-s`, so an override applies to the
    assembled document rather than to whichever fragment defined the key."""
    assert cli.main(["design", str(split_case)]) == 0
    baseline = capsys.readouterr().err

    assert cli.main(["design", str(split_case), "-s", "mean_line.psi=1.2"]) == 0
    changed = capsys.readouterr().err

    assert baseline != changed


def test_an_include_key_does_not_reach_the_config(split_case):
    """Popped during resolution, so the strict unknown-key check needs no
    exception for it and a written config carries no pointer to a file that
    may since have changed."""
    from turbigen import Config  # noqa: PLC0415

    config = Config.from_file(split_case)

    assert "include" not in config.to_dict()
    assert config == Config.from_dict(config.to_dict())


def test_a_run_writes_its_includes_out_expanded(run_case):
    """An archived run records what it ran, not where the pieces came from."""
    directory = run_case.parent

    fluid, _, rest = RUN_CASE.partition("mean_line:")
    (directory / "fluid.yaml").write_text(fluid)
    run_case.write_text(f"include: [fluid.yaml]\nmean_line:{rest}")

    assert cli.main(["run", str(run_case)]) == 0

    written = (directory / "output.yaml").read_text()
    assert "include" not in written
    # The included value is there in full, not as a pointer to a file that may
    # since have changed.
    assert "cp: 1005" in written


#
# THE chic VERB
#
# The characteristic sweep. What is worth checking here rather than in
# test_chic.py is the composition: that a restart resolves after an iterate,
# that a converged design is not re-iterated, and that the points land where
# the layout says they do.
#

SWEEP = """
chic:
  step: 0.4
  step_min: 0.2
  max_points: 3
"""
"""Deliberately coarse, so a sweep is three short runs rather than ten.

A cascade does not stall, so the steps are large enough to push the exit
pressure somewhere the solver will refuse.
"""

CHIC_CASE = RUN_CASE + SWEEP
"""No iterators, so the design phase is a single solve of the design point."""

CHIC_ITERATE_CASE = (
    ITERATE_CASE.replace(
        "  - type: deviation\n", "  - type: deviation\n    tolerance: 20.0\n"
    ).replace("  - type: incidence\n", "  - type: incidence\n    tolerance: 20.0\n")
    + SWEEP
)
"""And with iterators, loose enough that one pass settles the design.

The tolerances are absurd on purpose: what this checks is that the two phases
compose, and paying for a genuine convergence would be minutes of CFD to learn
nothing more.
"""


@pytest.fixture
def chic_case(tmp_path):
    directory = tmp_path / "chic"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(CHIC_CASE)
    return path


@pytest.fixture
def chic_iterate_case(tmp_path):
    directory = tmp_path / "chic_iterate"
    directory.mkdir()
    path = directory / "input.yaml"
    path.write_text(CHIC_ITERATE_CASE)
    return path


def test_iterate_leaves_a_field_where_restart_looks_for_it(iterate_case):
    """`run` left one beside the config and `iterate` did not, so `iterate`
    followed by `--restart` failed on a case that plainly had a field."""
    cli.main(["iterate", str(iterate_case), "-s", "max_iter=1"])

    field = iterate_case.parent / cli.RESTART_NAME
    assert field.is_symlink()
    assert field.resolve().is_file()

    # Which is exactly what the flag resolves, so the two agree by test rather
    # than by inspection.
    args = type("A", (), {"restart": True})()
    assert cli.resolve_restart(args, iterate_case) == field


def test_chic_solves_the_design_point_then_sweeps(chic_case):
    """With no iterators the design phase is a single solve, because the sweep
    still needs a field to start from and an answer to depart from."""
    out = chic_case.parent

    assert cli.main(["chic", str(chic_case)]) == 0

    assert (out / "iter_0000" / cli.OUTPUT_NAME).is_file()
    assert (out / "chic_0000" / cli.OUTPUT_NAME).is_file()
    assert (out / "final").resolve() == (out / "iter_0000").resolve()


def test_chic_iterates_the_design_first_when_it_can(chic_iterate_case, capsys):
    """The composition the verb exists for: converge, then sweep what settled."""
    out = chic_iterate_case.parent

    assert cli.main(["chic", str(chic_iterate_case), "-s", "max_iter=2"]) == 0

    assert "Converging the design first" in capsys.readouterr().err
    assert (out / "iter_0000" / cli.OUTPUT_NAME).is_file()
    assert (out / "chic_0000" / cli.OUTPUT_NAME).is_file()


def test_chic_refuses_to_sweep_a_design_that_did_not_settle(iterate_case, capsys):
    """A characteristic of a machine still being redesigned is a
    characteristic of no machine in particular."""
    unsettled = iterate_case.parent / "input.yaml"
    unsettled.write_text(ITERATE_CASE + SWEEP)

    assert cli.main(["chic", str(unsettled), "-s", "max_iter=1"]) == 1

    assert "no machine to sweep" in capsys.readouterr().err


def test_chic_skips_a_design_that_has_already_settled(chic_case, capsys):
    """The inference: a stored result that converged, with its design errors
    inside their tolerances, means the design phase is done."""
    cli.main(["chic", str(chic_case)])
    capsys.readouterr()

    # Run again in a directory carrying the answer the first one reached.
    settled = chic_case.parent / "settled"
    settled.mkdir()
    answer = (chic_case.parent / cli.OUTPUT_NAME).resolve()
    (settled / "input.yaml").write_text(answer.read_text())

    cli.main(["chic", str(settled / "input.yaml")])
    printed = capsys.readouterr().err

    assert "Sweeping straight away" in printed
    assert not (settled / "iter_0000").exists()
    assert (settled / "chic_0000").is_dir()


def test_chic_without_a_chic_section_says_so(iterate_case, capsys):
    assert cli.main(["chic", str(iterate_case)]) == 1

    assert "needs a chic: section" in capsys.readouterr().err

"""Tests for plugin discovery and the command line interface.

The work underneath the CLI is covered elsewhere, so these target what is
unique to the command line: that an ephemeral run really writes nothing, that
overrides reach the design, that a user's design is found without being told
where it is, and that a bad config reads as a message rather than a traceback.
"""

import textwrap

import pytest

from turbigen2 import cli, plugins

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
    return textwrap.dedent(f'''
        from typing import ClassVar
        from turbigen2.design import MeanLineDesign

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
    ''')


@pytest.fixture
def case(tmp_path):
    """A config file in a directory of its own."""
    path = tmp_path / "case.yaml"
    path.write_text(CASE)
    return path


@pytest.fixture
def clean_registry():
    """Restore the design registry after a test loads a plugin into it."""
    from turbigen2.design import MeanLineDesign
    from turbigen2.node import _REGISTRY

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
    from turbigen2.design import MeanLineDesign

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


def test_design_writes_nothing_without_out(case, capsys):
    before = set(case.parent.iterdir())

    assert cli.main(["design", str(case)]) == 0

    assert set(case.parent.iterdir()) == before
    assert "Mean line:" in capsys.readouterr().err


def test_design_writes_a_config_that_reads_back_equal(case, tmp_path):
    from turbigen2 import Config

    out = tmp_path / "out"
    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

    assert (out / "config.yaml").is_file()
    assert (out / "log_turbigen2.txt").is_file()
    assert Config.from_file(out / "config.yaml") == Config.from_file(case)


def test_out_star_takes_the_next_free_number(case, tmp_path):
    pattern = str(tmp_path / "run_*")

    cli.main(["design", str(case), "-o", pattern, "-q"])
    cli.main(["design", str(case), "-o", pattern, "-q"])

    assert (tmp_path / "run_0000").is_dir()
    assert (tmp_path / "run_0001").is_dir()


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


def test_quiet_suppresses_results_but_not_the_exit_code(case, capsys):
    assert cli.main(["design", str(case), "-q"]) == 0

    assert capsys.readouterr().out == ""


def test_missing_config_file_is_a_message_not_a_traceback(tmp_path, capsys):
    assert cli.main(["design", str(tmp_path / "nope.yaml")]) == 1

    captured = capsys.readouterr()
    assert "FileNotFoundError" in captured.err
    assert "Traceback" not in captured.err


def test_verbose_shows_the_traceback(tmp_path, capsys):
    assert cli.main(["design", str(tmp_path / "nope.yaml"), "-v"]) == 1

    assert "Traceback" in capsys.readouterr().err


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

    This is why no plugin path is recorded anywhere: the output directory sits
    under the case directory, so walking up from it reaches the same place.
    """
    plug_dir = tmp_path / plugins.PLUGIN_DIR_NAME
    plug_dir.mkdir()
    (plug_dir / "mine.py").write_text(plugin_source("_t_rerun", "RerunStage"))

    case = tmp_path / "stage.yaml"
    case.write_text(PLUGIN_CASE.format(name="_t_rerun"))

    out = tmp_path / "out"
    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

    assert cli.main(["design", str(out / "config.yaml"), "-q"]) == 0


def test_mesh_without_a_mesh_section_is_a_message_not_a_traceback(case, capsys):
    """`Config` has no make_grid to hold this check, so the verb holds it.

    A missing `mesh:` section is a fact about the command the user typed, not
    about the config in the abstract -- a `design` run of the same file is
    perfectly valid. Without the check `config.mesh.mesh(...)` would raise
    AttributeError on None, which is the unhelpful failure the strict config
    validation exists to avoid.
    """
    assert cli.main(["mesh", str(case)]) == 1

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
    path = tmp_path / "cascade.yaml"
    path.write_text(RUN_CASE)
    return path


def test_run_solves_a_case_end_to_end(run_case, tmp_path, capsys):
    out = tmp_path / "out"

    assert cli.main(["run", str(run_case), "-o", str(out)]) == 0

    printed = capsys.readouterr().err
    assert "Mean line:" in printed
    assert "Mesh:" in printed
    assert "Solver: converged" in printed


def test_run_writes_a_config_that_reads_back(run_case, tmp_path):
    """The archived config is the run, defaults and all."""
    from turbigen2 import Config  # noqa: PLC0415

    out = tmp_path / "out"
    cli.main(["run", str(run_case), "-o", str(out), "-q"])

    written = out / "config.yaml"
    assert written.exists()
    assert (out / "log_turbigen2.txt").exists()
    assert Config.from_file(written) == Config.from_file(run_case)


# The march is driven unstable on purpose, so ember's warning that the outlet
# has gone supersonic is the expected behaviour rather than a problem. Without
# this the suite's `filterwarnings = error` turns it into an exception, and the
# verb reports a config error (1) instead of a failed solve (2).
@pytest.mark.filterwarnings("ignore::ember.nonreflecting.UnsupportedMeanStateWarning")
@pytest.mark.filterwarnings("ignore:invalid value")
@pytest.mark.filterwarnings("ignore:divide by zero")
def test_run_reports_a_failed_solve_in_its_exit_code(run_case, tmp_path):
    """Exit 2, and the output is still written.

    A diverged run is exactly the one whose output someone needs to look at, so
    failing must not also throw away the evidence. A distinct code from 1 keeps
    "the solver did not converge" apart from "the config was wrong", which a
    script driving a sweep has to tell apart without parsing the log.
    """
    out = tmp_path / "out"

    # A CFL far past the stability limit, so it diverges within a few steps.
    code = cli.main(["run", str(run_case), "-o", str(out), "-q", "-s", "solver.cfl=50.0"])

    assert code == 2
    assert (out / "config.yaml").exists()


def test_run_requires_an_output_directory(run_case, capsys):
    """Unlike design and mesh, a run produces artefacts worth keeping."""
    assert cli.main(["run", str(run_case)]) == 1

    assert "--out" in capsys.readouterr().err


def test_run_without_a_solver_section_is_a_message(run_case, tmp_path, capsys):
    text = run_case.read_text()
    trimmed = "\n".join(
        line for line in text.splitlines() if not line.startswith("solver:")
    )
    run_case.write_text(trimmed)

    assert cli.main(["run", str(run_case), "-o", str(tmp_path / "out")]) == 1

    assert "solver: section" in capsys.readouterr().err


def test_run_writes_its_answer_beside_the_config(run_case, tmp_path):
    """The point of the whole arrangement: one file, loaded once.

    A run's mixed-out mean line goes into the same file under `result:`, so
    comparing what was achieved against what was asked for needs no second
    artefact and no repeat of the CFD.
    """
    from turbigen2 import case  # noqa: PLC0415

    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 0

    config, result = case.read(out / "config.yaml")

    assert result.converged is True
    assert result.actual is not None
    assert result.actual.shape == result.nominal.shape

    # The achieved design variables come back in the design's own vocabulary,
    # recomputed from the stored state rather than stored themselves.
    achieved = config.mean_line.backward(result.actual)
    assert 0.0 < float(achieved["Ma2"]) < 1.0


def test_mesh_restart_replots_a_previous_run(run_case, tmp_path):
    """Re-plotting needs no more than the config and the restart file.

    The grid is not serialised, so a re-plot re-designs and re-meshes to put
    the stored field back -- seconds against the minutes of the march it
    stands in for.
    """
    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 0

    replot = tmp_path / "replot"
    code = cli.main(
        [
            "mesh",
            str(out / "config.yaml"),
            "--restart",
            str(out / "restart.npz"),
            "-o",
            str(replot),
            "-q",
        ]
    )

    assert code == 0
    assert (replot / "post.pdf").is_file()


def test_bare_restart_replots_a_run_in_place(run_case, tmp_path):
    """The common case: redo the plots for a run that is already there."""
    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 0
    (out / "post.pdf").unlink()

    code = cli.main(
        ["mesh", str(out / "config.yaml"), "--restart", "-o", str(out), "-q"]
    )

    assert code == 0
    assert (out / "post.pdf").is_file()


def test_replotting_in_place_keeps_the_recorded_answer(run_case, tmp_path):
    """A verb that computed no answer must not erase the one already there.

    Rewriting the archived config with this verb's empty result would leave a
    converged run claiming it had never converged, and lose the mean line it
    mixed out to.
    """
    from turbigen2 import case  # noqa: PLC0415

    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 0
    _, before = case.read(out / "config.yaml", design=False)

    cli.main(["mesh", str(out / "config.yaml"), "--restart", "-o", str(out), "-q"])

    _, after = case.read(out / "config.yaml", design=False)
    assert after.converged is before.converged is True
    assert after.actual is not None


def test_run_writes_its_convergence_history(run_case, tmp_path):
    """Beside the field, so a re-plot can draw the page the run had."""
    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 0

    assert (out / cli.HISTORY_NAME).is_file()


def test_a_replot_recovers_the_convergence_history(run_case, tmp_path):
    out = tmp_path / "out"
    cli.main(["run", str(run_case), "-o", str(out), "-q"])

    history = cli.read_history(out / cli.HISTORY_NAME)

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


def test_bare_restart_needs_somewhere_to_look(run_case, capsys):
    assert cli.main(["mesh", str(run_case), "--restart"]) == 1

    assert "needs --out" in capsys.readouterr().err


def test_bare_restart_says_when_there_is_nothing_to_read(run_case, tmp_path, capsys):
    out = tmp_path / "new"

    assert cli.main(["mesh", str(run_case), "--restart", "-o", str(out)]) == 1

    assert "No restart.npz" in capsys.readouterr().err


def test_a_failing_plot_cannot_lose_the_solution(run_case, tmp_path, monkeypatch):
    """The flow field is written before the report, not after.

    Post-processing raises by design, and the standard plots run whether or not
    a config asks for them -- so with the two the other way round, a plot that
    fell over would throw away a march that had already been paid for.
    """
    from turbigen2 import SectionsPlot  # noqa: PLC0415

    def boom(self, config, result):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(SectionsPlot, "report", boom)

    out = tmp_path / "out"
    assert cli.main(["run", str(run_case), "-o", str(out), "-q"]) == 1

    assert (out / "restart.npz").is_file()

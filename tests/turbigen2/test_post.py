"""Tests for post-processing.

A post-processor returns figures rather than drawing into a shared document,
which is what makes these tests possible at all: each one can be run alone and
asserted on without a PdfPages or a filesystem.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from test_cli import RUN_CASE  # noqa: E402
from turbigen2 import (  # noqa: E402
    AnnulusPlot,
    Config,
    ConvergencePlot,
    Post,
    Result,
    SectionsPlot,
    SurfacePlot,
    cli,
    post,
)

FLUID = {"type": "perfect", "cp": 1005.0, "gamma": 1.4, "mu": 1.8e-5}
MEAN_LINE = {
    "type": "axial_turbine",
    "psi": 1.6,
    "phi2": 0.8,
    "Ma2": 0.9,
    "fac_Ma3_rel": 0.8,
    "mdot": 10.0,
    "Ys": [0.05, 0.05],
    "r_rms": 0.3,
}
ANNULUS = {
    "type": "fixed_axial_chord",
    "cx_row": [0.04, 0.04],
    "cx_gap": [0.06, 0.02, 0.08],
}
POST = [{"type": "annulus", "show_axis": True, "m_cut": [1.0, 2.0]}]

CASE = f"""
fluid: {FLUID}
mean_line: {MEAN_LINE}
annulus: {ANNULUS}
post_process: {POST}
"""


@pytest.fixture
def config():
    return Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "annulus": ANNULUS,
            "post_process": POST,
        }
    )


@pytest.fixture
def result(config):
    return Result(machine=config.design())


@pytest.fixture(scope="module")
def bladed():
    """The fast single-row case, which the flow plots need blades to draw."""
    return Config.from_dict(yaml.safe_load(RUN_CASE))


@pytest.fixture(scope="module")
def meshed(bladed):
    """Meshed and given an initial guess, but never marched."""
    machine, grid = cli.prepare(bladed)
    return Result(machine=machine, grid=grid)


@pytest.fixture(scope="module")
def solved(bladed):
    """The same case marched briefly, for a real field and a real history."""
    machine, grid = cli.prepare(bladed)
    history = bladed.solver.solve(grid)
    return Result(
        machine=machine,
        grid=grid,
        converged=True,
        history=history,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


#
# THE NODE PROTOCOL CARRIES A LIST OF THEM
#


def test_post_processors_are_nodes(config):
    assert isinstance(config.post_process, tuple)
    assert all(isinstance(p, Post) for p in config.post_process)
    assert isinstance(config.post_process[0], AnnulusPlot)


def test_a_list_of_post_processors_round_trips(config):
    assert Config.from_dict(config.to_dict()) == config


def test_post_processor_fields_round_trip(config):
    dumped = config.to_dict()["post_process"][0]

    assert dumped["type"] == "annulus"
    assert dumped["show_axis"] is True
    assert dumped["m_cut"] == [1.0, 2.0]


def test_nothing_is_added_implicitly():
    """What the config asks for is what runs.

    The old config inserted four processors at the front of the list in
    __post_init__ unless an isinstance check found them already there, so what
    ran was not what the file said.
    """
    bare = Config.from_dict({"fluid": FLUID, "mean_line": MEAN_LINE})

    assert bare.post_process == ()


def test_an_unknown_post_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown Post type"):
        Config.from_dict(
            {
                "fluid": FLUID,
                "mean_line": MEAN_LINE,
                "post_process": [{"type": "no_such_plot"}],
            }
        )


#
# REPORTING
#


def test_report_returns_figures(config, result):
    figures = config.post_process[0].report(config, result)

    assert len(figures) == 1
    assert isinstance(figures[0], plt.Figure)


def test_report_does_not_need_a_file(config, result):
    """A processor can be run alone, in a notebook or a test."""
    post = AnnulusPlot(m_cut=(1.0,))

    figures = post.report(config, result)

    assert len(figures) == 1


def test_an_absent_annulus_is_not_an_error():
    """Skipping is normal: the same processors run at every pipeline depth."""
    config = Config.from_dict(
        {"fluid": FLUID, "mean_line": MEAN_LINE, "post_process": POST}
    )
    result = Result(machine=config.design())

    assert config.post_process[0].report(config, result) == []


def test_the_plot_draws_the_hub_and_casing(config, result):
    figures = config.post_process[0].report(config, result)
    ax = figures[0].axes[0]

    # Two m_cut planes, then hub, casing, and the axis of rotation.
    assert len(ax.lines) == 2 + 3


def test_report_leaves_the_config_untouched(config, result):
    """A plot cannot mutate the design; Config is frozen.

    The old post-processors call config.apply_recamber() and
    config.undo_recamber() from inside a plot.
    """
    before = config.to_dict()

    config.post_process[0].report(config, result)

    assert config.to_dict() == before


#
# THE STANDARD PLOTS
#
# Each one is checked twice: that it draws what it should when its inputs are
# there, and that it returns nothing when they are not. The second half is what
# lets the set run for every verb without a switch -- a plot that raised on a
# mean-line-only design would take the whole command down with it.
#


def test_sections_plot_draws_each_row(bladed, meshed):
    figures = SectionsPlot().report(bladed, meshed)

    assert len(figures) == len(meshed.machine.rows)
    # One section, drawn as its two surfaces.
    assert len(figures[0].axes[0].lines) == 2


def test_sections_plot_takes_the_span_fractions_it_is_given(bladed, meshed):
    figures = SectionsPlot(spf=(0.25, 0.75)).report(bladed, meshed)

    assert len(figures[0].axes[0].lines) == 4


def test_sections_plot_without_blades_is_empty(config, result):
    assert SectionsPlot().report(config, result) == []


def test_convergence_plot_draws_residuals_and_errors(bladed, solved):
    residuals, errors = ConvergencePlot().report(bladed, solved)

    # One line per conserved variable.
    assert len(residuals.axes[0].lines) == 5
    # Mass, work and loss, side by side.
    assert len(errors.axes) == 3


def test_convergence_plot_without_a_march_is_empty(bladed, meshed):
    """Every verb but `run` reaches here with no history at all."""
    assert ConvergencePlot().report(bladed, meshed) == []


def test_surface_plot_draws_a_physical_distribution(bladed, solved):
    figures = SurfacePlot().report(bladed, solved)

    assert len(figures) == 1
    (line,) = figures[0].axes[0].lines
    zeta, mas = line.get_xdata(), line.get_ydata()

    # Both surfaces are folded onto the positive axis, running from the
    # stagnation point at the origin out to the trailing edge at one.
    assert zeta.min() == pytest.approx(0.0, abs=1e-6)
    assert zeta.max() == pytest.approx(1.0)
    assert mas.min() == pytest.approx(0.0, abs=1e-6)
    # A turbine cascade accelerating to a subsonic exit.
    assert 0.3 < mas.max() < 1.5


def test_surface_plot_says_when_it_is_drawing_a_guess(bladed, meshed):
    """Plotting an unmarched grid is a way to look at a mesh, not a mistake.

    It must not be mistaken for a solution, so the one thing that has to hold
    is that the figure says which it is.
    """
    figures = SurfacePlot().report(bladed, meshed)

    assert len(figures) == 1
    assert "Guess" in figures[0].axes[0].get_title()


def test_surface_plot_without_a_grid_is_empty(config, result):
    assert SurfacePlot().report(config, result) == []


def test_surface_plot_skips_a_diverged_march(bladed, solved):
    """A blown-up field is NaN throughout, and no state can be found in it.

    The plot has to notice, because the standard set runs unasked: raising here
    would report a diverged run as a broken config.
    """
    import dataclasses  # noqa: PLC0415

    history = solved.history.copy()
    history.diverged = True

    diverged = dataclasses.replace(solved, history=history)

    assert SurfacePlot().report(bladed, diverged) == []


#
# THE STANDARD SET
#


def test_the_standard_set_runs_unasked():
    bare = Config.from_dict({"fluid": FLUID, "mean_line": MEAN_LINE})

    running = cli.processors(bare)

    assert [p.type for p in running] == [p.type for p in post.STANDARD]


def test_a_configured_processor_replaces_its_standard_counterpart(config):
    """Tuning a standard plot must not give you two of them.

    The package this replaces had the same rule, but reached it by inserting
    into the user's own list from __post_init__, so the config that ran was not
    the config that was written.
    """
    running = cli.processors(config)

    annulus_plots = [p for p in running if p.type == "annulus"]
    assert annulus_plots == [config.post_process[0]]
    assert len(running) == len(post.STANDARD)


def test_a_configured_processor_of_a_new_type_is_added(config):
    class Extra(Post):
        type = "extra_for_this_test"

        def report(self, config, result):
            return []

    extended = Config.from_dict(
        {**config.to_dict(), "post_process": [{"type": Extra.type}]}
    )

    running = cli.processors(extended)

    # The whole standard set, and the extra one after it.
    assert len(running) == len(post.STANDARD) + 1
    assert running[-1].type == Extra.type


#
# THROUGH THE CLI
#


def test_pdf_is_written_to_the_output_directory(tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text(CASE)
    out = tmp_path / "out"

    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

    pdf = out / "post.pdf"
    assert pdf.is_file()
    assert pdf.read_bytes().startswith(b"%PDF")


def test_no_output_directory_means_no_pdf(tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text(CASE)
    before = set(tmp_path.iterdir())

    assert cli.main(["design", str(case), "-q"]) == 0

    assert set(tmp_path.iterdir()) == before


def test_a_report_is_written_without_any_post_processors(tmp_path):
    """The standard plots do not have to be asked for.

    They cost a fraction of a solve and each one degrades to nothing when what
    it needs is absent, so a run with somewhere to write always gets a report.
    """
    case = tmp_path / "case.yaml"
    case.write_text(f"fluid: {FLUID}\nmean_line: {MEAN_LINE}\nannulus: {ANNULUS}\n")
    out = tmp_path / "out"

    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

    assert (out / "post.pdf").is_file()


def test_a_design_with_nothing_to_draw_writes_no_report(tmp_path):
    """An empty document is not a report, and matplotlib writes no file for one."""
    case = tmp_path / "case.yaml"
    case.write_text(f"fluid: {FLUID}\nmean_line: {MEAN_LINE}\n")
    out = tmp_path / "out"

    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

    assert (out / "config.yaml").is_file()
    assert not (out / "post.pdf").exists()


def test_a_failing_post_processor_is_not_swallowed(tmp_path, monkeypatch):
    """A broken plot must not leave a silently incomplete report.

    The old post_process_all catches every exception, prints a traceback to
    stderr rather than the log, and carries on.
    """

    def boom(self, config, result):
        raise RuntimeError("plot exploded")

    monkeypatch.setattr(AnnulusPlot, "report", boom)

    case = tmp_path / "case.yaml"
    case.write_text(CASE)

    assert cli.main(["design", str(case), "-o", str(tmp_path / "out"), "-q"]) == 1

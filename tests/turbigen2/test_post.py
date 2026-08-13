"""Tests for post-processing.

A post-processor returns figures rather than drawing into a shared document,
which is what makes these tests possible at all: each one can be run alone and
asserted on without a PdfPages or a filesystem.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from turbigen2 import AnnulusPlot, Config, Post, Result, cli  # noqa: E402

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


def test_no_post_processors_means_no_pdf(tmp_path):
    case = tmp_path / "case.yaml"
    case.write_text(f"fluid: {FLUID}\nmean_line: {MEAN_LINE}\nannulus: {ANNULUS}\n")
    out = tmp_path / "out"

    assert cli.main(["design", str(case), "-o", str(out), "-q"]) == 0

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

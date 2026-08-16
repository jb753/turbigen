"""Tests for blade design.

The blade is the stage that most needed the config/result split, because the
package this replaces reaches its geometry by mutating the designer three times
over: a stream surface and a thickness scale are written onto it, the recamber
angles are overwritten in place with metal angles, and an ``is_recambered``
flag is toggled on and off by post-processors. So besides the geometry, these
check that none of that state exists any more.
"""

import dataclasses

import numpy as np
import pytest

import turbigen_ref.annulus
import turbigen_ref.blade
import turbigen_ref.nblade
from turbigen.blade import _interpolate
from turbigen import (
    Blade,
    BladeDesign,
    Config,
    Quadratic,
    Row,
    Section,
    Taylor,
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
SPF = (0.2, 0.5, 0.8)
CAMBER = {"type": "quadratic", "aft_loading": 0.0}
THICKNESS = {
    "type": "taylor",
    "R_LE": 0.04,
    "t_max": 0.12,
    "m_tmax": 0.35,
    "kappa_max": 0.0,
    "t_TE": 0.02,
    "tanwedge": 0.25,
}


def blade(dchi_LE=-8.0, dchi_TE=0.0, **kwargs):
    """A three-section blade, with the design overridden as asked."""
    return {
        "sections": [
            {
                "spf": spf,
                "dchi_LE": dchi_LE,
                "dchi_TE": dchi_TE,
                "camber": CAMBER,
                "thickness": THICKNESS,
            }
            for spf in SPF
        ],
        "count": {"type": "Co", "Co": 0.6},
        **kwargs,
    }


def build(blades=None, **kwargs):
    """A two-row config carrying blades, unless told otherwise."""
    if blades is None:
        blades = [blade(), blade(dchi_LE=2.0)]
    return Config.from_dict(
        {
            "fluid": FLUID,
            "mean_line": MEAN_LINE,
            "annulus": ANNULUS,
            "blades": blades,
            **kwargs,
        }
    )


@pytest.fixture
def machine():
    return build().design()


class OldRow:
    """The old package's mean-line interface, over a turbigen mean line.

    The two mean lines name the root-mean-square radius differently, `r_rms`
    against `r`, so comparing against the old blade code needs this much of a
    shim and no more.
    """

    def __init__(self, mean_line_row):
        ml = mean_line_row
        self.r_hub = np.asarray(ml.r_hub, dtype=float)
        self.r_cas = np.asarray(ml.r_cas, dtype=float)
        self.r_rms = np.asarray(ml.r, dtype=float)
        self.Vt = np.asarray(ml.Vt, dtype=float)
        self.Vm = np.asarray(ml.Vm, dtype=float)
        self.Omega = np.asarray(ml.Omega, dtype=float)
        self.Alpha = np.asarray(ml.Alpha, dtype=float)
        self.Alpha_rel = np.asarray(ml.Alpha_rel, dtype=float)
        self.Am = np.asarray(ml.Am, dtype=float)


def old_blade(machine, i_row, dchi_LE=-8.0, dchi_TE=0.0):
    """The same blade, built and recambered through the old package."""
    flat = machine.mean_line.flat
    cx_row = np.array(ANNULUS["cx_row"])
    cx_gap = np.array(ANNULUS["cx_gap"])
    annulus = turbigen_ref.annulus.MergedFixedAxialChord(
        {"cx_row": cx_row, "cx_gap": cx_gap}
    )
    annulus.forward(
        np.asarray(flat.r_mid, dtype=float),
        np.asarray(flat.span, dtype=float),
        np.asarray(flat.Beta, dtype=float),
        cx_row=cx_row,
        cx_gap=cx_gap,
        merge_weight=0.0,
    )

    designer = turbigen_ref.blade.BladeDesigner(
        spf=np.array(SPF),
        camber=np.tile([dchi_LE, dchi_TE, CAMBER["aft_loading"]], (len(SPF), 1)),
        thick=np.tile(
            [
                THICKNESS[key]
                for key in ("R_LE", "t_max", "m_tmax", "kappa_max", "t_TE", "tanwedge")
            ],
            (len(SPF), 1),
        ),
        camber_type="quadratic",
        thick_type="taylor",
    )
    designer.set_streamsurface(annulus.xr_row(i_row))
    designer.apply_recamber(OldRow(machine.mean_line.row(i_row)))
    return designer


#
# THE STAGE INTERFACE
#


def test_design_returns_a_machine_holding_every_stage(machine):
    assert len(machine.rows) == machine.mean_line.n_row
    assert all(isinstance(r, Row) for r in machine.rows)
    assert all(isinstance(r.blade, Blade) for r in machine.rows)


def test_a_config_without_blades_designs_the_annulus_alone():
    config = Config.from_dict(
        {"fluid": FLUID, "mean_line": MEAN_LINE, "annulus": ANNULUS}
    )

    machine = config.design()

    assert machine.rows == ()
    assert machine.annulus is not None


def test_the_design_stores_nothing_on_itself(machine):
    """No stream surface, no thickness scale, no recambered flag.

    All three are attributes the old designer grows during a design, which is
    what makes it unable to hold more than one.
    """
    config = build()
    before = dataclasses.asdict(config.blades[0])

    config.design()

    assert dataclasses.asdict(config.blades[0]) == before
    for name in ("streamsurface", "_thick_scale", "is_recambered"):
        assert not hasattr(config.blades[0], name)


def test_designing_twice_gives_two_independent_blades():
    config = build()
    mean_line = config.design().mean_line
    annulus = config.annulus.design(mean_line)

    first = config.blades[0].design(mean_line.row(0), annulus.row(0))
    second = config.blades[0].design(mean_line.row(0), annulus.row(0))

    assert first is not second
    np.testing.assert_allclose(first.blade.chi(0.5), second.blade.chi(0.5))


def test_wrong_number_of_blades_is_rejected():
    with pytest.raises(ValueError, match="one blade per row"):
        build(blades=[blade()]).design()


def test_blades_without_an_annulus_are_rejected():
    config = Config.from_dict(
        {"fluid": FLUID, "mean_line": MEAN_LINE, "blades": [blade(), blade()]}
    )

    with pytest.raises(ValueError, match="need an annulus"):
        config.design()


#
# GEOMETRY
#


def test_recamber_is_measured_from_the_local_flow_angle(machine):
    """Metal angle is flow angle plus recamber, resolved once at design time."""
    mean_line = machine.mean_line.row(0)
    chi = machine.rows[0].blade.chi(0.5)

    # At mid-span of a free vortex the radius is not the rms radius, so this
    # is only approximately the mean-line angle, but it is much closer to it
    # than the eight degrees of recamber.
    np.testing.assert_allclose(chi[0] - (-8.0), mean_line.Alpha_rel[0], atol=1.0)
    np.testing.assert_allclose(chi[1], mean_line.Alpha_rel[1], atol=1.0)


def test_metal_angles_over_ninety_degrees_are_rejected():
    with pytest.raises(ValueError, match="over 90 degrees"):
        build(blades=[blade(dchi_LE=120.0), blade()]).design()


def test_sections_are_stacked_at_the_stacking_point():
    """Every section passes through theta=0 at m_stack, by definition."""
    machine = build(blades=[blade(m_stack=0.25), blade()]).design()

    m = np.linspace(0.0, 1.0, 101)
    for spf in (0.0, 0.5, 1.0):
        xrtu, xrtl = machine.rows[0].blade.evaluate_section(spf, m=m)
        theta_camber = 0.5 * (xrtu[2] + xrtl[2])
        assert abs(np.interp(0.25, m, theta_camber)) < 1e-3


def test_theta_offset_rotates_the_whole_blade():
    offset = 0.1
    plain = build().design().rows[0].blade
    rotated = build(blades=[blade(theta_offset=offset), blade()]).design().rows[0].blade

    for spf in (0.0, 1.0):
        for turned, still in zip(
            rotated.evaluate_section(spf), plain.evaluate_section(spf)
        ):
            np.testing.assert_allclose(turned[:2], still[:2], atol=1e-12)
            np.testing.assert_allclose(turned[2] - still[2], offset, atol=1e-9)


def test_a_single_section_blade_is_constant_in_span():
    config = build(
        blades=[
            {
                "sections": [
                    {
                        "spf": 0.5,
                        "dchi_LE": -8.0,
                        "dchi_TE": 0.0,
                        "camber": CAMBER,
                        "thickness": THICKNESS,
                    }
                ],
                "count": {"type": "Nb", "Nb": 40},
            },
            blade(),
        ]
    )

    bld = config.design().rows[0].blade

    np.testing.assert_allclose(bld.chi(0.0), bld.chi(1.0), atol=1e-12)


def test_sections_of_different_design_cannot_be_interpolated():
    """There is no meaning to blending one camber shape into another."""
    sections = (
        Section(
            spf=0.0,
            dchi_LE=0.0,
            dchi_TE=0.0,
            camber=Quadratic(aft_loading=0.0),
            thickness=Taylor(**{k: v for k, v in THICKNESS.items() if k != "type"}),
        ),
        Section(
            spf=1.0,
            dchi_LE=0.0,
            dchi_TE=0.0,
            camber=OtherCamber(),
            thickness=Taylor(**{k: v for k, v in THICKNESS.items() if k != "type"}),
        ),
    )
    with pytest.raises(ValueError, match="same design"):
        dataclasses.replace(build().blades[0], sections=sections)


class OtherCamber(Quadratic):
    """A second camber shape, to check that mixing them is refused."""

    type = "other_for_testing"


#
# BLADE NUMBER AND TIP GAP
#


def test_fixed_count_returns_what_it_was_given():
    machine = build(blades=[blade(count={"type": "Nb", "Nb": 37}), blade()]).design()

    assert machine.rows[0].n_blade == 37


def test_circulation_count_matches_the_turbigen_implementation(machine):
    for i_row in range(2):
        dchi_LE = (-8.0, 2.0)[i_row]
        expected = turbigen_ref.nblade.Co(Co=0.6).get_blade_number(
            OldRow(machine.mean_line.row(i_row)),
            old_blade(machine, i_row, dchi_LE=dchi_LE),
        )

        assert machine.rows[i_row].n_blade == int(np.round(expected).item())


@pytest.mark.parametrize(
    "field, expected",
    [
        ("tip_span", "span"),
        ("tip_chord", "chord"),
        ("tip_metre", "metre"),
    ],
)
def test_each_tip_reference_gives_the_expected_gap(field, expected):
    machine = build(blades=[blade(**{field: 0.02}), blade()]).design()
    mean_line = machine.mean_line.row(0)

    reference = {
        "span": float(np.mean(mean_line.span)),
        "chord": machine.annulus.chords(0.5)[1],
        "metre": 1.0,
    }[expected]

    assert machine.rows[0].tip_gap == pytest.approx(0.02 * reference)


def test_no_tip_clearance_by_default(machine):
    assert machine.rows[0].tip_gap == 0.0


def test_two_tip_references_at_once_are_rejected():
    with pytest.raises(ValueError, match="one tip clearance"):
        BladeDesign.from_dict(blade(tip_span=0.01, tip_chord=0.01))


#
# SERIALISATION
#


def test_config_with_blades_round_trips():
    config = build()

    assert Config.from_dict(config.to_dict()) == config


def test_blade_defaults_are_written_out():
    data = build().to_dict()["blades"][0]

    assert data["vortex_exponent"] == -1.0
    assert data["m_stack"] == 0.5
    assert data["tip_span"] == 0.0
    assert data["sections"][0]["camber"]["type"] == "quadratic"


def test_sections_must_increase_in_span_fraction():
    reversed_sections = blade()
    reversed_sections["sections"] = reversed_sections["sections"][::-1]

    with pytest.raises(ValueError, match="increasing span fraction"):
        BladeDesign.from_dict(reversed_sections)


def test_a_blade_needs_a_section():
    with pytest.raises(ValueError, match="at least one section"):
        BladeDesign.from_dict({"sections": [], "count": {"type": "Nb", "Nb": 1}})


#
# AGREEMENT WITH THE PACKAGE THIS REPLACES
#


@pytest.mark.parametrize("i_row, dchi_LE", [(0, -8.0), (1, 2.0)])
def test_matches_the_turbigen_implementation(machine, i_row, dchi_LE):
    """The one test that exercises the whole port at once.

    Camber, thickness, recamber, stacking and the stream surface all feed
    `evaluate_section`, so agreeing with the old package here is agreement
    everywhere.
    """
    old = old_blade(machine, i_row, dchi_LE=dchi_LE)
    new = machine.rows[i_row].blade

    for spf in (0.0, 0.5, 1.0):
        for surface_new, surface_old in zip(
            new.evaluate_section(spf), old.evaluate_section(spf)
        ):
            np.testing.assert_allclose(
                surface_new,
                surface_old,
                rtol=1e-12,
                atol=1e-12,
                err_msg=f"differs from the turbigen blade at spf={spf}",
            )

        np.testing.assert_allclose(new.chi(spf), old.get_chi(spf), atol=1e-12)
        np.testing.assert_allclose(
            new.surface_length(spf), old.surface_length(spf), rtol=1e-12
        )
        np.testing.assert_allclose(new.chord(spf), old.chord(spf), rtol=1e-12)


#
# REPORTING
#


def test_machine_reports_the_blades(machine):
    report = machine.to_string()

    assert "Blades:" in report
    assert "N_blade" in report


def test_machine_reports_only_what_was_designed():
    machine = Config.from_dict(
        {"fluid": FLUID, "mean_line": MEAN_LINE, "annulus": ANNULUS}
    ).design()

    assert "Blades:" not in machine.to_string()


#
# IMMUTABILITY
#


def test_blade_cannot_be_rebound():
    """A result is frozen, so the mutation the old designer relied on -- writing
    a stream surface and then metal angles onto itself -- has nowhere to happen.
    """
    bld = build().design().rows[0].blade

    with pytest.raises(dataclasses.FrozenInstanceError):
        bld.n_blade = 1

    with pytest.raises(dataclasses.FrozenInstanceError):
        bld.tanchi = None


def test_camber_line_cannot_be_rebound():
    camber, _ = build().design().rows[0].blade.section(0.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        camber.tanchi_LE = 0.0


def test_repr_stays_readable():
    """The bulky fields are kept out of the generated repr, which would
    otherwise print every section, both arrays and the whole annulus."""
    bld = build().design().rows[0].blade

    assert repr(bld).startswith("Blade(m_stack=")
    assert "PchipInterpolator" not in repr(bld)


#
# SHAPE AND COUNT ARE SEPARATE
#


def test_a_shape_carries_no_count():
    """How a blade is shaped says nothing about how many of them there are.

    That independence is what lets the shape be built in one go: counting reads
    a shape, but a shape never reads a count, so there is no point at which a
    half-built object has to exist. The old code built the blade twice, passing
    one with `n_blade=None` to the counting rule.
    """
    row = build().design().rows[0]

    assert not hasattr(row.blade, "n_blade")
    assert not hasattr(row.blade, "tip_gap")
    assert isinstance(row.n_blade, int)


def test_a_count_cannot_get_out_of_step_with_its_shape():
    """Separate concerns, but held together rather than indexed apart.

    The package this replaces keeps counts in a second list indexed by row, so
    the two can disagree; `config.get_nblade()` calls `sys.exit(1)` when they
    do. There is no second list here to disagree with.
    """
    machine = build().design()

    assert len(machine.rows) == machine.mean_line.n_row
    assert all(isinstance(r.blade, Blade) for r in machine.rows)


def test_recounting_reuses_the_shape():
    """A different count over the same geometry is a field replacement, not a
    redesign -- which is the practical benefit of keeping the two apart."""
    row = build().design().rows[0]

    recounted = dataclasses.replace(row, n_blade=row.n_blade + 4)

    assert recounted.blade is row.blade
    assert recounted.n_blade == row.n_blade + 4


#
# THICKNESS VALIDITY
#


def test_an_overshooting_thickness_is_refused_when_it_is_built():
    """A distribution that exceeds its own t_max is invalid on construction.

    The parameters decide it, not the points anyone later evaluates, so it is
    settled once here rather than checked inside thick() -- where it could only
    ever cover the samples a caller happened to ask for, and would surface
    part-way through meshing a blade.
    """
    # A nose blunter than the blade is thick: the cubic has to bulge above
    # t_max to get from the leading edge radius down to the declared peak.
    with pytest.raises(ValueError, match="peaks at"):
        Taylor(R_LE=0.02, t_max=0.03, m_tmax=0.35)


def test_a_valid_thickness_evaluates_without_bound_checks():
    """thick() is pure evaluation, so a caller may sample it however it likes."""
    thickness = Taylor(**{k: v for k, v in THICKNESS.items() if k != "type"})

    t = thickness.thick(np.linspace(0.0, 1.0, 501))

    assert t.max() <= thickness.t_max + 1e-10
    assert thickness.peak() == pytest.approx(t.max(), abs=1e-6)


def test_the_peak_is_found_exactly_not_sampled():
    """`peak()` substitutes m = u**2 to get a polynomial, so it is exact.

    Dense sampling is the reference here; the point is that `peak` matches it
    without anyone having chosen a sample count.
    """
    thickness = Taylor(**{k: v for k, v in THICKNESS.items() if k != "type"})

    dense = thickness.thick(np.linspace(0.0, 1.0, 200001)).max()

    assert thickness.peak() == pytest.approx(dense, abs=1e-9)
    # Built to hit t_max at m_tmax, so the peak is the declared one.
    assert thickness.peak() == pytest.approx(thickness.t_max, abs=1e-9)


def test_an_invalid_interpolated_section_names_the_span_fraction():
    """Two valid sections can interpolate to an invalid one in between.

    Nothing constrains the path between them, so the failure has to name where
    it happened -- the offending parameters appear in no config file.
    """
    # Both ends are valid; their midpoint is a blunter nose on a thinner
    # blade than either, which is not.
    ends = [
        Taylor(R_LE=0.002, t_max=0.03, m_tmax=0.35),
        Taylor(R_LE=0.080, t_max=0.08, m_tmax=0.35),
    ]

    with pytest.raises(ValueError, match=r"spf=0\.5"):
        _interpolate(ends, np.array([0.0, 1.0]), 0.5)

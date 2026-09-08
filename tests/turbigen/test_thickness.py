"""Tests for the two-sided thickness distribution.

`ClarkThickness` is the first distribution that is not symmetric about the
camber line, so besides its own shape these check what that costs the rest of
the package: a section of two rows has to interpolate over the span, and a
blade built from one has to count, shape and mesh like any other.

`Taylor` is tested through `test_blade.py`, where it was written.
"""

import numpy as np
import pytest
from test_blade import SPF, blade, build

from turbigen import ClarkThickness, ThicknessDesign, shapespace

CLARK = {
    "type": "clark",
    "R_LE": 0.01,
    "tanwedge": [0.25, 0.25],
    "t_TE": 0.02,
    "coeff": [[0.3, 0.1], [-0.1, 0.05]],
}
"""A deliberately lopsided section: a fatter suction surface than pressure."""

MESH = {"type": "h", "dm_TE": 0.05, "resolution_factor": 0.5, "dspf_mid": 0.1}
"""A coarse mesh, so that meshing a blade costs about a second."""

M = np.linspace(0.0, 1.0, 201)


def design(**kwargs):
    """A ClarkThickness, with the parameters overridden as asked.

    Built the way a config file builds one, so that the coefficients arrive
    as the nested tuples a Node holds rather than as whatever was typed here.
    """
    return ClarkThickness.from_dict({**CLARK, **kwargs})


def blades(thicknesses=None):
    """A two-row config whose first row carries Clark sections.

    One thickness per section when `thicknesses` is given, so that a blade can
    be built with a distribution that changes over the span.
    """
    row = blade()
    for i, section in enumerate(row["sections"]):
        section["thickness"] = CLARK if thicknesses is None else thicknesses[i]
    return [row, blade()]


#
# SHAPE
#


def test_zero_coefficients_are_symmetric():
    """With nothing perturbing the interior, the two sides are the same.

    The endpoints are shared, so the coefficients are the only thing that can
    make an aerofoil lopsided --- and none of them is the default.
    """
    upper, lower = design(coeff=((), ())).thick_both(M)
    np.testing.assert_allclose(upper, lower, atol=1e-15)


def test_the_leading_edge_closes_on_both_surfaces():
    upper, lower = design().thick_both(0.0)
    assert upper == pytest.approx(0.0, abs=1e-15)
    assert lower == pytest.approx(0.0, abs=1e-15)


def test_each_surface_carries_half_the_trailing_edge():
    """`t_TE` is the total due to both sides, so it splits evenly.

    Which is what keeps a blunt trailing edge centred on the camber line even
    when the surfaces ahead of it are nothing alike.
    """
    upper, lower = design().thick_both(1.0)
    assert upper == pytest.approx(CLARK["t_TE"] / 2.0, abs=1e-15)
    assert lower == pytest.approx(CLARK["t_TE"] / 2.0, abs=1e-15)


def test_both_surfaces_share_the_nose_radius():
    """One radius, so the nose is a single curvature rather than two.

    Checked against the circle: a nose of radius `R_LE` has half-thickness
    `sqrt(2 R_LE m)` near `m=0`, and both surfaces have to land on it however
    different they are further back.
    """
    # Close enough to the nose that the terms growing with `m` -- the line
    # in shape space, and the perturbation on it -- have not yet arrived.
    m = np.array([1e-10, 4e-10, 9e-10])
    circle = np.sqrt(2.0 * CLARK["R_LE"] * m)

    for side in design().thick_both(m):
        np.testing.assert_allclose(side, circle, rtol=1e-5)


def test_the_wedge_is_the_closing_slope_of_its_own_surface():
    """Each surface closes at the angle written against it, and no other.

    Read on a closed trailing edge, where the wedge is the only thing setting
    the slope --- a finite `t_TE` steepens it by a further `t_TE / 2`.
    """
    tanwedge = (0.25, 0.15)
    m = 1.0 - np.array([1e-7, 2e-7])

    for side, expected in zip(design(tanwedge=tanwedge, t_TE=0.0).thick_both(m), tanwedge):
        slope = np.diff(side) / np.diff(m)
        np.testing.assert_allclose(slope, -expected, rtol=1e-5)


def test_coefficients_move_only_the_interior():
    """The pinned ends are what make `R_LE` and the wedge mean anything.

    Read in shape space, where the claim is exact: the perturbation is zero
    at both ends whatever its coefficients, so both surfaces start at the
    value the leading edge radius sets and finish at the one the trailing
    edge thickness and wedge angle set. In thickness the same statement is
    only a limit, the class function having taken both ends to zero.
    """
    ends = np.array([0.0, 1.0])
    expected = (
        shapespace.tau_LE(CLARK["R_LE"]),
        shapespace.tau_TE(CLARK["t_TE"], CLARK["tanwedge"][0]),
    )

    for coeff in (((), ()), ((2.0, -1.5), (-1.0, 3.0)), ((0.0, 9.0), (9.0, 0.0))):
        for side in design(coeff=coeff).tau(ends):
            np.testing.assert_allclose(side, expected, atol=1e-15)


def test_a_positive_coefficient_thickens_its_own_surface():
    """Which row is which surface: the first is the suction one.

    All a distribution on its own can say --- which side of a camber line is
    the suction side is the blade's to decide, and there is no blade here. What
    is checkable without one is that the rows do not cross over on the way to
    the answer.
    """
    suction, pressure = design(coeff=((0.5, 0.5), (0.0, 0.0))).thick_both(0.5)
    assert suction > pressure

    suction, pressure = design(coeff=((0.0, 0.0), (0.5, 0.5))).thick_both(0.5)
    assert pressure > suction


def test_a_scalar_position_gives_scalars_back():
    both = design().thick_both(0.5)
    assert all(isinstance(side, float) for side in both)


def test_there_is_no_single_thickness_to_ask_for():
    """A two-sided distribution has no one number, and says so."""
    with pytest.raises(NotImplementedError, match="Clark"):
        design().thick(0.5)


#
# VALIDITY
#


def test_coefficients_need_a_row_for_each_surface():
    with pytest.raises(ValueError, match="two rows"):
        design(coeff=((0.1, 0.2),))


def test_both_surfaces_need_the_same_number_of_coefficients():
    """Their length is the order of the curve, which is one curve family."""
    with pytest.raises(ValueError, match="same number of coefficients"):
        design(coeff=((0.1, 0.2), (0.3,)))


@pytest.mark.parametrize("R_LE", (0.0, -0.01))
def test_the_nose_needs_a_positive_radius(R_LE):
    with pytest.raises(ValueError, match="must be positive"):
        design(R_LE=R_LE)


#
# CONFIG
#


def test_selected_from_a_config_dict():
    thickness = ThicknessDesign.from_dict(dict(CLARK))
    assert isinstance(thickness, ClarkThickness)
    assert thickness.coeff == ((0.3, 0.1), (-0.1, 0.05))


def test_round_trips_through_a_config_dict():
    """Including the nested coefficients, which YAML holds as lists of lists."""
    thickness = design()
    assert ThicknessDesign.from_dict(thickness.to_dict()) == thickness
    assert thickness.to_dict()["coeff"] == [[0.3, 0.1], [-0.1, 0.05]]


#
# ON A BLADE
#


def test_a_clark_blade_is_shaped_and_counted():
    """A row of these is a row like any other, as far as the rest goes.

    Built on the perpendicular offset, which is what leaves the lopsidedness
    below a statement about the thickness. The blend the sections default to
    rotates the pressure surface toward the circumferential direction and so
    lengthens its `theta` departure without changing how thick that side is
    --- see `turbigen.blade.SectionDesign.fac_tangential`.
    """
    rows = blades()
    for section in rows[0]["sections"]:
        section["fac_tangential"] = 0.0
    machine = build(blades=rows).design()
    row = machine.rows[0]

    assert row.n_blade > 0
    for spf in SPF:
        suction, pressure = row.blade.evaluate_section(spf)
        assert np.all(np.isfinite(suction))
        assert np.all(np.isfinite(pressure))

        # Lopsided, so the surfaces are not reflections of one another about
        # the camber line the blade was built on. Measured as a distance from
        # it rather than as a signed angle, since which side of the camber the
        # suction surface falls on is the blade's business, not this test's.
        xrt = row.blade.evaluate_camber(spf)
        assert np.max(np.abs(suction[2] - xrt[2])) > 2.0 * np.max(
            np.abs(pressure[2] - xrt[2])
        )


def test_clark_sections_interpolate_over_the_span():
    """Two rows of coefficients blend element by element, like any parameter."""
    ends = [
        {**CLARK, "coeff": [[0.0, 0.0], [0.0, 0.0]]},
        {**CLARK, "coeff": [[0.4, 0.2], [0.2, 0.6]]},
        {**CLARK, "coeff": [[0.8, 0.4], [0.4, 1.2]]},
    ]
    machine = build(blades=blades(ends)).design()

    # SPF is (0.2, 0.5, 0.8), so the middle section is the mean of the ends
    # and the blade should read the same thing there.
    _, thickness = machine.rows[0].blade._get_cam_thick(SPF[1])
    assert thickness.coeff == ((0.4, 0.2), (0.2, 0.6))


def test_a_clark_blade_meshes():
    """The mesher reads surfaces, and never asked for a symmetric section."""
    config = build(blades=blades(), mesh=MESH)
    grid = config.mesh.mesh(config.design())

    assert len(grid) > 0
    assert all(np.all(np.isfinite(block.xrt)) for block in grid)


#
# SHAPE SPACE COEFFICIENTS
#
# The interior perturbation and the two endpoint values are one Bernstein
# curve written three ways round, which is the form an iterator drives: in it
# every coefficient does the same kind of thing, so one sensitivity sign
# covers all of them.
#


def test_the_order_is_the_length_of_a_coefficient_row():
    """Not declared, so there is no second place for it to disagree."""
    assert design(coeff=((0.1, 0.2), (0.3, 0.4))).order == 3
    assert design(coeff=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))).order == 4


def test_control_points_come_from_the_shared_basis():
    """One array for both surfaces, which carry the same number of coefficients."""
    thickness = design()
    np.testing.assert_allclose(
        thickness.m_ctl, shapespace.control_m(thickness.order)
    )
    assert thickness.m_ctl.shape == (thickness.order + 1,)


def test_the_end_coefficients_are_the_nose_and_the_wedge():
    """The two ends are not special cases of the curve; they *are* its ends."""
    c = design().tau_coeff

    assert c.shape == (2, design().order + 1)
    for row in c:
        assert row[0] == pytest.approx(shapespace.tau_LE(CLARK["R_LE"]))
        assert row[-1] == pytest.approx(
            shapespace.tau_TE(CLARK["t_TE"], CLARK["tanwedge"][0])
        )


def test_zero_perturbation_is_the_straight_line():
    """What `coeff` is a perturbation *of*, made explicit."""
    thickness = design(coeff=((0.0, 0.0), (0.0, 0.0)))
    line = np.linspace(
        shapespace.tau_LE(CLARK["R_LE"]),
        shapespace.tau_TE(CLARK["t_TE"], CLARK["tanwedge"][0]),
        thickness.order + 1,
    )

    for row in thickness.tau_coeff:
        np.testing.assert_allclose(row, line, atol=1e-15)


def test_coefficients_evaluate_the_same_curve():
    """The two forms are one polynomial, so they had better agree everywhere.

    `tau` builds it as a line plus a pinned perturbation; this reads it as a
    single Bernstein curve. If those disagreed, an iterator moving coefficients
    would be shaping a different aerofoil from the one that gets meshed.
    """
    thickness = design()
    for row, side in zip(thickness.tau_coeff, thickness.tau(M)):
        np.testing.assert_allclose(
            shapespace.evaluate_bernstein(row, M), side, atol=1e-14
        )


def test_coefficients_round_trip_through_the_fields():
    thickness = design()
    back = thickness.with_tau_coeff(thickness.tau_coeff)

    assert back.R_LE == pytest.approx(thickness.R_LE)
    assert back.tanwedge == pytest.approx(thickness.tanwedge)
    np.testing.assert_allclose(back.coeff, thickness.coeff, atol=1e-15)


def test_the_fields_round_trip_through_the_coefficients():
    """The other way round, which is the direction an iterator writes in."""
    thickness = design()
    c = thickness.tau_coeff.copy()
    c[:, 0] += 0.03
    c[0, 1] -= 0.10
    c[:, -1] += 0.05

    np.testing.assert_allclose(thickness.with_tau_coeff(c).tau_coeff, c, atol=1e-14)


def test_writing_coefficients_gives_back_plain_numbers():
    """A Node is written to a config file, and numpy scalars do not survive that."""
    back = design().with_tau_coeff(design().tau_coeff)

    assert isinstance(back.R_LE, float)
    assert all(isinstance(value, float) for value in back.tanwedge)
    assert all(isinstance(value, float) for row in back.coeff for value in row)


def test_moving_the_nose_leaves_the_interior_where_it_was():
    """The trap this method exists to absorb.

    `R_LE` sets one end of the straight line the perturbation is measured
    from, so moving it moves the line under the whole curve. A caller holding
    the fields and setting `R_LE` alone would drag every interior coefficient
    along behind the nose without meaning to; moving `c[0]` here changes the
    nose and nothing else.
    """
    thickness = design()
    c = thickness.tau_coeff.copy()
    c[:, 0] += 0.07

    moved = thickness.with_tau_coeff(c).tau_coeff

    assert moved[0][0] == pytest.approx(thickness.tau_coeff[0][0] + 0.07)
    np.testing.assert_allclose(
        moved[:, 1:], thickness.tau_coeff[:, 1:], atol=1e-14
    )


def test_the_two_surfaces_have_to_agree_at_the_nose():
    """One nose radius is what this class cannot do without.

    A caller who has broken it is asking for a section this cannot represent,
    so it says so rather than quietly taking the first row and discarding the
    other. The trailing edge is the opposite case --- see below.
    """
    thickness = design()
    c = thickness.tau_coeff.copy()
    c[1, 0] += 0.1

    with pytest.raises(ValueError, match="share one leading edge"):
        thickness.with_tau_coeff(c)


def test_the_two_surfaces_need_not_agree_at_the_trailing_edge():
    """Each surface leaves at its own wedge angle, if it is given one.

    Written through the coefficients rather than the field, which is the path
    an iterator takes: what comes back carries a pair, and reads back out the
    coefficients that were put in.
    """
    thickness = design()
    c = thickness.tau_coeff.copy()
    c[1, -1] += 0.1

    moved = thickness.with_tau_coeff(c)
    assert moved.tanwedge == pytest.approx(np.array(thickness.tanwedge) + [0.0, 0.1])
    np.testing.assert_allclose(moved.tau_coeff, c, atol=1e-14)


def test_a_split_wedge_leaves_each_surface_at_its_own_angle():
    """Which is what it is for, measured off the thickness rather than asserted."""
    m = np.linspace(0.9, 1.0, 5001)
    split = design(tanwedge=(0.30, 0.20), t_TE=0.02)

    slopes = [np.gradient(t, m)[-1] for t in split.thick_both(m)]

    # dt/dm at the trailing edge is -(tanwedge + t_TE / 2) on each surface.
    for slope, tanwedge in zip(slopes, split.tanwedge):
        assert slope == pytest.approx(-(tanwedge + 0.02 / 2.0), rel=1e-3)


def test_a_split_wedge_leaves_the_trailing_edge_where_it_was():
    """The point does not move; only the direction the surfaces reach it from.

    Half of `t_TE` sits on each surface at `m = 1` whatever the wedges do,
    that being the linear ramp rather than anything the shape space reaches.
    """
    for tanwedge in ((0.25, 0.25), (0.30, 0.20)):
        thickness = design(tanwedge=tanwedge, t_TE=0.02)
        assert thickness.thick_both(1.0) == pytest.approx((0.01, 0.01))


def test_a_level_pair_is_a_symmetric_trailing_edge_not_a_different_section():
    """Equal entries are how symmetry is asked for, and they stay a pair.

    A run that wants the loop to separate the two surfaces starts them level,
    so nothing may collapse that back to one number on the way through the
    coefficients --- the knob that would have separated them is the one that
    would go.
    """
    thickness = design(tanwedge=(0.25, 0.25))

    assert thickness.tanwedge == pytest.approx((0.25, 0.25))
    assert thickness.with_tau_coeff(thickness.tau_coeff).tanwedge == pytest.approx(
        (0.25, 0.25)
    )


def test_a_wedge_angle_is_written_once_per_surface():
    for tanwedge in ((0.3,), (0.3, 0.2, 0.1)):
        with pytest.raises(ValueError, match="per surface, so two of them"):
            design(tanwedge=tanwedge)


def test_a_split_wedge_round_trips_through_a_config_dict():
    """YAML holds the pair as a list, as it holds the coefficients."""
    thickness = design(tanwedge=(0.30, 0.20))
    assert ThicknessDesign.from_dict(thickness.to_dict()) == thickness
    assert thickness.to_dict()["tanwedge"] == [0.30, 0.20]


def test_coefficients_have_to_be_shaped_like_the_curve():
    with pytest.raises(ValueError, match="takes coefficients of shape"):
        design().with_tau_coeff(np.zeros((2, 7)))

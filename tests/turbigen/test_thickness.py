"""Tests for the two-sided thickness distribution.

`Clark` is the first distribution that is not symmetric about the camber
line, so besides its own shape these check what that costs the rest of the
package: a section of two rows has to interpolate over the span, and a blade
built from one has to count, shape and mesh like any other.

`Taylor` is tested through `test_blade.py`, where it was written.
"""

import numpy as np
import pytest
from test_blade import SPF, blade, build

from turbigen import Clark, ThicknessDesign, shapespace

CLARK = {
    "type": "clark",
    "R_LE": 0.01,
    "tanwedge": 0.25,
    "t_TE": 0.02,
    "coeff": [[0.3, 0.1], [-0.1, 0.05]],
}
"""A deliberately lopsided section: a fatter upper surface than lower."""

MESH = {"type": "h", "dm_TE": 0.05, "resolution_factor": 0.5, "dspf_mid": 0.1}
"""A coarse mesh, so that meshing a blade costs about a second."""

M = np.linspace(0.0, 1.0, 201)


def design(**kwargs):
    """A Clark thickness, with the parameters overridden as asked.

    Built the way a config file builds one, so that the coefficients arrive
    as the nested tuples a Node holds rather than as whatever was typed here.
    """
    return Clark.from_dict({**CLARK, **kwargs})


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


def test_the_wedge_is_the_closing_slope_of_both_surfaces():
    """One wedge angle, the camber line owning the trailing edge direction.

    Read on a closed trailing edge, where the wedge is the only thing setting
    the slope --- a finite `t_TE` steepens it by a further `t_TE / 2`.
    """
    tanwedge = 0.25
    m = 1.0 - np.array([1e-7, 2e-7])

    for side in design(tanwedge=tanwedge, t_TE=0.0).thick_both(m):
        slope = np.diff(side) / np.diff(m)
        np.testing.assert_allclose(slope, -tanwedge, rtol=1e-5)


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
        shapespace.tau_TE(CLARK["t_TE"], CLARK["tanwedge"]),
    )

    for coeff in (((), ()), ((2.0, -1.5), (-1.0, 3.0)), ((0.0, 9.0), (9.0, 0.0))):
        for side in design(coeff=coeff).tau(ends):
            np.testing.assert_allclose(side, expected, atol=1e-15)


def test_a_positive_coefficient_thickens_its_own_surface():
    """Which row is which surface: the first is the upper one."""
    upper, lower = design(coeff=((0.5, 0.5), (0.0, 0.0))).thick_both(0.5)
    assert upper > lower

    upper, lower = design(coeff=((0.0, 0.0), (0.5, 0.5))).thick_both(0.5)
    assert lower > upper


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
    assert isinstance(thickness, Clark)
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
    """A row of these is a row like any other, as far as the rest goes."""
    machine = build(blades=blades()).design()
    row = machine.rows[0]

    assert row.n_blade > 0
    for spf in SPF:
        xrtu, xrtl = row.blade.evaluate_section(spf)
        assert np.all(np.isfinite(xrtu))
        assert np.all(np.isfinite(xrtl))
        # Lopsided, so the surfaces are not reflections of one another about
        # the camber line the blade was built on.
        xrt = row.blade.evaluate_camber(spf)
        assert np.max(xrtu[2] - xrt[2]) > 2.0 * np.max(xrt[2] - xrtl[2])


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

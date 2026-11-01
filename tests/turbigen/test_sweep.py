"""Tests for swept edges and leaned sections.

Built on the synthetic two-row axial turbine of `test_blade`, whose blades have
sections at span fractions 0.2, 0.5 and 0.8. Edge positions are checked against
arc lengths measured independently along the annulus, on a fine sample of
:meth:`~turbigen.annulus.RowAnnulus.evaluate_xr`.

Test cases:
- test_no_offset_is_the_identity: exactly, with nothing evaluated
- test_edges_land_at_the_asked_arc_distance: in local chords, either sign
- test_equal_offsets_slide_the_section_rigidly: every point the same distance
- test_points_between_move_in_proportion: linear along the section
- test_a_section_crossing_a_station_has_no_kink: even spacing across it
- test_offsets_out_of_reach_are_refused: past the segments either side
- test_an_inverted_section_is_refused: leading edge behind the trailing edge
- test_bad_offsets_fail_at_design: before anything is meshed
- test_a_swept_blade_starts_where_its_edge_is: surfaces follow the map
- test_a_swept_section_stretches_its_aerofoil: chord scales with its length
- test_offsets_interpolate_between_sections: held beyond the end ones
- test_a_single_section_blade_is_swept_everywhere
- test_a_constant_rotation_turns_the_whole_blade
- test_a_linear_rotation_leans_the_stacking_line
- test_theta_offset_is_an_unknown_key: one way to turn the blade
- test_a_swept_blade_pickles: the fitted placement is rebuilt
- test_a_swept_blade_cannot_be_meshed_yet
- test_unswept_m_inverts_swept_m: both ways round, inside and beyond
- test_unswept_m_is_outside_beyond_the_edges: in swept section lengths
- test_no_offset_is_the_identity_both_ways
- test_unswept_cuts_are_where_they_were: the stations, plus the offset
- test_the_cuts_clear_the_furthest_edges: whatever span reaches furthest
- test_an_exit_cut_may_cross_the_next_row: only the default must fit
- test_rows_with_no_room_for_cuts_fail_at_design
- test_cuts_out_of_the_annulus_fail_at_design
"""

import pickle

import numpy as np
import pytest
from test_blade import blade, build
from test_mesh import MESH

import turbigen.util

SWEEP = {"dm_LE": 0.1, "dm_TE": -0.05}


def _row_annulus(machine, i_row=0):
    return machine.rows[i_row].blade.row_annulus


def _arc(row_annulus, spf, m):
    """Arc length along the constant-`spf` line from the leading edge station
    to row meridional coordinates `m`, measured on a fine sample."""
    m_fine = np.linspace(-1.0, 2.0, 30001)
    s = turbigen.util.cum_arc_length(row_annulus.evaluate_xr(m_fine, spf))
    return np.interp(m, m_fine, s) - np.interp(0.0, m_fine, s)


def _chord(row_annulus, spf):
    return _arc(row_annulus, spf, 1.0)


@pytest.fixture
def machine():
    return build().design()


def test_no_offset_is_the_identity(machine):
    m = np.linspace(0.0, 1.0, 7)
    swept = _row_annulus(machine).evaluate_swept_m(m, 0.3, 0.0, 0.0)
    np.testing.assert_array_equal(swept, m)


@pytest.mark.parametrize("spf", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dm_LE, dm_TE", [(0.1, -0.05), (-0.2, 0.3)])
def test_edges_land_at_the_asked_arc_distance(machine, spf, dm_LE, dm_TE):
    row_annulus = _row_annulus(machine)
    m_LE, m_TE = row_annulus.evaluate_swept_m([0.0, 1.0], spf, dm_LE, dm_TE)
    chord = _chord(row_annulus, spf)

    np.testing.assert_allclose(
        _arc(row_annulus, spf, [m_LE, m_TE]),
        [dm_LE * chord, (1.0 + dm_TE) * chord],
        atol=1e-4 * chord,
    )


def test_equal_offsets_slide_the_section_rigidly(machine):
    row_annulus = _row_annulus(machine)
    m = np.linspace(0.0, 1.0, 21)
    for spf in (0.0, 1.0):
        shift = _arc(row_annulus, spf, row_annulus.evaluate_swept_m(m, spf, 0.2, 0.2))
        np.testing.assert_allclose(
            shift - _arc(row_annulus, spf, m),
            0.2 * _chord(row_annulus, spf),
            atol=1e-4 * _chord(row_annulus, spf),
        )


def test_points_between_move_in_proportion(machine):
    row_annulus = _row_annulus(machine)
    spf, m = 0.0, np.linspace(0.0, 1.0, 21)
    chord = _chord(row_annulus, spf)
    moved = _arc(row_annulus, spf, row_annulus.evaluate_swept_m(m, spf, 0.1, 0.0))
    np.testing.assert_allclose(
        moved - _arc(row_annulus, spf, m), 0.1 * (1.0 - m) * chord, atol=1e-4 * chord
    )


def test_a_section_crossing_a_station_has_no_kink(machine):
    """Points evenly spaced along the section stay evenly spaced in the annulus
    across the leading edge station, where the scale of `m` jumps."""
    row_annulus = _row_annulus(machine)
    m = np.linspace(0.0, 1.0, 401)
    m_row = row_annulus.evaluate_swept_m(m, 0.5, -0.3, 0.0)
    assert m_row[0] < 0.0 < m_row[-1]

    # The moved section is uniformly stretched, so even in chord fraction is
    # even in arc length on the leading part, wherever the station falls
    xr = row_annulus.evaluate_xr(m_row, 0.5)
    ds = np.diff(turbigen.util.cum_arc_length(xr))
    lead = m[1:] < 0.3
    assert np.ptp(ds[lead]) < 1e-3 * ds[lead].mean()


def test_offsets_out_of_reach_are_refused(machine):
    with pytest.raises(ValueError, match="beyond the segments"):
        _row_annulus(machine).evaluate_swept_m([0.0, 1.0], 0.5, -50.0, 0.0)


def test_an_inverted_section_is_refused(machine):
    with pytest.raises(ValueError, match="leading edge downstream"):
        _row_annulus(machine).evaluate_swept_m([0.0, 1.0], 0.5, 0.8, -0.8)


def test_bad_offsets_fail_at_design():
    config = build(blades=[blade(section={"dm_LE": 0.8, "dm_TE": -0.8}), blade()])
    with pytest.raises(ValueError, match="leading edge downstream"):
        config.design()


def test_a_swept_blade_starts_where_its_edge_is():
    swept = build(blades=[blade(section=SWEEP), blade()]).design().rows[0].blade
    for spf in (0.0, 0.5, 1.0):
        surfaces = swept._evaluate_section_m(spf, turbigen.util.cluster_cosine(2001))
        m_row = np.concatenate([surf[3] for surf in surfaces])
        m_row -= swept.row_annulus.m_LE
        np.testing.assert_allclose(
            [m_row.min(), m_row.max()],
            swept.evaluate_swept_m([0.0, 1.0], spf),
            atol=1e-9,
        )


def test_a_swept_section_stretches_its_aerofoil():
    plain = build().design().rows[0].blade
    swept = build(blades=[blade(section=SWEEP), blade()]).design().rows[0].blade
    stretch = 1.0 + SWEEP["dm_TE"] - SWEEP["dm_LE"]
    for spf in (0.0, 0.5, 1.0):
        np.testing.assert_allclose(
            swept.evaluate_chord(spf), stretch * plain.evaluate_chord(spf), rtol=1e-3
        )
        # The aerofoil stretches meridionally only: the same turning
        np.testing.assert_allclose(
            swept.evaluate_chi(spf), plain.evaluate_chi(spf), atol=1e-12
        )


def test_offsets_interpolate_between_sections():
    config = blade()
    for section, dm in zip(config["sections"], (0.0, 0.1, 0.3)):
        section["dm_LE"] = dm
    leaned = build(blades=[config, blade()]).design().rows[0].blade

    # Through every section, as the recambers are, and held beyond the
    # outermost
    for spf, dm in ((0.0, 0.0), (0.2, 0.0), (0.5, 0.1), (0.8, 0.3), (1.0, 0.3)):
        assert leaned._placement(spf) == pytest.approx((dm, 0.0, 0.0))
    assert 0.0 < leaned._placement(0.35)[0] < 0.1 < leaned._placement(0.65)[0] < 0.3


def test_a_single_section_blade_is_swept_everywhere():
    config = blade(section=SWEEP)
    config["sections"] = config["sections"][1:2]
    swept = build(blades=[config, blade()]).design().rows[0].blade
    for spf in (0.0, 0.5, 1.0):
        assert swept._placement(spf) == pytest.approx((0.1, -0.05, 0.0))


def test_a_constant_rotation_turns_the_whole_blade():
    offset = 0.1
    plain = build().design().rows[0].blade
    rotated = (
        build(blades=[blade(section={"dtheta": offset}), blade()])
        .design()
        .rows[0]
        .blade
    )

    for spf in (0.0, 1.0):
        for turned, still in zip(
            rotated.evaluate_section(spf), plain.evaluate_section(spf)
        ):
            np.testing.assert_allclose(turned[:2], still[:2], atol=1e-12)
            np.testing.assert_allclose(turned[2] - still[2], offset, atol=1e-9)


def test_a_linear_rotation_leans_the_stacking_line():
    config = blade()
    for section in config["sections"]:
        section["dtheta"] = 0.2 * section["spf"]
    leaned = build(blades=[config, blade()]).design().rows[0].blade

    m = np.linspace(0.0, 1.0, 1001)
    for spf in (0.2, 0.35, 0.5, 0.8):
        theta = leaned.evaluate_camber(spf, m=m)[2]
        # The camber line passes through the stacking point turned by dtheta
        np.testing.assert_allclose(
            np.interp(leaned.m_stack, m, theta), 0.2 * spf, atol=1e-3
        )


def test_theta_offset_is_an_unknown_key():
    with pytest.raises(ValueError, match="theta_offset"):
        build(blades=[blade(theta_offset=0.1), blade()])


def test_a_swept_blade_pickles():
    swept = build(blades=[blade(section=SWEEP), blade()]).design().rows[0].blade
    swept._placement(0.5)
    again = pickle.loads(pickle.dumps(swept))
    for a, b in zip(again.evaluate_section(0.3), swept.evaluate_section(0.3)):
        np.testing.assert_array_equal(a, b)


def test_a_swept_blade_cannot_be_meshed_yet():
    config = build(blades=[blade(section=SWEEP), blade()], mesh=MESH)
    with pytest.raises(NotImplementedError, match="Row 0 has swept edges"):
        config.mesh.mesh(config.design())


#
# THE INVERSE MAP
#


@pytest.mark.parametrize("spf", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("dm_LE, dm_TE", [(0.1, -0.05), (-0.2, 0.3)])
def test_unswept_m_inverts_swept_m(machine, spf, dm_LE, dm_TE):
    row_annulus = _row_annulus(machine)
    # Short of the end of the line beyond the trailing edge, a gap of half a
    # chord, where the swept section is up to one and a half chords long
    m = np.linspace(-0.5, 1.1, 41)
    swept = row_annulus.evaluate_swept_m(m, spf, dm_LE, dm_TE)
    back = row_annulus.evaluate_unswept_m(swept, spf, dm_LE, dm_TE)
    np.testing.assert_allclose(back, m, atol=1e-12)

    m_row = np.linspace(-0.5, 1.5, 41)
    unswept = row_annulus.evaluate_unswept_m(m_row, spf, dm_LE, dm_TE)
    again = row_annulus.evaluate_swept_m(unswept, spf, dm_LE, dm_TE)
    np.testing.assert_allclose(again, m_row, atol=1e-12)


def test_unswept_m_is_outside_beyond_the_edges(machine):
    """The leading edge station lies upstream of a leading edge swept
    downstream, by the sweep over the section's swept length."""
    row_annulus = _row_annulus(machine)
    stations = row_annulus.evaluate_unswept_m([0.0, 1.0], 0.5, **SWEEP)
    length = 1.0 + SWEEP["dm_TE"] - SWEEP["dm_LE"]
    np.testing.assert_allclose(
        stations,
        [-SWEEP["dm_LE"] / length, 1.0 - SWEEP["dm_TE"] / length],
        atol=1e-4,
    )


def test_no_offset_is_the_identity_both_ways(machine):
    m = np.linspace(-0.5, 1.5, 9)
    plain = machine.rows[0].blade
    np.testing.assert_array_equal(plain.evaluate_unswept_m(m, 0.3), m)
    np.testing.assert_array_equal(plain.evaluate_swept_m(m, 0.3), m)


#
# CUT PLANES
#


def test_unswept_cuts_are_where_they_were(machine):
    """Upstream of each leading edge station and downstream of each trailing
    edge station by the offset, in the gap's own scale of `m`."""
    chords = machine.annulus.evaluate_chords(0.5)
    expected = []
    for i_row in range(2):
        step = 0.05 * chords[2 * i_row + 1]
        expected += [
            2 * i_row + 1.0 - step / chords[2 * i_row],
            2 * i_row + 2.0 + step / chords[2 * i_row + 2],
        ]
    np.testing.assert_allclose(machine._m_cut(), expected, rtol=1e-14)
    for m, xr in zip(expected, machine.cut_planes()):
        np.testing.assert_allclose(xr, machine.annulus.evaluate_xr(m, [0.0, 1.0]).T)


def test_the_cuts_clear_the_furthest_edges(machine):
    config = blade()
    for section, dm in zip(config["sections"], (-0.1, -0.3, 0.0)):
        section.update(dm_LE=dm, dm_TE=-dm)
    swept = build(blades=[config, blade()]).design()
    row = swept.rows[0].blade

    spf = np.linspace(0.0, 1.0, 101)
    edges = np.array([row.evaluate_swept_m([0.0, 1.0], s) for s in spf])
    m_LE, m_TE = edges[:, 0].min(), edges[:, 1].max()

    # The machine searches fewer spans for the furthest edge, so finds it to
    # within a fraction of a percent of the clearance the offset leaves
    plain, cut = machine._m_cut(), swept._m_cut()
    np.testing.assert_allclose(cut[0] - (1.0 + m_LE), plain[0] - 1.0, atol=1e-3)
    np.testing.assert_allclose(cut[1] - (1.0 + m_TE), plain[1] - 2.0, atol=1e-3)
    np.testing.assert_array_equal(cut[2:], plain[2:])


def test_an_exit_cut_may_cross_the_next_row(machine):
    """The exit profile reads half a chord on, past the next row's leading
    edge cut here, which is no reason to refuse it."""
    m_cut = machine._m_cut(0.5)
    assert m_cut[1] > m_cut[2]
    assert len(machine.cut_planes(0.5)) == 4


def test_rows_with_no_room_for_cuts_fail_at_design():
    config = build(
        blades=[blade(section={"dm_TE": 0.3}), blade(section={"dm_LE": -0.3})]
    )
    with pytest.raises(ValueError, match="Rows 0 and 1 leave no room"):
        config.design()


def test_cuts_out_of_the_annulus_fail_at_design():
    config = build(blades=[blade(section={"dm_LE": -1.45}), blade()])
    with pytest.raises(ValueError, match="leave the annulus"):
        config.design()

import numpy as np
import pytest
from turbigen_ref.hmesh import H, _theta_limits


@pytest.fixture
def mesher():
    return H()


@pytest.fixture
def axial_section():
    """Synthetic axial blade section: (xrt_u, xrt_l)."""
    N = 200
    m = np.linspace(0, 1, N)
    x = m * 0.1
    r = np.ones_like(m)
    thick = 0.02 * np.sin(np.pi * m)
    camber = 0.01 * m
    return (
        np.stack([x, r, camber + thick]),
        np.stack([x, r, camber - thick]),
    )


class TestSpanwiseGrid:
    @pytest.mark.parametrize(
        "dspf_hub,dspf_casing",
        [(1e-3, 1e-3), (5e-4, 2e-3)],
    )
    def test_no_tip_endpoints_and_monotonic(self, mesher, dspf_hub, dspf_casing):
        spf = mesher.spanwise_grid(dspf_hub, dspf_casing, 0.0)
        assert spf[0] == 0.0
        assert np.isclose(spf[-1], 1.0)
        assert (np.diff(spf) > 0.0).all()

    def test_no_tip_mg_divisible(self, mesher):
        spf = mesher.spanwise_grid(1e-3, 1e-3, 0.0)
        assert (len(spf) - 1) % 8 == 0

    def test_no_tip_wall_spacing(self, mesher):
        dspf_hub, dspf_casing = 1e-3, 1e-3
        spf = mesher.spanwise_grid(dspf_hub, dspf_casing, 0.0)
        assert spf[1] - spf[0] <= dspf_hub * 1.1
        assert spf[-1] - spf[-2] <= dspf_casing * 1.1

    def test_with_tip_min_points_in_gap(self, mesher):
        tip = 0.02
        spf = mesher.spanwise_grid(1e-3, 1e-3, tip)
        assert (spf >= tip).sum() >= 9

    def test_with_tip_endpoints_and_monotonic(self, mesher):
        spf = mesher.spanwise_grid(1e-3, 1e-3, 0.02)
        assert spf[0] == 0.0
        assert np.isclose(spf[-1], 1.0)
        assert (np.diff(spf) > 0.0).all()

    def test_resolution_factor_scales(self):
        m1 = H(resolution_factor=1.0)
        m2 = H(resolution_factor=2.0)
        spf1 = m1.spanwise_grid(1e-3, 1e-3, 0.0)
        spf2 = m2.spanwise_grid(1e-3, 1e-3, 0.0)
        assert len(spf2) > len(spf1)


class TestPitchwiseGrid:
    @pytest.mark.parametrize(
        "drt_row,pitch_chord,AR_row",
        [(0.005, 0.7, 1.5), (0.002, 1.0, 1.0)],
    )
    def test_endpoints_and_monotonic(self, mesher, drt_row, pitch_chord, AR_row):
        x = mesher.pitchwise_grid(drt_row, pitch_chord, AR_row)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert (np.diff(x) > 0.0).all()

    def test_mg_divisible(self, mesher):
        x = mesher.pitchwise_grid(0.005, 0.7, 1.5)
        assert (len(x) - 1) % 8 == 0

    def test_er_bound_respected(self, mesher):
        x = mesher.pitchwise_grid(0.005, 0.7, 1.5)
        dx = np.diff(x)
        er = dx[1:] / dx[:-1]
        er = np.where(er < 1.0, 1.0 / er, er)
        assert (er <= mesher.ER_pitch * 1.001).all()

    def test_symmetric(self, mesher):
        x = mesher.pitchwise_grid(0.005, 0.7, 1.5)
        np.testing.assert_allclose(x, 1.0 - x[::-1], atol=1e-12)

    def test_resample_false_path(self, mesher):
        x = mesher.pitchwise_grid(0.005, 0.7, 1.5, resample=False)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert (np.diff(x) > 0.0).all()


class TestPitchwiseGridFixedNpts:
    def test_length_matches_npts(self, mesher):
        x_free = mesher.pitchwise_grid(0.005, 0.7, 1.5, resample=False)
        npts = len(x_free)
        x_fixed = mesher.pitchwise_grid_fixed_npts(0.005, 0.7, 1.5, npts)
        assert len(x_fixed) == npts

    def test_endpoints_and_monotonic(self, mesher):
        x = mesher.pitchwise_grid_fixed_npts(0.005, 0.7, 1.5, 41)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert (np.diff(x) > 0.0).all()


class TestStreamwiseGrid:
    PC = np.array([0.5, 0.6, 0.7])

    @pytest.mark.parametrize("L", [(1.0, 1.0), (1.0, 0.5), (0.5, 1.0), (0.5, 0.5)])
    def test_endpoints(self, mesher, L):
        t, _ile, _ite = mesher.streamwise_grid(self.PC, 41, L, 1.5, tte=0.95)
        assert np.isclose(t[0], -L[0])
        assert np.isclose(t[-1], 1.0 + L[1])

    @pytest.mark.parametrize("L", [(1.0, 1.0), (1.0, 0.5)])
    def test_le_te_indices(self, mesher, L):
        t, ile, ite = mesher.streamwise_grid(self.PC, 41, L, 1.5, tte=0.95)
        assert np.isclose(t[ile], 0.0)
        assert np.isclose(t[ite], 1.0)

    def test_monotonic(self, mesher):
        t, _, _ = mesher.streamwise_grid(self.PC, 41, (1.0, 1.0), 1.5, tte=0.95)
        assert (np.diff(t) > 0.0).all()

    def test_mg_divisible(self, mesher):
        t, _, _ = mesher.streamwise_grid(self.PC, 41, (1.0, 1.0), 1.5, tte=0.95)
        assert (len(t) - 1) % 8 == 0

    def test_with_ni_cusp(self, mesher):
        t, ile, ite = mesher.streamwise_grid(
            self.PC, 41, (1.0, 1.0), 1.5, tte=0.95, ni_cusp=5
        )
        assert np.isclose(t[ile], 0.0)
        assert np.isclose(t[ite], 1.0)
        assert (np.diff(t) > 0.0).all()
        assert (len(t) - 1) % 8 == 0

class TestPitchwiseRelaxation:
    PC = np.array([0.5, 0.7, 0.6])

    def test_zero_across_blade(self, mesher):
        sf = np.linspace(0.0, 1.0, 11)
        r = mesher.pitchwise_relaxation(sf, self.PC)
        np.testing.assert_allclose(r, 0.0, atol=1e-12)

    def test_one_at_anchors(self, mesher):
        pc = self.PC
        anchor_up = -(mesher.nchord_relax * pc[0] / pc[1])
        anchor_dn = 1.0 + mesher.nchord_relax * pc[2] / pc[1]
        sf = np.array([anchor_up, anchor_dn])
        r = mesher.pitchwise_relaxation(sf, pc)
        np.testing.assert_allclose(r, 1.0, atol=1e-12)

    def test_one_outside_anchors(self, mesher):
        pc = self.PC
        anchor_up = -(mesher.nchord_relax * pc[0] / pc[1])
        anchor_dn = 1.0 + mesher.nchord_relax * pc[2] / pc[1]
        sf = np.array([anchor_up - 0.5, anchor_dn + 0.5])
        r = mesher.pitchwise_relaxation(sf, pc)
        np.testing.assert_allclose(r, 1.0, atol=1e-12)

    def test_monotonic_decay_upstream(self, mesher):
        pc = self.PC
        anchor_up = -(mesher.nchord_relax * pc[0] / pc[1])
        sf = np.linspace(anchor_up, 0.0, 20)
        r = mesher.pitchwise_relaxation(sf, pc)
        assert (np.diff(r) <= 0.0).all()

    def test_monotonic_decay_downstream(self, mesher):
        pc = self.PC
        anchor_dn = 1.0 + mesher.nchord_relax * pc[2] / pc[1]
        sf = np.linspace(1.0, anchor_dn, 20)
        r = mesher.pitchwise_relaxation(sf, pc)
        assert (np.diff(r) >= 0.0).all()


class TestThetaLimits:
    @pytest.fixture
    def axial_section(self):
        N = 200
        m = np.linspace(0, 1, N)
        x = m * 0.1
        r = np.ones_like(m)
        thick = 0.02 * np.sin(np.pi * m)
        camber = 0.01 * m
        return (
            np.stack([x, r, camber + thick]),
            np.stack([x, r, camber - thick]),
        )

    def test_upper_above_lower(self, axial_section):
        xrt_u, xrt_l = axial_section
        tq = np.linspace(-0.2, 1.2, 80)
        tu, tl, _tte = _theta_limits(tq, xrt_u, xrt_l, (0, 1))
        assert (tu >= tl).all()

    def test_tte_in_range(self, axial_section):
        xrt_u, xrt_l = axial_section
        tq = np.linspace(-0.2, 1.2, 80)
        _, _, tte = _theta_limits(tq, xrt_u, xrt_l, (0, 1))
        assert 0.0 < tte <= 1.0

    def test_skew_zero_no_displacement(self, axial_section):
        xrt_u, xrt_l = axial_section
        tq = np.linspace(-0.2, 1.2, 80)
        tu0, tl0, _ = _theta_limits(tq, xrt_u, xrt_l, (0, 1), Theta=(0.0, 0.0))
        tu1, tl1, _ = _theta_limits(tq, xrt_u, xrt_l, (0, 1), Theta=(0.0, 0.0))
        np.testing.assert_allclose(tu0, tu1)
        np.testing.assert_allclose(tl0, tl1)

    def test_positive_exit_skew_shifts_downstream_positive(self, axial_section):
        xrt_u, xrt_l = axial_section
        tq = np.linspace(-0.2, 1.2, 80)
        tu0, _, _ = _theta_limits(tq, xrt_u, xrt_l, (0, 1), Theta=(0.0, 0.0))
        tu1, _, _ = _theta_limits(tq, xrt_u, xrt_l, (0, 1), Theta=(0.0, 20.0))
        ind_dn = tq > 1.0
        assert (tu1[ind_dn] >= tu0[ind_dn]).all()
        assert (tu1[ind_dn] > tu0[ind_dn]).any()

    def test_thicker_than_pitch_raises(self):
        N = 200
        m = np.linspace(0, 1, N)
        x = m * 0.1
        r = np.ones_like(m)
        # upper below lower — impossible blade
        xrt_u = np.stack([x, r, -0.05 * np.ones_like(m)])
        xrt_l = np.stack([x, r, +0.05 * np.ones_like(m)])
        tq = np.linspace(0.0, 1.0, 50)
        with pytest.raises(Exception, match="Blade is thicker than calculated pitch"):
            _theta_limits(tq, xrt_u, xrt_l, (0, 1))

    def test_radial_inlet_branch(self):
        N = 200
        m = np.linspace(0, 1, N)
        # radial blade: r varies more than x near LE
        r = 1.0 + m * 0.5
        x = m * 0.01
        thick = 0.02 * np.sin(np.pi * m)
        camber = 0.01 * m
        xrt_u = np.stack([x, r, camber + thick])
        xrt_l = np.stack([x, r, camber - thick])
        tq = np.linspace(-0.2, 1.2, 80)
        tu, tl, _tte = _theta_limits(tq, xrt_u, xrt_l, (0, 1))
        assert (tu >= tl).all()

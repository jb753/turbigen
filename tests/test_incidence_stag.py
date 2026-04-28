"""Test that incidence_unstructured extracts stagnation coordinates via get_i_stag."""

import ember.block
import ember.cut
import ember.fluid
import ember.grid
import ember.util
import numpy as np

from turbigen.util_post import get_i_stag


def make_blade_surf(ni=41, nj=11, i_stag_true=20):
    """Construct a minimal 2D blade surface block (ni, nj, 1).

    i goes TE->suction->LE->pressure->TE so the LE is near the middle.
    A Gaussian pressure peak is planted at i_stag_true.
    """
    shape = (ni, nj, 1)
    xrt = ember.util.linmesh3([0.0, 1.0], [1.0, 1.5], [0.0, 0.0], shape)
    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)
    block.set_Omega(0.0)

    i_coords = np.arange(ni)
    P_base = np.exp(-((i_coords - i_stag_true) ** 2) / 20.0) * 1e5 + 1e5
    P = P_base[:, None, None] * np.ones((1, nj, 1))
    rho = np.ones(shape) * 1.2
    Vx = np.ones(shape) * 50.0
    Vr = np.zeros(shape)
    Vt = np.zeros(shape)
    block.set_primitive(rho, Vx, Vr, Vt, P)
    return block


def test_get_i_stag_on_meridional_cut():
    """structured_meridional cut then get_i_stag returns correct stag coordinates."""
    ni, nj = 41, 11
    i_stag_true = 20
    surf = make_blade_surf(ni=ni, nj=nj, i_stag_true=i_stag_true)

    # Meridional cut midway through the radial extent of the block
    r_mid = 0.5 * (surf.r.min() + surf.r.max())
    xr_cut = np.array([[surf.x.min(), r_mid], [surf.x.max(), r_mid]])
    C = ember.cut.structured_meridional(surf, xr_cut)
    assert len(C) == 1, "Expected exactly one cut block"

    cut = C[0]
    assert cut.ndim == 2

    istag = get_i_stag(cut)[0]
    xrt_stag = cut.xrt[istag, 0, :]

    # Stag point should be at the planted pressure maximum
    assert istag == i_stag_true
    # And its x-coordinate should match the surface
    np.testing.assert_allclose(xrt_stag[0], surf.x[i_stag_true, nj // 2, 0], atol=1e-6)

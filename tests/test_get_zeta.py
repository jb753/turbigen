import ember.block
import ember.util
import numpy as np
import pytest
from turbigen_ref.util_post import get_zeta


def test_get_zeta_1d_straight_line():
    """Test get_zeta on a 1D block (single gridline) with straight line geometry."""
    # Create a straight line from x=0 to x=1
    ni = 11
    shape = (ni, 1, 1)
    xrt = ember.util.linmesh3([0, 1], [1, 1], [0, 0], shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # Check zeta starts at zero
    assert zeta[0, 0, 0] == pytest.approx(0.0)

    # Check zeta ends at 1.0 (length of line)
    assert zeta[-1, 0, 0] == pytest.approx(1.0)

    # Check zeta increases monotonically
    assert np.all(np.diff(zeta[:, 0, 0]) > 0)

    # Check zeta is equally spaced for uniform grid
    dz = np.diff(zeta[:, 0, 0])
    np.testing.assert_allclose(dz, dz[0], rtol=1e-6)


def test_get_zeta_1d_circular_arc():
    """Test get_zeta on a 1D block with circular arc geometry."""
    # Create a quarter circle in r-t plane (x constant)
    ni = 101
    shape = (ni, 1, 1)

    block = ember.block.Block(shape=shape)
    x = np.zeros((ni, 1, 1))
    r = np.ones((ni, 1, 1))  # Radius = 1
    t = np.linspace(0, np.pi / 2, ni)[:, None, None]  # Quarter circle
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # Check zeta starts at zero
    assert zeta[0, 0, 0] == pytest.approx(0.0)

    # Arc length of quarter circle = pi*r/2
    expected_length = np.pi / 2
    assert zeta[-1, 0, 0] == pytest.approx(expected_length, rel=1e-3)

    # Check zeta increases monotonically
    assert np.all(np.diff(zeta[:, 0, 0]) > 0)


def test_get_zeta_2d_uniform_lines():
    """Test get_zeta on a 2D block with multiple parallel gridlines."""
    # Create multiple straight lines at different radii
    ni, nj = 10, 5
    shape = (ni, nj, 1)

    # Lines from x=0 to x=2, at different radii
    x = np.linspace(0, 2, ni)[:, None, None]
    r = np.linspace(1, 2, nj)[None, :, None]
    t = np.zeros((ni, nj, 1))

    block = ember.block.Block(shape=shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # All j-lines should start at zero
    np.testing.assert_allclose(zeta[0, :, 0], 0.0)

    # All j-lines should have same arc length (parallel lines)
    expected_length = 2.0
    np.testing.assert_allclose(zeta[-1, :, 0], expected_length, rtol=1e-10)

    # Each j-line should increase monotonically
    for j in range(nj):
        assert np.all(np.diff(zeta[:, j, 0]) > 0)


def test_get_zeta_2d_varying_lengths():
    """Test get_zeta on a 2D block where gridlines have different lengths."""
    # Create lines of different lengths in x-direction at fixed r
    ni, nj = 10, 3
    shape = (ni, nj, 1)

    block = ember.block.Block(shape=shape)

    # Line lengths: 1.0, 2.0, 3.0
    lengths = np.array([1.0, 2.0, 3.0])

    # Lines extend different distances in x, all at r=1
    x = np.linspace(0, 1, ni)[:, None, None] * lengths[None, :, None]
    r = np.ones((ni, nj, 1))
    t = np.zeros((ni, nj, 1))

    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # All lines start at zero
    np.testing.assert_allclose(zeta[0, :, 0], 0.0)

    # Each line should have its expected length
    for j in range(nj):
        assert zeta[-1, j, 0] == pytest.approx(lengths[j], rel=1e-6)


def test_get_zeta_3d_independent_gridlines():
    """Test get_zeta on a 3D block to verify each (j,k) gridline is independent."""
    ni, nj, nk = 8, 3, 4
    shape = (ni, nj, nk)

    # Create gridlines with varying x-extent based on j and k
    i_coords = np.linspace(0, 1, ni)

    # Create a pattern where length varies with j and k
    x = i_coords[:, None, None] * (
        1.0 + 0.5 * np.arange(nj)[None, :, None] + 0.2 * np.arange(nk)[None, None, :]
    )
    r = np.ones((ni, nj, nk))
    t = np.zeros((ni, nj, nk))

    block = ember.block.Block(shape=shape)
    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # All gridlines start at zero
    np.testing.assert_allclose(zeta[0, :, :], 0.0)

    # Each (j,k) gridline should have different ending zeta
    end_zeta = zeta[-1, :, :]
    # Should have variation across j and k
    assert np.std(end_zeta) > 0.1

    # Each gridline should increase monotonically
    for j in range(nj):
        for k in range(nk):
            assert np.all(np.diff(zeta[:, j, k]) > 0)


def test_get_zeta_helix():
    """Test get_zeta on a helical path (varying x, r, and t)."""
    ni = 51
    shape = (ni, 1, 1)

    # Create a helix: advancing in x and rotating in t
    block = ember.block.Block(shape=shape)

    i_coords = np.linspace(0, 1, ni)
    x = i_coords[:, None, None]
    r = np.ones((ni, 1, 1))
    t = 2 * np.pi * i_coords[:, None, None]  # One full rotation

    block.set_x(x)
    block.set_r(r)
    block.set_t(t)

    zeta = get_zeta(block)

    # Check shape
    assert zeta.shape == shape

    # Check zeta starts at zero
    assert zeta[0, 0, 0] == pytest.approx(0.0)

    # Helix length: sqrt(h^2 + (2*pi*r)^2) where h=1, r=1
    expected_length = np.sqrt(1.0**2 + (2 * np.pi) ** 2)
    assert zeta[-1, 0, 0] == pytest.approx(expected_length, rel=1e-3)

    # Check zeta increases monotonically
    assert np.all(np.diff(zeta[:, 0, 0]) > 0)


def test_get_zeta_single_point():
    """Test get_zeta on a degenerate case with single point."""
    shape = (1, 1, 1)
    xrt = ember.util.linmesh3([0, 0], [1, 1], [0, 0], shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)

    zeta = get_zeta(block)

    # Single point should have zeta = 0
    assert zeta.shape == shape
    assert zeta[0, 0, 0] == pytest.approx(0.0)


def test_get_zeta_two_points():
    """Test get_zeta on minimal case with two points."""
    shape = (2, 1, 1)
    xrt = ember.util.linmesh3([0, 1], [1, 1], [0, 0], shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)

    zeta = get_zeta(block)

    assert zeta.shape == shape
    assert zeta[0, 0, 0] == pytest.approx(0.0)
    assert zeta[1, 0, 0] == pytest.approx(1.0)

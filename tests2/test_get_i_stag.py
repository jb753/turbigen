import ember.block
import ember.fluid
import ember.util
import numpy as np
import pytest
from turbigen.util_post import get_i_stag


def test_get_i_stag_simple_2d():
    """Test get_i_stag on a simple 2D block with clear pressure maximum."""
    # Create a 2D block
    ni, nj = 21, 5
    shape = (ni, nj, 1)

    # Create straight line geometry
    xrt = ember.util.linmesh3([0, 1], [1, 1.5], [0, 0], shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)

    # Set fluid
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Create artificial pressure field with maximum at i=10 (middle)
    # Use Gaussian-like distribution centered at i=10
    i_coords = np.arange(ni)
    P = np.exp(-((i_coords - 10) ** 2) / 10.0)
    P = P[:, None, None] * np.ones((1, nj, 1))

    # Set primitive variables (rho, Vx, Vr, Vt, P)
    rho = np.ones((ni, nj, 1)) * 1.2  # kg/m^3
    Vx = np.ones((ni, nj, 1)) * 50.0  # m/s
    Vr = np.zeros((ni, nj, 1))
    Vt = np.zeros((ni, nj, 1))
    block.set_primitive(rho, Vx, Vr, Vt, P * 1e5)
    block.set_Omega(0.0)  # Non-rotating, so P_rot = P

    # Squeeze to 2D before calling get_i_stag
    i_stag = get_i_stag(block.squeeze())

    # Check shape
    assert i_stag.shape == (nj,)

    # All j-lines should find stagnation at i=10
    np.testing.assert_array_equal(i_stag, 10)


def test_get_i_stag_shape_validation():
    """Test that get_i_stag raises ValueError for non-2D blocks."""
    # Test 3D block (doesn't squeeze to 2D)
    shape_3d = (10, 5, 5)
    xrt_3d = ember.util.linmesh3([0, 1], [1, 2], [0, 0.1], shape_3d)
    block_3d = ember.block.Block(shape=shape_3d).set_xrt(xrt_3d)

    with pytest.raises(ValueError, match="Can only find stagnation point on 2D cuts"):
        get_i_stag(block_3d.squeeze())

    # Test 1D block (squeezes to 1D, should raise)
    shape_1d = (10, 1, 1)
    xrt_1d = ember.util.linmesh3([0, 1], [1, 1], [0, 0], shape_1d)
    block_1d = ember.block.Block(shape=shape_1d).set_xrt(xrt_1d)

    with pytest.raises(ValueError, match="Can only find stagnation point on 2D cuts"):
        get_i_stag(block_1d.squeeze())


def test_get_i_stag_multiple_j_lines():
    """Test get_i_stag with different stagnation points on different j-lines."""
    ni, nj = 21, 4
    shape = (ni, nj, 1)

    xrt = ember.util.linmesh3([0, 1], [1, 1.5], [0, 0], shape)

    block = ember.block.Block(shape=shape)
    block.set_xrt(xrt)

    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1e-5, Pr=0.72)
    block.set_fluid(fluid)

    # Create pressure field with different maxima on each j-line
    # j=0: max at i=8, j=1: max at i=9, j=2: max at i=10, j=3: max at i=11
    i_coords = np.arange(ni)
    P = np.zeros((ni, nj, 1))

    for j in range(nj):
        i_max = 8 + j
        P[:, j, 0] = np.exp(-((i_coords - i_max) ** 2) / 10.0)

    # Set primitive variables
    rho = np.ones((ni, nj, 1)) * 1.2
    Vx = np.ones((ni, nj, 1)) * 50.0
    Vr = np.zeros((ni, nj, 1))
    Vt = np.zeros((ni, nj, 1))
    block.set_primitive(rho, Vx, Vr, Vt, P * 1e5)
    block.set_Omega(0.0)

    i_stag = get_i_stag(block.squeeze())

    # Check that each j-line found its own maximum
    expected = np.array([8, 9, 10, 11])
    np.testing.assert_array_equal(i_stag, expected)

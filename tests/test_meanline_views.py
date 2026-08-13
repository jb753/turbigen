"""Every documented way of indexing a MeanLine must yield a writeable view.

The (2, n_row) shape is only worth having if slicing it gives live views of the
parent's storage rather than detached copies: the mix-out loop writes back
through ``ml.flat``, the annulus reads through it, and the row/station forms
are how the design functions address individual planes. A silent copy anywhere
in that table would lose writes without raising, so pin the whole table.
"""

import numpy as np
import pytest

import ember.fluid
from turbigen.meanline_new import MeanLine

N_ROW = 3


# Every indexing form documented in the MeanLine module docstring, keyed by the
# expression a caller would write.
VIEWS = {
    "ml.flat": lambda ml: ml.flat,
    "ml[0]": lambda ml: ml[0],
    "ml[1]": lambda ml: ml[1],
    "ml[:, 1]": lambda ml: ml[:, 1],
    "ml.row(1)": lambda ml: ml.row(1),
    "ml[0, 1]": lambda ml: ml[0, 1],
    "ml[1, 2]": lambda ml: ml[1, 2],
    "ml.inlet": lambda ml: ml.inlet,
    "ml.outlet": lambda ml: ml.outlet,
    "ml.flat[2]": lambda ml: ml.flat[2],
}


@pytest.fixture
def ml():
    """A fully initialised 3-row mean line, built without using any view."""
    fluid = ember.fluid.PerfectFluid(cp=1005.0, gamma=1.4, mu=1.8e-5, Pr=0.72)
    out = MeanLine(N_ROW)
    out.set_fluid(fluid)
    out.set_r(0.5)
    out.set_Am(1.0)
    out.set_P_T(1e5, 300.0)
    out.set_Vx(100.0)
    out.set_Vr(0.0)
    out.set_Vt(50.0)
    return out


@pytest.mark.parametrize("expr", list(VIEWS))
def test_indexing_shares_storage_with_parent(ml, expr):
    """Each indexing form returns something backed by the parent's array."""
    view = VIEWS[expr](ml)
    assert isinstance(view, MeanLine)
    assert np.shares_memory(view._data, ml._data), f"{expr} is a copy, not a view"


@pytest.mark.parametrize("expr", list(VIEWS))
def test_write_through_view_reaches_parent(ml, expr):
    """A setter called on a view modifies exactly its own stations of the parent."""
    view = VIEWS[expr](ml)
    n_view = view.size

    # Geometry setter: our own, backed by the added Am data key.
    view.set_Am(7.0)
    assert np.count_nonzero(ml.Am == 7.0) == n_view, f"{expr}: set_Am did not stick"
    assert np.count_nonzero(ml.Am != 7.0) == ml.size - n_view

    # Coordinate setter inherited from Block.
    view.set_r(9.0)
    assert np.count_nonzero(ml.r == 9.0) == n_view, f"{expr}: set_r did not stick"

    # Thermodynamic setter inherited from Block.
    view.set_P_T(2e5, 400.0)
    assert np.count_nonzero(np.isclose(ml.T, 400.0)) == n_view, (
        f"{expr}: set_P_T did not stick"
    )

    # Nodal Omega, which ember holds as scalar block metadata and we do not.
    view.set_Omega(1234.0)
    assert np.count_nonzero(ml.Omega == 1234.0) == n_view, (
        f"{expr}: set_Omega did not stick"
    )


@pytest.mark.parametrize("expr", list(VIEWS))
def test_parent_write_reaches_view(ml, expr):
    """The sharing runs both ways: a write to the parent shows up in the view."""
    view = VIEWS[expr](ml)
    ml.set_r(3.0)
    np.testing.assert_allclose(view.r, 3.0)


def test_flat_is_streamwise(ml):
    """ml.flat runs from machine inlet to machine outlet, row by row."""
    # Label each station with its streamwise position via the 2-D form, so the
    # expected ordering is not built with the flat view under test.
    label = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])  # [station, row]
    ml.set_Am(label + 1.0)
    np.testing.assert_allclose(ml.flat.Am, np.arange(6.0) + 1.0)


def test_inlet_and_outlet_are_the_machine_endpoints(ml):
    ml.set_Am(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert ml.inlet.Am == pytest.approx(1.0)  # row 0 inlet
    assert ml.outlet.Am == pytest.approx(6.0)  # last row outlet


def test_row_and_station_views_address_the_right_stations(ml):
    ml.set_Am(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(ml.row(1).Am, [2.0, 5.0])
    np.testing.assert_allclose(ml[0].Am, [1.0, 2.0, 3.0])  # every row's inlet
    np.testing.assert_allclose(ml[1].Am, [4.0, 5.0, 6.0])  # every row's outlet
    assert ml[0, 2].Am == pytest.approx(3.0)
    assert ml[1, 0].Am == pytest.approx(4.0)


def test_per_row_omega_survives_slicing(ml):
    """Omega is nodal data, so row views carry their own blade speed.

    As block metadata it could not: ember's view() shares the metadata dict,
    so setting Omega on one row would clobber every other row.
    """
    ml.set_Omega_row([0.0, 1000.0, 2000.0])
    assert np.all(ml.row(0).Omega == 0.0)
    assert np.all(ml.row(1).Omega == 1000.0)
    assert np.all(ml.row(2).Omega == 2000.0)

    # Setting one row must not disturb its neighbours.
    ml.row(1).set_Omega(50.0)
    assert np.all(ml.row(0).Omega == 0.0)
    assert np.all(ml.row(1).Omega == 50.0)
    assert np.all(ml.row(2).Omega == 2000.0)

    # Relative-frame properties follow Omega per row rather than per block.
    assert ml.row(2).U == pytest.approx(2000.0 * ml.row(2).r)
    assert np.all(ml.row(0).U == 0.0)


def test_flat_raises_rather_than_copying_a_strided_slice(ml):
    """A slice with no contiguous flattening must raise, not silently copy."""
    with pytest.raises(ValueError, match="without copying"):
        ml[:, ::2].flat

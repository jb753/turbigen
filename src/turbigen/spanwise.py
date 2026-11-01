"""Spanwise distribution of the flow the blades are designed against.

The mean line holds one value of each flow variable per station, at the
root-mean-square radius. A blade needs a flow angle at every span fraction,
because its metal angle is a recamber measured off the local flow, so something
has to say how the flow varies from hub to casing. That is this stage, read
from the ``spanwise:`` key of a config.

There are two alternatives.

:class:`VortexExpon` is a power-law swirl distribution, :math:`V_\\theta
\\propto r^n`, with the meridional velocity uniform over the span. It is the
default, with :math:`n=-1`, a free vortex.

:class:`VelocitySpline` holds a spanwise profile of meridional velocity and of
angular momentum at *every* station, the inlet and outlet of each row, as
values at that row's blade sections:

* The meridional velocity is a *shape*, rescaled to an area average of one at
  each station so that the mean line keeps the mass flow,

  .. math::

      V_m(s) = V_{m,\\mathrm{rms}}\\,
      \\frac{f(s)}{\\int_0^1 f\\,r\\,\\mathrm{d}s \\big/ \\int_0^1 r\\,\\mathrm{d}s} \\,.

* The angular momentum is an additive offset :math:`\\Delta` from the mean
  line's own, in units that do not vanish when the mean line has no swirl,

  .. math::

      r V_\\theta(s) = r_\\mathrm{rms} V_{\\theta,\\mathrm{rms}}
      + \\Delta(s)\\, r_\\mathrm{rms} V_{m,\\mathrm{rms}} \\,,

  so :math:`\\Delta=0` everywhere is a free vortex at the mean-line swirl,
  and an axial exit when that swirl is zero.

**The knots are the blade sections.** A blade reads the flow at its sections
to set their metal angles, so a value at each section is exactly what decides
the blade; the spline between them, linear beyond the outermost, serves the
spans in between, the initial guess and the fits to a solution.

The two kinds of station mean different things. A row *inlet* holds the flow
arriving there, which is a state the CFD measures; the
:class:`~turbigen.iterate.SpanwiseInflow` iterator copies it in. A row
*outlet* holds a setting of the blade: with the trailing-edge recamber held at
zero, the outlet :math:`\\Delta` *is* the trailing-edge metal angle, and differs
from the flow that leaves by the deviation. It is moved by
:class:`~turbigen.iterate.SpanwiseWork` on a rotor and
:class:`~turbigen.iterate.SpanwiseSwirl` on a stator until the CFD meets their
targets.

Span fractions are mapped to radius linearly between the mean-line hub and
casing radii of each station.
"""

from typing import ClassVar

import numpy as np

import turbigen.util
from turbigen.node import Node

N_QUAD = 1001
"""Points in the spanwise quadrature that normalises a shape."""


def spline(knots, values):
    """Return the spanwise curve through knot `values` at span fractions `knots`.

    A natural cubic inside the knots and straight beyond them; a constant for
    one knot, a line for two. Linear in the knot values, which is what lets
    :func:`fit_knots` fit a measured profile onto exactly the representation
    the design evaluates.
    """
    knots = np.asarray(knots, dtype=float)
    values = np.asarray(values, dtype=float)
    if knots.size == 1:
        return lambda spf: np.full(np.shape(spf), values[0])
    return turbigen.util.interp1d_linear_extrap(knots, values)


def fit_knots(knots, spf_meas, y_meas, weights=None):
    """Return the knot values whose curve best fits a measured profile.

    Weighted linear least squares, with the curve through each unit knot
    vector as a basis function. A profile sampled from a curve on the same
    knots is recovered exactly.

    Parameters
    ----------
    knots : array_like, shape (n_knot,)
        Knot span fractions.
    spf_meas, y_meas : array_like, shape (n_meas,)
        Where the profile was measured, and its values there.
    weights : array_like, shape (n_meas,), optional
        Relative weight of each measurement, uniform by default.

    """
    knots = np.asarray(knots, dtype=float)
    spf_meas = np.asarray(spf_meas, dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)
    w = np.ones_like(spf_meas) if weights is None else np.asarray(weights, float)

    basis = np.stack(
        [spline(knots, unit)(spf_meas) for unit in np.eye(knots.size)], axis=1
    )
    sqrt_w = np.sqrt(np.abs(w))
    fitted, *_ = np.linalg.lstsq(sqrt_w[:, None] * basis, sqrt_w * y_meas, rcond=None)
    return fitted


def station_radius(station, spf):
    """Return radius at span fractions on one mean-line station [m]."""
    spf = np.asarray(spf, dtype=float)
    return station.r_hub * (1.0 - spf) + station.r_cas * spf


def _quadrature(station, weight=None):
    """Return span fractions and normalised weights over a station's span.

    Area weights, radius on an annulus, times `weight` at each point if given.
    """
    s = np.linspace(0.0, 1.0, N_QUAD)
    w = station_radius(station, s)
    if weight is not None:
        w = w * weight(s)
    return s, w / np.trapezoid(w, s)


def area_average(knots, values, station):
    """Return the area average of the curve through knot `values` over the
    span of `station`."""
    s, w = _quadrature(station)
    return float(np.trapezoid(spline(knots, values)(s) * w, s))


def mass_average(knots, values, Vm_shape, station):
    """Return the mass average of the curve through knot `values` over the
    span of `station`.

    Weighted by the meridional velocity shape times radius, the mass flux of
    an annulus whose density is uniform, which is what a design assumes.
    """
    s, w = _quadrature(station, spline(knots, Vm_shape))
    return float(np.trapezoid(spline(knots, values)(s) * w, s))


class SpanwiseDesign(Node):
    """How the flow varies from hub to casing.

    A member gives the meridional velocity and angular momentum at any span
    fraction of any station; the relative flow angle a blade is designed
    against follows from those here. Stations are numbered as the mean line's
    are, ``k = 2 i_row`` at a row inlet and one more at its outlet, and
    `knots` is the span fraction of each blade section of that row.

    See :mod:`turbigen.spanwise` for the alternatives.
    """

    def Vm(self, station, k, knots, spf):
        """Return meridional velocity at span fractions `spf` [m/s].

        `station` is the mean-line station `k` the profile is on.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement Vm(self, station, k, knots, spf)"
        )

    def rVt(self, station, k, knots, spf):
        """Return angular momentum at span fractions `spf` [m^2/s]."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement rVt(self, station, k, knots, spf)"
        )

    def filled(self, n_sections):
        """Return this design checked against blades with `n_sections` per
        row, and with whatever it defaults written out. Itself by default."""
        del n_sections
        return self

    def Alpha_rel(self, mean_line_row, i_row, knots, spf):
        """Return relative flow angle at inlet and outlet of row `i_row`, shape
        (n, 2) [deg].

        Parameters
        ----------
        mean_line_row : MeanLine
            Inlet and outlet stations of the row, shape (2,).
        i_row : int
            Index of the row.
        knots : array_like
            Span fraction of each of its sections.
        spf : array_like
            Where to evaluate.

        """
        spf = np.asarray(spf, dtype=float).reshape(-1)
        Alpha_rel = np.empty((spf.size, 2))
        for j in range(2):
            station = mean_line_row[j]
            k = 2 * i_row + j
            r = station_radius(station, spf)
            Vt_rel = self.rVt(station, k, knots, spf) / r - station.Omega * r
            Vm = self.Vm(station, k, knots, spf)
            Alpha_rel[:, j] = np.degrees(np.arctan(Vt_rel / Vm))
        return Alpha_rel


class VortexExpon(SpanwiseDesign):
    """Swirl varying as a power of radius, with uniform meridional velocity."""

    type: ClassVar[str] = "vortex_expon"

    n: float | tuple[float, ...] = -1.0
    """Exponent of radius that tangential velocity varies with, one for every
    row or one per row [--]. The default is a free vortex."""

    def filled(self, n_sections):
        if isinstance(self.n, tuple) and len(self.n) != len(n_sections):
            raise ValueError(
                f"Expected one vortex exponent per row, but got {len(self.n)} "
                f"for {len(n_sections)} rows."
            )
        return self

    def _n(self, k):
        return float(self.n[k // 2] if isinstance(self.n, tuple) else self.n)

    def Vm(self, station, k, knots, spf):
        del k, knots
        return np.full(np.shape(spf), float(station.Vm))

    def rVt(self, station, k, knots, spf):
        del knots
        r = station_radius(station, spf)
        return r * station.Vt * (r / station.r) ** self._n(k)


class StationProfile(Node):
    """Spanwise meridional velocity and angular momentum at one station, one
    value per blade section of its row."""

    Vm: tuple[float, ...] = ()
    """Shape of the meridional velocity at each section, rescaled to an area
    average of one [--]. Uniform if omitted."""

    rVt: tuple[float, ...] = ()
    """Offset of angular momentum from the mean line at each section, in units
    of mean-line rms radius times meridional velocity [--]. Zero, a free vortex,
    if omitted."""


class VelocitySpline(SpanwiseDesign):
    """Meridional velocity and angular momentum at every station, given at the
    blade sections."""

    type: ClassVar[str] = "velocity_spline"

    stations: tuple[StationProfile, ...] = ()
    """Profile at every station, in streamwise order: first row inlet, first
    row outlet, second row inlet, and so on. Every station a free vortex at
    uniform meridional velocity if omitted."""

    def filled(self, n_sections):
        """Return this design with every station's defaults written out, so
        every knot is a leaf of the config that an iterator can name and a
        batch can set.

        Raises unless there is a profile per station, each with a value per
        section of its row.
        """
        n_station = 2 * len(n_sections)
        stations = self.stations or (StationProfile(),) * n_station
        if len(stations) != n_station:
            raise ValueError(
                f"Expected a spanwise profile at each of {n_station} stations, "
                f"an inlet and outlet per row, but got {len(stations)}."
            )

        filled = []
        for k, station in enumerate(stations):
            n = n_sections[k // 2]
            Vm = station.Vm or (1.0,) * n
            rVt = station.rVt or (0.0,) * n
            for name, values in (("Vm", Vm), ("rVt", rVt)):
                if len(values) != n:
                    raise ValueError(
                        f"Spanwise station {k} gives {len(values)} {name} "
                        f"value(s), but row {k // 2} has {n} blade section(s), "
                        "one value each."
                    )
            if np.any(np.asarray(Vm) <= 0.0):
                raise ValueError(
                    f"Spanwise station {k} meridional velocity shape must be "
                    f"positive, got {list(Vm)}."
                )
            filled.append(
                StationProfile(
                    Vm=tuple(float(v) for v in Vm), rVt=tuple(float(v) for v in rVt)
                )
            )

        return VelocitySpline(stations=tuple(filled))

    def Vm(self, station, k, knots, spf):
        shape = self.stations[k].Vm
        level = area_average(knots, shape, station)
        return float(station.Vm) * spline(knots, shape)(spf) / level

    def rVt(self, station, k, knots, spf):
        Delta = spline(knots, self.stations[k].rVt)(spf)
        return float(station.r * station.Vt) + Delta * float(station.r * station.Vm)

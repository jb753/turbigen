r"""Classes to calculate meridional coordinates of an axisymmetric annulus.

The purpose of these objects is to evaluate x/r coordinates over a turbomachine
annulus as a function of spanwise and streamwise location.

To make a new annulus, subclass the BaseAnnulus and implement:
    - __init__(self, rmid, span, Beta, [... your choice of design variables])
    - evaluate_xr(m, spf)
    - nrow

"""

from turbigen import util
from turbigen.geometry import MeridionalLine
from scipy.optimize import minimize, root_scalar
import scipy.interpolate
from abc import abstractmethod

import numpy as np

logger = util.make_logger()


class AnnulusDesigner(util.BaseDesigner):
    """Base class defining the interface for an annulus."""

    _supplied_design_vars = ("rmid", "span", "Beta")

    @abstractmethod
    def forward(self, rmid, span, Beta, *args, **kwargs):
        """Set the coordinates of the mean line of the annulus.

        Do whatever is necessary to set up the annulus geometry, such as
        calculating the hub and casing lines.

        Parameters
        ----------
        rmid : (nrow*2) array
            Mid-span radii at inlet and exit of all rows.
        span : (nrow*2) array
            Annulus span perpendicular to pitch angle at all stations.
        Beta : (nrow*2) array
            Pitch angles at all stations [deg].
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_xr(self, m, spf) -> np.ndarray:
        """Get meridional coordinates within the annulus.

        The input non-dimensional coordinates must be broadcastable
        to the same shape.

        Parameters
        ----------
        m: array_like
            Normalised meridional distance, where 0 is the inlet,
            1 is the first row LE, 2 is the first row TE, etc.
        spf : array_like
            Span fraction, where 0 is the hub and 1 is the casing.

        Returns
        -------
        xr : array_like (2, ...)
            Meridional coordinates of the requested points in the annulus.
            First axis is x or r coordinates, remaining axes are broadcasted
            shape of mnorm and spf.

        """
        raise NotImplementedError

    def setup_annulus(self, mean_line):
        """Setup the annulus using coordinates from a mean line."""
        self.forward(mean_line.rmid, mean_line.span, mean_line.Beta, **self.design_vars)

    @property
    @abstractmethod
    def nrow(self) -> int:
        """Number of blade rows in this annulus."""
        raise NotImplementedError

    @property
    def nseg(self) -> int:
        """Number of segments in this annulus, 2*nrow + 1."""
        return 2 * self.nrow + 1

    @property
    def mmax(self):
        """Maximum value of normalised meridional coordinate."""
        return float(self.nseg)

    def chords(self, spf):
        """Meridional chords of rows and row gaps.

        Parameters
        ----------
        spf : float
            Span fraction at which to evaluate the chords.
            0 is the hub, 1 is the casing.

        Returns
        -------
        cm: (nseg-1) array
            Meridional chord lengths of all segments.

        """

        # Preallocate chords
        nchord = self.nseg
        chords = np.zeros(nchord)

        # Loop over all chords
        for i in range(nchord):
            # Evaluate meridional coordinates and integrate arc length
            mq = np.linspace(i, i + 1, 100)
            chords[i] = util.arc_length(self.evaluate_xr(mq, spf))

        return chords

    def get_interfaces(self):
        """Meridional coordinates of row interfaces.

        Returns
        -------
        xr_interfaces : array_like (nrow-1, 2, 2)
            Meridional coordinates of the interfaces between blade rows.
            First axis is row index, second axis is x/r, third axis is
            the hub or casing.

        """
        mq = np.arange(2.5, (self.nrow + 1.5), 2.0).reshape(-1, 1)
        return self.get_cut_plane(mq)

    def get_cut_plane(self, m):
        """Coordinates on the hub and casing at a constant meridional posistion.

        Parameters
        ----------
        m: (n,) array
            Normalised meridional positions to evaluate the cut planes.

        Returns
        -------
        xr_cut : array_like (n, 2, 2)
            Meridional coordinates of the cut planes.
            Axes are: cut index, x/r,  hub/cas.

        """
        spf = np.reshape([0.0, 1.0], (1, -1))
        mq = np.reshape(m, (-1, 1))
        return self.evaluate_xr(mq, spf).transpose(1, 0, 2)

    def get_offset_planes(self, offset):
        """Meridional cut lines offset a distance up/downstream of each row.

        Parameters
        ----------
        offset : array_like
            Offset of the cut planes upstream from the LE and downstream
            from the TE of each row. Defaults to 2% meridional chord.

        Returns
        -------
        xr_cut : array_like (2*nrow, 2, 2)
            Meridional coordinates of the cut planes. Axes are: row, x/r, hub/cas.
        """

        # Separate out row and gap chords, repeat for LE/TE cuts
        chords = self.chords(0.5)
        chords_blade = np.repeat(chords[1::2], 2)
        chords_gap = chords[::2]
        chords_gap = np.concatenate(
            [[chords_gap[0]], np.repeat(chords_gap[1:-1], 2), [chords_gap[-1]]]
        )

        # Convert the offset specified as a fraction of blade chord
        # to a fraction of gap chord
        offsets = offset * np.ones((self.nrow * 2,))
        offsets[::2] *= -1.0
        offsets *= chords_blade / chords_gap

        # Calculate query meridional coordinates
        mq = np.arange(1.0, self.nseg) + offsets
        return self.get_cut_plane(mq)

    def get_coords(self, npts=50):
        """Sample the coordinates of hub and casing lines in AutoGrid format.

        Parameters
        ----------
        npts : int
            Number of sampled points per blade and gap.

        Returns
        -------
        xr: (2, nrow*npts, 2)
            Coordinates of the annulus lines. Axes are: hub/cas, streamwise, x/r.

        """
        m = np.linspace(0.0, self.mmax, self.nseg * npts + 1)
        return self.get_cut_plane(m).transpose(2, 0, 1)

    def get_span(self, m):
        """Span of the annulus at a given meridional position.

        Parameters
        ----------
        m : (n,) array
            Normalised meridional positions to evaluate the span.

        Returns
        -------
        span : (n,) array
            Span of the annulus at the given meridional position.

        """
        xr_span = self.get_cut_plane(m).transpose(1, 2, 0)
        return util.arc_length(xr_span)

    def get_span_curve(self, spf, n=201, mlim=None):
        """Meridional xr curve along a given span fraction.

        Parameters
        ----------
        spf : float
            Span fraction at which to evaluate the curve.
            0 is the hub, 1 is the casing.
        n : int
            Number of streamwise points to evaluate.
        mlim : tuple
            Normalised meridional limits to evaluate the curve.
            Defaults to the entire annulus.

        Returns
        -------
        xr : array_like (2, n)
            Meridional coordinates of the curve. Axes are: x/r, streamwise.

        """
        util.check_scalar(spf=spf)
        if mlim is None:
            mlim = (0.0, self.mmax)
        m_ref = np.linspace(*mlim, n)
        return self.evaluate_xr(m_ref, spf).squeeze()

    def xr_row(self, irow):
        """Make a meridional interpolator restricted to one blade row.

        Parameters
        ----------
        irow : int
            Index of the blade row to evaluate.

        Returns
        -------
        xr_row : callable
            Function to evaluate meridional coordinates inside the blade row
            Has signature xr_row(spf, m) where
                - m=0 is the row LE, m=1 is the row TE;
                - spf=0 is the hub, spf=1 is the casing.

        """

        mst = 2 * irow + 1

        def func(spf, m):
            return self.evaluate_xr(mst + m, spf)

        return func

    def get_mp_from_xr(self, spf, n=4999, mlim=None):
        """Return a function to find 1D unwrapped distance from a 2D xr point.

        Parameters
        ----------
        spf : float
            Span fraction at which to evaluate the curve.
            0 is the hub, 1 is the casing.
        n : int
            Number of streamwise points to evaluate.
        mlim : tuple
            Normalised meridional limits to evaluate the curve.
            Defaults to the entire annulus.

        Returns
        -------
        mp_from_xr : callable
            Function to evaluate the unwrapped meridional distance from a
            2D point in the meridional plane. Has signature mp_from_xr(xr)
            where xr is a (2,n) array of x/r coordinates.

        """
        # We want to plot along a general meridional surface
        # So brute force a mapping from x/r to meridional distance

        xr_ref = self.get_span_curve(spf, n, mlim)

        # Calculate normalised meridional distance (angles are angles)
        dxr = np.diff(xr_ref, n=1, axis=1)
        dm = np.sqrt(np.sum(dxr**2.0, axis=0))
        rc = 0.5 * (xr_ref[1, 1:] + xr_ref[1, :-1])
        mp_ref = util.cumsum0(dm / rc)
        assert (np.diff(mp_ref) > 0.0).all()

        func = scipy.interpolate.NearestNDInterpolator(xr_ref.T, mp_ref)

        def mp_from_xr(xr):
            xru = xr.reshape(2, -1)
            mpu = func(xru.T)  # % - mp_stack
            return mpu.reshape(xr.shape[1:])

        return mp_from_xr


class FixedAxialChord(AnnulusDesigner):
    def forward(
        self,
        rmid,
        span,
        Beta,
        cx_row,
        cx_gap,
        nozzle_ratio=1.0,
    ):
        # Check input data
        npt = len(rmid)
        nrow = npt // 2
        ngap = nrow + 1
        util.check_vector((npt,), rmid=rmid, span=span, Beta=Beta)
        util.check_vector((nrow,), cx_row=cx_row)
        util.check_vector((ngap,), cx_gap=cx_gap)

        self.span = span

        # Assemble vector of all cx
        cx = np.zeros(nrow * 2 + 1)
        cx[::2] = cx_gap
        cx[1::2] = cx_row

        # Integrate x
        xmid = util.cumsum0(cx)
        xmid -= xmid[1]  # Place x origin at first row LE

        # Extend r coords for inlet and exit ducts at constant Beta
        rmid = np.r_[0.0, rmid, 0.0]
        sinBeta = np.sin(np.radians(Beta))
        rmid[0] = rmid[1] - cx[0] * sinBeta[0]
        rmid[-1] = rmid[-2] + cx[-1] * sinBeta[-1]

        # The extensions have same span and pitch angle as first/last point
        span = np.pad(span, 1, "edge")
        Beta = np.pad(Beta, 1, "edge")

        # Adjust to nozzle exit area
        radius_ratio = rmid[-2] / rmid[-1]
        span[-1] *= nozzle_ratio * radius_ratio

        # We now have coordinates of the mid-span line
        # So make the hub and casing lines
        sinBeta = np.sin(np.radians(Beta))
        cosBeta = np.cos(np.radians(Beta))
        xhub = xmid + 0.5 * span * sinBeta
        xcas = xmid - 0.5 * span * sinBeta
        rhub = rmid - 0.5 * span * cosBeta
        rcas = rmid + 0.5 * span * cosBeta

        # Make hub and casing line splines
        self._hub = MeridionalLine(xhub, rhub, Beta).smooth()
        self._cas = MeridionalLine(xcas, rcas, Beta).smooth()

    def __repr__(self):
        try:
            cm = self.chords(0.5)[1::2]
            mq = np.arange(1.5, self.nrow + 1.5)
            span = self.get_span(mq)
            xrhub = self.evaluate_xr(mq, 0.0)
            xrcas = self.evaluate_xr(mq, 0.0)
            xrrms = np.sqrt(0.5 * (xrhub**2 + xrcas**2))
            return (
                "FixedAxialChord(\n"
                f"    x={util.format_array(xrrms[1])} m,\n"
                f"    r_rms={util.format_array(xrrms[1])} m,\n"
                f"    span={util.format_array(span)} m,\n"
                f"    AR={util.format_array(span / cm)}\n"
                ")"
            )
        except Exception:
            return "FixedAxialChord()"

    @property
    def nrow(self):
        return self._hub.N // 2 - 1

    def evaluate_xr(self, m, spf):
        tb, spfb = np.broadcast_arrays(m, spf)

        # t is a vector that describes grid spacings where each unit interval
        # corresponds to a gap or blade
        # We need to map to meridional distance fractions
        npts = self.nseg + 1
        tctrl = np.linspace(0, npts - 1, npts)
        mhub = np.interp(tb, tctrl, self._hub.mctrl)
        mcas = np.interp(tb, tctrl, self._cas.mctrl)

        # Evaluate hub and casing coordinates
        xr_hub = self._hub.xr(mhub)
        xr_cas = self._cas.xr(mcas)

        # Finally evaluate the meridional grid
        spf1 = np.expand_dims(np.stack((1.0 - spfb, spfb)), 1)
        xr_hc = np.stack((xr_hub, xr_cas))
        xr = np.sum(spf1 * xr_hc, axis=0)

        return xr


class Smooth(AnnulusDesigner):
    """Annlus defines the entire meridional geometry of the turbomachine."""

    def forward(
        self,
        rmid,
        span,
        Beta,
        AR_chord,
        AR_gap,
        nozzle_ratio=1.0,
        rcout_offset=0.0,
        smooth=True,
    ):
        r"""Construct an annulus from geometric parameters.

        Parameters
        ----------
        rmid : (nrow*2) array
            Mid-span radii at inlet and exit of all rows.
        span : (nrow*2) array
            Annulus span perpendicular to pitch angle at all stations.
        Beta : (nrow*2) array
            Pitch angles at all stations [deg].
        AR_chord : (nrow)
            Span to meridional chord aspect ratio for each blade row. When the
            pitch angle is 90 degrees, the aspect ratio is constrained by
            `rmid` and not a free parameter, so must be set to `NaN`.
        AR_gap : (nrow+1) array
            Meridional aspect ratio of inlet, exit and gaps between rows. When
            the pitch angle is 90 degrees, the aspect ratio is constrained by
            `rmid` and not a free parameter, so must be set to `NaN`.
        nozzle_ratio : float
            Area ratio of exit nozzle, default to 1. for no contraction.

        """

        npt = len(rmid)
        nrow = npt // 2
        ngap = nrow + 1
        nchord = nrow

        # Check input data
        util.check_scalar(nozzle_ratio=nozzle_ratio, rcout_offset=rcout_offset)
        util.check_vector((npt,), rmid=rmid, span=span, Beta=Beta)
        util.check_vector((ngap,), AR_gap=AR_gap)
        util.check_vector((nchord,), AR_chord=AR_chord)

        # Store input data
        self.rmid = np.reshape(rmid, (npt,))
        self.span = np.reshape(span, (npt,))
        self.Beta = np.reshape(Beta, (npt,))
        self.AR_chord = np.reshape(AR_chord, (nchord,))
        self.AR_gap = np.reshape(AR_gap, (ngap,))
        self.nozzle_ratio = nozzle_ratio

        # Assemble vectors of all ARs and spans
        AR = np.zeros(self.nrow * 2 + 1)
        AR[::2] = self.AR_gap
        AR[1::2] = self.AR_chord
        span_avg = 0.5 * (self.span[1:] + self.span[:-1])
        span_avg = np.append(np.insert(span_avg, 0, self.span[0]), self.span[-1])

        # Calculate meridional lengths, estimate axial lenghts
        AR_guess = AR + 0.0
        AR_guess[AR < 0.0] = 0.4
        Ds = span_avg / AR_guess
        cosBeta_avg = np.cos(np.radians(0.5 * (self.Beta[1:] + self.Beta[:-1])))
        cosBeta = np.cos(np.radians(self.Beta))
        cosBeta_avg = np.append(np.insert(cosBeta_avg, 0, cosBeta[0]), cosBeta[-1])
        Dx = cosBeta_avg * Ds
        Dx[cosBeta_avg < 1e-3] = 0.0
        Ds[AR < 0.0] = -1.0

        # Integrate x
        xmid = util.cumsum0(Dx)
        xmid -= xmid[1]  # Place x origin at first row LE

        # Extended r coords
        rmid = np.zeros((self.nrow + 1) * 2)

        # Fill in known radii
        rmid[1:-1] = self.rmid

        # Inlet/exit ducts
        sinBeta = np.sin(np.radians(Beta))
        rmid[0] = rmid[1] - Ds[0] * sinBeta[0]
        rmid[-1] = rmid[-2] + Ds[-1] * sinBeta[-1]

        # We now have an initial guess of axial coordinates
        # So make the hub and casing lines

        # Extract data
        span = np.pad(self.span, 1, "edge")
        Beta = np.pad(self.Beta, 1, "edge")
        cosBeta = np.cos(np.radians(Beta))
        sinBeta = np.sin(np.radians(Beta))

        # Adjust to meet nozzle exit area
        radius_ratio = rmid[-2] / rmid[-1]
        span[-1] *= self.nozzle_ratio * radius_ratio

        xhub = xmid + 0.5 * span * sinBeta
        xcas = xmid - 0.5 * span * sinBeta
        rhub = rmid - 0.5 * span * cosBeta
        rcas = rmid + 0.5 * span * cosBeta

        # Offset the exit casing radius (defaults to zero)
        rcas[-1] += rcout_offset * span[-1]

        # Smoothed the initial guess lines
        self.hub = MeridionalLine(xhub, rhub, Beta).smooth()
        self.cas = MeridionalLine(xcas, rcas, Beta).smooth()

        # Now we need to offset the x-coordinates of each control point in turn
        # to reach target aspect ratios

        # Initialise the offsets to zero
        Dx_AR = np.zeros_like(xhub)

        # To optimise the kth chord, we offset the k+1th and downstream control points
        def _iter_chord(delta, k):
            Dx_AR[k + 1 :] = delta
            self.hub.x = xhub + Dx_AR
            self.cas.x = xcas + Dx_AR
            self.hub._fit()
            self.cas._fit()
            chords = self.chords(0.5)
            err = chords - Ds
            return err[k]

        # To optimise the kth chord, we offset the k+1th and downstream control points
        def _iter_smooth(delta, k):
            Dx_AR[k + 1 :] = delta
            self.hub.x = xhub + Dx_AR
            self.cas.x = xcas + Dx_AR
            self.hub._fit()
            self.cas._fit()
            # self.hub.smooth()
            # self.cas.smooth()
            return self.hub.smoothness_metric + self.cas.smoothness_metric

        # Loop over all chords and iterate axial coordinates
        def _solve_k(k):
            dxref = np.max(np.abs((xmid[k + 1] - xmid[k], rmid[k + 1] - rmid[k])))

            if np.isnan(Ds[k]):
                return

            elif Ds[k] < 0.0:
                minimize(
                    _iter_smooth,
                    0.0,
                    args=(k,),
                    tol=dxref * 1e-6,
                    options={"maxiter": 200},
                )

            else:
                # Find a bracket safely
                dx_lower = None

                # High guess
                for rel_dx in (0.1, 0.2, 0.4, 0.8, 1.6):
                    dx_upper = dxref * rel_dx
                    err = _iter_chord(dx_upper, k)
                    if err > 0.0:
                        break
                    else:
                        dx_lower = dxref * rel_dx

                # Low guess
                if dx_lower is None:
                    for rel_dx in (0.1, 0.2, 0.4, 0.8, 1.6):
                        dx_lower = -dxref * rel_dx
                        err = _iter_chord(dx_lower, k)
                        if err < 0.0:
                            break

                try:
                    root_scalar(
                        _iter_chord,
                        bracket=(dx_lower, dx_upper),
                        args=(k,),
                        xtol=dxref * 1e-3,
                    )
                except ValueError:
                    pass

        if smooth:
            # for k in range(1, 3):
            for k in range(1, self.nseg - 1):
                _solve_k(k)

        # err_out_abs = self.chords(0.5) - Ds
        # err_out_rel = err_out_abs / self.chords(0.5)
        # assert (np.abs(err_out_rel[~np.isnan(err_out_rel)]) < 1e-2).all()

        self.cas.smooth()
        self.hub.smooth()

        if not rcout_offset:
            assert all(self.hub._is_straight() == self.cas._is_straight())

    @property
    def nrow(self):
        return self.rmid.size // 2

    def evaluate_xr(self, m, spf):
        tb, spfb = np.broadcast_arrays(m, spf)

        # t is a vector that describes grid spacings where each unit interval
        # corresponds to a gap or blade
        # We need to map to meridional distance fractions
        npts = self.nseg + 1
        tctrl = np.linspace(0, npts - 1, npts)
        mhub = np.interp(tb, tctrl, self.hub.mctrl)
        mcas = np.interp(tb, tctrl, self.cas.mctrl)

        # Evaluate hub and casing coordinates
        xr_hub = self.hub.xr(mhub)
        xr_cas = self.cas.xr(mcas)

        # Finally evaluate the meridional grid
        spf1 = np.expand_dims(np.stack((1.0 - spfb, spfb)), 1)
        xr_hc = np.stack((xr_hub, xr_cas))
        xr = np.sum(spf1 * xr_hc, axis=0)

        return xr

    def __repr__(self):
        try:
            cm = self.chords(0.5)[1::2]
            mq = np.arange(1.5, self.nrow + 1.5)
            span = self.get_span(mq)
            xr_mid = self.evaluate_xr(mq, 0.5)
            return f"FixedAR(nrow={self.nrow}, x={xr_mid[0]}, r={xr_mid[1]}, AR={span / cm})"
        except Exception:
            return "FixedAR()"


#
#
# def load_annulus(annulus_type):
#     """Get annulus class by string, including any custom classes."""
#     available_annulus_types = {a.__name__: a for a in BaseAnnulus.__subclasses__()}
#     if annulus_type not in available_annulus_types:
#         raise ValueError(
#             f"Unknown annulus type: {annulus_type}, should be one of {available_annulus_types.keys()}"
#         )
#     else:
#         return available_annulus_types[annulus_type]

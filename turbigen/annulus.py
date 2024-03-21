r"""Classes to calculate meridional coordinates of an axisymmetric annulus.

The purpose of these objects is to evaluate x/r coordinates over a turbomachine
annulus as a function of spanwise and streamwise location.

The streamwise coordinate is a normalised meridional distance :math:`m` defined
such that:
    - :math:`m=0` at the machine inlet;
    - :math:`m=1` at the first row LE;
    - :math:`m=2` at the first row TE;
    - ...
    - :math:`m=2N_\mathrm{row}-1` at the last row LE;
    - :math:`m=2N_\mathrm{row}` at the last row TE;
    - :math:`m=2N_\mathrm{row}+1` at the machine outlet;

Between each of these points, :math:`m` varies linearly with arc length along
the streamsurface; every non-dimensional unit interval corresponds to, in
general, a different dimensional arc length.

"""

import turbigen.util
from turbigen.geometry import MeridionalLine, LinearLine
from scipy.optimize import minimize, root_scalar, newton

import numpy as np

logger = turbigen.util.make_logger()


class Smooth:
    """Annlus defines the entire meridional geometry of the turbomachine."""

    def __init__(
        self, rmid, span, Beta, AR_chord, AR_gap, nozzle_ratio=1.0, rcout_offset=0.0
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

        # Store input data
        self.rmid = np.reshape(rmid, -1)
        self.span = np.reshape(span, -1)
        self.Beta = np.reshape(Beta, -1)
        self.AR_chord = np.reshape(AR_chord, -1)
        self.AR_gap = np.reshape(AR_gap, -1)
        self.nozzle_ratio = nozzle_ratio

        # Assemble vectors of all ARs and spans
        AR = np.empty(self.nrow * 2 + 1)
        AR[::2] = self.AR_gap
        AR[1::2] = self.AR_chord
        span_avg = 0.5 * (self.span[1:] + self.span[:-1])
        span_avg = np.append(np.insert(span_avg, 0, self.span[0]), self.span[-1])

        # Calculate meridional lengths, estimate axial lenghts
        AR_guess = AR + 0.0
        AR_guess[AR < 0.0] = 0.4
        Ds = span_avg / AR_guess
        cosBeta_avg = np.cos(np.radians(0.5 * (self.Beta[1:] + self.Beta[:-1])))
        cosBeta_avg = np.append(
            np.insert(cosBeta_avg, 0, self.cosBeta[0]), self.cosBeta[-1]
        )
        Dx = cosBeta_avg * Ds
        Dx[cosBeta_avg < 1e-3] = 0.0
        Ds[AR < 0.0] = -1.0

        # Integrate x
        xmid = turbigen.util.cumsum0(Dx)
        xmid -= xmid[1]  # Place x origin at first row LE

        # Extended r coords
        rmid = np.empty((self.nrow + 1) * 2)

        # Fill in known radii
        rmid[1:-1] = self.rmid

        # Inlet/exit ducts
        rmid[0] = rmid[1] - Ds[0] * self.sinBeta[0]
        rmid[-1] = rmid[-2] + Ds[-1] * self.sinBeta[-1]

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

                # print(f'Setting chord k={k}')
                # print('High guess')
                # High guess
                for rel_dx in (0.1, 0.2, 0.4, 0.8, 1.6):
                    dx_upper = dxref * rel_dx
                    err = _iter_chord(dx_upper, k)
                    # print(f'err={err}')
                    if err > 0.0:
                        break
                    else:
                        dx_lower = dxref * rel_dx

                # Low guess
                # print('Low guess')
                if dx_lower is None:
                    for rel_dx in (0.1, 0.2, 0.4, 0.8, 1.6):
                        dx_lower = -dxref * rel_dx
                        err = _iter_chord(dx_lower, k)
                        if err < 0.0:
                            break

                root_scalar(
                    _iter_chord,
                    bracket=(dx_lower, dx_upper),
                    args=(k,),
                    xtol=dxref * 1e-3,
                )

        for k in range(1, self.npts - 2):
            _solve_k(k)

        # err_out_abs = self.chords(0.5) - Ds
        # err_out_rel = err_out_abs / self.chords(0.5)
        # assert (np.abs(err_out_rel[~np.isnan(err_out_rel)]) < 1e-2).all()

        self.cas.smooth()
        self.hub.smooth()

        if not rcout_offset:
            assert all(self.hub._is_straight() == self.cas._is_straight())

    def __str__(self):
        xr_mid = self.xr_mid(self.mctrl[1:-1])
        xstr = np.array2string(xr_mid[0], precision=4)
        rstr = np.array2string(xr_mid[1], precision=4)
        sstr = np.array2string(self.span, precision=4)
        return f"""Annulus(
    xmid={xstr},
    rmid={rstr},
    span={sstr}
    )"""

    @classmethod
    def from_fit(cls, Beta, xr_hub, xr_cas, AR_gap, nozzle_ratio=1.0):
        xr_hub = np.array(xr_hub)
        xr_cas = np.array(xr_cas)

        rmid = 0.5 * (xr_hub[1, (0, -1)] + xr_cas[1, (0, -1)])
        xmid = 0.5 * (xr_hub[0, (0, -1)] + xr_cas[0, (0, -1)])
        span = np.sqrt(np.sum((xr_hub[:, (0, -1)] - xr_cas[:, (0, -1)]) ** 2.0, axis=0))
        AR_chord = np.nan
        return cls(rmid, span, Beta, AR_chord, AR_gap, nozzle_ratio, xmid)

    @property
    def mctrl(self):
        return 0.5 * (self.hub.mctrl + self.cas.mctrl)

    @property
    def _mctl(self):
        return np.linspace(0, self.npts - 1, self.npts)

    def chords(self, spf):
        """ "Meridional chords of gaps and rows at a specified span fraction."""

        chords = np.empty(self.npts - 1)
        for i in range(self.npts - 1):
            mhub = np.linspace(
                *self.hub.mctrl[
                    (i, i + 1),
                ]
            )
            mcas = np.linspace(
                *self.cas.mctrl[
                    (i, i + 1),
                ]
            )
            chords[i] = turbigen.util.arc_length(
                spf * self.cas.xr(mcas) + (1.0 - spf) * self.hub.xr(mhub)
            )
        return chords

    @property
    def nrow(self):
        return self.rmid.size // 2

    @property
    def npts(self):
        return self.hub.N

    @property
    def cosBeta(self):
        return np.cos(np.radians(self.Beta))

    @property
    def sinBeta(self):
        return np.sin(np.radians(self.Beta))

    @property
    def rhub(self):
        return self.rmid - 0.5 * self.span * self.cosBeta

    @property
    def rcas(self):
        return self.rmid + 0.5 * self.span * self.cosBeta

    def xr_mid(self, m):
        """Get coordinates on midspan streamsurface."""
        return 0.5 * self.cas.xr(m) + 0.5 * self.hub.xr(m)

    def xr_row(self, irow):
        """Return a streamsurface for a blade row."""

        # We need to map a fraction of meridional distance to mctrl on hub and casing
        ictrl = 1 + irow * 2
        mch = self.hub.mctrl[
            (ictrl, ictrl + 1),
        ]
        mcc = self.cas.mctrl[
            (ictrl, ictrl + 1),
        ]

        def func(spf, s):
            mh = s * mch[1] + (1.0 - s) * mch[0]
            mc = s * mcc[1] + (1.0 - s) * mcc[0]
            return spf * self.cas.xr(mc) + (1.0 - spf) * self.hub.xr(mh)

        return func

    def evaluate_xr(self, t, spf):
        tb, spfb = np.broadcast_arrays(t, spf)

        # t is a vector that describes grid spacings where each unit interval
        # corresponds to a gap or blade
        # We need to map to meridional distance fractions
        tctrl = np.linspace(0, self.npts - 1, self.npts)
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

    def get_coords(self, nseg=50):
        """Sample the coordinates of hub and casing lines in AutoGrid style."""
        N = self.hub.N
        s = np.linspace(0.0, 1.0, N * nseg + 1)
        return np.stack((self.hub.xr(s), self.cas.xr(s))).transpose(0, 2, 1)

    def get_interfaces(self):
        """Meridional coordinates of row interfaces."""
        t = np.arange(2.5, (self.nrow + 1.5), 2.0)
        xr_hub = self.hub._xr(t)
        xr_cas = self.cas._xr(t)
        return np.stack((xr_hub, xr_cas)).transpose(2, 0, 1)

    def get_cut_planes(self, offset):
        """For each row, return (x,r) points on hub casing offset chords up/downstream.

        Returns axes: [x or r, row, hub or cas]"""

        if offset is None:
            offset = 0.02 * np.ones((self.nrow * 2,))
            offset[::2] *= -1.0

        t = np.arange(0.0, self.nrow * 2.0) + 1.0
        chords_blades = np.repeat(self.chords(0.5)[1::2], 2)
        chords_gaps = self.chords(0.5)[0::2]
        chords_gaps = np.concatenate(
            [[chords_gaps[0]], np.repeat(chords_gaps[1:-1], 2), [chords_gaps[-1]]]
        )
        # t += np.tile((-offset, offset), (self.nrow,)) * chords_blades / chords_gaps
        t += offset * chords_blades / chords_gaps

        spf = np.reshape([0.0, 1.0], (1, -1))
        return self.evaluate_xr(t.reshape(-1, 1), spf).transpose(1, 0, 2)

    def get_cut_plane(self, t):
        """(x,r) points on hub and casing at given normalise merdional coord."""
        spf = np.reshape([0.0, 1.0], (1, -1))
        xrc = self.evaluate_xr(t, spf).transpose(1, 0, 2)
        return xrc

    def A(self, m):
        """Flow area as function of normalised meridional distance."""

        m0 = np.clip(np.atleast_1d(m), 0.005, 0.995)

        xr_mid = self.xr_mid(m0)

        def _iter(mh):
            """Distance between mid on midspan and mhub ."""
            xr_hub = self.hub.xr(mh)
            norm_hub = self.hub.normal(mh)
            dxr_span = xr_mid - xr_hub
            mag_span = np.linalg.norm(dxr_span, ord=2, axis=0, keepdims=True)
            dxr_span /= mag_span
            return np.sum(dxr_span * norm_hub, axis=0)

        m1 = m0.copy()
        m1[m1 < 0.5] = m1[m1 < 0.5] + 0.001
        m1[m1 >= 0.5] = m1[m1 >= 0.5] - 0.001
        mout = newton(_iter, x0=m0, x1=m1)

        # Deal with end points
        mout[m0 > 0.999] = 1.0
        mout[m0 < 0.001] = 0.0

        # Fill in any NaNs
        inan = np.logical_or(np.isnan(mout), np.abs(mout) > 10.0 * np.median(mout))
        m0_fill = np.interp(m, m[~inan], mout[~inan]) - 0.001
        m1_fill = m0_fill + 0.001
        mout = newton(_iter, x0=m0_fill, x1=m1_fill)

        span = 2.0 * np.sqrt(np.sum((xr_mid - self.hub.xr(mout)) ** 2.0, axis=0))
        A = 2 * np.pi * xr_mid[1] * span

        return A


class Aircon:
    def __init__(
        self,
        rmid12,
        span12,
        Beta12,
        drin,
        htrin,
        xleak,
        Lout,
        Lmotor,
        rmotor,
        ARnoz=None,
    ):
        logger.debug("STARTING AIRCON ANNULUS DESIGN")

        # Calculate inlet radius
        rsin = (rmid12 + 0.5 * span12 * np.cos(np.radians(Beta12)))[0] - drin

        rhin = htrin * rsin
        rmin = 0.5 * (rsin + rhin)
        Hin = rsin - rhin
        logger.debug(f"rsin={rsin:.4f}, rhin={rhin:.4f}")

        # Calculate outlet
        rout = rmid12[-1] + Lout
        if ARnoz:
            HRout = ARnoz / rout * rmid12[-1]
            Hout = span12[-1] * HRout
        else:
            Hout = span12[-1]

        # Add inlet and exit stations
        Beta = np.concatenate(((0.0,), Beta12, (90.0,)))
        rmid = np.concatenate(((rmin,), rmid12, (rout,)))
        span = np.concatenate(((Hin,), span12, (Hout,)))
        logger.debug(f"Beta={Beta}")
        logger.debug(f"rmid={rmid}")
        logger.debug(f"span={span}")

        x12 = -span12 * 0.5 * turbigen.util.sind(Beta12)
        xmid = np.concatenate(((xleak,), x12, (x12[-1],)))

        cosBeta = turbigen.util.cosd(Beta)
        sinBeta = turbigen.util.sind(Beta)

        # Hub/casing coordinates
        xhub = xmid + 0.5 * span * sinBeta
        xcas = xmid - 0.5 * span * sinBeta
        rhub = rmid - 0.5 * span * cosBeta
        rcas = rmid + 0.5 * span * cosBeta

        xcas[1] -= 0.01

        Beta_cas = Beta.copy()

        DLmotor = Lmotor * 0.1
        # rmin = rmid12[1] * 0.05
        rmin = rhub[0]
        thub = np.linspace(0.0, len(xhub) - 1, len(xhub))
        xmotor = np.array([-Lmotor - DLmotor, -Lmotor, 0.0])
        rmot = np.array([rmin, rmotor, rmotor + 0.065])
        tmotor = np.array([0.5, 0.6, 0.7])
        xhub = np.insert(xhub, 1, xmotor)
        rhub = np.insert(rhub, 1, rmot)
        rhub[0] = rmin
        thub = np.insert(thub, 1, tmotor)

        self.hub = LinearLine(xhub, rhub, thub)
        Beta_cas[1] = 30.0
        self.cas = MeridionalLine(xcas, rcas, Beta_cas).smooth(slope_max=10.0)
        # self.cas = LinearLine(xcas, rcas)

        self.span = span

    def xr_row(self, irow):
        """Return a streamsurface for a blade row."""

        # We need to map a fraction of meridional distance to mctrl on hub and casing
        ictrl = 1 + irow * 2
        mch = self.hub.mctrl[
            (ictrl, ictrl + 1),
        ]
        mcc = self.cas.mctrl[
            (ictrl, ictrl + 1),
        ]

        def func(spf, s):
            mh = s * mch[1] + (1.0 - s) * mch[0]
            mc = s * mcc[1] + (1.0 - s) * mcc[0]
            return spf * self.cas.xr(mc) + (1.0 - spf) * self.hub.xr(mh)

        return func

    def chords(self, spf):
        """Dimensional arc lengths between each control point."""
        mq = np.linspace(0.0, self._mctl[-1], 1000)
        xrq = self.evaluate_xr(mq, spf)
        sq = turbigen.util.cum_arc_length(xrq)
        sctl = np.interp(self._mctl, mq, sq)
        return np.diff(sctl)

    @property
    def npts(self):
        return self.cas.N

    @property
    def nrow(self):
        return (self.cas.N - 2) // 2

    @property
    def _mctl(self):
        return np.linspace(0, self.npts - 1, self.npts)

    def evaluate_xr(self, t, spf):
        tb, spfb = np.broadcast_arrays(t, spf)

        # t is a vector that describes grid spacings where each unit interval
        # corresponds to a gap or blade
        # We need to map to meridional distance fractions
        # tctrl = np.linspace(0, self.npts - 1, self.npts)
        mhub = np.interp(tb, self.hub.tctrl, self.hub.mctrl)
        mcas = np.interp(tb, self.cas.tctrl, self.cas.mctrl)

        # Evaluate hub and casing coordinates
        xr_hub = self.hub.xr(mhub)
        xr_cas = self.cas.xr(mcas)

        # Finally evaluate the meridional grid
        spf1 = np.expand_dims(np.stack((1.0 - spfb, spfb)), 1)
        xr_hc = np.stack((xr_hub, xr_cas))
        xr = np.sum(spf1 * xr_hc, axis=0)

        return xr

    def get_cut_planes(self, offset):
        """For each row, return (x,r) points on hub casing offset chords up/downstream.

        Returns axes: [x or r, row, hub or cas]"""

        if offset is None:
            offset = 0.02 * np.ones((self.nrow * 2,))
            offset[::2] *= -1.0

        t = np.arange(0.0, self.nrow * 2.0) + 1.0
        chords_blades = np.repeat(self.chords(0.5)[1::2], 2)
        chords_gaps = self.chords(0.5)[0::2]
        chords_gaps = np.concatenate(
            [[chords_gaps[0]], np.repeat(chords_gaps[1:-1], 2), [chords_gaps[-1]]]
        )
        # t += np.tile((-offset, offset), (self.nrow,)) * chords_blades / chords_gaps
        t += offset * chords_blades / chords_gaps

        spf = np.reshape([0.0, 1.0], (1, -1))
        return self.evaluate_xr(t.reshape(-1, 1), spf).transpose(1, 0, 2)

    def get_cut_plane(self, t):
        """(x,r) points on hub and casing at given normalise merdional coord."""
        spf = np.reshape([0.0, 1.0], (1, -1))
        xrc = self.evaluate_xr(t, spf).transpose(1, 0, 2)
        return xrc

    def get_coords(self, nseg=100):
        """Sample the coordinates of hub and casing lines in AutoGrid style."""
        N = 3
        s = np.linspace(0.0, N, N * nseg + 1)
        xr = self.evaluate_xr(s, [[0], [1]]).transpose(1, 2, 0)
        return xr

    def get_interfaces(self):
        """Meridional coordinates of row interfaces."""
        t = np.arange(2.5, (self.nrow + 1.5), 2.0)
        xr_hub = self.evaluate_xr(t, 0)
        xr_cas = self.evaluate_xr(t, 1)
        return np.stack((xr_hub, xr_cas)).transpose(2, 0, 1)

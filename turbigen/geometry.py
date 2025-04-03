"""Objects for constructing blades and annulus line geometries."""

import numpy as np
from scipy.optimize import minimize
import scipy.interpolate
from scipy.linalg import norm
from turbigen import util
import turbigen.thickness
import turbigen.camber


class LinearLine:
    def __init__(self, x, r, t=None, method="pchip"):
        """Simple linear interpolation between points."""

        self._x = x.reshape(-1)
        self._r = r.reshape(self._x.shape)
        self._xr = np.stack((self._x, self._r))
        self.N = len(self._x)
        if t is None:
            self._t = np.linspace(0.0, self.N - 1, self.N)
        else:
            self._t = t
        tmax = self._t[-1]
        if method == "pchip":
            self.xr_t = scipy.interpolate.PchipInterpolator(self._t, self._xr, axis=-1)
        elif method == "akima":
            self.xr_t = scipy.interpolate.Akima1DInterpolator(
                self._t, self._xr, axis=-1
            )
        elif method == "spline":
            self.xr_t = scipy.interpolate.CubicSpline(self._t, self._xr, axis=-1)

        Narc = 10000
        tarc = np.linspace(0, tmax, Narc)
        xrarc = self.xr_t(tarc)
        marc = util.cum_arc_length(xrarc)
        marc /= marc[-1]

        self.xr = scipy.interpolate.PchipInterpolator(marc, xrarc, axis=-1)

        self.tctrl = np.linspace(0.0, tmax, int(tmax) + 1)
        self.mctrl = np.interp(self.tctrl, tarc, marc)


class MeridionalLine:
    """A line in meridional plane from control points and pitch angles."""

    def __init__(self, x, r, Beta, t=None):
        r"""Initialise a curve in meridional plane with coordinates and pitch angles.

        This class produces a curvature-continous line running through the
        specified points with the specified pitch angles (or slopes). There is
        one degree of freedom at each point that alters the distribution of
        curvature along the segments.

        Parameters
        ----------
        x: (N,) array
            Axial coordinates of the control points, :math:`x_i\,`.
        r: (N,) array
            Radial coordinates of the control points, :math:`r_i\,`.
        Beta: (N,) array
            Pitch angles at the control points, :math:`\beta_i/^\circ\,`.

        Notes
        -----

        Each segment is a parametric spline :math:`[\tilde{x}(s),
        \tilde{r}(s)]` where the spline parameter :math:`i < s < i+1` for the
        :math:`i`-th segment (indexing from zero). A quintic polynomial spline
        has six degrees of freedom per control point, allowing control over
        value, slope, and curvature.

        The values are constrained as,

        .. math::
            \tilde{x}(i) = x_i\,, \quad \tilde{r}(i) = r_i\,.

        With parametric curves, by the chain rule, slope only fixes the ratio
        of derivatives,

        .. math::
            \tan\beta_i = \left.\frac{\dee r}{\dee x}\right|_i
            = \frac{\tilde{r}'(i)}{\tilde{x}'(i)}\,.

        This gives us our free parameter to control the shape of the curve. To
        avoid numerical problems at very small (:math:`\beta \rightarrow
        0^\circ`) or very large slopes (:math:`\beta \rightarrow 90^\circ`) we
        select the free parameter based on the value of :math:`\beta\,`:

        .. math::

            \begin{align}
                \beta_i &\le 45^\circ \quad \Rightarrow
                    \quad \text{free}\ \tilde{x}'(i)\,,
                    \quad \text{fix}\ \tilde{r}'(i) = \tilde{x}'(i)\tan\beta_i\,; \\
                \beta_i &\gt 45^\circ \quad \Rightarrow
                    \quad \text{free}\ \tilde{r}'(i)\,,
                    \quad \text{fix}\ \tilde{x}'(i) = \tilde{r}'(i)/\tan\beta_i\,.
            \end{align}

        Finally, to enable curved segments to join onto straight lines
        curvature must go to zero at each control point,

        .. math::
            \tilde{x}''(i) = 0\,, \quad \tilde{r}''(i) = 0\,.

        With the above assumptions, specifying one parameteric slope at each
        control point (by the `set_shape()` method on this class) is
        sufficient to uniquely determine the interpolating curve.


        """

        # We parameterise distance along the curve using two variables
        #   m --- normalised arc length along the curve, 0 at start, 1 at end
        #   t --- fit parameter, by convention each unit interval is either a
        #   intra-blade row gap or blade chord

        # Store input data
        self.x = np.reshape(x, -1)
        self.r = np.reshape(r, -1)
        self.Beta = np.reshape(Beta, -1)

        # Default fit parameter assumes two control points per blade
        if t is None:
            self.t = np.linspace(0.0, self.N - 1, self.N)
        else:
            self.t = t

        # Default parametric slope and curvature
        self.sslope = np.ones_like(self.x)
        self.scurve = np.zeros_like(self.x)

        # A vector of fit parameter, clustered near segment leading/trailing edges
        tclu1 = util.cluster_cosine(200)
        self._t_clu = np.unique(
            np.reshape(
                [i + tclu1 for i in range(int(self.t.max()))],
                (-1,),
            )
        )

        self.tctrl = np.linspace(0, self.N - 1, self.N)

        # Calculate coefficients
        self._fit()

    def set_shape(self, sslope, scurve=None):
        """Set the parametric slopes and curvatures at the control points.

        Parameters
        ----------
        sslope: (N,) array
            Parametric slopes for each control point, in terms of either
            :math:`x` or :math:`r` selected automatically for numerical
            stability. Appropriately non-dimensional such that unity is a sane
            initial guess.y

        scurve: (N,) array
            Parametric curvatures for each control point, in terms of either
            :math:`x` or :math:`r` selected automatically for numerical
            stability. Omit this argument to leave curvature alone.

        """
        self.sslope = sslope
        if scurve is not None:
            self.scurve = scurve
        self._fit()

    # @property
    # def mctrl(self):
    #     """Normalised meridional coordinates at the control points."""
    #     return self._pm(self.t)

    @property
    def N(self):
        """Number of control points."""
        return self.x.size

    def _fit(self):
        """Fit splines to control points with current slopes and curvatures."""

        # Convert pitch angles to slopes
        drdx = np.tan(np.radians(self.Beta))

        # Reference slopes using coordinate steps
        Dr = util.cell_to_node(np.diff(self.r))
        Dx = util.cell_to_node(np.diff(self.x))

        # Preallocate
        drdt = np.zeros_like(self.x)
        dxdt = np.zeros_like(self.x)
        d2rdt2 = np.zeros_like(self.x)
        d2xdt2 = np.zeros_like(self.x)

        # To accomodate very shallow or steep curves, we will use the
        # parametric slope and curvature either for x or r depending on the
        # value of Beta
        ii = np.abs(self.Beta) > 45.0

        # Set gradients
        drdt[ii] = self.sslope[ii] * Dr[ii]
        dxdt[ii] = drdt[ii] / drdx[ii]
        dxdt[~ii] = self.sslope[~ii] * Dx[~ii]
        drdt[~ii] = dxdt[~ii] * drdx[~ii]

        # Set curvatures
        C = self.scurve * (dxdt**2.0 + drdt**2.0) ** 1.5
        d2rdt2[ii] = self.scurve[ii] * Dr[ii] ** 2.0
        d2xdt2[ii] = (dxdt[ii] * d2rdt2[ii] - C[ii]) / drdt[ii]
        d2xdt2[~ii] = self.scurve[~ii] * Dx[~ii] ** 2.0
        d2rdt2[~ii] = (drdt[~ii] * d2xdt2[~ii] + C[~ii]) / dxdt[~ii]

        # Assemble all derivatives
        dx_all = np.stack((self.x, dxdt, d2xdt2)).T
        dr_all = np.stack((self.r, drdt, d2rdt2)).T

        # Fit polynomials
        # t = np.linspace(0.0, self.N - 1, self.N)
        self._px = scipy.interpolate.BPoly.from_derivatives(self.t, dx_all)
        self._pr = scipy.interpolate.BPoly.from_derivatives(self.t, dr_all)

        # Integrate arc length
        # tt = np.linspace(0.0, self.N - 1.0, 1 + (self.N - 1) * 100)

        xr = self._xr(self._t_clu)
        ds = np.sqrt(np.diff(xr[0]) ** 2.0 + np.diff(xr[1]) ** 2.0)
        s = np.insert(np.cumsum(ds), 0, 0.0)

        # Normalised meridional distance
        m = s + 0.0
        m /= m[-1]

        # Spline parameter to normalised meridional distance and inverse
        self._pt = scipy.interpolate.PchipInterpolator(m, self._t_clu)

        self.chords = np.diff(np.interp(self.t, self._t_clu, s))
        self.mctrl = np.interp(self.t, self._t_clu, m)
        self.mclu = m[1:-1]

    def _xr(self, t):
        """Meridional coordinates as function of spline parameter."""

        xr = np.stack((self._px(t), self._pr(t)))

        # Override if we are in a straight segment
        # We have to do this because when for example dr/dx = inf
        # there is freedom for dr/dt to vary non-monotonically and not show up
        # in the curvature metric because x is constant
        ind_straight = self._straight_indices(t)
        xr[:, ind_straight] = self._xr_straight(t[ind_straight])
        return xr

    def _straight_indices(self, t):
        """Return indices from t which correspond to a straight segment."""
        # tref = np.linspace(0.0, self.N - 1, self.N)
        ibin = np.digitize(t, self.t) - 1
        ibin[ibin == (self.N - 1)] -= 1  # Special case last bin
        return self._is_straight()[ibin]

    def xr(self, m):
        """Interpolate coordinates as a function of normalised meridional coordinate.

        The normalised meridional coordinate :math:`m=0` at the first control
        point and increases linearly with arc length along the line until the
        last control point at :math:`m=1`.

        Parameters
        ----------

        m: (Ns,) array
            Normalised meridional arc lengths to sample the curve at.

        Returns
        -------

        xr: (2,Ns) array
            Axial and radial coordinates at each of the requested locations.

        """
        return self._xr(self._pt(np.atleast_1d(m)))

    def _kappa(self, t):
        """Curvature as a function of fit parameter."""
        x1 = self._px.derivative(1)(t)
        r1 = self._pr.derivative(1)(t)
        x2 = self._px.derivative(2)(t)
        r2 = self._pr.derivative(2)(t)
        kappa = (x1 * r2 - r1 * x2) * (x1**2.0 + r1**2.0) ** (-1.5)

        # Override kappa in straight segments to avoid numerical problems
        ind_straight = self._straight_indices(t)
        kappa[ind_straight] = 0.0

        return kappa

    def kappa(self, m):
        r"""Curvature as function of normalised meridional coordinate.

        Note this is the 'proper' curvature, not just the second derivative.
        This is for two reasons: the second derivative only approximates
        curvature for small slopes, and more practically :math:`\dee^2r/\dee x^2`
        is numerically ill-posed for steep slopes.

        Parameters
        ----------
        m: (Ns,) array
            Normalised meridional arc lengths to sample the curvature, :math:`0
            \le m \le 1\,`.

        Returns
        -------
        kappa: (Ns) array
            Curvature at each requested arc length.

        """

        return self._kappa(self._pt(m))

    def normal(self, m):
        """Normal vector as function of normalised meridional coordinate"""
        t = self._pt(np.atleast_1d(m))
        dr = self._pr.derivative(1)(t)
        dx = self._px.derivative(1)(t)
        # dxr = np.stack((-dr, dx))
        dxr = np.stack((dx, dr))
        mag = np.linalg.norm(dxr, ord=2, axis=0, keepdims=True)
        return dxr / mag

    @property
    def strain(self):
        r"""Integrate curvature over the line to get scalar optimisation metric.

        The squared curvature integrated with respect to meridional distance,
        non-dimensional by the total arc length :math:`L`:

            .. math::
                \mathcal{S} = L^2 \int_{m=0}^1 \kappa(m)^2 \dee m

        """

        # Cluster integration samples towards the control points
        # tclu = np.reshape(
        # [i + util.cluster_cosine(50) for i in range(self.N - 1)], (-1,)
        # )
        # tclu = np.reshape(
        # [i + np.linspace(0.,1.,50) for i in range(self.N - 1)], (-1,)
        # )

        # m = np.linspace(0.,1.,40*(self.N-1))[1:-1]

        strain = np.trapezoid(self.kappa(self.mclu) ** 2.0, self.mclu)
        Ltot = np.sum(self.chords)
        return strain * Ltot**2.0

    @property
    def jerk(self):
        r"""Integrate rate of change of curvature to get scalar optimisation metric.

        The squared derivative of curvature integrated with respect to
        meridional distance, non-dimensional by the total arc length :math:`L`:

        .. math::
            \mathcal{J} = L^2 \int_{m=0}^1 \kappa'(m)^2 \dee m

        """

        # Cluster integration samples towards the control points
        # tclu = np.reshape(
        #     [i + util.cluster_cosine(50) for i in range(self.N - 1)], (-1,)
        # )
        jerk = np.trapezoid(
            np.gradient(self.kappa(self.mclu), self.mclu) ** 2.0, self.mclu
        )
        Ltot = np.sum(self.chords)
        return jerk * Ltot**2.0

    @property
    def non_monotonicity(self):
        """A metric that quantifies the amount of non-monotonicity in the curve.

        Roughly speaking, this is the area enclosed by each segment outside a
        bounding box with corners at successive control points, normalised by
        total arc length.

        """

        # Calculate bounding boxes for each segment
        xmin = np.minimum(self.x[:-1], self.x[1:])
        xmax = np.maximum(self.x[:-1], self.x[1:])
        rmin = np.minimum(self.r[:-1], self.r[1:])
        rmax = np.maximum(self.r[:-1], self.r[1:])

        # For a sampling vector, determine which bounding box should be checked
        m = np.linspace(0.0, 1.0, self.N * 100)[1:-1]
        t = self._pt(m)
        # tref = np.linspace(0.0, self.N - 1, self.N)
        ibin = np.digitize(t, self.t) - 1
        ibin[ibin == (self.N - 1)] -= 1  # Special case last bin

        # Evaluate coordinates
        xs, rs = self.xr(m)

        # Check magnitude of bounding box violations
        dx_lower = np.maximum(0.0, xmin[ibin] - xs)
        dx_upper = np.maximum(0.0, xs - xmax[ibin])
        dr_lower = np.maximum(0.0, rmin[ibin] - rs)
        dr_upper = np.maximum(0.0, rs - rmax[ibin])

        # Integrate
        delta = (
            np.trapezoid(dx_lower, m)
            + np.trapezoid(dx_upper, m)
            + np.trapezoid(dr_lower, m)
            + np.trapezoid(dr_upper, m)
        )
        Ltot = np.sum(self.chords)
        return delta / Ltot

    @property
    def smoothness_metric(self):
        # Relative weightings for optimisation metrics
        # TODO are these problem specific?
        weight_mono = 1e6
        # weight_jerk = 1e-2
        return (
            self.strain
            + self.non_monotonicity * weight_mono  # + self.jerk * weight_jerk
        )

    def smooth(self, slope_max=2.0):
        """Optimise parametric slope to minimise strain.

        This sets the parametric slope and parametric curvature to values that
        minimise integrated true curvature and rate of change of curvature
        along the meridional line, while maintaining a monotonic curve between
        each segment.

        This method is called automatically on initialisation, but can be
        called subsequently if the control points are changed.

        """

        # Slopes only
        def _iter(x):
            self.set_shape(x)
            return self.smoothness_metric

        x0 = self.sslope
        bnd_slope = ((1e-4, slope_max),)
        bound = bnd_slope * self.N
        opt = {"maxiter": 200}
        minimize(_iter, x0, bounds=bound, options=opt, tol=1e-7)

        # Set small curvatures to exactly zero
        # self.scurve[np.abs(self.scurve) < 0.05] = 0.0

        return self

    def _is_straight(self):
        """Determine which segments are straight lines."""
        tol_Beta = 0.1
        same_Beta = np.isclose(self.Beta[1:], self.Beta[:-1], atol=tol_Beta)
        dx = np.diff(self.x)
        dr = np.diff(self.r)
        dx[dx == 0.0] = 1e-16
        slope = dr / dx
        Beta_slope = np.degrees(np.arctan(slope))
        correct_slope = np.isclose(
            np.abs(Beta_slope), np.abs(self.Beta[1:]), atol=tol_Beta
        )
        return np.logical_and(same_Beta, correct_slope)

    def _xr_straight(self, t):
        """Evaluate meridional curve on straight path between control points."""
        xr = np.stack((self.x, self.r))
        text = self.t.copy()
        # Extend the interpolation range very slightly to avoid bounds errors
        # due to fluke numerical noise (but still raise error on gross violations)
        eps = 1e-9
        text[0] -= eps
        text[-1] += eps
        return scipy.interpolate.interp1d(text, xr)(t)


class Blade:
    """Encapsulate all information needed to generate a blade surface."""

    def __init__(
        self,
        spf,
        q_camber,
        q_thick,
        streamsurface,
        mstack,
        thick_type=None,
        camber_type=None,
        mlim=None,
        dtheta=None,
        theta_offset=0.0,
    ):
        """Initialise a blade row with geometry parameters.

        Parameters
        ----------
        spf : (nr,) array
            Span fractions to define section data at.
        thick_param : (nr, npt) array
            Vectors of thickness parameters at each radial location.
        cam_param : (nr, npc) array
            Vectors of camber parameters at each radial location.
        streamsurface : callable
            A function with the signature: ``xr = streamsurface(spf, m)``
            where ``0 <= spf <= 1`` is the spanwise location of a
            streamsurface, ``0 <= m <= 1`` is a 1-D vector of meridional
            locations on that streamsurface, and ``xr`` is a (2, len(m)) matrix
            of coordinates in the meridional plane.
        mlim : (nr, 2) array
            Meridional locations of LE and TE for each radial location.
        theta_offset : float
            Anular offset to be added to the coordinates.

        """
        self.thick_type = thick_type
        self.camber_type = camber_type
        self.theta_offset = theta_offset

        # Store input data
        self.streamsurface = streamsurface
        self.spf = np.reshape(spf, -1)
        # Sort everything by spf
        isort = np.argsort(self.spf)
        self.spf = self.spf[isort]
        N = len(self.spf)
        self.q_thick = np.reshape(q_thick, (N, -1))[isort]
        self.q_camber = np.reshape(q_camber, (N, -1))[isort]
        self.mstack = mstack
        if mlim is None:
            self.mlim = np.tile((0.0, 1.0), (N, 1))
        else:
            self.mlim = np.array(mlim)[isort]
        if dtheta is None:
            self.dtheta = np.zeros((N, 1))
        else:
            self.dtheta = np.array(dtheta)[isort]

    def get_pvec(self, isect=None):
        if isect is not None:
            qthick = self.q_thick[isect, :]
            qcam = self.q_camber[isect, :]
            mlim = self.mlim[isect, :]
        else:
            qthick = self.q_thick.reshape(-1)
            qcam = self.q_camber.reshape(-1)
            mlim = self.mlim.reshape(-1)
        toff = [
            self.theta_offset,
        ]
        return np.concatenate((qthick, qcam, mlim, toff))

    def get_bound(self, isect=None):
        Nspf, Nthick = self.q_thick.shape
        if isect is not None:
            Nspf = 1
        _, Ncam = self.q_camber.shape
        bound_thick = np.tile(self._Thick.qbound, (Nspf, 1))
        bound_cam = np.tile(self._Cam.qbound, (Nspf, 1))
        bound_mlim = np.tile(((-0.0, 0.0), (1.0, 1.0)), (Nspf, 1))
        dm = 1e-9
        if isect is not None:
            bound_mlim = np.array(
                [
                    [self.mlim[isect, 0] - dm, self.mlim[isect, 0] + dm],
                    [self.mlim[isect, 1] - dm, self.mlim[isect, 1] + dm],
                ]
            )
        else:
            bound_mlim = np.concatenate(
                [
                    [
                        [self.mlim[ispf, 0] - dm, self.mlim[ispf, 0] + dm],
                        [self.mlim[ispf, 1] - dm, self.mlim[ispf, 1] + dm],
                    ]
                    for ispf in range(len(self.spf))
                ],
                axis=0,
            )
        bound_toff = ((-np.pi, np.pi),)
        bound = np.concatenate((bound_thick, bound_cam, bound_mlim, bound_toff), axis=0)
        return bound

    def set_pvec(self, q, isect=None):
        self.theta_offset = q[-1]
        Nspf, Nthick = self.q_thick.shape
        _, Ncam = self.q_camber.shape
        if isect is not None:
            Nspf = 1
        ithick = Nthick * Nspf
        icam = ithick + (Ncam * Nspf)
        im = icam + 2 * Nspf
        if isect is not None:
            # print(q[icam:im],self.mlim[isect,:])
            # quit()
            self.q_thick[isect, :] = q[:ithick]
            self.q_camber[isect, :] = q[ithick:icam]
            self.mlim[isect, :] = q[icam:im]
        else:
            self.q_thick = q[:ithick].reshape(Nspf, Nthick)
            self.q_camber = q[ithick:icam].reshape(Nspf, Ncam)
            self.mlim = q[icam:im].reshape(Nspf, 2)

    @property
    def nsect(self):
        return len(self.spf)

    @property
    def _interp_method(self):
        if self.nsect == 1:
            return "const"
        elif self.nsect == 2:
            return "linear"
        elif self.nsect == 3:
            return "quadratic"
        else:
            return "cubic"

    @property
    def _Thick(self):
        if self.thick_type:
            Thick = getattr(turbigen.thickness, self.thick_type)
        else:
            Thick = turbigen.thickness.Taylor
        return Thick

    @property
    def _Cam(self):
        if self.camber_type:
            Cam = turbigen.camber.__dict__[self.camber_type]
        else:
            Cam = turbigen.camber.Quartic
        return Cam

    def _get_cam_thick(self, spf):
        # Create thickness and camber lines
        if len(self.spf) == 1:
            # Constant values
            thick = self._Thick(self.q_thick[0])
            cam = self._Cam(self.q_camber[0])
        else:
            # # Interpolate the parameters
            # qthick = scipy.interpolate.interp1d(
            #     self.spf,
            #     self.q_thick,
            #     fill_value="extrapolate",
            #     axis=0,
            #     kind=self._interp_method,
            # )
            # qcam = scipy.interpolate.interp1d(
            #     self.spf,
            #     self.q_camber,
            #     fill_value="extrapolate",
            #     axis=0,
            #     kind=self._interp_method,
            # )

            qcam = util.interp1d_linear_extrap(self.spf, self.q_camber)
            qthick = util.interp1d_linear_extrap(self.spf, self.q_thick)

            thick = self._Thick(qthick(spf).reshape(-1))
            cam = self._Cam(qcam(spf).reshape(-1))

        return cam, thick

    def _get_mlim(self, spf):
        return util.interp1d_linear_extrap(self.spf, self.mlim)(spf)

    def evaluate_section(self, spf, nchord=10000, ncusp=0, debug=False, m=None):
        """Coordinates of upper and lower surfaces at one span fraction."""

        cam, thick = self._get_cam_thick(spf)

        # Evaluate midspan meridional chord
        if m is None:
            m = util.cluster_cosine(nchord)

        # Get coordinates of the streamsurface to put this section on
        dydm = cam.dydm(m)
        chi = np.arctan(dydm)
        tau = thick.t(m)

        # Calculate offsets for perpendicular thickness in b2b plane
        Dm = -tau * np.sin(chi)
        Dy = tau * np.cos(chi)

        # Meridional positions for upper and lower surfaces
        mu = m + Dm
        ml = m - Dm

        # We need to convert the camber line meridional positions into LE/TE
        # meridional positions.
        mcam_LE = np.min((mu.min(), ml.min()))
        mcam_TE = np.max((mu.max(), ml.max()))
        mcam_ptp = mcam_TE - mcam_LE
        mu_LTE = (mu - mcam_LE) / mcam_ptp
        ml_LTE = (ml - mcam_LE) / mcam_ptp
        mcam = (m - mcam_LE) / mcam_ptp
        chord_full = util.arc_length(self.streamsurface(0.5, mcam))

        # now remap to mlim
        mlim = self._get_mlim(spf)
        mu_LTE = mlim[0] + np.ptp(mlim) * mu_LTE
        ml_LTE = mlim[0] + np.ptp(mlim) * ml_LTE
        mcam = mlim[0] + np.ptp(mlim) * mcam
        chord = util.arc_length(self.streamsurface(0.5, mcam))

        # Find coordinates on stream surface of upper/lower/camber points
        xru = self.streamsurface(spf, mu_LTE)
        xrl = self.streamsurface(spf, ml_LTE)
        xr = self.streamsurface(spf, mcam)

        # Project camber angle onto streamsurface
        theta = util.cumtrapz0(dydm / xr[1], mcam * chord_full)
        # Stack so that camber theta=0 at a certain position

        if self.mstack < 0.0:
            theta -= np.interp(-self.mstack, xr[1], theta)
        else:
            theta -= np.interp(self.mstack, m, theta)

        # Add on the whole blade angular offset
        theta += self.theta_offset
        theta += util.interp1d_linear_extrap(self.spf, self.dtheta)(spf)

        # drtu = Dy * chord
        # drtl = -Dy * chord

        # Change in theta if r were constant
        dtu = Dy * chord / xr[1]
        dtl = -Dy * chord / xr[1]
        # Change in rtheta at the average radius between camber and surf
        drtu = dtu * 0.5 * (xr[1] + xru[1])
        drtl = dtl * 0.5 * (xr[1] + xrl[1])

        xrrtu = np.stack((*xru, theta * xru[1] + drtu))
        xrrtl = np.stack((*xrl, theta * xrl[1] + drtl))

        xrtu = xrrtu + 0.0
        xrtu[2] /= xrtu[1]

        xrtl = xrrtl + 0.0
        xrtl[2] /= xrtl[1]

        # Add cusp if requested
        if ncusp:
            xrrtu = xrtu.copy()
            xrrtu[2] *= xrrtu[1]
            xrrtl = xrtl.copy()
            xrrtl[2] *= xrrtl[1]

            xrrt_te = 0.5 * (xrrtu[:, -1] + xrrtl[:, -1])
            dxrrt_te = xrrtu[:, -1] - xrrtl[:, -1]
            dxrrt_cam = np.mean(
                0.5
                * (np.diff(xrrtu, axis=-1)[:, -3:] + np.diff(xrrtl, axis=-1)[:, -3:]),
                axis=-1,
            )

            L_cusp = norm(dxrrt_te, 2) * 2.0
            dxrrt_cam /= norm(dxrrt_cam, 2)
            xrrt_point = (xrrt_te + L_cusp * dxrrt_cam).reshape(3, 1)
            cusp_frac = np.linspace(0.0, 1.0, ncusp).reshape(1, -1)
            xrrtu_cusp = cusp_frac * xrrt_point + (1.0 - cusp_frac) * xrrtu[
                :, -1
            ].reshape(3, 1)
            xrrtl_cusp = cusp_frac * xrrt_point + (1.0 - cusp_frac) * xrrtl[
                :, -1
            ].reshape(3, 1)
            xrrtu = np.concatenate((xrrtu, xrrtu_cusp), axis=-1)
            xrrtl = np.concatenate((xrrtl, xrrtl_cusp), axis=-1)

            xrtu = xrrtu.copy()
            xrtl = xrrtl.copy()
            xrtu[2] /= xrtu[1]
            xrtl[2] /= xrtl[1]

        return xrtu, xrtl

    def surface_length(self, spf):
        """Suction surface length."""
        xrtu, xrtl = self.evaluate_section(spf)
        xrrtu = np.stack((*xrtu[:2],) + (xrtu[1] * xrtu[2],))
        xrrtl = np.stack((*xrtl[:2],) + (xrtl[1] * xrtl[2],))
        Lu = util.arc_length(xrrtu)
        Ll = util.arc_length(xrrtl)
        return np.maximum(Lu, Ll)

    def get_camber_line(self, spf):
        cam, thick = self._get_cam_thick(spf)
        m = util.cluster_cosine(500)

        xrtul = np.stack(self.evaluate_section(spf, m=m), axis=0)
        xrtcam = np.mean(xrtul, axis=0)

        return xrtcam

    def get_LE_cent(self, spf, fac_Rle=1.0):
        """Get the centre of the leading edge."""

        # Make a meridional grid vector for just the le
        cam, thick = self._get_cam_thick(spf)
        Rle = thick.R_LE / fac_Rle
        m = util.cluster_cosine(500)
        xrtul = np.stack(self.evaluate_section(spf, m=m), axis=0)
        xrtul = xrtul[:, :, m < 2.0 * Rle]
        xrtcam = np.mean(xrtul, axis=0)
        xrtLE = xrtcam[:, np.argmax(m > Rle)]

        return xrtLE

    def get_nose(self, spf):
        """Get the nose of the aerofoil leading edge."""

        # Make a meridional grid vector for just the le
        cam, thick = self._get_cam_thick(spf)
        m = util.cluster_cosine(500)
        xrtul = np.stack(self.evaluate_section(spf, m=m), axis=0)
        xrtcam = np.mean(xrtul, axis=0)
        xrtLE = xrtcam[:, 0].squeeze()

        return xrtLE

    def get_coords(self, nspf=20, nchord=100, flip_theta=False):
        """3-D coordinates for this blade row in AutoGrid-style format.

        Parameters
        ----------
        nspf : int
            Number of sections in radial direction.
        nchord : int
            Number of chordwise points along each surface.

        Returns
        -------
        xrt : (2, nspf, nchord, 3) array
            Axial, radial, angular coordinates for this blade. `xrt[0]` is the
            upper surface, with highest theta, `xrt[1]` the lower surface.

        """

        eps = 1e-4
        xrt = np.stack(
            [
                self.evaluate_section(spf, nchord)
                for spf in np.linspace(-eps, 1.0 + eps, nspf)
            ]
        ).transpose(1, 0, 3, 2)

        if flip_theta:
            xrt[:, :, :, 2] *= -1.0

        return xrt

    def get_chi(self, spf):
        """Interpolate metal angles at a given span fraction."""
        cam, _ = self._get_cam_thick(spf)
        return cam.chi((0.0, 1.0))


class Machine:
    """Encapsulate all geometry needed for meshing in one object."""

    def __init__(self, ann, bld, Nb, tip, split):
        """Make a machine object from component objects."""

        self.ann = ann
        self.bld = bld
        self.Nb = Nb
        self.tip = tip
        self.split = split if split else None
        self.Nrow = len(self.bld)

    def get_coords(self, flip_theta=False):
        sections = []
        for b in self.bld:
            if b:
                sections.append(b[0].get_coords(flip_theta=flip_theta))
            else:
                sections.append([None, None])

        annulus = self.ann.get_coords()
        zcst = self.ann.get_interfaces()
        if self.split:
            split = []
            for irow in range(len(sections)):
                try:
                    split.append(self.split[irow].get_coords(flip_theta=flip_theta))
                except Exception:
                    split.append([None, None])
        else:
            split = None
        return sections, annulus, zcst, self.Nb, self.tip, split


class DiscreteMeridionalLine:
    def __init__(self, xr):
        """A discretised curve in the meridional plane."""

        xr = np.unique(xr.reshape(2, -1), axis=1)
        xr = xr[:, np.argsort(np.prod(xr, axis=0))]

        assert xr.shape[0] == 2
        assert xr.ndim == 2

        self._xr = xr

        self._poly_x = scipy.interpolate.PchipInterpolator(self.mp, self.xr[0])
        self._poly_r = scipy.interpolate.PchipInterpolator(self.mp, self.xr[1])
        self._nearest_mp = scipy.interpolate.NearestNDInterpolator(self.xr.T, self.mp)

        assert np.allclose(self.xr_from_mp(self.mp), self.xr)
        assert np.allclose(self.mp_from_xr(self.xr), self.mp)

    @property
    def xr(self):
        return self._xr

    @property
    def npts(self):
        return self.xr.shape[1]

    @property
    def rc(self):
        """Edge-centered radius."""
        return 0.5 * (self.xr[1, 1:] + self.xr[1, :-1])

    @property
    def mp(self):
        dxr = np.diff(self.xr, n=1, axis=1)
        dm = np.sqrt(np.sum(dxr**2.0, axis=0))
        mp = util.cumsum0(dm / self.rc)
        assert (np.diff(mp) > 0.0).all()
        return mp

    def mp_from_xr(self, xr):
        assert np.isin(self.xr, xr).all()
        xrf = xr.reshape(2, -1)
        return self._nearest_mp(*xrf).reshape(xr.shape[1:])

    def xr_from_mp(self, mp):
        return np.stack((self._poly_x(mp), self._poly_r(mp)))

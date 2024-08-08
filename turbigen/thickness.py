"""Classes to represent thickness distributions.

Each class needs to accept a 1D vector of parameters and have a `t(m)` method
that returns thickness as a function of meridional distance.

All lengths are normalised by the meridional chord.

The trailing edge thickness is input as the total thickness, with half
contributed by each side.

"""

import numpy as np
from scipy.optimize import minimize


class Taylor:
    """After Taylor (2016)."""

    def __init__(self, q_thick):
        r"""Two cubic splines in shape space.

        Parameters
        ----------
        q_thick: (6,) array
            Geometry parameter vector with elements:
                * Leading-edge radius, :math:`R_\LE\,`;
                * Maximum thickness, :math:`t_\max\,`;
                * Location of maximum thickness, :math:`m_\max\,`;
                * Curvature at maximum thickness, :math:`\kappa_\max\,`;
                * Trailing edge thickness, :math:`2t_\TE\,`;
                * Trailing edge wedge angle tangent, :math:`\tan\zeta\,`.


        Notes
        -----

        To construct a thickness distribution, we need a curvature-continuous
        function that starts with a rounded leading edge, increases to a
        maximum value :math:`t_\max` at a specified point :math:`m_\max`, and
        then monotonically decreases again. We accomplish this by defining two
        piecewise-cubic polynomials in in :cite:`Kulfan2008` shape-space as a
        function of normalised meridional coordinate,

        .. math::

            \tau(m) =
            \begin{cases}
                \tau_1(m) & \text{if } m \le m_\max\\
                \tau_2(m) & \text{if } m \gt m_\max
            \end{cases}\, ,

        where :math:`m=0` is the leading edge and :math:`m=1` is the trailing
        edge. In the case of a blunt trailing edge, the actual thickness is
        then obtained using the transformation,

        .. math::

            t(m) = \tau(m) \sqrt{m} (1 - m)  + m t_\TE\, ,

        where :math:`t_\TE` is the trailing edge thickness.

        To fix the point of maximum thickness, we need to constrain the value
        and slope of our shape-space polynomials at :math:`m_\max` such that
        :math:`t(m_\max) = t_\max` and :math:`t'(m_\max)=0`. Rearranging and
        differentiating the transformation equation implies,

        .. math::

            \begin{align}
                \tau(m_\max) &= \tau_\max =
                    \frac{t_\max - m_\max t_\TE}{\sqrt{m_\max}(1-m_\max)}\, ,\\
                \tau'(m_\max) &= \tau'_\max =
                    \frac{\tau_\max\left(\sqrt{m_\max}
                    - \frac{1-m_\max}{2\sqrt{m_\max}}\right)
                    - t_\TE}{\sqrt{m_\max}(1-m_\max)}\,.
            \end{align}

        We also introduce another two degrees of freedom: the curvature at
        maximum thickness :math:`\kappa_\max`, which sets how sharp or flat the
        thickness peak is; and the trailing edge wedge angle :math:`\tan\zeta`.

        The general form for a cubic contains four coefficients, thus requires
        four constraints to uniquely specify a curve. The first cubic extends
        from the leading edge to the maximum thickness location, with the
        following constraints:

        .. math::

            \begin{align}
                \tau_1(0) &= \sqrt{2R_\LE}\,, \\
                \tau_1(m_\max) &= \tau_\max\,, \\
                \tau'_1(m_\max) &= \tau'_\max\,, \\
                \tau''_1(m_\max) &= \kappa_\max\, . \\
            \end{align}

        Similarly for the second cubic from the maximum thickness
        location to the trailing edge:

        .. math::

            \begin{align}
                \tau_2(m_\max) &= \tau_\max\,, \\
                \tau'_2(m_\max) &= \tau'_\max\,, \\
                \tau''_2(m_\max) &= \kappa_\max\,, \\
                \tau_2(1) &= \tan\zeta + t_\TE\, . \\
            \end{align}

        Assembling our degrees of freedom into a parameter vector,

        .. math::

            q_\mathrm{thick} = [R_\LE, t_\max, m_\max,
            \kappa_\max, 2t_\TE, \tan\zeta]\, ,

        the value of :math:`q_\mathrm{thick}` allows us to solve for all
        polynomial coefficients and gives a unique solution for thickness distribution.

        To produce a rounded trailing edge, we use the alternative shape-space
        transformation,

        .. math::

            t(m) = \tau(m) \sqrt{m} \sqrt{1-m}\, .

        This class will produce a rounded trailing edge if
        :math:`t_\mathrm{TE}` is set to `NaN`. The trailing edge radius is then
        controlled by the value of :math:`\tan\zeta`.


        References
        ----------

        This parameterisation is based on :cite:`Taylor2016`. The shape-space
        transformation for aerofoil parameterisation was proposed by
        :cite:`Kulfan2008`.

        """

        # Record inputs
        self.q_thick = np.reshape(q_thick, 6)

        # Cache for polynomial coefficients
        self._coeff_cache = {}
        self._use_cache = True

    @property
    def R_LE(self):
        return self.q_thick[0]

    @property
    def t_max(self):
        return self.q_thick[1]

    @property
    def s_tmax(self):
        return self.q_thick[2]

    @property
    def kappa_max(self):
        return self.q_thick[3]

    @property
    def t_te(self):
        return self.q_thick[4]

    @property
    def tanwedge(self):
        return self.q_thick[5]

    eps = 1e-3
    qbound = (
        (eps, 0.1),
        (eps, 0.5),
        (0.05, 0.95),
        (-10.0, 10.0),
        (0.02, 0.2),
        (eps, 1.0),
    )

    @classmethod
    def from_fit(cls, m, t):
        r"""Initialise by fitting geometry parameters to coordinates.

        Given a set of points forming a discrete thickness distribution, fit
        geometry parameters to approximate the continuous distribution.

        Parameters
        ----------
        m : (N,) array
            Normalised meridional distance of coordinates to fit.
        t : (N,) array
            Thickness to fit, normalised by meridional chord length.

        Example
        -------

        Fit parameters to a fictional quadratic thickness distribution:

        .. plot::

            from turbigen.geometry import ThicknessDistribution
            m = np.linspace(0.,1,20)
            t = 0.2*m*(1.-m) + 0.01*m
            thk = ThicknessDistribution.from_fit(m,t)
            fig, ax = plt.subplots()
            ax.plot(m,t,'rx',label='Input')
            ax.plot(m,thk.t(m),'k-', label='Fit')
            ax.set_ylabel('Normalised Thickness, $t$')
            ax.set_xlabel('Normalised Meridional Coordinate, $m$')
            ax.legend()
            plt.tight_layout()
            plt.show()

        """

        p0 = np.array([0.05, 0.05, 0.35, 0.02, 0.02, 0.2])

        def _iter(p):
            thk_now = cls(p)
            t_fit = thk_now.t(m)
            return np.sum((t_fit - t) ** 2.0)

        eps = 1e-3
        bounds = (
            (eps, 0.5),
            (eps, 0.5),
            (eps, 1.0 - eps),
            (eps, 0.5),
            (eps, 0.5),
            (eps, 0.5),
        )

        popt = minimize(_iter, p0, bounds=bounds).x
        return cls(popt)

    def __hash__(self):
        """Hash based on tuple of the geometry parameters."""
        return hash(tuple(self.q_thick))

    def _to_shape(self, x, t, eps=1e-5):
        """Transform real thickness to shape space."""
        # Ignore singularities at leading and trailing edges
        ii = np.abs(x - 0.5) < (0.5 - eps)
        s = np.ones(x.shape) * np.nan
        if np.isnan(self.t_te):
            s[ii] = t[ii] / np.sqrt(x[ii]) / np.sqrt(1.0 - x[ii])
        else:
            s[ii] = (t[ii] - x[ii] * self.t_te / 2.0) / np.sqrt(x[ii]) / (1.0 - x[ii])
        return s

    def _from_shape(self, x, s):
        """Transform shape space to real coordinates."""
        if np.isnan(self.t_te):
            return np.sqrt(x) * np.sqrt(1.0 - x) * s
        else:
            return np.sqrt(x) * (1.0 - x) * s + x * self.t_te / 2.0

    @property
    def _coeff(self):
        """Coefficients for piecewise polynomials in shape space."""

        # Skip if we have already fitted polynomials for current params
        if self._use_cache and (hash(self) in self._coeff_cache):
            return self._coeff_cache[hash(self)]

        # Evaluate control points
        sle = np.sqrt(2.0 * self.R_LE)
        t_te = self.t_te
        if np.isnan(t_te):
            t_te = 0.0
            smax = self.t_max / np.sqrt(self.s_tmax) / np.sqrt(1.0 - self.s_tmax)
            dsmax = (
                smax
                / 2.0
                * (2.0 * self.s_tmax - 1.0)
                / self.s_tmax
                / (1.0 - self.s_tmax)
            )
        else:
            smax = (
                (self.t_max - self.s_tmax * t_te / 2.0)
                / np.sqrt(self.s_tmax)
                / (1.0 - self.s_tmax)
            )
            dsmax = (
                (
                    smax
                    * (
                        np.sqrt(self.s_tmax)
                        - (1.0 - self.s_tmax) / 2.0 / np.sqrt(self.s_tmax)
                    )
                    - t_te / 2.0
                )
                / np.sqrt(self.s_tmax)
                / (1.0 - self.s_tmax)
            )

        ste = t_te + self.tanwedge

        # For brevity
        x3 = self.s_tmax**3.0
        x2 = self.s_tmax**2.0
        x1 = self.s_tmax

        # Fit front cubic
        A = np.empty((4, 4))
        b = np.empty((4, 1))

        # LE radius
        A[0] = [0.0, 0.0, 0.0, 1.0]
        b[0] = sle

        # Value of max thickness
        A[1] = [x3, x2, x1, 1.0]
        b[1] = smax

        # Slope at max thickness
        A[2] = [3.0 * x2, 2.0 * x1, 1.0, 0.0]
        b[2] = dsmax

        # Curvature at max thickness
        A[3] = [6.0 * x1, 2.0, 0.0, 0.0]
        b[3] = self.kappa_max

        coeff_front = np.linalg.solve(A, b).reshape(-1)

        # Fit rear cubic
        # TE thick/wedge (other points are the same)
        A[0] = [1.0, 1.0, 1.0, 1.0]
        b[0] = ste

        coeff_rear = np.linalg.solve(A, b).reshape(-1)

        coeff = np.stack((coeff_front, coeff_rear))

        if self._use_cache:
            self._coeff_cache[hash(self)] = coeff

        return coeff

    def tau(self, s):
        r"""Thickness in shape space as function of normalised meridional distance.

        Parameters
        ----------
        s: array
            Fractions of normalised meridional distance to evaluate at.

        Returns
        -------
        t: array
            Samples of thickness distribution at the requested points.
        """

        s = np.array(s)

        coeff_front, coeff_rear = self._coeff

        tau = np.empty_like(s)
        tau[s <= self.s_tmax] = np.polyval(coeff_front, s[s <= self.s_tmax])
        tau[s > self.s_tmax] = np.polyval(coeff_rear, s[s > self.s_tmax])
        return tau

    def t(self, m):
        r"""Thickness as function of normalised meridional distance.

        Parameters
        ----------
        m: (N) array
            Fractions of normalised meridional distance to evaluate at.

        Returns
        -------
        t: (N) array
            Samples of thickness distribution at the requested points :math:`t(m)`.

        """
        return self._from_shape(m, self.tau(m))


class Impeller:
    """For radial impellers."""

    def __init__(self, q_thick):
        r"""A constant thickness with curvature-continuous elliptical leading edge

        Parameters
        ----------
        q_thick: (1,) array
            Geometry parameter vector with elements:
                * Maximum thickness, :math:`t_\max\,`;
                * Ellipse ratio, :math:`a\,`;
                * Meridional location of maximum thickness, :math:`m_\max\,`.
                * Trailing edge thickness, :math:`t_\TE\,`;

        """

        # Record inputs
        self.q_thick = np.reshape(q_thick, 4)

        # Make a circular LE curvature-continuous by adding a blending polynomial
        # t = sqrt(x(2R-x)) + x^3/2/R^2 - x^2/R + x/2
        # Then scale x-coord to make ellipse

        # Calculate coefficients for blending polynomial
        # Scaling factor on x coord
        # 1.25 empirically chosen to fit a real curvature-discontinous circle
        # Then divide by ellipse ratio
        fm = 1.0 / 1.25 / self.a
        R = self.tmax
        A = 0.5 / R**2.0 * fm**3.0
        B = -1.0 / R * fm**2.0
        C = 0.5 * fm
        self._coeffs = np.array([A, B, C])
        self._fm = fm

        # Trailing edge blend

        b = np.empty((4, 1))
        A = np.empty((4, 4))

        # t(1) = tTE
        A[0] = [1.0, 1.0, 1.0, 1.0]
        b[0] = self.tTE / 2.0

        # t(mm) = tmax
        mm = self.mmax
        A[1] = [mm**3, mm**2, mm, 1.0]
        b[1] = self.tmax

        # t'(mm) = 0
        A[2] = [3.0 * mm**2.0, 2.0 * mm, 1.0, 0.0]
        b[2] = 0.0

        # t''(mm) = 0
        A[3] = [6.0 * mm, 2.0, 0.0, 0.0]
        b[3] = 0.0

        self._coeffTE = np.linalg.solve(A, b).reshape(-1)

    @property
    def a(self):
        return self.q_thick[1]

    @property
    def tmax(self):
        return self.q_thick[0]

    @property
    def tTE(self):
        return self.q_thick[3]

    @property
    def R_LE(self):
        return self.tmax / self.a * 5.0 / 2.0

    @property
    def mmax(self):
        return self.q_thick[2]

    def t(self, m):
        r"""Thickness as function of normalised meridional distance.

        Parameters
        ----------
        m: (N) array
            Fractions of normalised meridional distance to evaluate at.

        Returns
        -------
        t: (N) array
            Samples of thickness distribution at the requested points :math:`t(m)`.

        """

        t = np.ones_like(m) * self.tmax
        fm = self._fm
        R = self.tmax
        iLE = m <= (R / fm)
        iTE = m > (self.mmax)
        A, B, C = self._coeffs

        t[iLE] = (
            np.sqrt(m[iLE] * fm * (2.0 * R - m[iLE] * fm))
            + A * m[iLE] ** 3
            + B * m[iLE] ** 2
            + C * m[iLE]
        )
        t[iTE] = np.polyval(self._coeffTE, m[iTE])

        return t

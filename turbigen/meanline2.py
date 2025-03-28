"""Define the interface for mean-line designers."""

from abc import abstractmethod
from turbigen import util
import numpy as np
import turbigen.flowfield


class MeanLineDesigner(util.BaseDesigner):
    """Define the interface for a mean-line designer."""

    _supplied_design_vars = "So1"

    nominal: None
    actual: None

    rtol: float = 0.05
    atol: float = 0.01

    @staticmethod
    @abstractmethod
    def forward(*args, **kwargs):
        """Use design variables to calculate flow field along mean line."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(mean_line):
        """Calculate design variables from mean line flow field."""
        raise NotImplementedError

    def setup_mean_line(self, So1):
        """Calculate the nominal mean line flow field from stored design variables."""
        self.nominal = turbigen.flowfield.make_mean_line(
            *self.forward(So1=So1, **self.design_vars)
        )

    def check_backward(self, mean_line):
        """Check the backward calculation of design variables."""
        params_inv = self.backward(mean_line)
        # Compare forward and inverse params, check within a tolerance
        for k, v in self.design_vars.items():
            if k not in params_inv:
                raise Exception(
                    f"Design variable {k} not returned by inverse function."
                )
            # Allow uncalculated variables to be None
            if params_inv[k] is None:
                continue

            # Compare the value of the design variable to nominal
            if np.all(v == 0.0):
                # Absolute tolerance for zero values
                if np.allclose(v, params_inv[k], atol=self.atol):
                    continue
            else:
                # Relative tolerance for non-zero values
                if np.allclose(v, params_inv[k], rtol=self.rtol):
                    continue

            raise Exception(
                f"Meanline inverted {k}={params_inv[k]} not same as nominal value {v}"
            )


class TurbineCascade(MeanLineDesigner):
    @staticmethod
    def forward(So1, span, Alpha, Ma2, Yh=0.0, htr=0.99, RR=1.0, Beta=(0.0, 0.0)):
        r"""A single-row stationary turbine cascade.

        Parameters
        ----------
        span: (2,) array
            Inlet and outlet spans [m].
        Alpha: (2,) array
            Inlet and outlet yaw angles [deg].
        Ma2: float
            Exit Mach number [--].
        Yh: float
            Estimate of the row energy loss coefficient [--].
        htr: float
            Inlet hub-to-tip radius ratio [--]. Defaults to just less than
            unity to approximate a linear cascade.
        RR: float
            Outlet to inlet radius ratio [--].
        Beta: (2,) array
            Inlet and outlet pitch angles [deg] Only makes sense
            to be non-zero if radius ratio is not unity.

        Returns
        -------
        rrms: (2,) array
            Mean radii at inlet and outlet, [m].
        A: (2,) array
            Annulus areas at inlet and outlet, [m^2].
        Omega: (2,) array
            Shaft angular velocities, zero for this case.
        Vxrt: (3, 2) array
            Velocity components at inlet and outlet [m/s].
        S: (2,) FlowField
            Static states at inlet and outlet.

        """

        util.check_scalar(Ma2=Ma2, Yh=Yh, htr=htr)
        util.check_vector((2,), span=span, Alpha=Alpha, Beta=Beta)

        # Trig
        cosBeta = util.cosd(Beta)
        cosAlpha = util.cosd(Alpha)
        tanAlpha = util.tand(Alpha)

        # Evaluate geometry first
        span_rm1 = (1.0 - htr) / (1.0 + htr) * 2.0 / cosBeta[0]
        rm1 = span[0] / span_rm1
        rm = np.array([1.0, RR]) * rm1
        rh = rm - 0.5 * span * cosBeta
        rt = rm + 0.5 * span * cosBeta
        rrms = np.sqrt(0.5 * (rh**2.0 + rt**2.0))
        A = 2.0 * np.pi * rm * span
        Aflow = A * cosAlpha

        # We will have to guess an entropy rise, then update it according to the
        # loss coefficients and Mach number
        ds = 0.0
        err = np.inf
        atol_Ma = 1e-7
        Ma1 = 0.0

        for _ in range(10):
            # Conserve energy to get exit stagnation state
            So2 = So1.copy().set_h_s(So1.h, So1.s + ds)

            # Static states
            S2 = So2.to_static(Ma2)
            S1 = So1.to_static(Ma1)

            # Velocities from Mach number
            V2 = S2.a * Ma2
            Vt2 = V2 * np.sqrt(tanAlpha[1] ** 2.0 / (1.0 + tanAlpha[1] ** 2.0))
            Vm2 = np.sqrt(V2**2.0 - Vt2**2.0)

            # Mass flow and inlet static state
            mdot = S2.rho * Vm2 * A[-1]
            Vm1 = mdot / S1.rho / A[0]
            Vt1 = tanAlpha[0] * Vm1
            V1 = np.sqrt(Vm1**2.0 + Vt1**2.0)

            # Update inlet Mach
            Ma1_new = V1 / S1.a
            err = Ma1 - Ma1_new
            Ma1 = Ma1_new

            if np.abs(err) < atol_Ma:
                break

            # Update loss using appropriate definition
            horef = So1.h
            href = S2.h

            # Ideal state is isentropic to the exit static pressure
            S2s = S2.copy().set_P_s(S2.P, So1.s)
            h2_new = S2s.h + Yh * (horef - href)
            S2_new = S2.copy().set_P_h(S2.P, h2_new)
            ds = S2_new.s - So1.s

        # Verify the loop has converged
        Yh_out = (S2.h - S2s.h) / (horef - href)
        assert np.isclose(Yh_out, Yh, atol=1e-3)

        # Assemble the data
        S = S1.stack((S1, S2))
        Ma = np.array((Ma1, Ma2))
        V = S.a * Ma
        Vxrt = np.stack(util.angles_to_velocities(V, Alpha, Beta))
        Omega = np.zeros_like(Vxrt[0])

        return rrms, A, Omega, Vxrt, S

    @staticmethod
    def backward(mean_line):
        """Reverse a cascade mean-line to design variables.

        Parameters
        ----------
        ml: MeanLine
            A mean-line object specifying the flow in a cascade.

        Returns
        -------
        out : dict
            Dictionary of aerodynamic design parameters with fields:
            `span1`, `span2`, `Alpha1`, `Alpha2`, `Ma2`, `Yh`, `htr`, `RR`, `Beta`.
            The fields have the same meanings as in :func:`forward`.
        """
        ml = mean_line
        # Pull out states
        S2s = ml.empty().set_P_s(ml.P[-1], ml.s[0])

        # Loss coefficient
        horef = ml.ho[0]
        if ml.ARflow[0] >= 1.0:
            # Compressor
            href = ml.h[0]
        else:
            # Turbine
            href = ml.h[1]
        Yh_out = (ml.h[1] - S2s.h) / (horef - href)
        Ys = ml.T[1] * (ml.s[1] - ml.s[0]) / (horef - href)

        out = {
            "span": ml.span,
            "Alpha": ml.Alpha,
            "Ma2": ml.Ma[1],
            "Yh": Yh_out,
            "Ys": Ys,
            "htr": ml.htr[0],
            "RR": ml.RR[0],
            "Beta": ml.Beta.tolist(),
        }

        return out

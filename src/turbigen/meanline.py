"""Define the interface for mean-line designers."""

from abc import abstractmethod
from turbigen import util
import numpy as np
import turbigen.flowfield
from scipy.optimize import fsolve, root_scalar

logger = util.make_logger()


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


class AxialTurbine(MeanLineDesigner):
    @staticmethod
    def forward(
        So1,
        psi,
        phi2,
        zeta,
        Ma2,
        fac_Ma3_rel,
        mdot,
        Ys,
        rrms,
    ):
        def iter_Alpha1(
            So1,
            psi,
            phi2,
            zeta,
            Ma2,
            fac_Ma3_rel,
            Alpha1,
            mdot,
            Ys,
            rrms,
        ):
            r"""Design the mean-line for an axial turbine stage.

            Parameters
            ----------
            So1: State
                Object specifing the working fluid and its state at inlet.


            Returns
            -------
            ml: MeanLine
                An object specifying the flow along the mean line.

            """

            # Can we change to controlling Ma2_rel?

            # Verify input scalars
            util.check_scalar(
                psi=psi,
                phi2=phi2,
                Ma2=Ma2,
                fac_Ma3_rel=fac_Ma3_rel,
                Alpha1=Alpha1,
                mdot=mdot,
                rrms=rrms,
            )

            # Check shapes of vectors
            util.check_vector((2,), zeta=zeta, Ys=Ys)

            # Use pseudo entropy loss coefficient to guess entropy
            # throughout the machine (update later based on CFD solution)
            Tref = So1.T
            dhead_ref = 0.5 * So1.a**2
            # Ys = To1*(s-s1)/(0.5*a01^2)
            s = np.concatenate(((0.0,), (Ys[0],), Ys)) * dhead_ref / Tref + So1.s

            # Define rotor Mach as offset from stator Mach
            Ma3_rel = fac_Ma3_rel * Ma2

            # Guess a blade speed
            U = So1.a * Ma2 * 0.5

            # Preallocate and loop
            So = So1.empty(shape=(4,)).set_h_s(So1.h, s)
            S = So.copy()
            MAXITER = 100
            RTOL = 1e-6
            for _ in range(MAXITER):
                # Axial velocities
                Vx2 = U * phi2
                Vx = np.array([zeta[0], 1.0, 1.0, zeta[1]]) * Vx2

                # Inlet flow angle sets inlet tangential velocity
                Vt1 = Vx[0] * np.tan(np.radians(Alpha1))

                # Stator exit velocity from Mach
                V2 = Ma2 * S.a[1]
                assert V2 > Vx2
                Vt2 = np.sqrt(V2**2 - Vx2**2)

                # Rotor exit relative velocity from rel Mach
                V3_rel = Ma3_rel * S.a[3]
                Vt3_rel = -np.sqrt(V3_rel**2 - Vx[3] ** 2)
                Vt3 = Vt3_rel + U

                # Stagnation enthalpy using Euler work equation
                Vt = np.array([Vt1, Vt2, Vt2, Vt3])
                ho1 = ho2 = So.h[0]
                ho3 = ho2 + U * (Vt3 - Vt2)
                ho = np.array([ho1, ho2, ho2, ho3])
                h = ho - 0.5 * (Vx**2 + Vt**2)

                # Update the states
                So.set_h_s(ho, s)
                S.set_h_s(h, s)

                # New guess for blade speed
                Unew = np.sqrt((ho1 - ho3) / psi)

                # Check convergence
                dU = Unew - U
                if np.abs(dU) < RTOL * U:
                    break
                else:
                    U = Unew

            # Conservation of mass to get areas
            A = mdot / S.rho / Vx

            # Prescribe the rotor radius
            rrms = np.full((4,), rrms)

            # Angular velocity
            Omega = U / rrms * np.array([0, 0, 1, 1])

            # Assemble velocity components
            Vxrt = np.stack((Vx, np.zeros_like(Vx), Vt))

            Alpha3 = np.arctan2(Vt[-1], Vx[-1]) * 180 / np.pi

            return (rrms, A, Omega, Vxrt, S), Alpha3

        # Guess Alpha1
        Alpha1 = 0.0
        atol = 0.1

        MAXITER = 100
        converged = False
        for _ in range(MAXITER):
            out, Alpha3 = iter_Alpha1(
                So1,
                psi,
                phi2,
                zeta,
                Ma2,
                fac_Ma3_rel,
                Alpha1,
                mdot,
                Ys,
                rrms,
            )
            err = np.abs(Alpha3 - Alpha1)

            if err < atol:
                converged = True
                break
            else:
                Alpha1 = Alpha3

        if not converged:
            raise ValueError(f"Alpha1 iteration did not converge: {Alpha1} -> {Alpha3}")

        return out

    @staticmethod
    def backward(mean_line):
        """Reverse a turbine stage mean-line to design variables.

        Parameters
        ----------
        mean_line: MeanLine
            A mean-line object specifying the flow in an axial turbine.

        Returns
        -------
        out : dict
            Dictionary of aerodynamic design parameters with fields:
                - So1: State
                - PRtt: float
                - psi: float
                - phi2: float
                - zeta: float
                - Ma2: float
                - DMa3_rel: float
                - Alpha1: float
                - mdot: float
                - Ys: (float, float, float)

        """

        U2 = mean_line.U[2]
        Vx2 = mean_line.Vx[2]
        Ma2 = mean_line.Ma[2]

        # Calculate pseudo entropy loss coefficient
        Tref = mean_line.To[0]
        dhead_ref = 0.5 * mean_line.ao[0] ** 2
        sref = mean_line.s[0]
        s = mean_line.s[
            (1, 3),
        ]
        Ys = (s - sref) * Tref / dhead_ref

        # Calculate axial velocity ratios
        zeta = (
            mean_line.Vx[
                (0, 3),
            ]
            / Vx2
        )

        # Reaction
        h = mean_line.h
        Lam = (h[1] - h[0]) / (h[3] - h[0])

        phi2 = Vx2 / U2
        Alpha1 = mean_line.Alpha[0]
        psi_rep = 2 * (1 - Lam - phi2 * np.tan(np.radians(Alpha1)))

        # Assemble the dict
        out = {
            "PR_tt": mean_line.PR_tt,
            "psi": -(mean_line.ho[3] - mean_line.ho[0]) / U2**2,
            "phi2": phi2,
            "zeta": zeta,
            "Ma2": Ma2,
            "fac_Ma3_rel": mean_line.Ma_rel[3] / mean_line.Ma[1],
            "Ma3_rel": mean_line.Ma_rel[3],
            "Alpha1": mean_line.Alpha[0],
            "mdot": mean_line.mdot[0],
            "Lam": Lam,
            "Ys": tuple(Ys),
            "htr2": mean_line.htr[1],
            "rrms": mean_line.rrms[0],
        }

        return out

"""Define the interface for mean-line designers."""

from turbigen import util
import numpy as np
import turbigen.flowfield
from scipy.optimize import brentq
import turbigen.meanline

logger = util.make_logger()


class AxialTurbinePsi(turbigen.meanline.MeanLineDesigner):
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
            "Alpha1": mean_line.Alpha[0],
            "mdot": mean_line.mdot[0],
            "Lam": Lam,
            "Ys": tuple(Ys),
            "htr2": mean_line.htr[1],
            "rrms": mean_line.rrms[0],
        }

        return out

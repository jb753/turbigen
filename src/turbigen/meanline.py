"""Define the interface for mean-line designers."""

from abc import abstractmethod
from turbigen import util
import numpy as np
import turbigen.flowfield

logger = util.make_logger()


class MeanLineDesigner(util.BaseDesigner):
    r"""
    Mean-line
    =========

    The first step in turbomachinery design is a one-dimensional analysis along
    a representative 'mean-line', a simplified model of the true
    three-dimensional flow. We consider an axisymmetric streamsurface at an
    intermediate position between hub and casing, with stations at the
    inlet and exit of each blade row.

    The inputs to the mean-line design are the inlet condition and the machine duty.
    Specifying some aerodynamic design variables and applying conservation of
    mass, momentum and energy, we can calculate the outputs of
    annulus areas, mean radii and flow angles at each station.

    The mean-line design process is different for each machine architecture:
    compressor/turbine, axial/radial, and so on. :program:`turbigen` provides
    the built-in architectures listed below, and also allows considerable flexibility
    in defining your own :ref:`ml-custom`.

    xxx

    .. _ml-custom:

    Custom architectures
    --------------------

    Custom architectures are defined by subclassing the :class:`MeanLineDesigner`.
    First, set the user plugin directory in the configuration file, e.g. to a new folder called `plug` in the current directory by adding the following line:

    .. code-block:: yaml

        plugdir: ./plug

    Then, create a new Python file in the `plug` directory, e.g.
    `custom.py`, and define a new class that inherits from
    :class:`MeanLineDesigner` like this:

    .. code-block:: python

        # File: ./plug/custom.py

        import turbigen.meanline

        class MyCustomMeanLine(turbigen.meanline.MeanLineDesigner):

            # Your design variables are arguments to the forward method
            @staticmethod
            def forward(So1, phi, psi, Ma1):
                '''Use design variables to calculate flow field.

                Parameters
                ----------
                So1: Fluid
                    The working fluid and its thermodynamic state at inlet.
                phi: float
                    Flow coefficient at inlet.
                psi: float
                    Stage loading coefficient.
                Ma1: float
                    Inlet Mach number.
                ... your chosen design variables ...

                Returns
                -------
                rrms: (2*nrow,) array
                    Mean radii at all stations [m].
                A: (2*nrow,) array
                    Annulus areas at all stations [m^2].
                Omega: (2*nrow,) array
                    Shaft angular velocities at all stations [rad/s].
                Vxrt: (3, 2*nrow) array
                    Velocity components at all stations [m/s].
                S: (2*nrow,) list of Fluid
                    Static states for all stations.

                '''

                # Your code here...
                raise NotImplementedError("Implement the forward method")

                # Manipulate thermodynamic states by copying the inlet
                # add setting new property values, say
                V1 = Ma1 * So1.a  # Approx, should iterate this
                h1 = So1.h - 0.5 * V1**2
                S1 = So1.copy().set_h_s(h1 So1.s)

                # Collect the static states
                S = [S1, S2]

                return rrms, A, Omega, Vxrt, S

            @staticmethod
            def backward(mean_line):
                '''Calculate design variables from flow field.

                Parameters
                ----------
                mean_line: MeanLine
                    Flow field along the mean line.

                Returns
                -------
                out : dict
                    Dictionary of design variables, keyed by arguments
                    to the `forward` method.

                '''

                # The mean_line object has all the flow field data
                # and calculates most composite quantities like
                # velocity components, stagnation enthalpy, for you

                # Blade speed at station 1 (first row inlet)
                U = mean_line.U[0]

                return {
                    # Inlet flow coefficient
                    'phi': mean_line.Vm[0] / U,
                    # Stage loading coefficient
                    'psi': (mean_line.ho[-1] - mean_line.ho[0]) / U**2,
                    # Mach number at inlet
                    'Ma1': mean_line.Ma[0],
                    # Your design variables here...
                    # ...
                    # Other keys are printed to the log file and saved to the
                    # output configuration file
                    'eta_tt': mean_line.eta_tt,
                    'Alpha1': mean_line.Alpha[0],
                    'DH': mean_line.V_rel[1] / mean_line.V_rel[0],
                }

    You will need to implement two static methods: `forward()` and `backward()`.

    The `forward()` function takes as arguments an inlet state object and
    some duty, geometric, or aerodynamic design variables. This function
    returns all information required to lay out the mean-line: radii, annulus
    areas, angular velocities, velocity components, and thermodynamic states at
    the inlet and exit of each blade row.

    The entries in the `mean_line` part of the configuration file are fed into
    `forward()` as keyword arguments. For example, if the configuration file
    contains:

    .. code-block:: yaml

        mean_line:
            type: my_custom_mean_line
            phi: 0.8
            psi: 1.6
            Ma1: 0.5

    Then, within :program:`turbigen`, the `type` key identifies
    the `MyCustomMeanLine` class and calls its `forward()` method like:

    .. code-block:: python

        MyCustomMeanLine.forward(
            So1,  # Inlet state calculated elsewhere, positional arg
            phi=0.8, # Keys from `mean_line` config unpacked as kwargs
            psi=1.6,
            Ma1=0.5,
        )

    Some notes on implementing the `forward()` method:

    * Retain generality of the working fluid by using the set property methods
      of the `Fluid` class, as in the example above. This is preferable to
      hard-coding calculations assuming a specific equation of state such as
      ideal gas.
    * Specify aerodynamic design variables instead of geometric ones, e.g.
      flow coefficient and Mach number instead of shaft
      angular velocity and radius ratio. Constraining geometry can lead to
      feasible designs only over a narrow range of duty, with many infeasible designs with no solution to the mean-line equations. It is more
      straighforward to map out a design space by varying independent variables
      within their natural aerodynamic bounds.
    * Controlling Mach number prevents choking when moving around the
      design space. It is imperative to limit deceleration through compressors
      to avoid flow separation, so specifying a relative velocity ratio or de
      Haller number is a good idea. Letting meridional velocity float can lead
      to wide variations in span and hence unfavourable high-curvature annulus
      lines, so controlling the change in meridional velocity through the
      machine is advisable.
    * Loss is best handled by guessing a vector of entropy rise at each
      station, which does not depend on the frame of reference. The values can then be
      updated using CFD results.
    * Iteration is required to solve for density in compressible cases. It is
      often easiest to guess a value for the blade speed, then iterate to converge
      on a blade speed that satisfies the requested duty. Matching a
      total-to-static pressure ratio requires iteration because the exit
      dynamic head is not known a priori.

    The `backward()` function takes a mean-line flow field as its only
    argument, and calculates a dictionary of the arguments to `forward()`.
    Given a suitably averaged CFD solution, `backward()` is a post-processing
    step that allows comparison of the three-dimensional simulated flow field
    to the one-dimensional design intent. Also, feeding the output of
    `forward()` straight into `backward()` acts as a check that the mean-line
    design is consistent with the requested inputs. `backward()` can also be
    used to post-process other quantities of interest from the mixed-out CFD
    flow field. Extra keys are printed the log file and saved to the output configuration
    file; only keys that are also design variables will be fed back
    into `forward()` for the consistency check.

    """

    _supplied_design_vars = "So1"

    nominal: None
    actual: None

    rtol: float = 0.05
    atol: float = 0.01

    @staticmethod
    @abstractmethod
    def forward(So1, *args, **kwargs):
        r"""Use design variables to calculate flow field along mean line.

        Parameters
        ----------
        So1: Fluid
            The working fluid and its thermodynamic state at inlet.
        *args, *kwargs:
            Design variables for the mean line, as defined in the subclass.

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
        S: (2,) Fluid
            Static states at inlet and outlet.

        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def backward(mean_line):
        """Calculate design variables from mean line flow field.

        Parameters
        ----------
        mean_line: MeanLine
            Flow field along the mean line.

        Returns
        -------
        out : dict
            Dictionary of design variables, keyed by arguments to the `forward`
            method.

        """
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
    """
    Turbine cascade
    ---------------

    A single-row, stationary turbine cascade. The geometry is defined by spans,
    flow angles and an exit Mach number. By default, the cascade is approximately linear,
    with a hub-to-tip ratio close to unity, no radius change, and no pitch
    angle. An annular cascade can be defined by specifying a radius ratio and
    pitch angles.


    """

    _design_vars = {
        "span": ("Inlet and outlet spans [m]", (2,), None),
        "Alpha": ("Inlet and outlet yaw angles [deg]", (2,), None),
        "Ma2": ("Exit Mach number [--]", (), None),
        "Yh": ("Energy loss coefficient [--]", (), None),
        "htr": ("Inlet hub-to-tip radius ratio [--]", (), 0.99),
        "RR": ("Outlet to inlet radius ratio [--]", (), 1.0),
        "Beta": ("Inlet and outlet pitch angles [deg]", (2,), (0.0, 0.0)),
    }

    @staticmethod
    def forward(So1, span, Alpha, Ma2, Yh, htr=0.99, RR=1.0, Beta=(0.0, 0.0)):
        util.check_scalar(Ma2=Ma2, Yh=Yh, htr=htr)
        util.check_vector((2,), span=span, Alpha=Alpha, Beta=Beta)

        # Trig
        cosBeta = util.cosd(Beta)
        tanAlpha = util.tand(Alpha)

        # Evaluate geometry first
        span_rm1 = (1.0 - htr) / (1.0 + htr) * 2.0 / cosBeta[0]
        rm1 = span[0] / span_rm1
        rm = np.array([1.0, RR]) * rm1
        rh = rm - 0.5 * span * cosBeta
        rt = rm + 0.5 * span * cosBeta
        rrms = np.sqrt(0.5 * (rh**2.0 + rt**2.0))
        A = 2.0 * np.pi * rm * span

        # We will have to guess an entropy rise, then update it according to the
        # loss coefficients and Mach number
        ds = 0.0
        err = np.inf
        atol_Ma = 1e-7
        Ma1 = 0.0

        for _ in range(20):
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
        S = [S1, S2]
        Ma = np.array((Ma1, Ma2))
        a = np.array((S1.a, S2.a))
        V = a * Ma
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
    """
    Axial turbine
    -------------

    A repeating-stage axial turbine. Duty is set by a mass flow rate and a constant mean radius.
    Vane exit Mach number is set directly, while the rotor exit relative Mach
    number is set by a scaling factor off the vane value. This allows compressibility effects to be predominantly controlled by `Ma2`; the degree of reaction is controlled by `fac_Ma3_rel`, with 50% reaction corresponding approximately to unity.
    The default is constant axial velocity, but this can be controlled by `zeta`.
    Pressure ratio and shaft speed are dependent variables under this parameterisation.

    """

    _design_vars = {
        "mdot": ("Mass flow rate [kg/s]", (), None),
        "rrms": ("Root-mean-square radius [m]", (), None),
        "psi": ("Stage loading coefficient [--]", (), None),
        "phi2": ("Rotor inlet flow coefficient [--]", (), None),
        "Ma2": ("Vane exit Mach number [--]", (), None),
        "fac_Ma3_rel": ("Rotor exit relative Mach factor [--]", (), None),
        "Ys": ("Entropy loss coefficients [--]", (2,), None),
        "zeta": ("Axial velocity ratios [--]", (2,), (1.0, 1.0)),
    }

    @staticmethod
    def forward(
        So1,
        psi,
        phi2,
        Ma2,
        fac_Ma3_rel,
        mdot,
        Ys,
        rrms,
        zeta=(1.0, 1.0),
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
            MAXITER = 500
            RTOL = 1e-6
            rf = 0.5
            convU = False
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
                    convU = True
                    break
                else:
                    U = Unew * rf + U * (1.0 - rf)

            if not convU:
                raise ValueError(f"U iteration did not converge: {U} -> {Unew}")

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

        # Assemble the dict
        out = {
            "PR_ts": mean_line.PR_ts,
            "PR_tt": mean_line.PR_tt,
            "psi": (mean_line.ho[0] - mean_line.ho[3]) / U2**2,
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
            "eta_tt": mean_line.eta_tt,
            "eta_ts": mean_line.eta_ts,
            "Omega": mean_line.Omega[-1],
        }

        return out

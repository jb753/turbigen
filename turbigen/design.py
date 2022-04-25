"""
Generate turbine stage geometry parameters from aerodynamic design parameters.
"""
import scipy.optimize
import scipy.integrate
from . import compflow
import numpy as np
from collections import namedtuple
import warnings

expon = 0.62
muref = 1.8e-5
Tref = 288.0


def _merge_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c


def _make_namedtuple_with_docstrings(class_name, class_doc, field_doc):
    nt = namedtuple(class_name, field_doc.keys())
    # Apply documentation, only works for Python 3
    try:
        nt.__doc__ = class_doc
        for fi, vi in field_doc.items():
            getattr(nt, fi).__doc__ = vi
    except AttributeError:
        pass
    return nt


# Define a namedtuple to store all information about a non-dim stage design
stage_vars = {
    "Yp": r"Stagnation pressure loss coefficients :math:`Y_p` [--]",
    "Ma": r"Mach numbers :math:`\Ma` [--]",
    "Marel": r"Rotor-relative Mach numbers :math:`\Ma^\rel` [--]",
    "Al": r"Yaw angles :math:`\alpha` [deg]",
    "Alrel": r"Rotor-relative yaw angles :math:`\alpha^\rel` [deg]",
    "Lam": r"Degree of reaction :math:`\Lambda` [--]",
    "Ax_Ax1": r"Annulus area ratios :math:`A_x/A_{x1}` [--]",
    "U_sqrt_cpTo1": r"Non-dimensional blade speed :math:`U/\sqrt{c_p T_{01}}` [--]",
    "Po_Po1": r"Stagnation pressure ratios :math:`p_0/p_{01}` [--]",
    "To_To1": r"Stagnation temperature ratios :math:`T_0/T_{01}` [--]",
    "ga": r"Ratio of specific heats, :math:`\gamma` [--]",
    "phi": r"Flow coefficient, :math:`\phi` [--]",
    "psi": r"Stage loading coefficient, :math:`\psi` [--]",
    "Vt_U": r"Normalised tangential velocities, :math:`V_\theta/U` [--]",
    "Vtrel_U": r"Normalised relative tangential velocities, :math:`V^\rel_\theta/U` [--]",
    "V_U": r"Normalised velocities, :math:`V/U` [--]",
    "Vrel_U": r"Normalised relative velocities, :math:`V/U` [--]",
    "P3_Po1": r"Total-to-static stage pressure ratio, :math:`p_3/p_{01}` [--]",
    "eta": r"Polytropic efficiency, :math:`\eta` [--]",
}
docstring_NonDimStage = (
    "Data class to hold non-dimensional geometry and derived flow parameters "
    "of a turbine stage mean-line design."
)
NonDimStage = _make_namedtuple_with_docstrings(
    "NonDimStage", docstring_NonDimStage, stage_vars
)

# The dimensional stage has additional variables
dim_stage_vars = {
    "htr": r"Hub-to-tip radius ratio ratio at rotor inlet, :math:`\HTR`.",
    "Omega": r"Shaft angular velocity, :math:`\Omega` [rad/s].",
    "To1": r"Inlet stagnation temperature, :math:`T_{01}` [K].",
    "Po1": r"Inlet stagnation pressure, :math:`p_{01}` [K].",
    "rgas": r"Specific gas constant, :math:`R` [J/kg/K].",
    "Re": r"Axial chord Reynolds number at vane exit, :math:`\Rey` [--].",
    "Co": r"Circulation coefficient :math:`C_0` [--].",
    "rm": r"Mean radius, :math:`r_\mean` [m].",
    "rh": r"Hub radii, :math:`r_\hub` [m].",
    "rc": r"Casing radii, :math:`r_\cas` [m].",
    "Dr": r"Annulus spans, :math:`\Delta r` [m].",
    "s_cx": r"Pitch-to-chord ratios, :math:`s/c_x` [m].",
    "s": r"Row pitches, :math:`s` [m].",
    "cx": r"Axial chords, :math:`c_x` [m].",
}
docstring_DimStage = (
    "Data class to hold dimensional geometry and derived flow parameters "
    "of a turbine stage mean-line design."
)

# We want the dimensional stage to have some methods, but also immutable data
# So make a namedtuple to start with, then subclass it to add the methods
_DimStage = _make_namedtuple_with_docstrings(
    "DimStage", docstring_DimStage, _merge_dicts(stage_vars, dim_stage_vars)
)


class DimStage(_DimStage):
    def free_vortex_vane(self, spf):
        """Evaluate vane flow angles assuming a free vortex."""

        rh_vane = self.rh[:2].reshape(-1, 1)
        rc_vane = self.rc[:2].reshape(-1, 1)
        Al_vane = self.Al[:2].reshape(-1, 1)

        r_rm = (
            np.reshape(spf, (1, -1)) * (rh_vane - rc_vane) + rh_vane
        ) / self.rm

        return np.degrees(np.arctan(np.tan(np.radians(Al_vane)) / r_rm))

    def free_vortex_blade(self, spf):
        """Evaluate blade flow angles assuming a free vortex."""

        rh_blade = self.rh[1:].reshape(-1, 1)
        rc_blade = self.rc[1:].reshape(-1, 1)
        Al_blade = self.Al[1:].reshape(-1, 1)

        r_rm = (
            np.reshape(spf, (1, -1)) * (rc_blade - rh_blade) + rh_blade
        ) / self.rm

        return np.degrees(
            np.arctan(np.tan(np.radians(Al_blade)) / r_rm - r_rm / self.phi)
        )


def _integrate_length(chi):
    """Integrate quadratic camber line length given angles."""
    xhat = np.linspace(0.0, 1.0)
    tanchi_lim = np.tan(np.radians(chi))
    tanchi = np.diff(tanchi_lim) * xhat + tanchi_lim[0]
    return np.trapz(np.sqrt(1.0 + tanchi ** 2.0), xhat)


def nondim_stage_from_Al(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Al13,  # Yaw angles [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Fraction of stator loss [--]
):
    r"""Get geometry for an aerodynamic parameter set specifying outlet swirl.

    This routine calculates the non-dimensional *geometric* parameters that
    correspond to an input set of non-dimensional *aerodynamic* parameters. In
    this way, a turbine designer can directly specify meaningful quantities
    that characterise the desired fluid dynamics while the precise blade
    and annulus geometry are abstracted away.

    The working fluid is a perfect gas obeying the standard compressible flow
    relations. The mean radius, angular velocity, and hence blade speed are
    constant throughout the turbine stage.

    From the output of this function, arbitrarily choosing one of angular
    velocity or mean radius, and providing an inlet stagnation state, will
    completely define the stage in dimensional terms.

    Parameters
    ----------
    phi : float
        Flow coefficient, :math:`\phi`.
    psi : float
        Stage loading coefficient, :math:`\psi`.
    Al13 : array
        Yaw angles at stage inlet and exit, :math:`(\alpha_1,\alpha_3)`.
    Ma2 : float
        Vane exit Mach number, :math:`\Ma_2`.
    ga : float
        Ratio of specific heats, :math:`\gamma`.
    eta : float
        Polytropic efficiency, :math:`\eta`.
    Vx_rat : array, default=(1.,1.)
        Axial velocity ratios, :math:`(\zeta_1,\zeta_3)`.

    Returns
    -------
    stg : NonDimStage
        Stage geometry and some secondary calculated aerodynamic parameters
        represented as a NonDimStage object.
    """

    #
    # First, construct velocity triangles
    #

    # Euler work equation sets tangential velocity upstream of rotor
    # This gives us absolute flow angles everywhere
    tanAl2 = np.tan(np.radians(Al13[1])) * Vx_rat[1] + psi / phi
    Al2 = np.degrees(np.arctan(tanAl2))
    Al = np.insert(Al13, 1, Al2)
    cosAl = np.cos(np.radians(Al))

    # Get non-dimensional velocities from definition of flow coefficient
    Vx_U = np.array([Vx_rat[0], 1.0, Vx_rat[1]]) * phi
    Vt_U = Vx_U * np.tan(np.radians(Al))
    V_U = np.sqrt(Vx_U ** 2.0 + Vt_U ** 2.0)

    # Change reference frame for rotor-relative velocities and angles
    Vtrel_U = Vt_U - 1.0
    Vrel_U = np.sqrt(Vx_U ** 2.0 + Vtrel_U ** 2.0)
    Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Use Mach number to get U/cpTo1 = U/cpTo2
    V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(Ma2, ga)
    U_sqrtcpTo1 = V_sqrtcpTo2 / V_U[1]

    # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
    cpTo1_Usq = 1.0 / U_sqrtcpTo1 ** 2
    cpTo3_Usq = cpTo1_Usq - psi
    cpTo_Usq = np.array([cpTo1_Usq, cpTo1_Usq, cpTo3_Usq])

    # Mach numbers and capacity from compressible flow relations
    Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    Marel = Ma * Vrel_U / V_U
    Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
    Q_Q1 = Q / Q[0]

    #
    # Second, construct annulus line
    #

    # Use polytropic effy to get entropy change
    To_To1 = cpTo_Usq / cpTo_Usq[0]
    Ds_cp = -(1.0 - 1.0 / eta) * np.log(To_To1[-1])

    # Somewhat arbitrarily, split loss 50% on stator, 50% on rotor
    s_cp = np.hstack((0.0, loss_rat, 1.0)) * Ds_cp

    # Convert to stagnation pressures
    Po_Po1 = np.exp((ga / (ga - 1.0)) * (np.log(To_To1) + s_cp))

    # Use definition of capacity to get flow area ratios
    # Area ratios = span ratios because rm = const
    Dr_Drin = np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

    # Evaluate some other useful secondary aerodynamic parameters
    T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
    P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
    Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)
    Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])

    # Reformulate loss as stagnation pressure loss coefficients
    # referenced to inlet dynamic head as in a compressor
    Yp_vane = (Po_Po1[0] - Po_Po1[1]) / (Po_Po1[0] - P_Po1[0])
    Yp_blade = (Porel_Po1[1] - Porel_Po1[2]) / (Porel_Po1[1] - P_Po1[1])

    # Assemble all of the data into the output object
    stg = NonDimStage(
        Yp=(Yp_vane, Yp_blade),
        Al=Al,
        Alrel=Alrel,
        Ma=Ma,
        Marel=Marel,
        Ax_Ax1=Dr_Drin,
        Lam=Lam,
        U_sqrt_cpTo1=U_sqrtcpTo1,
        Po_Po1=Po_Po1,
        To_To1=To_To1,
        phi=phi,
        psi=psi,
        ga=ga,
        Vt_U=Vt_U,
        Vtrel_U=Vtrel_U,
        V_U=V_U,
        Vrel_U=Vrel_U,
        P3_Po1=P_Po1[2],
        eta=eta,
    )

    return stg


def nondim_stage_from_Lam(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Lam,  # Degree of reaction [--]
    Al1,  # Inlet yaw angle [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Axial velocity ratios [--]
):
    r"""Get geometry for an aerodynamic parameter set specifying reaction.

    A turbine designer is more interested in the degree of reaction of a stage,
    which controls the balance of loading between rotor and stator, rather than
    the exit yaw angle which has no such general physical interpretation.
    However, there is no analytical solution that will yield geometry at a
    fixed reaction.

    This function iterates exit yaw angle in :func:`nondim_stage_from_Al` to
    find the value which corresponds to the desired reaction, and returns a
    non-dimensional stage geometry.

    Parameters
    ----------
    phi : float
        Flow coefficient, :math:`\phi`.
    psi : float
        Stage loading coefficient, :math:`\psi`.
    Lam : float
        Degree of reaction, :math:`\Lambda`.
    Al1 : float
        Inlet yaw angle, :math:`\alpha_1`.
    Ma2 : float
        Vane exit Mach number, :math:`\Ma_2`.
    ga : float
        Ratio of specific heats, :math:`\gamma`.
    eta : float
        Polytropic efficiency, :math:`\eta`.
    Vx_rat : array, default=(1.,1.)
        Axial velocity ratios, :math:`(\zeta_1,\zeta_3)`.

    Returns
    -------
    stg : NonDimStage
        Stage geometry and some secondary calculated aerodynamic parameters
        represented as a NonDimStage object.
    """

    # Iteration step: returns error in reaction as function of exit yaw angle
    def iter_Al(x):
        stg_now = nondim_stage_from_Al(phi, psi, [Al1, x], Ma2, ga, eta, Vx_rat)
        return stg_now.Lam - Lam

    # Solving for Lam in general is tricky
    # Our strategy is to map out a coarse curve first, pick a point
    # close to the desired reaction, then Newton iterate

    # Evaluate guesses over entire possible yaw angle range
    Al_guess = np.linspace(-89.0, 89.0, 21)
    Lam_guess = np.zeros_like(Al_guess)

    # Catch errors if this guess of angle is horrible/non-physical
    for i in range(len(Al_guess)):
        with np.errstate(invalid="ignore"):
            try:
                Lam_guess[i] = iter_Al(Al_guess[i])
            except ValueError:
                Lam_guess[i] = np.nan

    # Remove invalid values
    Al_guess = Al_guess[~np.isnan(Lam_guess)]
    Lam_guess = Lam_guess[~np.isnan(Lam_guess)]

    # Trim to the region between minimum and maximum reaction
    # Now the slope will be monotonic
    i1, i2 = np.argmax(Lam_guess), np.argmin(Lam_guess)
    Al_guess, Lam_guess = Al_guess[i1:i2], Lam_guess[i1:i2]

    # Start the Newton iteration at minimum error point
    i0 = np.argmin(np.abs(Lam_guess))
    # Catch the warning from scipy that derivatives are zero
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            Al_soln = scipy.optimize.newton(
                iter_Al, x0=Al_guess[i0], x1=Al_guess[i0 - 1]
            )
        except:
            print('scipy warns derivatives are zero.')
            print('debug info..')
            print('Al_guess',Al_guess[i0], Al_guess[i0-1.])
            print('Lam errors',iter_Al[i0:(i0+2)])
            print('Al_soln', Al_soln)


    # Once we have a solution for the exit flow angle, evaluate stage geometry
    stg_out = nondim_stage_from_Al(
        phi, psi, [Al1, Al_soln], Ma2, ga, eta, Vx_rat
    )

    return stg_out


def annulus_line(stg, htr, cpTo1, Omega):
    r"""Return dimensional annulus line from given non-dim' geometry and inlet state.

    The parameter :math:`U/\sqrt{c_p T_{01}}` characterises blade speed in a
    non-dimensional sense. To scale a design to specific dimensional conditions
    is to choose two of: inlet enthalpy, angular velocity, and mean radius.
    This function calculates the latter given the former two.

    Then, choosing a hub-to-tip radius ratio fixes the blade span as a
    proportion of the mean radius. Given input annulus area ratios, this then
    yields dimensional values for blade span throughout the stage.

    This method of specifying the annulus line leaves mass flow as a free
    variable.

    Parameters
    ----------
    stg : NonDimStage
        A non-dimensional turbine stage mean-line design.
    htr : float
        Hub-to-tip radius ratio at rotor inlet, :math:`\HTR`.
    cpTo1 : float
        Inlet specific stagnation enthalpy, :math:`c_p T_{01}` [J/kg].
    Omega : float
        Shaft angular velocity, :math:`\Omega` [rad/s].

    Returns
    -------
    rm : float
        Mean radius, :math:`r_\mean` [m].
    Dr : array, length 3
        Annulus spans :math:`\Delta r` [m].
    """

    # Use non-dimensional blade speed to get U, hence mean radius
    U = stg.U_sqrt_cpTo1 * np.sqrt(cpTo1)
    rm = U / Omega

    # Use hub-to-tip ratio to set span (mdot will therefore float)
    Dr_rm = 2.0 * (1.0 - htr) / (1.0 + htr)
    Dr = rm * Dr_rm * np.array(stg.Ax_Ax1) / stg.Ax_Ax1[1]

    return rm, Dr


def pitch_Zweifel(stg, Z):
    r"""Calculate pitch-to-chord ratio from Zweifel coefficient.

    The Zweifel loading coefficient :math:`Z` is given by,

    .. math ::

        Z = \frac{\text{actual loading}}{\text{ideal loading}} =
        \frac{\dot{m}(V_{\theta 1} + V_{\theta 2})}{\Delta r
        c_x (p_{01}-p_2) } \, ,

    which can be rearranged to evaluate the pitch-to-chord ratio :math:`s/c_x`
    in purely non-dimensional terms (see Dixon and Hall p. 88). The calculation
    is performed for the stator in the absolute frame, and for the rotor in the
    relative frame.

    This equation does not assume constant axial velocity or incompressible
    flow.

    Parameters
    ----------
    stg : NonDimStage
        A non-dimensional turbine stage mean-line design.
    Z : tuple
        Zweifel loading coefficients for stator and rotor, :math:`Z` [--].

    Returns
    -------
    s_cx : 2-tuple
        Pitch-to-chord ratios for stator and rotor, :math:`s/c_x` [--].

    """

    # Angles
    Alr = np.radians(stg.Al)
    cosAl = np.cos(Alr)
    V_cpTo_sinAl = compflow.V_cpTo_from_Ma(stg.Ma, stg.ga) * np.sin(Alr)

    Alrelr = np.radians(stg.Alrel)
    cosAlrel = np.cos(Alrelr)
    V_cpTo_sinAlrel = compflow.V_cpTo_from_Ma(stg.Marel, stg.ga) * np.sin(
        Alrelr
    )

    P2_Po1 = stg.Po_Po1[1] / compflow.Po_P_from_Ma(stg.Ma[1], stg.ga)
    P3_Po2_rel = (
        stg.Po_Po1[2]
        / stg.Po_Po1[1]
        / compflow.Po_P_from_Ma(stg.Marel[1], stg.ga)
        * compflow.Po_P_from_Ma(stg.Ma[1], stg.ga)
        / compflow.Po_P_from_Ma(stg.Ma[2], stg.ga)
    )

    Q_stator = compflow.mcpTo_APo_from_Ma(stg.Ma[0], stg.ga)
    Q_rotor = compflow.mcpTo_APo_from_Ma(stg.Marel[1], stg.ga)

    # Evaluate pitch to chord
    s_c_stator = (
        Z[0]
        * (1.0 - P2_Po1)
        / Q_stator
        / cosAl[0]
        / (V_cpTo_sinAl[1] - V_cpTo_sinAl[0])
        / stg.Ax_Ax1[0]
        * np.mean(stg.Ax_Ax1[2:])
    )

    # Evaluate pitch to chord
    s_c_rotor = (
        Z[1]
        * (1.0 - P3_Po2_rel)
        / Q_rotor
        / cosAlrel[1]
        / np.abs(V_cpTo_sinAlrel[2] - V_cpTo_sinAlrel[1])
        / stg.Ax_Ax1[1]
        * np.mean(stg.Ax_Ax1[1:])
    )

    return s_c_stator, s_c_rotor


def pitch_circulation(stg, C0):
    r"""Calculate pitch-to-chord ratios using circulation coefficient.

    The circulation coefficient measure of loading was proposed by,

    Coull, J. D. and Hodson, H. P. (2012). "Blade Loading and Its
    Application in the Mean-Line Design of Low Pressure Turbines."
    ASME. J. Turbomach. 135(2)
    https://doi.org/10.1115/1.4006588

    They argue that the traditional Zweifel coefficient does not account for
    camber, and hence that blades with different levels of turning have a
    different value Zweifel coefficient for optimum pitch-to-chord.

    Here we assume constant axial velocity and incompressible flow to yield,

    .. math ::

        \frac{s}{c_x} = C_0 \frac{S}{c_x} \frac{\sec \alpha_2}{\tan \alpha_2 - \tan_\alpha_1}

    """

    chi = np.array((stg.Al[:2], stg.Alrel[1:]))
    V2 = np.array((stg.V_U[1], stg.Vrel_U[2]))
    Vt2 = np.array((stg.Vt_U[1], stg.Vtrel_U[2]))
    Vt1 = np.array((stg.Vt_U[0], stg.Vtrel_U[1]))
    S0_c = np.array([_integrate_length(chii) for chii in chi])

    return C0 * S0_c * V2 / np.abs(Vt1 - Vt2)


def chord_from_Re(stg, Re, cpTo1, Po1, rgas, viscosity=None):
    r"""Set axial chord length using Reynolds number and vane exit state.

    To define a Reynolds number, we must select a characteristic state and
    length scale. The suction-side boundary layer dominates profile loss, so
    the suction-side surface length should be our characteristic, but that
    depends on the thickness distribution. We approximate the suction-side
    length with the camber line length. Conventional wisdom suggests the vane
    exit state (highest possible velocity) scales performance well.

    So basing our Reynolds number on exit state and camberline length,

    .. math ::

        \Rey = \frac{\rho_{2} V_{2} \ell}{\mu_{2}} \, .

    We can solve for the dimensional camber line length :math:`\ell` by
    specifying a value for :math:`\Rey`, calculating the dimensional
    thermodynamic state at vane exit, and specifying the fluid viscosity
    :math:`\mu_2`.

    Fixing the inlet stagnation enthalpy and pressures, and a value for the
    specific gas constant yields :math:`\rho_2 V_2` for a given non-dimensional
    turbine stage design. Either, we can specify a constant viscosity, or,
    assuming the working fluid is air, the viscosity is approximated by,

    Finally, assuming a quadratic camber line :math:`\ell` relates the camber
    line length to axial chord :math:`c_x`.

    .. math ::

        \mu(T) = \mu_\rf \left(T/T_\rf\right)^{0.62}

    with :math:`\mu_\rf = 1.8 \times 10^5` [kg/m/s] at :math:`T_\rf = 288` [K].

    Parameters
    ----------
    stg : NonDimStage
        A non-dimensional turbine stage design object.
    Re : float
        Axial chord based Reynolds number, :math:`\Rey` [--].
    cpTo1 : float
        Inlet specific stagnation enthalpy, :math:`c_p T_{01}` [J/kg].
    Po1 : float
        Inlet stagnation pressure, :math:`P_{01}` [Pa].
    rgas : float
        Specific gas constant, :math:`R` [J/kg/K].
    viscosity : float, default None
        Kinematic viscosity, :math:`\mu` [kg/m^2/s]. If None, evaluate power
        law at vane exit static temperature.

    Returns
    -------
    cx : float
        Dimensional vane axial chord, :math:`c_x` [m].
    """

    # Get vane exit static state
    Ma2 = stg.Ma[1]
    ga = stg.ga
    cp = rgas * ga / (ga - 1.0)
    To1 = cpTo1 / cp
    P2 = Po1 * stg.Po_Po1[1] / compflow.Po_P_from_Ma(Ma2, ga)
    T2 = To1 * stg.To_To1[1] / compflow.To_T_from_Ma(Ma2, ga)
    rho2 = P2 / rgas / T2
    V2 = compflow.V_cpTo_from_Ma(Ma2, ga) * np.sqrt(To1 * cp)

    # Choose viscosity
    if viscosity:
        # Fixed constant viscosity
        mu2 = viscosity
    else:
        # Get viscosity using 0.62 power approximation
        mu2 = muref * (T2 / Tref) ** 0.62

    # Get camber line length using Rey
    ell = Re * mu2 / rho2 / V2

    # Scale by camber line length
    ell_cx = _integrate_length(stg.Al[:2])

    return ell / ell_cx


def free_vortex(stg, r_rm, dev):
    r"""Return free-vortex radial distributions of metal angles.

    Given the mean-line design and a vector of radius ratios across the span,
    twist vanes and blades such that angular momentum :math:`rV_\theta` is
    constant and hence axial velocity :math:`V_x` will be radially uniform.

    Deviation is always positive. The turning through the row is
    increased to counteract the specifed amount of deviation.

    Parameters
    ----------
    stg : NonDimStage
        A non-dimensional turbine stage mean-line design.
    r_rm : array 3-by-n
        Radius ratios at each station, :math:`r/r_\mathrm{m}` [--].
    dev: array length 2
        Flow deviation to counteract for each row, :math:`\delta` [deg].

    Returns
    -------
    chi_vane : array 2-by-n
        Vane inlet and exit metal angles, :math:`\chi` [deg].
    chi_blade : array 2-by-n
        Blade inlet and exit metal angles, :math:`\chi` [deg].
    """

    # Twist blades in a free vortex
    chi_vane = np.degrees(
        np.arctan(np.tan(np.radians(np.atleast_2d(stg.Al[:2])).T) / r_rm[:2, :])
    )
    chi_blade = np.degrees(
        np.arctan(
            np.tan(np.radians(np.atleast_2d(stg.Al[1:])).T) / r_rm[2:, :]
            - r_rm[2:, :] / stg.phi
        )
    )

    # Determine the direction of turning
    turn_dir_vane = 1.0 if (stg.Al[1] - stg.Al[0]) > 0.0 else -1.0
    turn_dir_blade = 1.0 if (stg.Alrel[1] - stg.Alrel[0]) > 0.0 else -1.0

    # Apply deviation
    chi_vane[1, :] += turn_dir_vane * dev[0]
    chi_blade[1, :] -= turn_dir_blade * dev[1]

    return chi_vane, chi_blade


def scale_geometry(stg, htr, Omega, To1, Po1, rgas, Re, Co, cx_rat=1.0):
    """Scale a mean-line design and evaluate geometry."""

    # Assuming perfect gas get cp
    cp = rgas * stg.ga / (stg.ga - 1.0)
    cpTo1 = cp * To1

    # Annulus line
    rm, Dr = annulus_line(stg, htr, cpTo1, Omega)

    # Chord
    cx = chord_from_Re(stg, Re, cpTo1, Po1, rgas) * np.array([1.0, cx_rat])

    # Pitches
    s_cx = pitch_circulation(stg, Co)

    # Assemble into dict
    geometry = {
        "rm": rm,
        "Dr": Dr,
        "cx": cx,
        "s_cx": s_cx,
        "s": s_cx * cx,
        "Omega": Omega,
        "To1": To1,
        "Po1": Po1,
        "rgas": rgas,
        "Re": Re,
        "Co": np.array(Co),
        "htr": htr,
        "rh": rm - Dr / 2.0,
        "rc": rm + Dr / 2.0,
    }

    # Return as a dimensional stage object
    return DimStage(**_merge_dicts(stg._asdict(), geometry))


def write_geomturbo(fname, ps, ss, h, c, nb, tips=(None, None), cascade=False):
    """Write blade and annulus geometry to AutoGrid GeomTurbo file.

    Parameters
    ----------

    fname : File name to write
    ps    : Nested list of arrays of pressure-side coordinates,
            ps[row][section][point on section, x/r/rt]
            We allow different sizes for each section and row.
    ss    : Same for suction-side coordinates.
    h     : Array of hub line coordinates, h[axial location, x/r].
    c     : Same for casing line.
    nb    : Iterable of numbers of blades for each row."""

    # Determine numbers of points
    ni_h = np.shape(h)[0]
    ni_c = np.shape(c)[0]

    n_row = len(ps)
    n_sect = [len(psi) for psi in ps]
    ni_ps = [[np.shape(psii)[0] for psii in psi] for psi in ps]
    ni_ss = [[np.shape(ssii)[0] for ssii in ssi] for ssi in ss]

    fid = open(fname, "w")

    # # Autogrid requires R,X,T coords
    # ps = ps[i][:,[1,0,2]]
    # ss = ss[i][:,[1,0,2]]

    if cascade:
        # Swap the coordinates
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k] = ps[i][k][:, (1, 2, 0)]
                ss[i][k] = ss[i][k][:, (1, 2, 0)]
    else:
        # Convert RT to T
        for i in range(n_row):
            for k in range(n_sect[i]):
                ps[i][k][:, 2] = ps[i][k][:, 2] / ps[i][k][:, 1]
                ss[i][k][:, 2] = ss[i][k][:, 2] / ss[i][k][:, 1]

    # Write the header
    fid.write("%s\n" % "GEOMETRY TURBO")
    fid.write("%s\n" % "VERSION 5.5")
    fid.write("%s\n" % "bypass no")
    if cascade:
        fid.write("%s\n\n" % "cascade yes")
    else:
        fid.write("%s\n\n" % "cascade no")

    # Write hub and casing lines (channel definition)
    fid.write("%s\n" % "NI_BEGIN CHANNEL")

    # Build the hub and casing line out of basic curves
    # Start the data definition
    fid.write("%s\n" % "NI_BEGIN basic_curve")
    fid.write("%s\n" % "NAME thehub")
    fid.write("%s %i\n" % ("DISCRETISATION", 10))
    fid.write("%s %i\n" % ("DATA_REDUCTION", 0))
    fid.write("%s\n" % "NI_BEGIN zrcurve")
    fid.write("%s\n" % "ZR")

    # Write the length of hub line
    fid.write("%i\n" % ni_h)

    # Write all the points in x,r
    for i in range(ni_h):
        fid.write("%1.11f\t%1.11f\n" % tuple(h[i, :]))

    fid.write("%s\n" % "NI_END zrcurve")
    fid.write("%s\n" % "NI_END basic_curve")

    # Now basic curve for shroud
    fid.write("%s\n" % "NI_BEGIN basic_curve")
    fid.write("%s\n" % "NAME theshroud")

    fid.write("%s %i\n" % ("DISCRETISATION", 10))
    fid.write("%s %i\n" % ("DATA_REDUCTION", 0))
    fid.write("%s\n" % "NI_BEGIN zrcurve")
    fid.write("%s\n" % "ZR")

    # Write the length of shroud
    fid.write("%i\n" % ni_c)

    # Write all the points in x,r
    for i in range(ni_c):
        fid.write("%1.11f\t%1.11f\n" % tuple(c[i, :]))

    fid.write("%s\n" % "NI_END zrcurve")
    fid.write("%s\n" % "NI_END basic_curve")

    # Now lay out the real shroud and hub using the basic curves
    fid.write("%s\n" % "NI_BEGIN channel_curve hub")
    fid.write("%s\n" % "NAME hub")
    fid.write("%s\n" % "VERTEX CURVE_P thehub 0")
    fid.write("%s\n" % "VERTEX CURVE_P thehub 1")
    fid.write("%s\n" % "NI_END channel_curve hub")

    fid.write("%s\n" % "NI_BEGIN channel_curve shroud")
    fid.write("%s\n" % "NAME shroud")
    fid.write("%s\n" % "VERTEX CURVE_P theshroud 0")
    fid.write("%s\n" % "VERTEX CURVE_P theshroud 1")
    fid.write("%s\n" % "NI_END channel_curve shroud")

    fid.write("%s\n" % "NI_END CHANNEL")

    # CHANNEL STUFF DONE
    # NOW DEFINE ROWS
    for i in range(n_row):
        fid.write("%s\n" % "NI_BEGIN nirow")
        fid.write("%s%i\n" % ("NAME r", i + 1))
        fid.write("%s\n" % "TYPE normal")
        fid.write("%s %i\n" % ("PERIODICITY", nb[i]))
        fid.write("%s %i\n" % ("ROTATION_SPEED", 0))

        hdr = [
            "NI_BEGIN NINonAxiSurfaces hub",
            "NAME non axisymmetric hub",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces hub",
            "NI_BEGIN NINonAxiSurfaces shroud",
            "NAME non axisymmetric shroud",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces shroud",
            "NI_BEGIN NINonAxiSurfaces tip_gap",
            "NAME non axisymmetric tip gap",
            "REPETITION 0",
            "NI_END   NINonAxiSurfaces tip_gap",
        ]

        fid.writelines("%s\n" % l for l in hdr)

        fid.write("%s\n" % "NI_BEGIN NIBlade")
        fid.write("%s\n" % "NAME Main Blade")

        if tips[i] is not None:
            fid.write("%s\n" % "NI_BEGIN NITipGap")
            fid.write("%s %f\n" % ("WIDTH_AT_LEADING_EDGE", tips[i][0]))
            fid.write("%s %f\n" % ("WIDTH_AT_TRAILING_EDGE", tips[i][1]))
            fid.write("%s\n" % "NI_END NITipGap")

        fid.write("%s\n" % "NI_BEGIN nibladegeometry")
        fid.write("%s\n" % "TYPE GEOMTURBO")
        fid.write("%s\n" % "GEOMETRY_MODIFIED 0")
        fid.write("%s\n" % "GEOMETRY TURBO VERSION 5")
        fid.write("%s %f\n" % ("blade_expansion_factor_hub", 0.1))
        fid.write("%s %f\n" % ("blade_expansion_factor_shroud", 0.1))
        fid.write("%s %i\n" % ("intersection_npts", 10))
        fid.write("%s %i\n" % ("intersection_control", 1))
        fid.write("%s %i\n" % ("data_reduction", 0))
        fid.write("%s %f\n" % ("data_reduction_spacing_tolerance", 1e-006))
        fid.write(
            "%s\n"
            % (
                "control_points_distribution "
                "0 9 77 9 50 0.00622408226922942 0.119480980447523"
            )
        )
        fid.write("%s %i\n" % ("units", 1))
        fid.write("%s %i\n" % ("number_of_blades", 1))

        fid.write("%s\n" % "suction")
        fid.write("%s\n" % "SECTIONAL")
        fid.write("%i\n" % n_sect[i])
        for k in range(n_sect[i]):
            fid.write("%s %i\n" % ("# section", k + 1))
            if cascade:
                fid.write("%s\n" % "XYZ")
            else:
                fid.write("%s\n" % "ZRTH")
            fid.write("%i\n" % ni_ss[i][k])
            for j in range(ni_ss[i][k]):
                fid.write("%1.11f\t%1.11f\t%1.11f\n" % tuple(ss[i][k][j, :]))

        fid.write("%s\n" % "pressure")
        fid.write("%s\n" % "SECTIONAL")
        fid.write("%i\n" % n_sect[i])
        for k in range(n_sect[i]):
            fid.write("%s %i\n" % ("# section", k + 1))
            if cascade:
                fid.write("%s\n" % "XYZ")
            else:
                fid.write("%s\n" % "ZRTH")
            fid.write("%i\n" % ni_ps[i][k])
            for j in range(ni_ps[i][k]):
                fid.write("%1.11f\t%1.11f\t%1.11f\n" % tuple(ps[i][k][j, :]))
        fid.write("%s\n" % "NI_END nibladegeometry")

        # choose a leading and trailing edge treatment

        #    fid.write('%s\n' % 'BLUNT_AT_LEADING_EDGE')
        fid.write("%s\n" % "BLENT_AT_LEADING_EDGE")
        #    fid.write('%s\n' % 'BLENT_TREATMENT_AT_TRAILING_EDGE')
        fid.write("%s\n" % "NI_END NIBlade")

        fid.write("%s\n" % "NI_END nirow")

    fid.write("%s\n" % "NI_END GEOMTURBO")

    fid.close()

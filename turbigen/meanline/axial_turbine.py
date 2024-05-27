"""Mean-line design for an axial turbine stage, assuming perfect gas."""
from turbigen import compflow_native as compflow
import warnings
from scipy.optimize import root_scalar
import turbigen.flowfield
import turbigen.fluid
import numpy as np


def forward(
    So1,
    htr,
    Omega,
    Alpha1,
    phi,
    psi,
    Lam,
    Ma2,
    eta,
    loss_split=0.5,
    VmR12=1.0,
    VmR23=1.0,
):
    r"""Design the mean-line for an axial turbine stage.

    Parameters
    ----------
    So1: State
        Object specifing the working fluid and its state at inlet.
    htr: float
        Hub-to-tip radius ratio, :math:`\htr`.
    Omega: float
        Shaft angular velocity, :math:`\Omega` [rad/s].
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
        Guess of polytropic efficiency, :math:`\eta`.
    loss_split : float
        Guess of entropy rise split between stator and rotor.
    VmR12 : float
        Meridional velocity ratio
    VmR23 : float
        Meridional velocity ratio

    Returns
    -------
    ml: MeanLine
        An object specifying the flow along the mean line.

    """

    # if not isinstance(So1, fluid.PerfectState):
    #     raise NotImplementedError("axial_turbine does not support real gases")

    ga = So1.gamma
    rgas = So1.rgas

    P_Po1, T_To1, Vx_U, Vt_U, Dr_Drin, U_sqrt_cpTo1 = _nondim_stage_from_Lam(
        phi,  # Flow coefficient [--]
        psi,  # Stage loading coefficient [--]
        Lam,  # Degree of reaction [--]
        Alpha1,  # Inlet yaw angle [deg]
        Ma2,  # Vane exit Mach number [--]
        ga,  # Ratio of specific heats [--]
        eta,  # Polytropic efficiency [--]
        (VmR12, VmR23),  # Axial velocity ratios [--]
        loss_split,  # Fraction of stator loss [--]
    )

    cp = rgas * ga / (ga - 1.0)

    # Use non-dimensional blade speed to get U, hence mean radius
    U = U_sqrt_cpTo1 * np.sqrt(cp * So1.T)

    P = np.repeat(P_Po1 * So1.P, [1, 2, 1])
    T = np.repeat(T_To1 * So1.T, [1, 2, 1])
    Dr_Drin = np.repeat(Dr_Drin, [1, 2, 1])
    Vx = np.repeat(Vx_U, [1, 2, 1]) * U
    Vt = np.repeat(Vt_U, [1, 2, 1]) * U
    Vr = np.zeros_like(Vx)

    rm = U / Omega * np.ones_like(P)

    # Use hub-to-tip ratio to set span (mdot will therefore float)
    Dr_rm = 2.0 * (1.0 - htr) / (1.0 + htr)
    span = rm * Dr_rm * np.array(Dr_Drin)
    A = 2.0 * np.pi * rm * span
    rhub = rm - 0.5 * span
    rtip = rm + 0.5 * span
    rrms = np.sqrt(0.5 * (rhub**2.0 + rtip**2.0))
    Omega = np.array([0.0, 0.0, Omega, Omega])

    Vxrt = np.stack((Vx, Vr, Vt))

    S = turbigen.fluid.PerfectState.from_properties(cp, ga, So1.mu, shape=rm.shape)
    S.set_P_T(P, T)

    ml = turbigen.flowfield.make_mean_line(rrms, A, Omega, Vxrt, S)

    return ml


def inverse(ml):
    """Given mean-line flow, extract design parameters for a turbine stage.

    Parameters
    ----------
    ml: MeanLine
        A mean-line object specifying the flow in an axial turbine.

    Returns
    -------
    out : dict
        Dictionary of aerodynamic design parameters with fields: `So1`, `htr`, `Omega`,
        `Alpha1`, `phi`, `psi`, `Lam`, `Ma2`, `eta`, `loss_split`, `VmR12`,
        `VmR23`. The fields have the same meanings as in :func:`forward`.
    """
    psi = -(ml.ho[-1] - ml.ho[0]) / (ml.U[2] ** 2.0)
    Lam = (ml.h[3] - ml.h[2]) / (ml.h[3] - ml.h[0])
    loss_split = (ml.s[1] - ml.s[0]) / ml.s.ptp()
    out = {
        "So1": ml.empty().set_P_h(ml.Po[0], ml.ho[0]),
        "htr": ml.htr[0],
        "Omega": ml.Omega[-1],
        "Alpha1": ml.Alpha[0],
        "phi": ml.phi[2],
        "psi": psi,
        "Lam": Lam,
        "Ma2": ml.Ma[1],
        "eta": ml.eta_poly,
        "loss_split": loss_split,
        "VmR12": ml.VmR[0],
        "VmR23": ml.VmR[-1],
    }
    return out


def _nondim_stage_from_Lam(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Lam,  # Degree of reaction [--]
    Al1,  # Inlet yaw angle [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Fraction of stator loss [--]
    mdotc_mdot1=(0.0, 0.0),  # Coolant flows as fraction of inlet [--]
    Toc_Toinf=(1.0, 1.0),  # Local coolant temperature ratios [--]
    preswirl_factor=0.0,  # Preswirl factor [--]
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
        Lam_now = _nondim_stage_from_Al(
            phi,
            psi,
            [Al1, x],
            Ma2,
            ga,
            eta,
            Vx_rat,
            loss_rat,
            mdotc_mdot1,
            Toc_Toinf,
            preswirl_factor,
        )[0]
        return Lam_now - Lam

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
            except (ValueError, FloatingPointError):
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
        warnings.filterwarnings("error")
        try:
            Al_soln = root_scalar(
                iter_Al,
                bracket=tuple(
                    Al_guess[
                        (i0 - 1, i0 + 1),
                    ]
                ),
            ).root
        except ValueError as e:
            print("debug info..")
            print("Al_guess", Al_guess)
            print("Lam errors", Lam_guess)
            raise e

    # Once we have a solution for the exit flow angle, evaluate stage geometry
    _, P_Po1, T_To1, Vx_U, Vt_U, Dr_Drin, U_sqrt_cpTo1 = _nondim_stage_from_Al(
        phi,
        psi,
        [Al1, Al_soln],
        Ma2,
        ga,
        eta,
        Vx_rat,
        loss_rat,
        mdotc_mdot1,
        Toc_Toinf,
        preswirl_factor,
    )

    return P_Po1, T_To1, Vx_U, Vt_U, Dr_Drin, U_sqrt_cpTo1


def _nondim_stage_from_Al(
    phi,  # Flow coefficient [--]
    psi,  # Stage loading coefficient [--]
    Al13,  # Yaw angles [deg]
    Ma2,  # Vane exit Mach number [--]
    ga,  # Ratio of specific heats [--]
    eta,  # Polytropic efficiency [--]
    Vx_rat=(1.0, 1.0),  # Axial velocity ratios [--]
    loss_rat=0.5,  # Fraction of stator loss [--]
    mdotc_mdot1=(0.0, 0.0),  # Coolant flows as fraction of inlet [--]
    Toc_Toinf=(1.0, 1.0),  # Local coolant temperature ratios [--]
    preswirl_factor=0.0,  # Preswirl factor [--]
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

    Coolant mass flows are specified as a fraction of the *inlet* mass flow.

    Coolant temperatures are specified as a ratio of coolant stagnation
    temperature to row inlet stagnation temperature, in the relative frame for
    rotors. Converting coolant temperatures to the absolute frame then requires
    knowledge of coolant preswirl. Change in relative stagnation temperature
    due to preswirl is a function of the factor,

    .. math ::

        \newcommand{\PS}{\mathrm{PS}}

        \frac{V_{\theta,\PS} r_\PS}{U r_\mathrm{m}


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

    # Rename coolant parameters for brevity
    fc = mdotc_mdot1

    #
    # First, construct velocity triangles
    #

    # Euler work eqn from 2-3 sets tangential velocity upstream of rotor
    tanAl2 = (
        np.tan(np.radians(Al13[1])) * Vx_rat[1] * (1.0 + fc[0] + fc[1]) / (1.0 + fc[0])
        + psi / phi
    )

    Al2 = np.degrees(np.arctan(tanAl2))
    Al = np.insert(Al13, 1, Al2)
    cosAl = np.cos(np.radians(Al))

    # Get non-dimensional velocities from definition of flow coefficient
    Vx_U = np.array([Vx_rat[0], 1.0, Vx_rat[1]]) * phi
    Vt_U = Vx_U * np.tan(np.radians(Al))
    V_U = np.sqrt(Vx_U**2.0 + Vt_U**2.0)

    # Change reference frame for rotor-relative velocities and angles
    # Vtrel_U = Vt_U - 1.0
    # Vrel_U = np.sqrt(Vx_U**2.0 + Vtrel_U**2.0)
    # Alrel = np.degrees(np.arctan2(Vtrel_U, Vx_U))

    # Branch depending on compressor or turbine (cooling or bleed flow)

    # Vane coolant will reduce To2
    # Simple because in absolute frame
    To2_To1 = (1.0 + fc[0] * Toc_Toinf[0]) / (1.0 + fc[0])

    # Use Mach number to get U/cpTo1
    V_sqrtcpTo2 = compflow.V_cpTo_from_Ma(Ma2, ga)
    U_sqrtcpTo1 = V_sqrtcpTo2 * np.sqrt(To2_To1) / V_U[1]
    Usq_cpTo1 = U_sqrtcpTo1**2.0

    # Now determine absolute stagnation temperature for rotor coolant
    rel_factor_inf = 0.5 * Usq_cpTo1 * (1.0 - 2.0 * phi * tanAl2)
    rel_factor_cool = 0.5 * Usq_cpTo1 * (1.0 - 2.0 * preswirl_factor)
    Toc2_To1 = Toc_Toinf[1] * (To2_To1 + rel_factor_inf) - rel_factor_cool
    Toc2_To2 = Toc2_To1 / To2_To1

    # Non-dimensional temperatures from U/cpTo Ma and stage loading definition
    cpTo1_Usq = 1.0 / Usq_cpTo1
    cpTo2_Usq = cpTo1_Usq * To2_To1
    cpTo3_Usq = (cpTo2_Usq * (1.0 + fc[0] + fc[1] * Toc2_To2) - psi) / (
        1.0 + fc[0] + fc[1]
    )

    cpTo_Usq = np.array([cpTo1_Usq, cpTo2_Usq, cpTo3_Usq])

    # Mach numbers and capacity from compressible flow relations
    Ma = compflow.Ma_from_V_cpTo(V_U / np.sqrt(cpTo_Usq), ga)
    # Marel = Ma * Vrel_U / V_U
    Q = compflow.mcpTo_APo_from_Ma(Ma, ga)
    Q_Q1 = Q / Q[0]

    #
    # Second, construct annulus line
    #

    # Use polytropic effy to get entropy change
    To_To1 = cpTo_Usq / cpTo_Usq[0]
    Ds_cp = -(1.0 - 1.0 / eta) * np.log(To_To1[-1])

    # Somewhat arbitrarily, split loss using loss ratio (default 0.5)
    s_cp = np.hstack((0.0, loss_rat, 1.0)) * Ds_cp

    # Convert to stagnation pressures
    Po_Po1 = np.exp((ga / (ga - 1.0)) * (np.log(To_To1) + s_cp))

    # Account for cooling or bleed flows
    mdot_mdot1 = np.array([1.0, 1.0 + fc[0], 1 + fc[0] + fc[1]])

    # Use definition of capacity to get flow area ratios
    # Area ratios = span ratios because rm = const
    Dr_Drin = mdot_mdot1 * np.sqrt(To_To1) / Po_Po1 / Q_Q1 * cosAl[0] / cosAl

    # Evaluate some other useful secondary aerodynamic parameters
    T_To1 = To_To1 / compflow.To_T_from_Ma(Ma, ga)
    P_Po1 = Po_Po1 / compflow.Po_P_from_Ma(Ma, ga)
    # Porel_Po1 = P_Po1 * compflow.Po_P_from_Ma(Marel, ga)

    # Reformulate loss as stagnation pressure loss coefficients
    # Turbine
    Lam = (T_To1[2] - T_To1[1]) / (T_To1[2] - T_To1[0])
    # Yp_vane = (Po_Po1[0] - Po_Po1[1]) / (Po_Po1[0] - P_Po1[0])
    # Yp_blade = (Porel_Po1[1] - Porel_Po1[2]) / (Porel_Po1[1] - P_Po1[1])

    return Lam, P_Po1, T_To1, Vx_U, Vt_U, Dr_Drin, np.sqrt(Usq_cpTo1)

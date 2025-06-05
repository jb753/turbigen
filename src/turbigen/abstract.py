"""Define interfaces using abstract base classes.

Documentation is automatically generated from these definitions
and therefore will not be broken by changes to the concrete
implementations, no matter what chaos ensues underneath. Implementations are
banned in this file.

"""

from abc import ABC, abstractmethod


class State(ABC):
    """Working fluids
    -------------

    Both perfect and real working fluids are represented by a :class:`State`
    class, which has a common interface for setting and reading thermodynamic
    properties. The interface allows the same mean-line design code to work
    with any working fluid. :class:`State` does not store velocity information
    and hence makes no distinction between static and stagnation states, the
    handling of which is left to the calling code.

    Setter methods
    ^^^^^^^^^^^^^^

    The following methods are used to set the thermodynamic state of the fluid
    to a new value. The object is updated in-place; a copy can be explicitly
    created using :meth:`State.copy`. By the two-property rule, the setters all
    take two arguments to uniquely specify the thermodynamic state. The
    following methods are available:

    xxx

    Property attributes
    ^^^^^^^^^^^^^^^^^^^

    Thermodynamic and transport properties of the fluid are accessed as attributes of the
    :class:`State` object. The following properties are available:

    yyy

    """

    @abstractmethod
    def set_rho_u(self, rho, u):
        """Set density and internal energy.

        Parameters
        ----------
        rho : float
            Density [kg/m^3].
        u : float
            Internal energy [J/kg].

        Returns
        -------
        self : Fluid
            The current instance with updated thermodynamic properties.

        """
        raise NotImplementedError()

    @abstractmethod
    def set_h_s(self, h, s):
        """Set enthalpy and entropy.

        Parameters
        ----------
        h : float
            Enthalpy [J/kg].
        s : float
            Entropy [J/kg/K].

        Returns
        -------
        self : Fluid
            The current instance with thermodynamic properties.

        """
        raise NotImplementedError()

    @abstractmethod
    def set_P_T(self, P, T):
        """Set pressure and temperature.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        T : float
            Temperature [K].

        Returns
        -------
        self : Fluid
            The current instance with thermodynamic properties.

        """
        raise NotImplementedError()

    @abstractmethod
    def set_P_s(self, P, s):
        """Set pressure and entropy.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        s : float
            Entropy [J/kg/K].

        Returns
        -------
        self : Fluid
            The current instance with thermodynamic properties.

        """
        raise NotImplementedError()

    @abstractmethod
    def set_P_h(self, P, h):
        """Set pressure and enthalpy.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        h : float
            Specific enthalpy [J/kgK].

        Returns
        -------
        self : Fluid
            The current instance with thermodynamic properties.

        """
        raise NotImplementedError()

    @abstractmethod
    def set_P_rho(self, P, rho):
        """Set pressure and density.

        Parameters
        ----------
        P : float
            Pressure [Pa].
        rho : float
            Density [kg/m^3].

        Returns
        -------
        self : Fluid
            The current instance with thermodynamic properties.

        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def gamma(self):
        """Ratio of specific heats [--]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rgas(self):
        """Specific gas constant [J/kg/K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def u(self):
        """Specific internal energy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def h(self):
        """Specific enthalpy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def cp(self):
        """Specific heat at constant pressure [J/kg/K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def cv(self):
        """Specific heat at constant volume [J/kg/K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def a(self):
        """Acoustic speed [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def P(self):
        """Pressure [Pa]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def T(self):
        """Temperature [K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def s(self):
        """Specific entropy [J/kg/K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def mu(self):
        """Kinematic viscosity [m^2/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Pr(self):
        """Prandtl number [--]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rho(self):
        """Density [kg/m^3]."""
        raise NotImplementedError()

    #
    # Thermodynamic derivatives
    #

    @property
    @abstractmethod
    def dsdrho_P(self):
        """Derivative of entropy with respect to density at constant pressure."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dsdP_rho(self):
        """Derivative of entropy with respect to pressure at constant density."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dhdP_rho(self):
        """Derivative of enthalpy with respect to pressure at constant density."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dhdrho_P(self):
        """Derivative of enthalpy with respect to density at constant pressure."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dudrho_P(self):
        """Derivative of internal energy with respect to density at constant pressure."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def dudP_rho(self):
        """Derivative of internal energy with respect to pressure at constant density."""
        raise NotImplementedError()


class FlowField(ABC):
    """
    Flow fields
    -----------

    Augmenting a thermodynamic state with velocity and coordinate data
    allows the :class:`FlowField` class to represent a flow field.
    Composite properties such as stagnation pressure and Mach number
    can then be computed from the thermodynamic state and velocity vector.
    Setting an angular velocity allows evaluation of quantities in a rotating
    frame. Circumferential periodicity is represented by a number of blades.

    Setter methods
    ^^^^^^^^^^^^^^

    The :class:`FlowField` class has extra setter methods as well as those
    defined in :class:`State`. Omitting a coordinate or velocity argument
    will leave the corresponding values unchanged.

    xxx

    Property attributes
    ^^^^^^^^^^^^^^^^^^^

    In addition to all the pure thermodynamic properties defined in
    :class:`State`, incorporating velocity and coordinate data allow the
    :class:`FlowField` to provide the following other properties:

    yyy

    """

    #
    # Independent coordinates
    #

    @property
    @abstractmethod
    def x(self):
        """Axial coordinate [m]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def r(self):
        """Radial coordinate [m]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def t(self):
        """Circumferential coordinate [rad]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def xrt(self):
        """Vector of polar coordinates []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def xyz(self):
        """Vector of Cartesian coordinates []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Nb(self):
        """Number of blades, circumferential periodicity [--]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def pitch(self):
        """Angular blade pitch, circumferential period [rad]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Omega(self):
        """Reference frame angular velocity [rad/s]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rpm(self):
        """Reference frame revolutions per minute [rpm]"""
        raise NotImplementedError()

    @abstractmethod
    def set_xrt(self, x, r, t):
        """Set polar coordinates.

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        r : float
            Radial coordinate [m].
        t : float
            Circumferential coordinate [rad].

        """
        raise NotImplementedError()

    @abstractmethod
    def set_xyz(self, x, y, z):
        """Set Cartesian coordinates.

        Parameters
        ----------
        x : float
            Axial coordinate [m].
        y : float
            Vertical coordinate [m].
        z : float
            Horizontal coordinate [m].

        """
        raise NotImplementedError()

    @abstractmethod
    def set_Omega(self, Omega):
        """Set reference frame angular velocity.

        Parameters
        ----------
        Omega : float
            Angular velocity [rad/s].

        """
        raise NotImplementedError()

    @abstractmethod
    def set_Nb(self, Nb):
        """Number of blades.

        Parameters
        ----------
        Nb : int
            Number of blades, circumferential periodicity [--].

        """
        raise NotImplementedError()

    @abstractmethod
    def set_conserved(self, conserved):
        """Vector of conserved variables.

        Parameters
        ----------
        conserved : float
            Vector of [rho, rhoVx, rhoVr, rhorVt, rhoe]

        """
        raise NotImplementedError()

    @abstractmethod
    def set_primitive(self, primitive):
        """Vector of primitive variables.

        Parameters
        ----------
        primitive : float
            Vector of [rho, Vx, Vr, Vt, P]

        """
        raise NotImplementedError()

    @abstractmethod
    def set_Vxrt(self, Vx, Vr, Vt):
        """Polar velocity vector.

        Parameters
        ----------
        Vx : float
            Axial velocity [m/s].
        Vr : float
            Radial velocity [m/s].
        Vt : float
            Tangential velocity [m/s].

        """
        raise NotImplementedError()

    def set_Vxyz(self, Vx, Vy, Vz):
        """Cartesian velocity vector.
        Parameters
        ----------
        Vx : float
            Axial velocity [m/s].
        Vy : float
            Vertical velocity [m/s].
        Vz : float
            Horizontal velocity [m/s].

        """
        raise NotImplementedError()

    @abstractmethod
    def set_V_Alpha_Beta(self, V, Alpha, Beta):
        """Velocity magnitude and angles.

        Parameters
        ----------
        V : float
            Absolute velocity magnitude [m/s].
        Alpha : float
            Yaw angle [deg].
        Beta : float
            Pitch angle [deg].

        """
        raise NotImplementedError()

    #
    # Independent velocities
    #

    @property
    @abstractmethod
    def Vx(self):
        """Axial velocity [m/s]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Vr(self):
        """Radial velocity [m/s]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Vt(self):
        """Circumferential velocity [m/s]"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def U(self):
        """Blade speed [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def V(self):
        """Absolute velocity magnitude [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Vm(self):
        """Meridional velocity magnitude [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Vt_rel(self):
        """Relative frame circumferential velocity [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def V_rel(self):
        """Relative frame velocity magnitude [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def halfVsq(self):
        """Specific kinetic energy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def halfVsq_rel(self):
        """Relative frame specific kinetic energy [J/kg]."""
        raise NotImplementedError()

    #
    # Angles
    #

    @property
    @abstractmethod
    def Alpha_rel(self):
        """Relative frame yaw angle [deg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Alpha(self):
        """Yaw angle [deg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Beta(self):
        """Pitch angle [deg]."""
        raise NotImplementedError()

    #
    # Composite properties
    #

    @property
    @abstractmethod
    def conserved(self):
        """Vector of conserved variables []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def primitive(self):
        """Vector of primitive variables []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fluxx(self):
        """Vector of fluxes in x-direction []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fluxr(self):
        """Vector of fluxes in r-direction []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def fluxt(self):
        """Vector of fluxes in t-direction []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def bcond(self):
        """Vector of boundary condition variables []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def e(self):
        """Specific total energy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Ma(self):
        """Mach number [--]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Ma_rel(self):
        """Relative frame Mach number [--]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def I(self):
        """Rothalpy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Po(self):
        """Stagnation pressure [Pa]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def To(self):
        """Stagnation temperature [K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def ao(self):
        """Stagnation acoustic speed [m/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def ho(self):
        """Stagnation specific enthalpy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def Po_rel(self):
        """Relative frame stagnation pressure [Pa]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def To_rel(self):
        """Relative frame stagnation temperature [K]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def ho_rel(self):
        """Relative frame stagnation specific enthalpy [J/kg]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rhoVx(self):
        """Volumetric axial momentum [kg/m^2/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rhoVr(self):
        """Volumetric radial momentum [kg/m^2/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rhoVt(self):
        """Volumetric angular momentum [kg/m^2/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rhorVt(self):
        """Volumetric angular momentum [kg/m^2/s]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def rhoe(self):
        """Volumetric total energy [J/m^3]."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def drhoe_drho_P(self):
        """Derivative of volumetric total energy with respect to density at constant pressure []."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def drhoe_dP_rho(self):
        """Derivative of volumetric total energy with respect to pressure at constant density []."""
        raise NotImplementedError()

    # @property
    # @abstractmethod
    # def dAi(self):
    #     """Areas of constant i faces [m^2]."""
    #     raise NotImplementedError()
    #
    # @property
    # @abstractmethod
    # def dAj(self):
    #     """Areas of constant j faces [m^2]."""
    #     raise NotImplementedError()
    #
    # @property
    # @abstractmethod
    # def dAk(self):
    #     """Areas of constant k faces [m^2]."""
    #     raise NotImplementedError()


class Block(ABC):
    """
    Blocks
    ------

    A 'block' is a three-dimensional volume of flow field that is
    defined by a structured array of points forming hexahedral cells.
    Blocks may be connected to each other by 'patches'


    """

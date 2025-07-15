"""Classes to represent flow fields."""

import numpy as np
import turbigen.base
import turbigen.fluid
import turbigen.yaml
import turbigen.abstract
from turbigen.base import dependent_property
from turbigen import util


class BaseFlowField(
    turbigen.base.StructuredData,
    turbigen.abstract.FlowField,
):
    def check_flow(self):
        assert np.isfinite(self.Vxrt).all()
        assert np.isfinite(self.P).all()
        assert np.isfinite(self.T).all()
        assert (self.P > 0.0).all()
        assert (self.T > 0.0).all()

    #
    # Independent coordinates
    #

    @property
    def x(self):
        """Axial coordinate [m]"""
        return self._get_data_by_key("x")

    @x.setter
    def x(self, value):
        self._set_data_by_key("x", value)

    @property
    def r(self):
        return self._get_data_by_key("r")

    @r.setter
    def r(self, value):
        self._set_data_by_key("r", value)

    @property
    def t(self):
        return self._get_data_by_key("t")

    @t.setter
    def t(self, value):
        self._set_data_by_key("t", value)

    @property
    def xrt(self):
        return self._get_data_by_key(("x", "r", "t"))

    @xrt.setter
    def xrt(self, value):
        return self._set_data_by_key(("x", "r", "t"), value)

    @property
    def xr(self):
        return self._get_data_by_key(("x", "r"))

    @xr.setter
    def xr(self, value):
        return self._set_data_by_key(("x", "r"), value)

    #
    # Independent velocities
    #

    @property
    def Vx(self):
        return self._get_data_by_key("Vx")

    @Vx.setter
    def Vx(self, value):
        self._set_data_by_key("Vx", value)

    @property
    def Vr(self):
        return self._get_data_by_key("Vr")

    @Vr.setter
    def Vr(self, value):
        self._set_data_by_key("Vr", value)

    @property
    def Vt(self):
        return self._get_data_by_key("Vt")

    @Vt.setter
    def Vt(self, value):
        self._set_data_by_key("Vt", value)

    @property
    def Vxrt(self):
        return self._get_data_by_key(("Vx", "Vr", "Vt"))

    @Vxrt.setter
    def Vxrt(self, value):
        self._set_data_by_key(("Vx", "Vr", "Vt"), value)

    @property
    def Omega(self):
        return self._get_data_by_key("Omega")

    @Omega.setter
    def Omega(self, Omega):
        self._set_data_by_key("Omega", Omega)

    #
    # Coordinaets
    #

    @dependent_property
    def rt(self):
        return self.r * self.t

    @dependent_property
    def xrrt(self):
        return np.stack((self.x, self.r, self.rt), axis=0)

    @dependent_property
    def xyz(self):
        return np.stack((self.x, self.y, self.z))

    @dependent_property
    def yz(self):
        return np.stack((self.y, self.z))

    @dependent_property
    def xy(self):
        return np.stack((self.x, self.y))

    @dependent_property
    def y(self):
        return self.r * np.sin(self.t)

    @dependent_property
    def z(self):
        return self.r * np.cos(self.t)

    @dependent_property
    def Vxrt_rel(self):
        return np.stack((self.Vx, self.Vr, self.Vt_rel))

    @dependent_property
    def Vxr(self):
        return np.stack((self.Vx, self.Vr))

    @dependent_property
    def U(self):
        return self.r * self.Omega

    @dependent_property
    def V(self):
        return util.vecnorm(self.Vxrt)

    @dependent_property
    def Vm(self):
        return util.vecnorm(self.Vxrt[:2])

    @dependent_property
    def Vt_rel(self):
        return self.Vt - self.U

    @dependent_property
    def V_rel(self):
        return np.sqrt(self.Vm**2.0 + self.Vt_rel**2.0)

    @dependent_property
    def halfVsq(self):
        return 0.5 * self.V**2

    @dependent_property
    def halfVsq_rel(self):
        return 0.5 * self.V_rel**2

    #
    # Angles
    #

    @dependent_property
    def Alpha_rel(self):
        return np.degrees(np.arctan2(self.Vt_rel, self.Vm))

    @dependent_property
    def Alpha(self):
        return np.degrees(np.arctan2(self.Vt, self.Vm))

    @dependent_property
    def Beta(self):
        return np.degrees(np.arctan2(self.Vr, self.Vx))

    @dependent_property
    def tanBeta(self):
        return self.Vr / self.Vx

    @dependent_property
    def tanAlpha(self):
        return self.Vt / self.Vm

    @dependent_property
    def tanAlpha_rel(self):
        return self.Vt_rel / self.Vm

    @dependent_property
    def cosBeta(self):
        return self.Vx / self.Vm

    @dependent_property
    def cosAlpha(self):
        return self.Vm / self.V

    @dependent_property
    def cosAlpha_rel(self):
        return self.Vm / self.V_rel

    #
    # Misc
    #

    @dependent_property
    def rpm(self):
        return self.Omega / 2.0 / np.pi * 60.0

    @dependent_property
    def conserved(self):
        return np.stack((self.rho, self.rhoVx, self.rhoVr, self.rhorVt, self.rhoe))

    @dependent_property
    def rhoVx(self):
        return self.rho * self.Vx

    @dependent_property
    def rhoVr(self):
        return self.rho * self.Vr

    @dependent_property
    def rhoVt(self):
        return self.rho * self.Vt

    @dependent_property
    def rhorVt(self):
        return self.r * self.rhoVt

    @dependent_property
    def rVt(self):
        return self.r * self.Vt

    @dependent_property
    def rhoe(self):
        return self.rho * self.e

    @dependent_property
    def rhoVm(self):
        return self.rho * self.Vm

    @dependent_property
    def e(self):
        return self.u + 0.5 * self.V**2.0

    @dependent_property
    def Ma(self):
        return self.V / self.a

    @dependent_property
    def Ma_rel(self):
        return self.V_rel / self.a

    @dependent_property
    def Mam(self):
        return self.Vm / self.a

    @dependent_property
    def I(self):
        return self.h + 0.5 * self.V**2.0 - self.U * self.Vt

    @dependent_property
    def _stagnation(self):
        return self.to_stagnation(self.Ma)

    @dependent_property
    def _stagnation_rel(self):
        return self.to_stagnation(self.Ma_rel)

    @property
    def Po(self):
        """Stagnation pressure [Pa]."""
        return self._stagnation.P

    @property
    def To(self):
        return self._stagnation.T

    @property
    def ao(self):
        return self._stagnation.a

    @property
    def ho(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V**2.0

    @property
    def Po_rel(self):
        return self._stagnation_rel.P

    @property
    def To_rel(self):
        return self._stagnation_rel.T

    @property
    def ho_rel(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V_rel**2.0

    @dependent_property
    def Vy(self):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        return self.Vr * cost - self.Vt * sint

    @dependent_property
    def Vz(self):
        cost = np.cos(self.t)
        sint = np.sin(self.t)
        return -self.Vr * sint - self.Vt * cost

    @dependent_property
    def P_rot(self):
        # Rotary static pressure
        if self.Omega.any():
            S = self.copy()
            # In rotating frame
            # Replace horel with rothalpy
            # i.e. subtract blade speed dyn head from h
            S.set_h_s(self.h - 0.5 * self.U**2, self.s)
            P = S.P
        else:
            # Just use normal static pressure in stationary frame
            P = self.P
        return P

    #
    # Fluxes
    #

    @dependent_property
    def flux_mass(self):
        # Mass fluxes in x and r dirns
        return np.stack((self.rhoVx, self.rhoVr, self.rhoVt))

    @dependent_property
    def flux_xmom(self):
        # Axial momentum fluxes in x and r dirns
        return np.stack(
            (self.rhoVx * self.Vx + self.P, self.rhoVr * self.Vx, self.rhoVt * self.Vx)
        )

    @dependent_property
    def flux_rmom(self):
        # Radial momentum fluxes in x and r dirns
        return np.stack(
            (
                self.rhoVx * self.Vr,
                self.rhoVr * self.Vr + self.P,
                self.rhoVt * self.Vr,
            )
        )

    @dependent_property
    def flux_rtmom(self):
        # Moment of angular momentum fluxes in x and r dirns
        return np.stack(
            (
                self.Vx * self.rhorVt,
                self.Vr * self.rhorVt,
                self.Vt * self.rhorVt + self.r * self.P,
            )
        )

    @dependent_property
    def flux_rothalpy(self):
        # Stagnation rothalpy fluxes in x an r dirns
        return self.flux_mass * self.I

    @dependent_property
    def flux_energy(self):
        # Stagnation entahlpy fluxes in x an r dirns
        return np.stack(
            (
                self.rhoVx * self.ho,
                self.rhoVr * self.ho,
                self.rhoVt * self.ho + self.Omega * self.r * self.P,
            )
        )

    @dependent_property
    def flux_entropy(self):
        # Mass fluxes in x and r dirns
        return self.flux_mass * self.s

    def set_conserved(self, conserved):
        rho, *rhoVxrt, rhoe = conserved
        Vxrt = rhoVxrt / rho
        Vxrt[2] /= self.r
        self.Vxrt = Vxrt
        u = rhoe / rho - 0.5 * self.V**2
        self.set_rho_u(rho, u)

    @dependent_property
    def fluxes(self):
        return np.stack(
            (
                self.rhoVx,
                self.rhoVx * self.Vx + self.P,
                self.rhoVx * self.Vr,
                self.rhoVx * self.rVt,
                self.rhoVx * self.ho,
            )
        )

    @dependent_property
    def bcond(self):
        return np.stack(
            (
                self.ho,
                self.s,
                self.tanAlpha,
                self.tanBeta,
                self.P,
            )
        )

    @dependent_property
    def drhoe_drho_P(self):
        return self.e + self.rho * self.dudrho_P

    @dependent_property
    def drhoe_dP_rho(self):
        return self.rho * self.dudP_rho

    @dependent_property
    def prim(self):
        return np.stack((self.rho, self.Vx, self.Vr, self.Vt, self.P))

    def set_prim(self, prim):
        rho, *Vxrt, P = prim
        self.set_P_rho(P, rho)
        self.Vxrt = Vxrt
        return self

    def mix_out(self):
        """Mix out the cut to a scalar state, conserving mass, momentum and energy."""
        return turbigen.average.mix_out(self)

    def area_average(self, prop):
        """Take area average of property over the cut surface.

        prop_avg = integral (prop  dA) / integral (dA)

        Parameters
        ----------
        prop : array
            Nodal property to average over the cut surface, same shape as self.

        Returns
        -------
            prop_avg : float
            Area average of the property over the cut surface.

        """
        dA = util.vecnorm(self.dA_node)
        if prop.ndim == 3:
            dA = dA[..., None]  # Add a dimension for broadcasting
        return np.sum(prop * dA) / np.sum(dA)

    @dependent_property
    def mass_flow(self):
        """Integrate the mass flow through a cut surface."""
        if not self.ndim == 2:
            raise Exception("Mass flow is only defined for 2D cuts")
        return np.sum(util.dot(self.flux_mass, self.dA_node))

    def mass_average(self, prop):
        """Take mass average of property through the cut surface.

        prop_avg = integral (prop  rho V dot dA) / integral (rho V dot dA)

        Parameters
        ----------
        prop : array
            Nodal property to average over the cut surface, same shape as self.

        Returns
        -------
            prop_avg : float
            Area average of the property over the cut surface.

        """
        if not self.ndim == 2:
            raise Exception("Mass average is only defined for 2D cuts")
        return np.sum(prop * util.dot(self.flux_mass, self.dA_node)) / self.mass_flow

    @dependent_property
    def dA_node(self):
        """Nodal weighted area.

        sum(dA_node) = A
        sum(prop * dA_node) = integral(prop dA)

        """
        if not self.ndim == 2:
            raise Exception("nodal area is only defined for 2D grids")

        # Face area magnitudes
        dA_face = self.dA

        # Distribute face area to nodes
        dA_node = np.zeros((3,) + self.shape)
        dA_node[:, :-1, :-1] += dA_face
        dA_node[:, :-1, 1:] += dA_face
        dA_node[:, 1:, :-1] += dA_face
        dA_node[:, 1:, 1:] += dA_face
        dA_node /= 4.0

        return dA_node

    @dependent_property
    def dA(self):
        # Vector area for 2D cuts, Gauss' theorem method
        if not self.ndim == 2:
            raise Exception("Face area is only defined for 2D grids")

        # Define four vertices ABCD
        #    B      C
        #     *----*
        #  ^  |    |
        #  k  *----*
        #    A      D
        #      i>
        #
        v = self.xrrt
        A = v[:, :-1, :-1, None]
        B = v[:, :-1, 1:, None]
        C = v[:, 1:, 1:, None]
        D = v[:, 1:, :-1, None]
        return util.dA_Gauss(A, B, C, D)[..., 0]


class PerfectFlowField(turbigen.fluid.PerfectState, BaseFlowField):
    """Flow and thermodynamic properties of a perfect gas."""

    _data_rows = (
        "x",
        "r",
        "t",
        "Vx",
        "Vr",
        "Vt",
        "rho",
        "u",
        "Omega",
    )

    @classmethod
    def from_properties(cls, xrt, Vxrt, PT, cp, ga, mu, Omega):
        # Make an empty class
        F = cls(np.shape(xrt)[1:])

        # Insert our data
        F.cp, F.gamma, F.mu, F.Omega = cp, ga, mu, Omega
        F.set_P_T(*PT)
        F.Vxrt = Vxrt
        F.xrt = xrt

        return F


class RealFlowField(turbigen.fluid.RealState, BaseFlowField):
    """Flow and thermodynamic properties of a perfect gas."""

    _data_rows = (
        "x",
        "r",
        "t",
        "Vx",
        "Vr",
        "Vt",
        "rho",
        "u",
        "Omega",
    )

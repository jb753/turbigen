"""Classes to represent flow fields."""

import numpy as np
import turbigen.base
import turbigen.fluid
import turbigen.yaml
import turbigen.abstract
from multiprocessing import Pool
import os
from turbigen import util

from turbigen.base import dependent_property


def make_mean_line(rrms, A, Omega, Vxrt, S):
    """Assemble a perfect or real mean-line data structure from input states."""
    try:
        S = S[0].stack(S)
    except AttributeError:
        pass
    if isinstance(S, turbigen.fluid.PerfectState):
        ml_class = PerfectMeanLine
    elif isinstance(S, turbigen.fluid.RealState):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(S)}")
    return ml_class.from_states(rrms, A, Omega, Vxrt, S)


def make_mean_line_from_flowfield(A, F, Ds_mix=0.0):
    """Assemble a perfect or real mean-line data structure from input states."""
    if isinstance(F, PerfectFlowField):
        ml_class = PerfectMeanLine
    elif isinstance(F, RealFlowField):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(F)}")
    ml = ml_class.from_states(F.r, A, F.Omega, F.Vxrt, F, F.Nb)
    ml.Ds_mix = Ds_mix
    ml._metadata.pop("patches")
    ml._metadata.pop("Nb")
    return ml


class BaseFlowField(turbigen.base.StructuredData, turbigen.abstract.FlowField):
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

    @property
    def Nb(self):
        return self._get_data_by_key("Nb")

    @Nb.setter
    def Nb(self, val):
        return self._set_data_by_key("Nb", val)

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
    def I(self):
        return self.h + 0.5 * self.V**2.0 - self.U * self.Vt

    @dependent_property
    def e(self):
        return self.u + 0.5 * self.V**2.0

    @dependent_property
    def Ma(self):
        return self.V / self.a

    @dependent_property
    def Ma_rel(self):
        return self.V_rel / self.a

    @property
    def Po(self):
        """Stagnation pressure [Pa]."""
        return self.stagnation.P

    @property
    def To(self):
        return self.stagnation.T

    @property
    def ao(self):
        return self.stagnation.a

    @property
    def ho(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V**2

    @property
    def halfVsq(self):
        return 0.5 * self.V**2

    @dependent_property
    def halfVsq_rel(self):
        return 0.5 * self.V_rel**2

    @property
    def Po_rel(self):
        return self.stagnation_rel.P

    @property
    def To_rel(self):
        return self.stagnation_rel.T

    @property
    def ho_rel(self):
        # We can directly use static enthalpy and velocity
        return self.h + 0.5 * self.V_rel**2.0

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
        return self.r * self.rho * self.Vt

    @dependent_property
    def rhoe(self):
        return self.rho * self.e

    @dependent_property
    def V_rel(self):
        return np.sqrt(self.Vm**2.0 + self.Vt_rel**2.0)

    @dependent_property
    def drhoe_drho_P(self):
        return self.e + self.rho * self.dudrho_P

    @dependent_property
    def drhoe_dP_rho(self):
        return self.rho * self.dudP_rho

    @dependent_property
    def conserved(self):
        return np.stack((self.rho, self.rhoVx, self.rhoVr, self.rhorVt, self.rhoe))

    @dependent_property
    def rpm(self):
        return self.Omega / 2.0 / np.pi * 60.0

    @property
    def x(self):
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

    @dependent_property
    def y(self):
        return self.r * np.sin(self.t)

    @dependent_property
    def z(self):
        return self.r * np.cos(self.t)

    @dependent_property
    def xyz(self):
        return np.stack((self.x, self.y, self.z))

    @xrt.setter
    def xrt(self, value):
        return self._set_data_by_key(("x", "r", "t"), value)

    @property
    def pitch(self):
        return 2.0 * np.pi / self.Nb

    def set_Nb(self, Nb):
        self.Nb = Nb
        return self

    def set_Omega(self, Omega):
        self.Omega = Omega
        return self

    def set_V_Alpha_Beta(self, V=None, Alpha=None, Beta=None):
        if V is None:
            V = self.V

        if Alpha is None:
            Alpha = self.Alpha

        if Beta is None:
            Beta = self.Beta

        tanAl = np.tan(np.radians(Alpha))
        tanBe = np.tan(np.radians(Beta))
        Vm = V / np.sqrt(1.0 + tanAl**2)
        Vx = V / np.sqrt((1.0 + tanBe**2) * (1.0 + tanAl**2))
        Vt = Vm * tanAl
        Vr = Vx * tanBe
        return self.set_Vxrt(Vx, Vr, Vt)

    def set_Vxrt(self, Vx=None, Vr=None, Vt=None):
        """Set the axial, radial and tangential velocity components."""
        if Vx is not None:
            self.Vx = Vx

        if Vr is not None:
            self.Vr = Vr

        if Vt is not None:
            self.Vt = Vt

        return self

    def set_conserved(self, conserved):
        rho, *rhoVxrt, rhoe = conserved
        Vxrt = rhoVxrt / rho
        Vxrt[2] /= self.r
        self.Vxrt = Vxrt
        u = rhoe / rho - 0.5 * self.V**2
        return self.set_rho_u(rho, u)

    def set_primitive(self, primitive):
        rho, Vx, Vr, Vt, P = primitive
        self.Vx = Vx
        self.Vr = Vr
        self.Vt = Vt
        self.set_P_rho(P, rho)
        return self

    @property
    def bcond(self):
        return np.stack(
            (
                self.ho,
                self.s,
                self.Vt / self.Vm,  # tanAlpha
                self.Vr / self.Vx,  # tanBeta
                self.P,
            )
        )

    @property
    def primitive(self):
        return np.stack(
            (
                self.rho,
                self.Vx,
                self.Vr,
                self.Vt,
                self.P,
            )
        )

    @property
    def fluxx(self):
        return np.stack(
            (
                self.rhoVx,
                self.rhoVx * self.Vx + self.P,
                self.rhoVx * self.Vr,
                self.rhoVx * self.r * self.Vt,
                self.rhoVx * self.ho,
            )
        )

    @property
    def fluxr(self):
        return np.stack(
            (
                self.rhoVr,
                self.rhoVr * self.Vx + self.P,
                self.rhoVr * self.Vr,
                self.rhoVr * self.r * self.Vt,
                self.rhoVr * self.ho,
            )
        )

    @property
    def fluxt(self):
        return np.stack(
            (
                self.rhoVt,
                self.rhoVt * self.Vx + self.P,
                self.rhoVt * self.Vr,
                self.rhoVt * self.r * self.Vt,
                self.rhoVt * self.ho,
            )
        )

    @property
    def vol_Cartesian(self):
        if not self.ndim == 3:
            raise Exception("Cell volume is only defined for 3D grids")

        # Numpy cross function assumes that the components are in last axis
        xyz = np.moveaxis(self.xyz, 0, -1).astype(np.float64)

        # Vectors for cell sides
        qi = np.diff(xyz[:, :-1, :-1, :], axis=0)
        qj = np.diff(xyz[:-1, :, :-1, :], axis=1)
        qk = np.diff(xyz[:-1, :-1, :, :], axis=2)

        return -np.sum(qk * np.cross(qi, qj), axis=-1)

    def set_xrt(self, x=None, r=None, t=None):
        if x is not None:
            self.x = x

        if r is not None:
            self.r = r

        if t is not None:
            self.t = t

        return self

    def set_xyz(self, x=None, y=None, t=None):
        """Set the x, y and t coordinates."""
        if x is not None:
            self.x = x

        if y is not None:
            self.r = y

        if t is not None:
            self.t = t

        return self


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


class PerfectMeanLine(turbigen.base.MeanLine, PerfectFlowField):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


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


class RealMeanLine(turbigen.base.MeanLine, RealFlowField):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


def mean_line_from_dict(d):
    if d["class"] == "PerfectMeanLine":
        return PerfectMeanLine.from_dict(d)
    elif d["class"] == "RealMeanLine":
        return RealMeanLine.from_dict(d)
    else:
        raise Exception(f"Unrecognised mean line class {d['class']}")


def read_mean_line_database(database_file):
    """Load a list of mean_lines from a database file."""
    # Initialise the objects in parallel
    Nworker = os.cpu_count()
    with Pool(Nworker) as p:
        ml = p.map(mean_line_from_dict, turbigen.yaml.read_yaml_list(database_file))
    return ml

import turbigen.fluid
import turbigen.base
import turbigen.abstract
import turbigen.flowfield
from turbigen.base import dependent_property
import numpy as np

from turbigen import util

logger = util.make_logger()


def make_mean_line_from_states(rrms, A, Omega, Vxrt, S):
    """Assemble a perfect or real mean-line data structure from input states."""
    try:
        S = turbigen.base.stack(S)
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
    if isinstance(F, turbigen.flowfield.PerfectFlowField):
        ml_class = PerfectMeanLine
    elif isinstance(F, turbigen.flowfield.RealFlowField):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(F)}")
    ml = ml_class.from_states(F.r, A, F.Omega, F.Vxrt, F, F.Nb)
    ml.Ds_mix = Ds_mix
    ml._metadata.pop("patches")
    ml._metadata.pop("Nb")
    return ml


class MeanLine:
    """Encapsulate flow and geometry on a nomial mean streamsurface."""

    @property
    def A(self):
        return self._get_data_by_key("A")

    @A.setter
    def A(self, val):
        return self._set_data_by_key("A", val)

    @property
    def Nb(self):
        return self._get_data_by_key("Nb")

    @Nb.setter
    def Nb(self, val):
        return self._set_data_by_key("Nb", val)

    @classmethod
    def from_states(cls, rrms, A, Omega, Vxrt, S, Nb=None):
        """Construct a mean-line from generic state objects."""

        # Preallocate class of correct shape
        F = cls(S.shape)

        # Reference the metadata (which defines fluid properties)
        F._metadata = S._metadata

        # Set mean-line variables
        F.r = rrms
        F.A = A
        F.Vxrt = Vxrt
        F.Omega = Omega
        F.set_P_T(S.P, S.T)

        if Nb is not None:
            F.Nb = Nb

        return F

    def get_row(self, irow):
        ist = irow * 2
        ien = ist + 2
        return self[ist:ien]

    def interpolate_guess(self, ann):
        # Get meridional coordinates along mean-line
        npts = 100
        sg = np.linspace(0.0, ann.mmax, npts)
        xg, rg = ann.evaluate_xr(sg, 0.5)

        # Get variations in thermodynamic state at inlet,exit and row boundaries
        # From the mean-line
        ro_mid = np.pad(self.rho, 1, "edge")
        u_mid = np.pad(self.u, 1, "edge")
        Vx_mid = np.pad(self.Vx, 1, "edge")
        Vr_mid = np.pad(self.Vr, 1, "edge")
        Vt_mid = np.pad(self.Vt, 1, "edge")

        # Interpolate flow properties linearly along mean-line
        sq = np.linspace(0.0, ann.mmax, ann.nseg + 1)
        rog = np.interp(sg, sq, ro_mid)
        ug = np.interp(sg, sq, u_mid)
        Vxg = np.interp(sg, sq, Vx_mid)
        Vrg = np.interp(sg, sq, Vr_mid)
        Vtg = np.interp(sg, sq, Vt_mid)

        Fg = self.empty(shape=(npts,)).set_rho_u(rog, ug)
        Fg.xr = np.stack((xg, rg))
        Fg.Vxrt = np.stack((Vxg, Vrg, Vtg))

        return Fg

    #
    # Override Omega methods to make it a vector
    #

    @staticmethod
    def _check_vectors(*args):
        """Ensure that some inputs have same shape and len multiple of 2."""
        shp = np.shape(args[0])
        assert np.mod(shp[0], 2) == 0
        assert len(shp) == 1
        for arg in args:
            assert np.shape(arg) == shp

    @property
    def rrms(self):
        return self.r

    @dependent_property
    def mdot(self):
        return self.rho * self.Vm * self.A

    @dependent_property
    def span(self):
        return self.A / 2.0 / np.pi / self.rmid

    @dependent_property
    def rmid(self):
        return (self.rtip + self.rhub) * 0.5

    @dependent_property
    def rhub(self):
        return np.sqrt(2.0 * self.rrms**2.0 - self.rtip**2.0)

    @dependent_property
    def rtip(self):
        return np.sqrt(self.A * self.cosBeta / 2.0 / np.pi + self.rrms**2.0)

    @dependent_property
    def htr(self):
        return self.rhub / self.rtip

    @dependent_property
    def Aflow(self):
        return self.A * self.cosAlpha_rel

    @dependent_property
    def ARflow(self):
        return self.Aflow[1:] / self.Aflow[:-1]

    def warn(self):
        """Print a warning if there are any suspicious values."""

        # Warn for very high flow angles
        if np.abs(self.Alpha_rel).max() > 85.0:
            logger.warning(
                """WARNING: Relative flow angles are approaching 90 degrees.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
            )

        # Warn for wobbly annulus
        is_radial = np.abs(self.Beta).max() > 10.0
        is_multirow = len(self.x) > 2
        if is_radial and is_multirow:
            if np.diff(np.sign(np.diff(self.rrms))).any():
                logger.warning(
                    """WARNING: Radii do not vary monotonically.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
                )

    def check(self):
        # """Assert that conserved quantities are in fact conserved"""

        check_failed = False

        # Check mass is conserved
        rtol = 1e-2
        mdot = self.mdot
        if np.isnan(mdot).any():
            check_failed = True
            logger.iter("NaN mass flow rate")

        if np.ptp(mdot) > (mdot[0] * rtol):
            check_failed = True
            logger.iter(f"Mass is not conserved, mdot={mdot}")

        # Get a sensible rothalpy tolerance
        Itol = (self.a.mean() ** 2) * 1e-3

        # Split the rothalpy into rows (where Omega changes)
        if self.Omega.all() or not self.Omega.any():
            isplit = []
        else:
            isplit = np.where(np.abs(np.diff(self.Omega)) > 0.0)[0] + 1
        Irow = np.stack(np.array_split(self.I, isplit))
        assert Irow.shape[1] == 2
        logger.debug("Checking row rothalpies")
        for irow in range(Irow.shape[0]):
            Iirow = Irow[irow, :]
            if np.ptp(Iirow) > Itol:
                check_failed = True
                logger.iter(
                    f"Rothalpy not conserved in row {irow}: I = {Iirow}\n"
                    f"tolerance: {Itol}, diff: {np.ptp(Iirow)}"
                )

        # Check that stagnation enthalpy is conserved between blade rows
        if isplit:
            logger.debug("Checking gap enthalpies")
            hogap = np.array_split(self.ho[:-1], isplit - 1)[1:]
            for igap in range(len(hogap)):
                if not np.ptp(hogap[igap]).item() < Itol:
                    check_failed = True
                    logger.iter(
                        f"Absolute stagnation enthalpy not conserved across gap {igap}:\n"
                        f"hogap = {hogap[igap]}"
                    )

        return not check_failed

    def show_debug(self):
        np.set_printoptions(linewidth=np.inf, precision=4, floatmode="maxprec_equal")
        logger.iter(f"rrms: {self.rrms}")
        logger.iter(f"rhub: {self.rhub}")
        logger.iter(f"rtip: {self.rtip}")
        logger.iter(f"A: {self.A}")
        logger.iter(f"span: {self.span}")
        logger.iter(f"I: {self.I}")
        logger.iter(f"To: {self.To}")
        logger.iter(f"T: {self.T}")
        logger.iter(f"Po: {self.Po}")
        logger.iter(f"P: {self.P}")
        logger.iter(f"Vx: {self.Vx}")
        logger.iter(f"Vr: {self.Vr}")
        logger.iter(f"Vt: {self.Vt}")
        logger.iter(f"Vt_rel: {self.Vt_rel}")
        logger.iter(f"Ma: {self.Ma}")
        logger.iter(f"Ma_rel {self.Ma_rel}")
        logger.iter(f"U: {self.U}")
        logger.iter(f"Al: {self.Alpha}")
        logger.iter(f"Al_rel: {self.Alpha_rel}")
        logger.iter(f"Beta: {self.Beta}")
        logger.iter(f"Omega: {self.Omega}")
        logger.iter(f"mdot: {self.mdot}")
        logger.iter(f"ho: {self.ho}")
        logger.iter(f"rho: {self.rho}")
        logger.iter(f"s: {self.s}")

    def __str__(self):
        Pstr = np.array2string(self.Po / 1e5, precision=4)
        Tstr = np.array2string(self.To, precision=4)
        Mastr = np.array2string(self.Ma, precision=3)
        Alrstr = np.array2string(self.Alpha_rel, precision=2)
        Alstr = np.array2string(self.Alpha, precision=2)
        Vxstr = np.array2string(self.Vx, precision=1)
        Vrstr = np.array2string(self.Vr, precision=1)
        Vtstr = np.array2string(self.Vt, precision=1)
        Vtrstr = np.array2string(self.Vt_rel, precision=1)
        rpmstr = np.array2string(self.rpm, precision=0)
        mstr = np.array2string(self.mdot, precision=2)
        return f"""MeanLine(
    Po={Pstr} bar,
    To={Tstr} K,
    Ma={Mastr},
    Vx={Vxstr} m/s,
    Vr={Vrstr} m/s,
    Vt={Vtstr} m/s,
    Vt_rel={Vtrstr} m/s,
    Al={Alstr} deg,
    Al_rel={Alrstr} deg,
    rpm={rpmstr},
    mdot={mstr} kg/s
    )"""

    def rspf(self, spf):
        if not np.shape(spf) == ():
            spf = spf.reshape(-1, 1)
        return self.rhub * (1.0 - spf) + self.rtip * spf

    def Vt_free_vortex(self, spf, n=-1):
        return self.Vt * (self.rspf(spf) / self.rrms) ** n

    def Vt_rel_free_vortex(self, spf, n=-1):
        return self.Vt_free_vortex(spf, n) - self.Omega * self.rspf(spf)

    def Alpha_free_vortex(self, spf, n=-1):
        return np.degrees(np.arctan(self.Vt_free_vortex(spf, n) / self.Vm))

    def Alpha_rel_free_vortex(self, spf, n=-1):
        return np.degrees(np.arctan(self.Vt_rel_free_vortex(spf, n) / self.Vm))

    def _get_ref(self, key):
        """Return a variable at inlet/exit of rows, for compressor/turbine."""
        x = getattr(self, key)
        try:
            return np.where(self.ARflow[::2] > 1.0, x[::2], x[1::2])
        except (IndexError, TypeError):
            return x

    @dependent_property
    def rho_ref(self):
        return self._get_ref("rho")

    @dependent_property
    def V_ref(self):
        return self._get_ref("V_rel")

    @dependent_property
    def mu_ref(self):
        return self._get_ref("mu")

    @dependent_property
    def L_visc(self):
        return self.mu_ref / self.rho_ref / self.V_ref

    @dependent_property
    def RR(self):
        return self.rrms[1:] / self.rrms[:-1]

    @dependent_property
    def PR_tt(self):
        PR = self.Po[-1] / self.Po[0]
        if PR < 1.0:
            PR = 1.0 / PR
        return PR

    @dependent_property
    def PR_ts(self):
        PR = self.P[-1] / self.Po[0]
        if PR < 1.0:
            PR = 1.0 / PR
        return PR

    @dependent_property
    def eta_tt(self):
        hos = self.copy().set_P_s(self.Po, self.s[0]).h
        ho = self.ho
        Dho = ho[-1] - ho[0]
        Dhos = hos[-1] - hos[0]
        if Dho == 0.0:
            return 1.0
        eta_tt = Dhos / Dho
        if eta_tt > 1.0:
            eta_tt = 1.0 / eta_tt
        return eta_tt

    @dependent_property
    def eta_ts(self):
        hs = self.copy().set_P_s(self.P, self.s[0]).h
        ho = self.ho
        Dho = ho[-1] - ho[0]
        Dhs = hs[-1] - ho[0]
        if Dho == 0.0:
            return 1.0
        eta_ts = Dhs / Dho
        if eta_ts > 1.0:
            eta_ts = 1.0 / eta_ts
        return eta_ts

    @dependent_property
    def eta_poly(self):
        eta_poly = (
            self.gamma
            / (self.gamma - 1.0)
            * np.log(self.To[-1] / self.To[0])
            / np.log(self.Po[-1] / self.Po[0])
        )
        if eta_poly > 1.0:
            eta_poly = 1.0 / eta_poly
        return eta_poly

    def to_dump(self):
        """Make a simple dict dump of the mean-line data."""
        # Without any state information
        return {
            "rrms": self.rrms.tolist(),
            "A": self.A.tolist(),
            "Omega": self.Omega.tolist(),
            "Vxrt": self.Vxrt.tolist(),
            "Nb": self.Nb.tolist(),
            "rho": self.rho.tolist(),
            "u": self.u.tolist(),
            "Ds_mix": self.Ds_mix,
        }


class PerfectMeanLine(
    MeanLine, turbigen.flowfield.PerfectFlowField, turbigen.abstract.MeanLine
):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


class RealMeanLine(
    MeanLine, turbigen.flowfield.RealFlowField, turbigen.abstract.MeanLine
):
    """Encapsulate the mean-line flow and geometry of a turbomachine."""

    _data_rows = ("x", "r", "A", "Vx", "Vr", "Vt", "rho", "u", "Omega", "Nb")


def meanline_from_dump(dump, So1):
    """Load a mean-line from a simple dict dump."""
    n = len(dump["rrms"])
    S = So1.empty(shape=(n,))
    S.set_rho_u(dump["rho"], dump["u"])

    if isinstance(So1, turbigen.fluid.PerfectState):
        ml_class = PerfectMeanLine
    elif isinstance(So1, turbigen.fluid.RealState):
        ml_class = RealMeanLine
    else:
        raise Exception(f"Unknown fluid class {type(So1)}")
    ml = ml_class.from_states(
        dump["rrms"], dump["A"], dump["Omega"], dump["Vxrt"], S, dump["Nb"]
    )
    ml.Ds_mix = dump["Ds_mix"]
    ml._metadata.pop("patches", None)
    ml._metadata.pop("Nb", None)
    return ml

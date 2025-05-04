from time import sleep
from dataclasses import dataclass
from timeit import default_timer as timer
import shutil
from glob import glob
import h5py
import numpy as np
import turbigen.grid
from turbigen.exceptions import ConvergenceError
import subprocess
import os
from pathlib import Path
import signal
import sys
import re
import grp
import getpass
from copy import copy
from turbigen.solvers.base import BaseSolver, ConvergenceHistory
import time

import turbigen.util

logger = turbigen.util.make_logger()


@dataclass
class ts3(BaseSolver):
    # Override base attributes
    _name = "ts3"

    workdir: Path = None
    """Working directory to run the simulation in."""

    environment_script: Path = Path(
        "/usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100"
    )
    """Setup environment shell script to be sourced before running."""

    atol_eta: float = 0.005
    """Absolute tolerance on drift in isentropic efficiency."""

    cfl: float = 0.4
    """Courant--Friedrichs--Lewy number, reduce for more stability."""

    dampin: float = 25.0
    """Negative feedback factor, reduce for more stability."""

    facsecin: float = 0.005
    """Fourth-order smoothing feedback factor, increase for more stability."""

    fmgrid: float = 0.2
    """Multigrid factor, reduce for more stability."""

    ilos: int = 2
    """Viscous model, 0 for inviscid, 1 for mixing-length, 2 for Spalart-Allmaras."""

    Lref_xllim: str = "pitch"
    """Mixing length characteristic dimension, "pitch" or "span"."""

    nchange: int = 2000
    """At start of simulation, ramp smoothing and damping over this many time steps."""

    nstep: int = 10000
    """Number of time steps."""

    nstep_avg: int = 5000
    """Average over the last `nstep_avg` steps of the calculation."""

    rfmix: float = 0.0
    """Mixing plane relaxation factor."""

    rtol_mdot: float = 0.01
    """Relative tolerance on mass flow conservation error and drift."""

    sfin: float = 0.5
    """Proportion of second-order smoothing, increase for more stability."""

    tvr: float = 10.0
    """Initial guess of turbulent viscosity ratio."""

    xllim: float = 0.03
    """Mixing length limit as a fraction of characteristic dimension."""

    rfin: float = 0.5
    """Inlet relaxation factor, reduce for low-Mach flows."""

    nstep_soft: int = 0
    """Number of steps for soft start precursor simulation."""

    sa_helicity_option: int = 0
    """Spalart--Allmaras turbulence model helicity correction."""

    smooth_scale_dts_option: int = 0
    smooth_scale_directional_option: int = 0

    show_yplus: bool = False
    laminar: bool = False
    """Enable laminar boundary layers on all walls."""

    fac_st0: float = 1.0
    ipout: int = 3
    convert_sliding: bool = False
    precon: int = 0
    dts: int = 0
    nstep_cycle: int = 72
    nstep_inner: int = 200
    ncycle: int = 0
    frequency: float = 0.0
    nstep_save_probe: int = 0
    nstep_save_start_probe: int = 0
    xllim_free: float = 0.1
    free_turb: float = 0.05
    turbvis_lim: float = 3000.0

    sa_ch1: float = 0.71
    sa_ch2: float = 0.6

    def __post_init__(self):
        if isinstance(self.workdir, str):
            self.workdir = Path(self.workdir)
        if isinstance(self.environment_script, str):
            self.environment_script = Path(self.environment_script)

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        config = super().to_dict()
        config.pop("workdir")
        config["environment_script"] = str(self.environment_script)
        return config

    def application_variables(self, ga, cp, mu):
        # """Make a complete set of applications variables, with defaults overriden
        av = DEFAULT_AV.copy()
        for k in av:
            if hasattr(self, k):
                av[k] = getattr(self, k)
        av["ga"] = ga
        av["cp"] = cp
        av["viscosity"] = mu

        # Never use the built-in guesses
        # Always restart from our own flow field
        av["restart"] = 1

        if av["dts"]:
            av["nstep_save_start"] = av["ncycle"] * av["nstep_cycle"] - self.nstep_avg
        else:
            if av["nstep"] < 0:
                av["nstep"] = int(2 * self.nstep_avg)
            av["nstep_save_start"] = av["nstep"] - self.nstep_avg

        # Raise error if averaging not OK
        nstep = av["nstep"]
        nstep_save_start = av["nstep_save_start"]
        if nstep_save_start >= nstep and nstep > 0 and not av["dts"]:
            raise Exception(f"nstep_save_start={nstep_save_start} is > nstep={nstep}")
        if nstep_save_start < 0:
            raise Exception(f"nstep_save_start={nstep_save_start} is < 0")

        return av

    def block_variables(self, block, rref, laminar):
        # """Make a dictionary of block variables, with defaults overriden as needed."""
        bv = DEFAULT_BV.copy()
        for k in bv:
            if hasattr(self, k):
                bv[k] = getattr(self, k)
        # Set some block variables using the block data
        bv["nblade"] = block.Nb
        bv["fblade"] = float(block.Nb)

        assert np.ptp(block.rpm) == 0.0
        rpm = block.rpm.mean()
        if self.convert_sliding and rpm == 0.0:
            for patch in block.patches:
                if isinstance(patch, turbigen.grid.MixingPatch):
                    assert np.ptp(patch.match.block.rpm) == 0.0
                    rpm = patch.match.block.rpm.mean()
                    break
        bv["rpm"] = rpm
        bv["fracann"] = 1.0 / float(block.Nb)
        if self.Lref_xllim == "pitch":
            bv["xllim"] = 2.0 * np.pi * rref / float(block.Nb) * self.xllim
        elif self.Lref_xllim == "span":
            span = np.ptp(block.r)
            bv["xllim"] = span * self.xllim
        elif self.Lref_xllim == "fix":
            bv["xllim"] = self.xllim
        else:
            raise Exception(f"Unrecognised Lref_xllim={self.Lref_xllim}")
        bv.update(_get_wall_rpms(block))

        if laminar:
            # Set laminar from i=0 to i=ni on every block
            ni1 = block.ni - 1
            bv["itrans"] = -1
            bv["itrans_j1_st"] = 0
            bv["itrans_j1_en"] = ni1
            bv["itrans_j2_st"] = 0
            bv["itrans_j2_en"] = ni1
            bv["itrans_k1_st"] = 0
            bv["itrans_k1_en"] = ni1
            bv["itrans_k2_st"] = 0
            bv["itrans_k2_en"] = ni1
            bv["itrans_j1_frac"] = 0.1
            bv["itrans_j2_frac"] = 0.1
            bv["itrans_k1_frac"] = 0.1
            bv["itrans_k2_frac"] = 0.1

        return bv

    def robust(self):
        """Increase damping and smoothing, lower CFL, and use mixing-length model."""
        return self.replace(
            ilos = 1,
            dampin = 3.0,
            facsecin = 0.02,
            sfin = 2.0,
            cfl = 0.3,
            fmgrid = 0.0,
            soft_start = False,
            precon = 0,
            dts = 0,
        )

    def restart(self):
        """Restart the simulation from a previous solution."""
        return self.replace(
            nchange = 0,
        )

    def run(self, grid, machine, workdir):
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        run(grid, self, machine, workdir)

# Block attributes that must be present
# Where we should not set a default, use None
DEFAULT_BA = {
    "bid": None,
    "nc": 0,
    "np": None,
    "ncl": 0,
    "ni": None,
    "nj": None,
    "nk": None,
    "procid": 0,
    "threadid": 0,
}

# Patch attributes that must be present
# Where we should not set a default, use None
DEFAULT_PA = {
    "pid": None,
    "bid": None,
    "ist": None,
    "jst": None,
    "kst": None,
    "ien": None,
    "jen": None,
    "ken": None,
    "idir": None,
    "jdir": None,
    "kdir": None,
    "kind": None,
    "nface": 0,
    "nt": 1,
    "nxbid": None,
    "nxpid": None,
}

# Application variables that must be present
# Where we should not set a default, use None
DEFAULT_AV = {
    "adaptive_smoothing": 1,
    "cfl": 0.4,
    "cfl_en_ko": 0.4,
    "cfl_ko": 0.4,
    "cfl_st_ko": 0.01,
    "cp": None,
    "cp0_0": 1005.0,
    "cp0_1": 1005.0,
    "cp1_0": 0.0,
    "cp1_1": 0.0,
    "cp2_0": 0.0,
    "cp2_1": 0.0,
    "cp3_0": 0.0,
    "cp3_1": 0.0,
    "cp4_0": 0.0,
    "cp4_1": 0.0,
    "cp5_0": 0.0,
    "cp5_1": 0.0,
    "dampin": 25.0,
    "dts": 0,
    "dts_conv": 0.0,
    "fac_sa_smth": 4.0,
    "fac_sa_step": 1.0,
    "fac_st0": 1.0,
    "fac_st0_option": 0,
    "fac_st1": 1.0,
    "fac_st2": 1.0,
    "fac_st3": 1.0,
    "fac_stmix": 0.0,
    "fac_wall": 1.0,
    "facsafe": 0.2,
    "facsecin": 0.005,
    "frequency": 0.0,
    "ga": None,
    "if_ale": 0,
    "if_no_mg": 0,
    "ifgas": 0,
    "ifsuperfac": 0,
    "ilos": 2,
    "ko_dist": 1e-4,
    "ko_restart": 0,
    "nchange": 1000,
    "ncycle": 0,
    "nlos": 5,
    "nomatch_int": 1,
    "nspecies": 1,
    "nstep": 10000,
    "nstep_cycle": 0,
    "nstep_inner": 0,
    "nstep_save": 0,
    "nstep_save_probe": 0,
    "nstep_save_start": 0,
    "nstep_save_start_probe": 0,
    "poisson_cfl": 0.7,
    "poisson_limit": 0,
    "poisson_nsmooth": 10,
    "poisson_nstep": 0,
    "poisson_restart": 2,
    "poisson_sfin": 0.02,
    "prandtl": 1.0,
    "precon": 0,
    "pref": 1e5,
    "rfmix": 0.0,
    "rfvis": 0.2,
    "rg_cp0": 1005.0,
    "rg_cp1": 0.0,
    "rg_cp2": 0.0,
    "rg_cp3": 0.0,
    "rg_cp4": 0.0,
    "rg_cp5": 0.0,
    "rg_rgas": 287.15,
    "rgas_0": 287.15,
    "rgas_1": 287.15,
    "sa_ch1": 1.0,
    "sa_ch2": 1.0,
    "sa_helicity_option": 0,
    "schmidt_0": 1.0,
    "schmidt_1": 1.0,
    "sf_scalar": 0.05,
    "sfin": 0.5,
    "sfin_ko": 0.05,
    "sfin_sa": 0.05,
    "smooth_scale_directional_option": 0,
    "smooth_scale_dts_option": 0,
    "smooth_scale_precon_option": 0,
    "tref": 300.0,
    "turb_vis_damp": 1.0,
    "turbvis_lim": 3000.0,
    "use_temperature_sensor": 0,
    "viscosity": None,
    "viscosity_a1": 0.0,
    "viscosity_a2": 0.0,
    "viscosity_a3": 0.0,
    "viscosity_a4": 0.0,
    "viscosity_a5": 0.0,
    "viscosity_law": 0,
    "wall_law": 0,
    "write_egen": 0,
    "write_force": 0,
    "write_tdamp": 0,
    "write_yplus": 1,
}

# Block variables that must be present
# Where we should not set a default, use None
DEFAULT_BV = {
    "dampin_mul": 1.0,
    "fac_st0": 1.0,
    "facsecin_mul": 1.0,
    "fblade": None,
    "fl_ibpa": 0.0,
    "fmgrid": 0.2,
    "fracann": None,
    "free_turb": 0.05,
    "fsturb": 1.0,
    "ftype": 0,
    "itrans": 0,
    "itrans_j1_en": 0,
    "itrans_j1_frac": 0.0,
    "itrans_j1_st": 0,
    "itrans_j2_en": 0,
    "itrans_j2_frac": 0.0,
    "itrans_j2_st": 0,
    "itrans_k1_en": 0,
    "itrans_k1_frac": 0.0,
    "itrans_k1_st": 0,
    "itrans_k2_en": 0,
    "itrans_k2_frac": 0.0,
    "itrans_k2_st": 0,
    "jtrans": 0,
    "jtrans_i1_en": 0,
    "jtrans_i1_frac": 0.0,
    "jtrans_i1_st": 0,
    "jtrans_i2_en": 0,
    "jtrans_i2_frac": 0.0,
    "jtrans_i2_st": 0,
    "jtrans_k1_en": 0,
    "jtrans_k1_frac": 0.0,
    "jtrans_k1_st": 0,
    "jtrans_k2_en": 0,
    "jtrans_k2_frac": 0.0,
    "jtrans_k2_st": 0,
    "ktrans": 0,
    "ktrans_i1_en": 0,
    "ktrans_i1_frac": 0.0,
    "ktrans_i1_st": 0,
    "ktrans_i2_en": 0,
    "ktrans_i2_frac": 0.0,
    "ktrans_i2_st": 0,
    "ktrans_j1_en": 0,
    "ktrans_j1_frac": 0.0,
    "ktrans_j1_st": 0,
    "ktrans_j2_en": 0,
    "ktrans_j2_frac": 0.0,
    "ktrans_j2_st": 0,
    "nblade": None,
    "ndup_phaselag": 1,
    "nimixl": 0,
    "poisson_fmgrid": 0.0,
    "pstatin": 800000.0,
    "pstatout": 800000.0,
    "rpm": None,
    "rpmi1": None,
    "rpmi2": None,
    "rpmj1": None,
    "rpmj2": None,
    "rpmk1": None,
    "rpmk2": None,
    "sfin_mul": 1.0,
    "srough_i0": 0.0,
    "srough_i1": 0.0,
    "srough_j0": 0.0,
    "srough_j1": 0.0,
    "srough_k0": 0.0,
    "srough_k1": 0.0,
    "superfac": 0.0,
    "tstagin": 1200.0,
    "tstagout": 1200.0,
    "turb_intensity": 5.0,
    "vgridin": 50.0,
    "vgridout": 50.0,
    "xllim": None,
    "xllim_free": 0.1,
}

# Patch variables that must be present on inlet patches
# Where we should not set a default, use None
DEFAULT_INLET_PV = {
    "rfin": 0.5,
    "sfinlet": 0.0,
}


# Patch variables that must be present on outlet patches
# Where we should not set a default, use None
DEFAULT_OUTLET_PV = {
    "ipout": 3,
    "throttle_type": 0,
    "throttle_target": 0.0,
    "throttle_k0": 0.0,
    "throttle_k1": 0.0,
    "throttle_k2": 0.0,
    "fthrottle": 0.0,
    "pout": None,
    "pout_st": 0.0,
    "pout_en": 0.0,
    "pout_nchange": 0,
}

# Patch variables that must be present on outlet patches
# Where we should not set a default, use None
DEFAULT_POROUS_PV = {"porous_fac_loss": 1.0, "porous_rf": 0.9}


def _get_patch_kind(patch):
    """Choose TS3 patch kind integer based on patch class."""
    if isinstance(patch, turbigen.grid.InletPatch):
        return 0
    elif isinstance(patch, turbigen.grid.OutletPatch):
        if patch.force:
            return 19  # for 'outlet2d'
        else:
            return 1
    elif isinstance(patch, turbigen.grid.MixingPatch):
        return 2
    elif isinstance(patch, turbigen.grid.PorousPatch):
        return 17
    elif isinstance(patch, turbigen.grid.PeriodicPatch):
        if patch.cartesian:
            return 16
        else:
            return 5
    elif isinstance(patch, turbigen.grid.InviscidPatch):
        return 7
    elif isinstance(patch, turbigen.grid.ProbePatch):
        return 8
    elif isinstance(patch, turbigen.grid.NonMatchPatch):
        return 15
    elif isinstance(patch, turbigen.grid.CoolingPatch):
        return 6
    else:
        raise Exception(f"No TS3 patch kind defined for {patch}")


def _get_patch_sten(patch):
    """Patch attributes describing patch start and end indices."""
    # Convert negative indices to zero indexed
    ijk_lim = patch.ijk_limits.copy()
    nijk = np.reshape(patch.block.shape, (3, 1))
    ijk_lim[ijk_lim < 0] = (nijk + ijk_lim)[ijk_lim < 0]
    # Add one to end indices to make them exclusive
    ijk_lim[:, 1] += 1
    # Return as a dictionary of patch attributes
    keys = ["ist", "ien", "jst", "jen", "kst", "ken"]
    return dict(zip(keys, ijk_lim.flat))


def _get_patch_connectivity(patch, rtol=1e-4):
    """Patch attributes describing periodic or mixing connectivity."""

    # Deal with mixing patches
    if isinstance(patch, turbigen.grid.MixingPatch):
        pm = patch.match
        # Find ids for the connected block and patch
        nxbid = pm.block.grid.index(pm.block)
        not_rot = [
            p
            for p in pm.block.patches
            if not isinstance(p, turbigen.grid.RotatingPatch)
        ]
        nxpid = not_rot.index(pm)

        if not patch.slide:
            return {"idir": 0, "jdir": 0, "kdir": 0, "nxbid": nxbid, "nxpid": nxpid}
        else:
            return {
                "kind": 3,
                "idir": 0,
                "jdir": 0,
                "kdir": 0,
                "nface": 1,
                "nxbid": nxbid,
                "nxpid": nxpid,
                "slide_nxbid": nxbid,
                "slide_nxpid": nxpid,
            }

    is_periodic = isinstance(patch, turbigen.grid.PeriodicPatch)
    is_nonmatch = isinstance(patch, turbigen.grid.NonMatchPatch)
    is_porous = isinstance(patch, turbigen.grid.PorousPatch)
    if is_periodic or is_nonmatch or is_porous:
        pm = patch.match
        # Find ids for the connected block and patch
        nxbid = pm.block.grid.index(pm.block)
        not_rot = [
            p
            for p in pm.block.patches
            if not isinstance(p, turbigen.grid.RotatingPatch)
        ]
        nxpid = not_rot.index(pm)

        return {
            "idir": patch.idir,
            "jdir": patch.jdir,
            "kdir": patch.kdir,
            "nxbid": nxbid,
            "nxpid": nxpid,
        }

    else:
        # If the patch is not mixing and not periodic, set dummy values
        return {"idir": 0, "jdir": 0, "kdir": 0, "nxbid": 0, "nxpid": 0}


def _block_attributes(block, bid):
    """Make a dictionary of block attributes."""
    ba = DEFAULT_BA.copy()
    ba["bid"] = bid
    ba["ni"], ba["nj"], ba["nk"] = block.shape
    ba["np"] = len(
        [p for p in block.patches if not isinstance(p, turbigen.grid.RotatingPatch)]
    )
    return ba


def _patch_attributes(patch, bid, pid):
    """Make a dictionary of patch attributes."""
    pa = DEFAULT_PA.copy()
    pa["pid"] = pid
    pa["bid"] = bid
    pa["kind"] = _get_patch_kind(patch)
    pa.update(_get_patch_sten(patch))
    pa.update(_get_patch_connectivity(patch))
    return pa


def _patch_variables(patch, ts3_config):
    """Make a dictionary of patch attributes."""
    pv = {}
    if isinstance(patch, turbigen.grid.InletPatch):
        pv.update(DEFAULT_INLET_PV)
        pv["rfin"] = float(patch.rfin)
    elif isinstance(patch, turbigen.grid.PorousPatch):
        pv.update(DEFAULT_POROUS_PV)
        pv["porous_fac_loss"] = float(patch.porous_fac_loss)
    elif isinstance(patch, turbigen.grid.ProbePatch):
        pv["probe_append"] = 1
    elif isinstance(patch, turbigen.grid.OutletPatch):
        pv.update(DEFAULT_OUTLET_PV)
        pv["pout"] = float(patch.Pout)
        pv["ipout"] = ts3_config.ipout
        if patch.mdot_target:
            pv["throttle_type"] = 1
            pv["throttle_target"] = float(patch.mdot_target)
            # Turbostream uses 'PDI' control, not 'PID'
            pv["throttle_k0"], pv["throttle_k2"], pv["throttle_k1"] = patch.Kpid
        if patch.force:
            pv.pop("pout")
    elif isinstance(patch, turbigen.grid.CoolingPatch):
        patch.check()
        pv.update(
            {
                "cool_mass": patch.cool_mass,
                "cool_pstag": patch.cool_pstag,
                "cool_tstag": patch.cool_tstag,
                "cool_type": patch.cool_type,
                "cool_angle_def": patch.cool_angle_def,
                "cool_sangle": patch.cool_sangle,
                "cool_xangle": patch.cool_xangle,
                "cool_frac_area": 1.0,
                "cool_mach": patch.cool_mach,
            }
        )

    return pv


def _patch_properties(patch):
    """Make a dictionary of patch properties."""
    pp = {}
    if isinstance(patch, turbigen.grid.InletPatch):
        if patch.state.shape == ():
            x = np.ones_like(patch.get_cut().x)
            pp["pstag"] = patch.state.P * x
            # So that the TS3->TS4 converter works for real gases
            pp["tstag"] = patch.state.h / patch.state.cp * x
            pp["pitch"] = patch.Beta * x
            pp["yaw"] = patch.Alpha * x
            pp["fsturb_mul"] = x
        else:
            raise NotImplementedError("Inlet shape not supported")
    elif isinstance(patch, turbigen.grid.OutletPatch):
        if patch.force:
            pp["pout"] = patch.Pout * np.ones_like(patch.get_cut().x)
    return pp


def _write_variable(group, name, suffix, val):
    """Save a scalar to an hdf5 file."""
    key = name + suffix
    if isinstance(val, int):
        dtype = np.dtype("i4")
    else:
        dtype = np.dtype("f4")
    if val is None:
        raise Exception(f"Unspecified value for variable {name}")
    try:
        group.create_dataset(key, data=np.reshape(val, (1,)), dtype=dtype)
    except Exception:
        raise Exception(f"Could not write key={key}, val={val}")


def _write_property(group, name, suffix, val, flat=False):
    """Save an array to an hdf5 file."""
    key = name + suffix
    dtype = np.dtype("f4")
    if val is None:
        raise Exception(f"Unspecified value for variable {name}")
    if np.isnan(val).any():
        raise Exception(f"NaN in variable {name}")

    val_out = np.ones(val.shape, dtype=np.float32)
    val_out.flat = val.transpose().flat

    if flat:
        val_out = val_out.flatten()

    group.create_dataset(key, data=val_out, dtype=dtype)


def _get_wall_rpms(block):
    """Dictionary of block variables describing wall rotations."""
    keys = [
        "rpmi1",
        "rpmj1",
        "rpmk1",
        "rpmi2",
        "rpmj2",
        "rpmk2",
    ]
    vals = np.zeros((6,))
    for patch in block.rotating_patches:
        st_set = np.logical_and(
            patch.ijk_limits[:, 0] == 0, patch.ijk_limits[:, 1] == 0
        )
        en_set = np.logical_and(
            patch.ijk_limits[:, 0] == -1, patch.ijk_limits[:, 1] == -1
        )
        rpm_patch = patch.Omega * 30.0 / np.pi
        vals[:3][st_set] = rpm_patch
        vals[3:][en_set] = rpm_patch
    # assert block.rpm.ptp() == 0.0
    # vals *= block.rpm.mean()
    return dict(zip(keys, vals))


def _write_hdf5(grid, ts3_config, fname="input.hdf5"):
    """Using a given configuration, write grid object to an hdf5."""

    # Store old internal energy datum
    # Then set to zero as assumed by TS
    Tu0_old = grid[0].Tu0 + 0.0
    for b in grid:
        b.set_Tu0(0.0)
    for p in grid.inlet_patches:
        p.state.set_Tu0(0.0)

    # Determine reference radii for mixing length limit
    rref = np.empty((grid.nrow,))
    for irow, row_block in enumerate(grid.row_blocks):
        rref[irow] = np.mean([0.5 * (b.r.max() + b.r.min()) for b in row_block])

    input_file_path = os.path.join(ts3_config.workdir, fname)
    if not os.path.exists(ts3_config.workdir):
        raise Exception(f"Working directory {ts3_config.workdir} does not exist.")
    f = h5py.File(input_file_path, "w")

    # Get gas properties from the inlet
    So1 = grid.inlet_patches[0].state
    cp = float(So1.cp)
    ga = float(So1.gamma)
    mu = float(So1.mu)

    grid.inlet_patches[0].rfin = ts3_config.rfin

    # Grid attributes
    nb = len(grid)
    f.attrs["nb"] = nb
    f.attrs["ntb"] = 0

    # Application variables
    for name, val in ts3_config.application_variables(ga, cp, mu).items():
        _write_variable(f, name, "_av", val)

    procids = grid.partition(ts3_config.ntask)

    # Loop over blocks
    for ib in range(nb):
        key = f"block{ib}"
        block = grid[ib]

        # Make group to hold block data
        block_group = f.create_group(key)

        # Block attributes
        block_group.attrs.update(_block_attributes(block, ib))
        block_group.attrs["procid"] = procids[ib]

        # Block variables
        if ts3_config.Lref_xllim == "pitch":
            rref_block = rref[grid.row_index(block)]
        else:
            rref_block = np.nan
        for name, val in ts3_config.block_variables(
            block, rref_block, ts3_config.laminar
        ).items():
            _write_variable(block_group, name, "_bv", val)

        # Block properties
        _write_property(block_group, "x", "_bp", block.x)
        _write_property(block_group, "r", "_bp", block.r)
        _write_property(block_group, "rt", "_bp", block.rt)
        _write_property(block_group, "ro", "_bp", block.rho)
        _write_property(block_group, "rovx", "_bp", block.rhoVx)
        _write_property(block_group, "rovr", "_bp", block.rhoVr)
        _write_property(block_group, "rorvt", "_bp", block.rhorVt)
        _write_property(block_group, "roe", "_bp", block.rhoe)
        _write_property(block_group, "phi", "_bp", block.w)
        if np.isnan(block.mu_turb).all():
            turb_visc = ts3_config.tvr * np.ones_like(block.x) * mu
            _write_property(block_group, "trans_dyn_vis", "_bp", turb_visc)
        else:
            _write_property(block_group, "trans_dyn_vis", "_bp", block.mu_turb)

        # Loop over patches
        ip = 0
        for patch in block.patches:
            # Skip rotating patches - set wall rpms using block variables
            if isinstance(patch, turbigen.grid.RotatingPatch):
                logger.debug(f"Skipping patch {patch}, ip={ip}")
                continue
            else:
                logger.debug(f"Writing patch {patch}, ip={ip}")

            # Make group to hold patch data
            patch_key = f"patch{ip}"
            patch_group = block_group.create_group(patch_key)

            # Patch attributes
            pa = _patch_attributes(patch, ib, ip)
            if "slide_nxbid" in pa:
                pv_slide = {k: pa.pop(k) for k in ("slide_nxbid", "slide_nxpid")}
                if ts3_config.convert_sliding:
                    for name, val in pv_slide.items():
                        _write_variable(patch_group, name, "", val)
                else:
                    pa["kind"] = 2

            # Patch variables
            for name, val in _patch_variables(patch, ts3_config).items():
                _write_variable(patch_group, name, "_pv", val)

            # Patch properties
            for name, val in _patch_properties(patch).items():
                # Make boundary conditions unsteady if needed
                if isinstance(patch, turbigen.grid.InletPatch):
                    if force := patch.force and ts3_config.dts:
                        t = _get_time_vector(ts3_config)
                        nt = len(t)
                        F = np.ones((1, 1, 1, nt))
                        for n in patch.harmonics:
                            phase = np.pi * n**2 / 2.5338  # For minimum crest factor
                            F += patch.amplitude * np.sin(
                                2.0 * np.pi * ts3_config.frequency * n * t + phase
                            )
                        ga = patch.state.gamma

                        if force == "isentropic":
                            Po_Poav = F
                            To_Toav = Po_Poav ** ((ga - 1.0) / ga)
                        elif force == "entropic":
                            Po_Poav = np.ones_like(F)
                            To_Toav = F
                        else:
                            raise Exception(f"Unknown inlet forcing type {force}")

                        val = np.expand_dims(val, 3)
                        if name == "pstag":
                            val = val * Po_Poav
                        elif name == "tstag":
                            val = val * To_Toav
                        else:
                            val = np.tile(val, (1, 1, 1, nt))

                        pa["nt"] = nt

                if isinstance(patch, turbigen.grid.OutletPatch):
                    if patch.force and ts3_config.dts:
                        t = _get_time_vector(ts3_config)
                        F = 1.0 + patch.amplitude * np.sin(
                            2.0 * np.pi * ts3_config.frequency * t
                        ).reshape(1, 1, 1, -1)
                        val = np.expand_dims(val, 3) * F
                        pa["nt"] = len(t)

                _write_property(patch_group, name, "_pp", val, flat=True)

            patch_group.attrs.update(pa)

            ip += 1

        assert ip == block_group.attrs["np"]

    # Now check that patch and block ids are consistent
    logger.debug("Checking np")
    for ib in range(nb):
        blk = f[f"block{ib}"]
        nptch = blk.attrs["np"]
        logger.debug(f"bid={ib}, np={nptch}, len(patches)={len(grid[ib].patches)}")
        for ip in range(nptch):
            pch = blk[f"patch{ip}"]
            bid = pch.attrs["bid"]
            pid = pch.attrs["pid"]
            nxbid = pch.attrs["nxbid"]
            nxblk = f[f"block{nxbid}/"]
            nxpid = pch.attrs["nxpid"]
            logger.debug(
                f"ip={ip}, kind={pch.attrs['kind']}, "
                f"bid={bid}, pid={pid}, "
                f"nxbid={nxbid}, nxpid={nxpid}, nxnp={nxblk.attrs['np']}"
            )
            ni = blk.attrs["ni"]
            nj = blk.attrs["nj"]
            nk = blk.attrs["nk"]
            ist = pch.attrs["ist"]
            jst = pch.attrs["jst"]
            kst = pch.attrs["kst"]
            ien = pch.attrs["ien"]
            jen = pch.attrs["jen"]
            ken = pch.attrs["ken"]

            assert ist < ni
            assert ien < (ni + 1)
            assert jst < nj
            assert jen < (nj + 1)
            assert kst < nk
            assert ken < (nk + 1)

            assert nxbid < nb
            assert nxpid < nxblk.attrs["np"]

    f.close()

    # Reset the internal energy datum
    for b in grid:
        b.set_Tu0(Tu0_old)
    for p in grid.inlet_patches:
        p.state.set_Tu0(Tu0_old)


def _execute(ts3_config):
    """Using a given configuration, execute TS3."""

    # Store old working directory and change to this config's
    old_workdir = os.getcwd()
    os.chdir(ts3_config.workdir)

    if not os.path.exists(ts3_config.environment_script):
        raise Exception(
            f"""Could not locate TS3 env script {ts3_config.environment_script}
Are you on a HPC compute node gpu-q-* (not a login node)?
If you have recently been added to the turbostream user group, log out
and then back in to refresh your access permissions.
"""
        )

    # Open a subshell, source the environment and run the solver
    ngpu = ts3_config.ntask
    nnode = ts3_config.nnode
    npernode = ngpu // nnode
    logger.info(f"Using {ngpu} GPUs on {nnode} nodes, {npernode} per node.")
    if ngpu == 1:
        cmd_str = (
            f". {ts3_config.environment_script};"
            "turbostream input.hdf5 output 1 > log.txt"
        )
    else:
        cmd_str = (
            f". {ts3_config.environment_script};"
            f" mpirun -npernode {npernode} -np {ngpu} turbostream"
            f" input.hdf5 output {npernode} > log.txt"
        )

    # Remove old probe data
    probe_dat = glob("output_probe_*.dat")
    for fname in probe_dat:
        os.remove(fname)

    # Start the Turbostream process
    with subprocess.Popen(
        cmd_str, shell=True, stderr=subprocess.PIPE, preexec_fn=os.setsid
    ) as proc:
        # Until process has finished, check regularly for divergence
        try:
            while proc.poll() is None:
                timeout = 60
                start = timer()
                while (timer() - start) < timeout:
                    sleep(10)
                    if os.path.isfile("log.txt"):
                        break
                if not os.path.isfile("log.txt"):
                    raise Exception(
                        f"Timed out after {timeout}s waiting for TS3 log file to appear"
                    )
                if istep_nan := _check_nan("log.txt"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    raise ConvergenceError(
                        f"TS3 diverged at step {istep_nan}"
                    ) from None
        except KeyboardInterrupt:
            logger.iter("******")
            logger.iter("Caught interrupt, killing solver...")
            logger.iter("******")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            logger.iter("Killed solver.")

        proc.wait()

        # If we have an error code, prind debugging info
        if proc.returncode:
            raise Exception(
                f"""TS3 failed, exit code {proc.returncode}
COMMAND: {cmd_str}
STDERR: {proc.stderr.read().decode(sys.getfilesystemencoding()).strip()}

Are you on a HPC compute node, i.e. gpu-q-x not login-q-x?"""
            ) from None

    # Delete extraneous files
    for f in ("stopit", "output_avg.xdmf", "output.xdmf"):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

    # Remove empty hdf5 probes (we don't use them)
    probe_hdf5 = glob("output_probe_*.hdf5")
    for fname in probe_hdf5:
        os.remove(fname)

    os.chdir(old_workdir)


def _read_hdf5(grid, ts3_config):
    """Using a given configuration, load flow solution and insert into grid."""

    output_file_path = os.path.join(ts3_config.workdir, "output_avg.hdf5")
    output_inst_file_path = os.path.join(ts3_config.workdir, "output.hdf5")
    if not os.path.exists(output_file_path):
        raise Exception(f"""No Turbostream output file found at: {output_file_path}""")

    f = h5py.File(output_file_path, "r")
    fi = h5py.File(output_inst_file_path, "r")

    # Although the TS3 hdf5 reports the shape of the data as ni x nj x nk, this
    # is not correct and actually the underlying data is stored in nk x nj x ni order.
    # So we reshape and swap the axes back
    def _unflip(x):
        ni, nj, nk = x.shape
        return np.swapaxes(np.reshape(x, (nk, nj, ni)), 0, 2)

    # Loop over blocks
    nb = len(grid)
    for ib in range(nb):
        block = grid[ib]
        block_group = f[f"block{ib}"]

        # Pull some properties first
        rho = _unflip(block_group["ro_bp"])
        roe = _unflip(block_group["roe_bp"])
        trans_dyn_vis = _unflip(fi[f"block{ib}"]["trans_dyn_vis_bp"])

        # Check for divergence
        if not np.isfinite(rho).all():
            raise ConvergenceError("TS3 solution has NAN density.")
        if (rho < 0.0).any():
            raise ConvergenceError("TS3 solution has negative density.")
        if not np.isfinite(roe).all():
            raise ConvergenceError("TS3 solution has NAN total energy.")
        if (roe < 0.0).any():
            raise ConvergenceError("TS3 solution has negative total energy.")
        if not np.isfinite(trans_dyn_vis).all():
            raise ConvergenceError("TS3 solution has NAN turbulent viscosity.")

        # Set the velocities
        block.Vx = _unflip(block_group["rovx_bp"]) / rho
        block.Vr = _unflip(block_group["rovr_bp"]) / rho
        block.Vt = _unflip(block_group["rorvt_bp"]) / rho / block.r

        # Convert total energy to internal energy
        u = roe / rho - 0.5 * block.V**2.0

        # Make sure that u is positive
        if not (u > 0.0).all():
            raise ConvergenceError("TS3 solution has negative internal energy.")

        # Set the thermodynamic state
        Tu0_old = block.Tu0 + 0.0
        block.Tu0 = 0.0
        block.set_rho_u(rho, u)
        block.set_Tu0(Tu0_old)

        # Set turbulent viscosity
        block.mu_turb = trans_dyn_vis

        # Print yplus if requested
        if ts3_config.show_yplus:
            yplus = _unflip(block_group["yplus_bp"])
            # Remove not-wall nodes
            yplus = yplus[yplus > 0.0]
            logger.info(f"Block {ib} ({block.label}): mean yplus={yplus.mean():.1f}")

    f.close()
    fi.close()


def _run(grid, ts3_config):
    """Perform all steps on a grid and config."""

    _write_hdf5(grid, ts3_config)
    _execute(ts3_config)
    _read_hdf5(grid, ts3_config)


def run(grid, ts3_conf, machine, workdir):
    """Write, run, and read TS3 results for a grid object, specifying some settings.

    Parameters
    ----------
    grid
    ts3_conf
    machine
    """

    del machine

    ts3_conf.workdir = workdir

    # Check that the user is a member of the turbostream group
    try:
        ts_users = grp.getgrnam("turbostream").gr_mem
        current_user = getpass.getuser()
        if current_user not in ts_users:
            raise Exception(
                f"Current user {current_user} is not a member of the turbostream group"
            )
    except KeyError:
        raise Exception("Cannot locate turbostream - are you on the HPC?") from None

    # Load balancing
    try:
        ts3_conf.ntask = int(np.minimum(int(os.environ["SLURM_NTASKS"]), len(grid)))
        ts3_conf.nnode = int(os.environ["SLURM_NNODES"])
    except KeyError:
        ts3_conf.ntask = 1
        ts3_conf.nnode = 1
        logger.info(
            "Could not establish number of GPUs, assuming serial "
            "(are you on a compute node?)"
        )

    # Keep old log file if it exists (e.g. after a soft start)
    log_path = os.path.join(ts3_conf.workdir, "log.txt")
    if os.path.exists(log_path):
        os.rename(log_path, log_path.replace("log.txt", "log_old.txt"))

    _run(grid, ts3_conf)

    # Parse the log file
    istep_save_start = ts3_conf.application_variables(0.0, 0.0, 0.0)["nstep_save_start"]

    try:
        istep, mdot, ho, Po, resid = parse_log(log_path)
        state_log = grid.inlet_patches[0].state.copy().empty(shape=mdot.shape)
        state_log.set_Tu0(0.0)
        state_log.set_P_h(ho, Po)
        conv = ConvergenceHistory(istep, istep_save_start, resid, mdot, state_log)
    except Exception as e:
        logger.iter(f"Failed to parse log file {log_path}")
        logger.iter(f"Exception: {e}")
        conv = None

    return conv


re_nstep = re.compile(r"nstep\s*:\s*(\d*)$")
re_cp = re.compile(r"cp\s*:\s*(\d*\.\d*)$")
re_dts = re.compile(r"dts\s*:\s*(\d*)$")
re_ncycle = re.compile(r"ncycle\s*:\s*(\d*)$")
re_davg = re.compile(r"TOTAL DAVG \s*(\d*\.\d*)E([+-]\d*)")
re_nstep_cycle = re.compile(r"nstep_cycle\s*:\s*(\d*)$")
re_nstep_save_start = re.compile(r"nstep_save_start\s*:\s*(\d*)$")
re_mdot = re.compile(r"^INLET FLOW =\s*(-?\d*\.\d*)\s*OUTLET FLOW =\s*(-?\d*\.\d*)$")
re_Po = re.compile(
    r"^AVG INLET STAG P =\s*(-?\d*\.\d*)\s*AVG OUTLET STAG P =\s*(-?\d*\.\d*)$"
)
re_To = re.compile(
    r"^AVG INLET STAG T =\s*(-?\d*\.\d*)\s*AVG OUTLET STAG T =\s*(-?\d*\.\d*)$"
)
re_eta = re.compile(r"EFFICIENCY\s*=\s*(-?\d*.\d*)$")
re_nan = re.compile(r".*NAN.*")
re_current_step = re.compile(r"^O?U?T?E?R? ?STEP No\.\s*(\d*)", flags=re.MULTILINE)


def parse_log(fname):
    """Read residuals and boundary properties from log file.

    Parameters
    ----------
    fname: string
        File name of a Turbostream 3 log.

    Returns
    -------
    istep: (nlog) array
    mdot: (2, nlog) array
    ho: (2, nlog) array
    Po: (2, nlog) array
    resid: (nlog) array


    """

    logger.debug(f"Opening log file {fname}...")

    # Loop over lines in the file
    with open(fname, "r") as f:
        # Look for cp
        for line in f:
            match = re_cp.search(line)
            if match:
                cp = float(match.group(1))
                break

        # Look for number of steps
        for line in f:
            match = re_nstep.search(line)
            if match:
                nstep = int(match.group(1))
                break

        # Look for number of steps
        for line in f:
            match = re_dts.search(line)
            if match:
                dts = int(match.group(1))
                break

        # Preallocate
        step_now = 0
        dn = 1 if dts else 50
        nlog = nstep // dn
        istep = np.arange(nlog) * dn
        mdot = np.zeros((2, nlog))
        Po = np.zeros((2, nlog))
        To = np.zeros((2, nlog))
        resid = np.zeros((nlog,))

        for ilog in range(nlog):
            logger.debug(f"* Parsing istep={istep[ilog]}")

            # Look for residual
            if ilog > 0:
                for line in f:
                    if davg_match := re_davg.search(line):
                        logger.debug(f'Found: "{line.strip()}"')
                        sig = float(davg_match.group(1))
                        expon = int(davg_match.group(2))
                        resid[ilog] = sig * 10 ** (expon)
                        break
            else:
                resid[ilog] = np.nan

            try:
                if not dts:
                    # Loop over lines until we find mdot
                    logger.debug("Finding mass flow rate...")

                    for line in f:
                        if mdot_match := re_mdot.search(line):
                            logger.debug(f'Found: "{line.strip()}"')
                            mdot[:, ilog] = [float(m) for m in mdot_match.group(1, 2)]
                            break

                else:
                    for line in f:
                        if re_nstep.search(line):
                            logger.debug(f'Found: "{line.strip()}"')
                            break

                # Skip flow ratio
                _ = f.readline()

                # Stagnation pressures
                ln = f.readline()
                logger.debug(f'Reading Po from "{ln.strip()}"')
                match_Po = re_Po.search(ln)
                Po[:, ilog] = [float(m) for m in match_Po.group(1, 2)]

                # Stagnation temperatures
                ln = f.readline()
                logger.debug(f'Reading To from "{ln.strip()}"')
                match_To = re_To.search(ln)
                To[:, ilog] = [float(m) for m in match_To.group(1, 2)]

                # Skip power and effy
                _ = f.readline()
                _ = f.readline()
                _ = f.readline()

                # Next step number
                if ilog < nlog - 1:
                    logger.debug("Finding next step No...")
                    step_next = None
                    for line in f:
                        if step_match := re_current_step.search(line):
                            step_next = int(step_match.group(1))
                            if step_next > step_now:
                                logger.debug(f" Found next istep={step_next}")
                                step_now = step_next
                                break
                            else:
                                continue
                    if not step_next == istep[ilog + 1]:
                        raise Exception(f"Log step mismatch at {step_now}, {step_next}")

            except AttributeError:
                logger.debug("Failed to parse, breaking")
                break

    return istep, mdot, To * cp, Po, resid

    @property
    def nstep(self):
        return np.linspace(0, self._nlog - 1, self._nlog, dtype=int) * self._step_fac

    def __str__(self):
        return (
            f"TS3Log(mdot_drift={self.mdot_drift * 100.0:.2f}%,"
            f"mdot_err={self.mdot_err * 100:.2f}%,"
            f"eta_drift={self.eta_drift * 100:.2f}℅"
        )

    @property
    def eta_drift(self):
        eta_avg = self._eta[self.nstep >= self._nstep_save_start]
        # Consider instantaneous eta
        # drift = eta_avg - eta_avg[-1]
        # return drift[np.argmax(np.abs(drift))]
        # Split averaging period into two and check eta same for both
        n2 = len(eta_avg) // 2
        return eta_avg[:n2].mean() - eta_avg[n2:].mean()

    @property
    def mdot_err(self):
        err = (self._mdot[0] / self._mdot[1] - 1.0)[
            self.nstep >= self._nstep_save_start
        ]
        return err[np.argmax(np.abs(err))]

    @property
    def mdot_drift(self):
        drift = self._mdot[0] / self._mdot[0, -1] - 1.0
        err = drift[self.nstep >= self._nstep_save_start]
        return err[np.argmax(np.abs(err))]


def read_probe_dat_dir(dname):
    """Load all probe text files in a directory into one big array.

    This function will write out an npz into the directory containing the data
    for faster processing on subsequent runs. If the npz exists already but is
    older than any of the dat files, it will be overwritten; otherwise it will
    be loaded.

    Parameters
    ----------
    dname: string
        Directory name containing Turbostream 3 probe dat files.

    Returns
    -------
    data: (8, nprobe,  nstep) array
        First axis is x, r, rt, ro, rovx, rovr, rorvt, roe.
        Second axis is which probe.
        Third axis are time steps.

    """

    # Check for npz file and modification time
    npz_fname = os.path.join(dname, "probe_data.npz")
    if os.path.exists(npz_fname):
        npz_mtime = os.path.getmtime(npz_fname)
    else:
        npz_mtime = 0

    # Get all dat files and their modification times
    fnames = glob(os.path.join(dname, "*.dat"))
    dat_mtimes = [os.path.getmtime(f) for f in fnames]
    if not fnames:
        max_dat_mtime = 0
    else:
        max_dat_mtime = max(dat_mtimes)

    # Load the npz if it exists and is newer than all dat files
    if os.path.exists(npz_fname) and npz_mtime > max_dat_mtime:
        logger.info(f"Loading probe data from {npz_fname}")
        with np.load(npz_fname) as d:
            data = d["data"]
    # Otherwise load the dat files
    else:
        logger.info(f"Loading probe data from {dname}")
        # Load each dat and trim to the same length
        data_all = [read_probe_dat(f) for f in fnames]
        nstep = min([d.shape[1] for d in data_all])
        data = np.stack([d[:, :nstep] for d in data_all], axis=1)
        np.savez(npz_fname, data=data)

    # If the probes are more than 48 hours old, then the calculation has
    # finished and we can delete the raw dat files
    if (time.time() - max_dat_mtime) > 48 * 3600:
        for fname in fnames:
            logger.info(f"Removing {fname}")
            os.remove(fname)

    return data


def read_probe_dat(fname):
    """Load a probe text file into a big array.

    Note that this returns flattened arrays, i.e. the shape of the probe patch
    is lost

    Parameters
    ----------
    fname: string
        File name of a Turbostream 3 probe dat file.

    Returns:
    --------
    data: (8, nstep) array
        Columns are x, r, rt, ro, rovx, rovr, rorvt, roe.
        Rows are time steps.

    """
    return np.loadtxt(fname, skiprows=1).T


def read_probe_flow(dname, S, shape=()):
    """Load all probes from a directory into a flowfield object.

    Parameters
    ----------
    dname: string
        Directory name containing Turbostream 3 probe dat files.
    S:
        Reference flowfield object.
    shape: tuple
        Shape of the flowfield to be returned. Defaults to a point probe.

    """

    # Get the raw data and split into conserved variables
    x, r, rt, ro, rovx, rovr, rorvt, roe = read_probe_dat_dir(dname)
    nprobe = x.shape[1]

    # Reshape if requested
    Fshape = (nprobe,) + shape + (-1,)
    if shape:
        x = x.reshape(Fshape, order="F")
        r = r.reshape(Fshape, order="F")
        rt = rt.reshape(Fshape, order="F")
        ro = ro.reshape(Fshape, order="F")
        rovx = rovx.reshape(Fshape, order="F")
        rovr = rovr.reshape(Fshape, order="F")
        rorvt = rorvt.reshape(Fshape, order="F")
        roe = roe.reshape(Fshape, order="F")
    Fshape = x.shape

    # Make a perfect flowfield object
    F = turbigen.flowfield.PerfectFlowField(Fshape)
    F.Tu0 = 0.0
    F.cp = S.cp
    F.gamma = S.gamma
    F.mu = S.mu
    F.Omega = 0.0

    # Insert the coordinates and velocities
    F.xrt = np.stack((x, r, rt / r))
    F.Vxrt = np.stack((rovx, rovr, rorvt / r)) / ro

    # Insert the thermodynamic state
    u = roe / ro - 0.5 * F.V**2.0
    F.set_rho_u(ro, u)
    F.set_Tu0(S.Tu0)

    return F


def _check_nan(fname):
    """Return step number of divergence from TS3 log, or zero if no NANs found."""
    NBYTES = 2048
    with open(fname, "r") as f:
        while chunk := f.read(NBYTES):
            if re_nan.match(chunk):
                try:
                    return int(re_current_step.findall(chunk)[-1])
                except Exception:
                    return -1
    return 0


def _get_time_vector(ts3_config):
    freq = ts3_config.frequency
    nstep_cycle = ts3_config.nstep_cycle
    nt = nstep_cycle * ts3_config.ncycle
    it = np.arange(nt)
    dt = 1.0 / freq / nstep_cycle
    t = it * dt
    return t

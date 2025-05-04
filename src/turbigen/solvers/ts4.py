"""Functions to write, run, and read for the Turbostream 3 solver."""

import numpy as np
from copy import copy
from timeit import default_timer as timer
from dataclasses import dataclass
from pathlib import Path
import turbigen.grid
import turbigen.solvers.ts3
import h5py
import sys
import shutil
import os
import subprocess
from scipy.spatial import KDTree
from time import sleep
import turbigen.util
import turbigen.flowfield
import json
import re
from turbigen.solvers.base import BaseSolver, ConvergenceHistory

logger = turbigen.util.make_logger()


@dataclass
class ts4(BaseSolver):
    """Settings with default values for the TS4 solver."""

    _name = "ts4"

    workdir: Path = None
    """Working directory to run the simulation in."""

    environment_script: Path = Path(
        "/usr/local/software/turbostream/ts42111/bashrc_module_ts42111_a100"
    )
    """Setup environment shell script to be sourced before running."""

    cfl: float = 25.0
    """Courant--Friedrichs--Lewy number, reduce for more stability."""

    cfl_ramp_nstep: int = 500
    """Ramp the CFL number up over the first `cfl_ramp_nstep` steps."""

    cfl_ramp_st: int = 1.0
    """Starting value for CFL ramp."""

    custom_pipeline: str = ""
    """Specify a custom pipeline to convert Turbostream 3 to 4 input file. Should run
    using pvpython and take two command-line arguments like `pvpython
    custom_pipeline.py input_ts3.hdf5 input_ts4`"""

    implicit_scheme: int = 1
    """Whether to use implicit time-marching scheme."""

    nstep: int = 5000
    """Number of time steps for the calculation."""

    nstep_avg: int = 1000
    """Number of time steps to average over."""

    nstep_ts3: int = 0
    """Number of steps for a Turbostream 3 initial guess"""

    nstep_soft: int = 5000
    """Number of time steps for soft start."""

    spf_probe: list = None
    """List of span fractions to place probes."""

    point_probe: list = None
    """Specify point probes."""

    logical_probe: list = None
    """Specify logical probes."""

    tables_path: str = ""
    """Path to gas tables for real working fluids."""

    body_force_template: str = ""
    """Path to a body_force.ofp definition file template."""

    body_force_params = {}
    """Parameters to be added to body force template."""

    viscous_model: int = 2
    """Turbulence model, 0 for inviscid, 1 for laminar, 2 for Spalart-Allmaras."""


    kappa2: float = 1.0
    kappa4: float = 1.0 / 128.0

    outlet_tag: str = "Outlet"
    """Identifier string for the outlet patch."""

    area_avg_pout: bool = True

    pout_fac_ramp_nstep: int = 0
    mixing_rf_ramp_nstep: int = 0

    inlet_relax_fac: float = 0.5
    nstep_save_start_probe_1d: int = 0
    nstep_save_probe_1d: int = 100
    nstep_save_start_probe_2d: int = 0
    nstep_save_probe_2d: int = 100
    cfl_turb_fac: float = 0.5

    precon: int = 0
    precon_fac_ramp_nstep: int = 100
    precon_fac_ramp_st: float = 0.1
    precon_fac_ramp_en: float = 1.0
    precon_sigma_pgr: float = 3.0

    if_dts: int = 0
    frequency: float = 0.0
    ncycle: int = 0
    nstep_cycle: int = 72

    halo_implementation: int = 1
    ncycle_avg: int = 1

    interpolation_update: int = 1  # 1 to freeze interpolating plane posn

    def to_dict(self):
        conf = super().to_dict()
        conf.pop("workdir")
        conf["environment_script"] = str(self.environment_script)
        return conf

    def robust(self):
        """Explicit with a slow CFL ramp."""
        return self.replace(
            implicit_scheme = 0,
            cfl = 3.5,
            cfl_ramp_st = 0.1,
            nstep = self.nstep_soft,
            cfl_ramp_nstep = self.nstep,
            nstep_avg = 50,
        )

    def restart(self):
        """Restart from a previous solution."""
        return self.replace(
            cfl_ramp_nstep = 0,
            precon_fac_ramp_nstep = 0,
            pout_fac_ramp_nstep = 0,
            mixing_rf_ramp_nstep = 0,
        )

    def run(self, grid, machine, workdir):
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        return run(grid, self, machine, workdir)

    @property
    def config_path(self):
        return os.path.join(self.workdir, "config.ofp")

    def to_ofp(self):
        v = DEFAULT_CONFIG.copy()
        for k in v:
            if hasattr(self, k):
                v[k] = getattr(self, k)

        # Raise error if averaging not OK
        nstep = v["nstep"]
        istep_avg_start = v["istep_avg_start"] = v["nstep"] - self.nstep_avg
        if istep_avg_start >= nstep and nstep > 0:
            raise Exception(f"istep_avg_start={istep_avg_start} is > nstep={nstep}")

        if v["cfl_ramp"]:
            v["cfl_ramp_en"] = v["cfl"]

        if v.get("pout_fac_ramp_nstep"):
            v["pout_fac_ramp"] = 1

        if v["precon"]:
            v["precon_fac_ramp"] = 1

        # Disable ramping in unsteady runs
        if self.if_dts:
            v["cfl_ramp"] = 0
            dnstep_avg = self.ncycle_avg * self.nstep_cycle
            nstep_tot = self.ncycle * self.nstep_cycle
            v["istep_avg_start"] = nstep_tot - dnstep_avg

        return v


re_nan = re.compile(r"^detected NAN$", flags=re.MULTILINE)
re_current_step = re.compile(r"^RK LOOP NO:\s*(\d*)$", flags=re.MULTILINE)


def parse_log(fname, nstep, Nb):
    """Parse a log file to get convergence history."""

    # Preallocate
    resid = np.full((nstep,), np.nan)
    inlet = np.full((nstep, 3), np.nan)
    outlet = np.full((nstep, 3), np.nan)

    # Loop over lines in file
    istep = 0
    with open(fname, "r") as f:
        for line in f:
            if line.startswith("ROVX DELTA:"):
                resid[istep] = float(line.split()[-1])
            elif line.startswith("INLET: "):
                inlet[istep, :] = [float(x) for x in line.split()[1:]]
            elif line.startswith("OUTLET: "):
                outlet[istep, :] = [float(x) for x in line.split()[1:]]
                istep += 1

    # Multiply by Nb to get mass flow for entire annulus
    inlet[:, 0] *= Nb[0]
    outlet[:, 0] *= Nb[1]

    # Extract the separate variables
    istep = np.arange(nstep)
    mdot, ho, Po = np.stack((inlet, outlet)).transpose(2, 0, 1)

    return istep, mdot, ho, Po, resid


def _check_nan(fname):
    """Return step number of divergence from TS4 log, or zero if no NANs found."""
    NBYTES = 2048
    with open(fname, "r") as f:
        while chunk := f.read(NBYTES):
            if re_nan.search(chunk):
                try:
                    return int(re_current_step.findall(chunk)[-1])
                except Exception:
                    return -1
    return 0


def _write_wall_distance(g, fname):
    """Given a TS4 input file, use wall nodes from this grid to get wall distance."""

    # Read the unstructured data from TS4 hdf5
    f = h5py.File(fname, "r+")

    # Extract coordinates
    x = np.copy(f["node_array_x"])
    y = np.copy(f["node_array_y"])
    z = np.copy(f["node_array_z"])

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2.0 + z**2.0)
    t = -np.arctan2(z, y)
    xrrt = np.stack((x, r, r * t), axis=1)

    # Initialise a kdtree of wall points
    kdtree = KDTree(g.get_wall_nodes().T)

    # Query the wall distances
    wdist, _ = kdtree.query(xrrt, workers=32)

    # Insert into grid
    f["node_array_wdist"][:] = wdist

    f.close()


def _write_ofp(fname, var):
    """Write a dictionary of configuration variables in ofp format."""
    with open(fname, "w") as f:
        for k, v in var.items():
            if isinstance(v, list):
                f.write(k + " = [" + ", ".join([str(vi) for vi in v]) + "]\n")
            else:
                f.write(k + " = " + str(v) + "\n")


def _read_ofp(fname):
    """Read a dictionary of configuration variables in ofp format."""
    var = {}
    with open(fname, "w") as f:
        for line in f:
            k, v = line.split()
            if "." in v or "e" in v:
                var[k] = float(v)
            else:
                var[k] = int(v)
    return var


DEFAULT_CONFIG = {
    "cfl": 50.0,
    "cfl_ramp": 1,
    "cfl_ramp_nstep": 500,
    "cfl_ramp_st": 1.0,
    "cfl_turb_fac": 0.25,
    "fluid_model": 0,
    "implicit_scheme": 1,
    "inlet_relax_fac": 0.5,
    "interpolation_extend_fac": 0.5,
    "interpolation_time_step_fac": 0.1,
    "istep_avg_start": 4000,
    "halo_implementation": 1,
    "kappa2": 1.0,
    "kappa4": 1.0 / 128.0,
    "mixing_alpha": 1.0,
    "mixing_bc_method": 2,
    "mixing_distribution_kind": 1,
    "mixing_distribution_spacing": 5e-4,
    "mixing_method": 2,
    "mixing_nstation": 200,
    "mixing_rf": 0.25,
    "mixing_rf_ramp": 1,
    "mixing_rf_ramp_nstep": 500,
    "mixing_rf_ramp_st": 0.05,
    "mixing_rf_ramp_en": 0.25,
    "node_ordering_option": 2,
    "nstep": 5000,
    "prandtl_turbulent": 0.9,
    "sa_helicity_model": 0,
    "sa_lim_neg": 1,
    "time_step_mod_fac_nsmooth": 0,
    "time_step_mod_fac_smooth_frac": 0.5,
    "update_mixing_nstep": 5,
    "viscous_model": 2,
    "precon": 0,
    "precon_dissipation": 0,
    "precon_sigma_pgr": 3.0,
    "precon_beta": 3.0,
    "precon_mach_ref": 0.1,
    "precon_mach_lim": 0.1,
    "precon_fac_ramp": 0,
    "precon_fac_ramp_nstep": 100,
    "precon_fac_ramp_st": 0.1,
    "precon_fac_ramp_en": 1.0,
    "pout_fac_ramp_nstep": 0,
    "pout_fac_ramp": 0,
    "pout_fac_ramp_st": 0.8,
    "pout_fac_ramp_en": 1.0,
    "use_gpu_direct": 1,
    "if_dts": 0,
    "ncycle": 0,
    "nstep_cycle": 0,
    "frequency": 0.0,
    "nstep_save": 1e9,
}


def _read_flow(grid, fname, fname_avg):
    start = timer()

    # Read the unstructured data from TS4 hdf5
    f = h5py.File(fname, "r")
    fa = h5py.File(fname_avg, "r")
    x = np.copy(f["node_array_x"])
    y = np.copy(f["node_array_y"])
    z = np.copy(f["node_array_z"])
    ro = np.copy(fa["node_array_ro"])
    rovx = np.copy(fa["node_array_rovx"])
    rovy = np.copy(fa["node_array_rovy"])
    rovz = np.copy(fa["node_array_rovz"])
    roe = np.copy(fa["node_array_roe"])
    turb0 = np.copy(fa["node_array_turb0"])
    f.close()
    fa.close()

    end = timer()
    logger.debug(f"Read flow in {end - start} s")

    # Divide out density
    vx = rovx / ro
    vy = rovy / ro
    vz = rovz / ro
    vsq = vx**2.0 + vy**2.0 + vz**2.0

    # Convert Cartesian coordinates to polar
    r = np.sqrt(y**2.0 + z**2.0)
    t = np.arctan2(-z, y)
    rt = r * t

    # Convert Cartesian velocities to polar
    vt = -(vy * np.sin(t) + vz * np.cos(t))
    vr = vy * np.cos(t) - vz * np.sin(t)

    # Calculate internal energy
    u = roe / ro - 0.5 * vsq

    # Build kdtree for nodes in the TS4 hdf5
    kdtree = KDTree(np.column_stack((x, r, rt)))

    # Now loop over blocks in the grid
    for block in grid:
        # Extract unstructure coordinates for this block
        xrrtb = block.to_unstructured().xrrt.T

        # Get nearest neighbours from the TS4 grid
        _, ind_pts = kdtree.query(xrrtb, workers=-1)

        # Check these really are the points we want
        assert np.allclose(block.x, x[ind_pts].reshape(block.shape))
        assert np.allclose(block.r, r[ind_pts].reshape(block.shape))
        assert np.allclose(block.t, t[ind_pts].reshape(block.shape))

        # Assign the velocities
        block.Vx = vx[ind_pts].reshape(block.shape)
        block.Vr = vr[ind_pts].reshape(block.shape)
        block.Vt = vt[ind_pts].reshape(block.shape)

        # Set thermodynamic properties
        Tu0_old = block.Tu0 + 0.0
        block.Tu0 = 0.0
        block.set_rho_u(
            ro[ind_pts].reshape(block.shape), u[ind_pts].reshape(block.shape)
        )
        block.set_Tu0(Tu0_old)

        # Assign turbulent viscosity
        block.mu_turb = turb0[ind_pts].reshape(block.shape)


def _write_throttle(ts4_conf, grid, fname):
    # Get indices for outlet bcells
    with h5py.File(fname, "r") as f:
        bcell_names = list(f["bcell_names"].attrs["names"])
        bcell_ind = [i for i, b in enumerate(bcell_names) if ts4_conf.outlet_tag == b]
        if not bcell_ind:
            raise Exception(
                f"Could not find throttle outlet tag {ts4_conf.outlet_tag}, "
                f"should be one of {bcell_names}"
            )

    # Loop over outlet patches
    mass_target = []
    throttle_kind = []
    throttle_k0 = []
    mdot_flag = False
    for patch in grid.outlet_patches:
        # TS4 only has integral control. Multiply by Nb because targets *patch*
        # not *annulus* mass flow. Have to reduce a bit because TS4 converges
        # in fewer steps than TS3.
        Nb = patch.block.Nb

        fac_adjust = 0.5 if ts4_conf.implicit_scheme else 1.0

        if patch.Kpid is not None:
            mdot_flag = True
            mass_target.append(patch.mdot_target / float(Nb))
            throttle_kind.append(0)
            throttle_k0.append(patch.Kpid[1] * fac_adjust)

        else:
            mass_target.append(0.0)
            throttle_kind.append(1)
            throttle_k0.append(0.0)

    # Assemble throttle data
    throt_ofp = {
        "throttle_tag_list": bcell_ind,
        "throttle_target_list": mass_target,
        "throttle_k0_list": throttle_k0,
        "throttle_correction_initial_list": [0.0 for _ in bcell_ind],
        "throttle_kind_list": throttle_kind,
    }

    throt_file_path = os.path.join(ts4_conf.workdir, "throttle_config.ofp")
    if os.path.exists(throt_file_path):
        os.remove(throt_file_path)

    if mdot_flag or ts4_conf.area_avg_pout:
        _write_ofp(throt_file_path, throt_ofp)

    return mdot_flag


def _write_probes_ofp(ts4_conf):
    probe_ofp = os.path.join(ts4_conf.workdir, "probes.ofp")
    with open(probe_ofp, "w") as f:
        # Initialise probe list
        f.write(
            """
import ts.process.probe.probe_definition
probe_list = []

"""
        )


def _check_probes(ts4_conf):
    probe_ofp = os.path.join(ts4_conf.workdir, "probes.ofp")
    with open(probe_ofp, "r") as f:
        probe_str = f.read()

    # Check syntax
    try:
        compile(probe_str, "dummy.py", "exec")
    except Exception as e:
        raise Exception("Error in probes file!") from e


def _write_point_probe(ts4_conf, xyzp, dom, label):
    probe_ofp = os.path.join(ts4_conf.workdir, "probes.ofp")

    with open(probe_ofp, "a") as f:
        # Initialise probe list
        f.write(
            f"""
p = ts.process.probe.probe_definition.ProbeDefinition()
p.kind = "point"
p.x = {xyzp[0].tolist()}
p.y = {xyzp[1].tolist()}
p.z = {xyzp[2].tolist()}
p.idomain = {dom}
p.absolute_frame = True
p.fname_root = "point_probe_{label}"
p.write_2d = True
p.nstep_save_start_1d = {ts4_conf.nstep_save_start_probe_1d}
p.nstep_save_1d = {ts4_conf.nstep_save_probe_1d}
p.nstep_save_start_2d = {ts4_conf.nstep_save_start_probe_1d}
p.nstep_save_2d = {ts4_conf.nstep_save_probe_1d}
p.time_average = False  # Must be False in a steady calc?
probe_list.append(p)

"""
        )


def _write_logical_probe(ts4_conf, tag, label):
    probe_ofp = os.path.join(ts4_conf.workdir, "probes.ofp")

    istep_save_start = ts4_conf.nstep - ts4_conf.nstep_avg

    with open(probe_ofp, "a") as f:
        # Initialise probe list
        f.write(
            f"""
p = ts.process.probe.probe_definition.ProbeDefinition()
p.kind = "logical"
p.val = "{tag}"
p.fname_root = "logical_probe_{label}"
p.write_2d = False
p.nstep_save_start_1d = {istep_save_start}
p.nstep_save_1d = {ts4_conf.nstep_save_probe_1d}
p.nstep_save_start_2d = 99999
p.nstep_save_2d = 0
probe_list.append(p)

"""
        )


def _write_xr_probe(machine, ts4_conf, spf):
    probe_ofp = os.path.join(ts4_conf.workdir, "probes.ofp")

    # Evaluate the xr curve, extrapolate a little
    eps = 0.01
    mq = np.linspace(-eps, machine.ann.npts - 1 + eps, 100)
    xr = machine.ann.evaluate_xr(mq, spf)

    logger.debug(f"min xr={xr.min(axis=1)}, max xr={xr.max(axis=1)}")

    istep_save_start = ts4_conf.nstep - ts4_conf.nstep_avg

    # Fill values into template
    pstr = []
    for irow in range(machine.ann.nrow):
        pstr.append(
            f"""
p = ts.process.probe.probe_definition.ProbeDefinition()
p.kind = "xr"
p.s1 = {xr[0].tolist()}
p.s2 = {xr[1].tolist()}
p.idomain = {irow}
p.fname_root = "xr_probe_row_{irow}_spf_{str(spf).replace(".", "")}"
p.write_2d = True
p.nstep_save_start_1d = {istep_save_start}
p.nstep_save_1d = {ts4_conf.nstep_save_probe_1d}
p.nstep_save_start_2d = {istep_save_start}
p.nstep_save_2d = {ts4_conf.nstep_save_probe_2d}
probe_list.append(p)

"""
        )

    with open(probe_ofp, "a") as f:
        f.writelines(pstr)


def run(grid, ts4_conf, machine, workdir):
    """Write, run, and read TS4 results for a grid object, specifying some settings."""

    ts4_conf.workdir = workdir

    input_file_path = os.path.join(ts4_conf.workdir, "input_ts4.hdf5")
    output_file_path = os.path.join(ts4_conf.workdir, "output_ts4.hdf5")
    output_avg_file_path = os.path.join(ts4_conf.workdir, "output_ts4_avg.hdf5")
    soln_exists = os.path.exists(output_file_path)
    if ts4_conf.skip:
        if soln_exists:
            logger.info("Skipping running, loading previous solution.")
            _read_flow(grid, output_file_path, output_avg_file_path)
            # Write out for debugging
            ts3_conf = turbigen.solvers.ts3.ts3(workdir=ts4_conf.workdir)
            turbigen.solvers.ts3._write_hdf5(grid, ts3_conf, fname="output_ts3.hdf5")
        else:
            logger.info("Skipping running, keeping initial guess.")
        return

    ofp = ts4_conf.to_ofp()

    # Invert the sign of inlet pitch angle
    inpatch = grid.inlet_patches[0]
    inpatch.Beta *= -1.0

    # Extract nominal fluid properties
    ofp["gamma"] = inpatch.state.gamma
    ofp["cp"] = inpatch.state.cp
    ofp["viscosity"] = inpatch.state.mu
    ofp["prandtl_laminar"] = inpatch.state.Pr
    ofp["prandtl_turbulent"] = inpatch.state.Pr * 1.25
    ofp["fluid_model"] = 0 if isinstance(grid[0], turbigen.grid.PerfectBlock) else 1

    if ofp["fluid_model"]:
        if not ts4_conf.tables_path:
            raise Exception("No `tables_path` specified for real gas fluid")
        dest_tables = os.path.join(ts4_conf.workdir, "fluid_model_tables.npz")
        shutil.copyfile(ts4_conf.tables_path, dest_tables)

    # Put body force definition into working directory, with substitutions
    if ts4_conf.body_force_template:
        with open(ts4_conf.body_force_template, "r") as f:
            body_force_str = f.read()

        for k, v in ts4_conf.body_force_params.items():
            body_force_str = re.sub(
                rf"{k}\s*=.*$", f"{k} = {v}", body_force_str, flags=re.MULTILINE
            )

        # Check syntax
        try:
            compile(body_force_str, "dummy.py", "exec")
        except Exception as e:
            raise Exception("Error in body force template!") from e

        dest_body_force = os.path.join(ts4_conf.workdir, "body_force_config.ofp")
        with open(dest_body_force, "w") as f:
            f.write(body_force_str)

        ofp["body_force_mode"] = 1

    _write_ofp(ts4_conf.config_path, ofp)

    # Check inlet patches for forcing
    for patch in grid.inlet_patches:
        if patch.force:
            # Evaluate the forcing function for the current time discretisation
            fac_Po, fac_ho = patch.get_unsteady_multipliers(
                ts4_conf.frequency,
                ts4_conf.nstep_cycle,
                ts4_conf.ncycle,
            )

            # Save to a numpy file
            np.savez(
                os.path.join(ts4_conf.workdir, "forcing.npz"),
                fac_Po=fac_Po,
                fac_ho=fac_ho,
            )

            # Write out a forcing file
            bcond_path = os.path.join(ts4_conf.workdir, "bcond_unsteady_config.ofp")
            with open(bcond_path, "w") as f:
                f.write(
                    """
import numpy

hstag_ramp = {}
pstag_ramp = {}
pstat_ramp = {}

data = numpy.load("forcing.npz")

# inlet
hstag_ramp[0] = data["fac_ho"]
pstag_ramp[0] = data["fac_Po"]

# outlet
pstat_ramp[1] = numpy.ones_like(pstag_ramp[0])

"""
                )
            logger.info("Wrote out unsteady boundary conditions.")

    ts3_conf = turbigen.solvers.ts3.ts3(dts=0, workdir=ts4_conf.workdir).robust()

    # Get number of GPUs from environment var
    ngpu = int(os.environ.get("SLURM_NTASKS", 1))
    nnode = int(os.environ.get("SLURM_NNODES", 1))
    npernode = ngpu // nnode
    ts3_conf.ntask = ngpu
    ts3_conf.nnode = nnode

    if ts4_conf.nstep_ts3:
        logger.info("Running TS3 initial guess...")
        ts3_conf.nstep = ts4_conf.nstep_ts3
        turbigen.solvers.ts3._run(grid, ts3_conf)
        grid.update_outlet()

    turbigen.solvers.ts3._write_hdf5(grid, ts3_conf)

    if pipeline := ts4_conf.custom_pipeline:
        # Write a dictionary of information that *might* be needed by custom pipelines
        # In particular, TS4 cannot work out Nblade for wall distance filter by iself
        pipeline_params = {
            "Nb_row": [int(blks[0].Nb) for blks in grid.row_blocks],
            "cp": inpatch.state.cp,
            "ho": inpatch.state.h,
        }
        pipeline_params_path = os.path.join(ts4_conf.workdir, "pipeline_params.json")
        with open(pipeline_params_path, "w") as f:
            json.dump(pipeline_params, f)

        convert_cmd = (
            f"pvpython {os.path.abspath(pipeline)} input.hdf5 input_ts4 > convert.log"
        )

        logger.iter(f"Converting TS3->TS4 using custom pipeline {pipeline}...")

        # When runing headless, paraview seems to segfault on exit, despite the
        # pipeline completing successfully. So we ignore errors when executing
        # pvpython. We could verify success by checking for existence of TS4
        # input file instead.
        check = False

    else:
        convert_script = os.path.join(
            os.path.dirname(__file__), "convert_ts3_to_ts4_native.py"
        )
        convert_cmd = f"{convert_script} input.hdf5 input_ts4 > convert.log"
        check = True
        logger.info("Converting TS3->TS4...")

    cmd_str = (
        f"source {ts4_conf.environment_script} 2> /dev/null;"
        f"cd {ts4_conf.workdir}; {convert_cmd}"
    )
    try:
        subprocess.run(cmd_str, shell=True, check=check, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise Exception(
            f"""Running TS3->TS4 conversion failed, exit code {e.returncode}
COMMAND: {cmd_str}
STDERR: {e.stderr.decode(sys.getfilesystemencoding()).strip()}
"""
        ) from None

    input_file_path = os.path.join(ts4_conf.workdir, "input_ts4.hdf5")

    # Write throttle config file
    _write_throttle(ts4_conf, grid, input_file_path)

    # Assume that the custom pipeline will set wall distances for us
    if not ts4_conf.custom_pipeline:
        _write_wall_distance(grid, input_file_path)

    # Write probe config file
    # if ts4_conf.spf_probe or ts4_conf.point_probe:
    _write_probes_ofp(ts4_conf)

    # Write point probes
    if ts4_conf.point_probe:
        # Loop over a list of probes
        for pp_conf in ts4_conf.point_probe:
            xyz = np.array(pp_conf["xyz"]).T
            idomain = int(pp_conf["domain"])
            # Get dimensions of input data
            assert xyz.ndim == 2
            assert xyz.shape[0] == 3
            _write_point_probe(ts4_conf, xyz, idomain, pp_conf["label"])
    else:
        # Check for point probes in the grid
        xyzp = []
        idomain = 0
        label = "grid"
        for patch in grid.probe_patches:
            if patch.is_point:
                xyzp.append(patch.get_cut().xyz.squeeze())
        if xyzp:
            xyz = np.stack(xyzp).T
            xyz = xyz[
                (0, 2, 1),
            ]  # Swap y and z for TS4 coord system
            _write_point_probe(ts4_conf, xyz, idomain, label)

    # Write span fraction probes
    if ts4_conf.spf_probe:
        for spf in ts4_conf.spf_probe:
            _write_xr_probe(machine, ts4_conf, spf)

    # Write logical probes
    if ts4_conf.logical_probe:
        # Loop over a list of probes
        for pp_conf in ts4_conf.logical_probe:
            _write_logical_probe(ts4_conf, **pp_conf)

    logger.info("Checking probes...")
    _check_probes(ts4_conf)

    logger.info(f"Using {ngpu} GPUs on {nnode} nodes, {npernode} per node.")
    logger.info("Running TS4...")
    cmd_str = (
        f"source {ts4_conf.environment_script} 2> /dev/null;"
        f"cd {ts4_conf.workdir};"
        f"mpirun -np {ngpu} python $TSHOME/$TSDIR/bin/turbostream.py"
        " config.ofp input_ts4.hdf5 input_ts4.hdf5 input_ts4.hdf5 output_ts4"
        f" {npernode} --fname_out_procid=procids.hdf5 > log_ts4.txt 2> err.txt"
    )
    sleep(1)

    try:
        subprocess.run(cmd_str, shell=True, check=check, stderr=subprocess.PIPE)

    # TODO catch keyboard interrupt here
    except subprocess.CalledProcessError as e:
        raise Exception(
            f"""Running TS4 failed, exit code {e.returncode}
COMMAND: {cmd_str}
STDERR: {e.stderr.decode(sys.getfilesystemencoding()).strip()}
"""
        ) from None

    log_path = os.path.join(ts4_conf.workdir, "log_ts4.txt")
    if istep_nan := _check_nan(log_path):
        raise Exception(f"TS4 diverged at step {istep_nan}")

    # output_file_path = os.path.join(ts4_conf.workdir, "output_ts4.hdf5")
    _read_flow(grid, output_file_path, output_avg_file_path)

    # Restore the sign of inlet pitch angle
    inpatch.Beta *= -1.0

    # Write out the ts4 flowfield to ts3 for debugging
    turbigen.solvers.ts3._write_hdf5(grid, ts3_conf, fname="output_ts3.hdf5")

    # Parse convergence history
    Nb = [grid.inlet_patches[0].block.Nb, grid.outlet_patches[0].block.Nb]
    try:
        istep, mdot, ho, Po, resid = parse_log(log_path, ts4_conf.nstep, Nb)

        # State to hold inlet and outlet flow properties
        state_log = grid.inlet_patches[0].state.copy().empty(shape=mdot.shape)
        state_log.set_Tu0(0.0)
        state_log.set_P_h(ho, Po)
        istep_save_start = ts4_conf.nstep - ts4_conf.nstep_avg
        conv = ConvergenceHistory(istep, istep_save_start, resid, mdot, state_log)
    except Exception:
        conv = None

    return conv


def read_probe_flow(fname, state):
    """Load unsteady flow field data from TS4 point probes."""

    # Read the unstructured data from TS4 hdf5
    with h5py.File(fname, "r") as f:
        # Get a sorted list of time steps
        steps = [int(i) for i in f["x"].keys()]
        steps.sort()

        # Function to pull out data for each property
        def _f(p):
            return np.array([f[p][str(i)] for i in steps])

        x = _f("x")
        y = _f("y")
        z = _f("z")
        ro = _f("ro")
        e = _f("roe") / ro
        Vx = _f("rovx") / ro
        Vy = _f("rovy") / ro
        Vz = _f("rovz") / ro

    # Convert to polars
    r = np.sqrt(y**2.0 + z**2.0)
    t = np.arctan2(-z, y)
    Vt = -(Vy * np.sin(t) + Vz * np.cos(t))
    Vr = Vy * np.cos(t) - Vz * np.sin(t)

    # Assemble vector components
    xrt = np.stack((x, r, t))
    Vxrt = np.stack((Vx, Vr, Vt))
    Vsq = np.sum(Vxrt**2, axis=0)

    # Internal energy
    u = e - 0.5 * Vsq

    # Make flow field data structure
    Omega = np.zeros_like(r)
    F = turbigen.flowfield.PerfectFlowField(Omega.shape)
    F.Tu0 = 0.0
    F.xrt = xrt
    F.Vxrt = Vxrt
    F.cp = state.cp
    F.gamma = state.gamma
    F.mu = state.mu
    F.set_rho_u(ro, u)
    F = F.transpose()

    return F

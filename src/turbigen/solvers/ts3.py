import logging
from time import sleep
from dataclasses import dataclass, fields
from timeit import default_timer as timer
from glob import glob
import numpy as np
import ember.ts3
from turbigen.exceptions import ConvergenceError
import subprocess
import os
from pathlib import Path
import signal
import sys
import re
import grp
import getpass
from turbigen.solvers.base import BaseSolver, ConvergenceHistory

logger = logging.getLogger("turbigen")

# Typed config fields that map to TS3 application/block variables of the same
# name, forwarded via writer.set_av / writer.set_bv. Used both to forward the
# values and to detect overlap with the raw av/bv override dicts.
_AV_FIELDS = frozenset(
    {
        "cfl",
        "dampin",
        "facsecin",
        "ilos",
        "nchange",
        "viscosity_law",
        "nstep",
        "rfmix",
        "sfin",
    }
)
_BV_FIELDS = frozenset({"fmgrid"})


@dataclass
class ts3(BaseSolver):
    """

    .. _solver-ts3:

    Turbostream 3
    -------------

    Turbostream 3 is a multi-block structured, GPU-accelerated Reynolds-averaged
    Navier--Stokes code developed by :cite:t:`Brandvik2011`.

    To use this solver, add the following to your configuration file:

    .. code-block:: yaml

        solver:
          type: ts3
          nstep: 10000  # Case-dependent
          nstep_avg: 2500  # Typically ~0.25 nstep

    """

    # Override base attributes
    _name = "ts3"

    soft_start: bool = False

    workdir: Path = None
    """Working directory to run the simulation in."""

    environment_script: Path = Path(
        "/usr/local/software/turbostream/ts3610_a100/bashrc_module_ts3610_a100"
    )
    """Setup environment shell script to be sourced before running."""

    cfl: float = 0.4
    """Courant--Friedrichs--Lewy number, reduce for more stability."""

    dampin: float = 25.0
    """Negative feedback factor, reduce for more stability."""

    facsecin: float = 0.005
    """Fourth-order smoothing factor, increase for more stability."""

    fmgrid: float = 0.2
    """Multigrid factor, reduce for more stability."""

    ilos: int = 2
    """Viscous model, 0 for inviscid, 1 for mixing-length, 2 for Spalart-Allmaras."""

    nchange: int = 2000
    """At start of simulation, ramp smoothing and damping over this many time steps."""

    viscosity_law: int = 0
    """Variation of viscosity with temperature, 0 for constant, 1 for 0.62 power law."""

    nstep: int = 10000
    """Number of time steps."""

    nstep_avg: int = 5000
    """Average over the last `nstep_avg` steps of the calculation."""

    rfmix: float = 0.0
    """Mixing plane relaxation factor."""

    sfin: float = 0.5
    """Proportion of second-order smoothing, increase for more stability."""

    rfin: float = 0.5
    """Inlet relaxation factor, reduce for low-Mach flows."""

    nstep_soft: int = 0
    """Number of steps for soft start precursor simulation."""

    av: dict = None
    """Raw application-variable overrides, ``av[name] = value``, for any TS3
    variable without a typed field. Applied after the typed fields; setting a
    name that is also driven by a non-default typed field is an error."""

    bv: dict = None
    """Raw block-variable overrides, ``bv[bid][name] = value``, for any TS3
    variable without a typed field. Applied after the typed fields; setting a
    name that is also driven by a non-default typed field is an error."""

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

    def robust(self):
        """Increase damping and smoothing, lower CFL, and use mixing-length model."""
        return self.replace(
            ilos=1,
            dampin=3.0,
            facsecin=0.02,
            sfin=2.0,
            cfl=0.3,
            fmgrid=0.0,
            soft_start=False,
        )

    def restart(self):
        """Restart the simulation from a previous solution."""
        return self.replace(
            nchange=0,
        )

    def run(self, grid, machine, workdir):
        if not workdir.exists():
            workdir.mkdir(parents=True, exist_ok=True)
        self.convergence = run(grid, self, machine, workdir)


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
            logger.warning("******")
            logger.warning("Caught interrupt, killing solver...")
            logger.warning("******")
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait()
            logger.warning("Killed solver.")

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
    for f in ("stopit", "output_avg.xdmf", "output.xdmf", "input.hdf5"):
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
    """Using a given configuration, load flow solution and insert into grid.

    The averaged flow field is read from ``output_avg.hdf5`` and the turbulent
    viscosity from the instantaneous ``output.hdf5`` (Turbostream only writes
    ``trans_dyn_vis`` to the latter). A diverged solution surfaces as a
    :class:`~turbigen.exceptions.ConvergenceError`.
    """

    output_file_path = os.path.join(ts3_config.workdir, "output_avg.hdf5")
    output_inst_file_path = os.path.join(ts3_config.workdir, "output.hdf5")
    if not os.path.exists(output_file_path):
        raise Exception(f"""No Turbostream output file found at: {output_file_path}""")

    # ember.ts3 owns the datum-correct read into the existing grid; a diverged
    # solution (negative/NaN density, etc.) is raised as ValueError there and
    # translated to ConvergenceError for the iterator.
    try:
        ember.ts3.read_conserved(grid, output_file_path)
        ember.ts3.read_mu_turb(grid, output_inst_file_path)
    except ValueError as e:
        raise ConvergenceError(f"TS3 solution diverged: {e}") from e


def _non_default_fields(ts3_config, names):
    """Return the subset of `names` whose config value differs from the default.

    Used to decide which typed fields actively drive a TS3 variable, so the
    raw av/bv override dicts can be rejected when they would compete with one.
    """
    defaults = {f.name: f.default for f in fields(ts3_config)}
    return {name for name in names if getattr(ts3_config, name) != defaults[name]}


def _check_overlap(ts3_config):
    """Reject raw av/bv overrides that collide with a non-default typed field.

    A TS3 variable must have a single source. The typed fields and the raw
    override dicts can each set the same name; forbid that when the typed field
    has been moved off its default (so e.g. `robust()` setting cfl and an `av`
    dict also setting cfl will correctly error).
    """
    av = ts3_config.av or {}
    bv = ts3_config.bv or {}

    av_clash = _non_default_fields(ts3_config, _AV_FIELDS) & av.keys()
    if av_clash:
        raise ValueError(
            f"av override(s) {sorted(av_clash)} also set by a non-default typed "
            "field; remove one of the two competing sources."
        )

    bv_fields = _non_default_fields(ts3_config, _BV_FIELDS)
    for bid, block_bv in bv.items():
        bv_clash = bv_fields & block_bv.keys()
        if bv_clash:
            raise ValueError(
                f"bv override(s) {sorted(bv_clash)} for block {bid} also set by a "
                "non-default typed field; remove one of the two competing sources."
            )


def _write_input(grid, ts3_config):
    """Write the TS3 input file from an ember grid + config via the writer.

    Drives ``ember.ts3.TS3Writer`` so turbigen can layer its typed fields and
    raw av/bv override dicts on top of the grid-derived defaults. Mirrors
    ``ember.ts3.write_ts3`` but with the extra config surface.

    The grid must already carry ``wdist`` (it holds the baked-in mixing-length
    limit); ``get_blocks`` reads it and also writes total energy on
    Turbostream's zero datum internally, so no datum handling is needed here.
    """
    # Forbid two competing sources for one variable before writing anything.
    _check_overlap(ts3_config)

    # strict=True: turbigen's contract is to hand TS3 a complete, runnable grid,
    # so a missing flow field, fluid, or periodic connectivity is a bug here and
    # must fail at write time rather than silently omitting variables.
    writer = ember.ts3.TS3Writer()
    writer.get(grid, strict=True)

    # Typed application variables (forwarded by matching name).
    writer.set_av(**{name: getattr(ts3_config, name) for name in _AV_FIELDS})

    # Derived application variables.
    writer.set_av(
        nstep_save_start=ts3_config.nstep - ts3_config.nstep_avg,
        restart=1,
    )

    # Typed block variables, per block. xllim is left at the writer's
    # non-clamping default since the limit is already baked into wdist.
    for bid in range(len(grid)):
        writer.set_bv(bid, fmgrid=ts3_config.fmgrid)

    # Inlet relaxation factor, applied to each inlet patch as a patch variable.
    # Iterate the writer's own pv (keyed by its renumbered pid, rotating patches
    # excluded); inlets are the patches carrying an rfin variable.
    for bid, block_pv in enumerate(writer.pv):
        for pid, pv in block_pv.items():
            if "rfin" in pv:
                writer.set_pv(bid, pid, rfin=ts3_config.rfin)

    # Raw overrides last (validated for overlap above). set_av/set_bv validate
    # unknown names, type-cast, and reject NaN.
    writer.set_av(**(ts3_config.av or {}))
    for bid, block_bv in (ts3_config.bv or {}).items():
        writer.set_bv(bid, **block_bv)

    writer.check()
    input_path = os.path.join(ts3_config.workdir, "input.hdf5")
    writer.write(input_path)
    writer.write_probe_meta(input_path)


def _run(grid, ts3_config):
    """Perform all steps on a grid and config."""

    _write_input(grid, ts3_config)
    _execute(ts3_config)
    _read_hdf5(grid, ts3_config)

    # Probe post-run caching disabled: the read functions moved to ember.ts3.
    # TODO: re-enable once probe_meta.yaml is written from an ember.grid via the
    # ember TS3 API, then cache via ember.ts3.read_probe_dat.
    # if ts3_config.nstep_save_probe:
    #     probe_fnames = list(read_probe_metadata(ts3_config.workdir).keys())
    #     # Use multiprocessing to read and save compressed probes in parallel
    #     with multiprocessing.Pool(processes=8) as pool:
    #         pool.map(read_probe_dat, probe_fnames)


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
    istep_save_start = ts3_conf.nstep - ts3_conf.nstep_avg

    try:
        istep, mdot, ho, Po, resid = parse_log(log_path)
        state_log = grid.inlet_patches[0].state.copy().empty(shape=mdot.shape)
        state_log.set_Tu0(0.0)
        state_log.set_P_h(ho, Po)
        conv = ConvergenceHistory(istep, istep_save_start, resid, mdot, state_log)
    except Exception as e:
        logger.warning(f"Failed to parse log file {log_path}")
        logger.warning(f"Exception: {e}")
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

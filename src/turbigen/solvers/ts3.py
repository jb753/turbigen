import logging
from dataclasses import dataclass, fields
from timeit import default_timer as timer
from glob import glob
import ember.ts3
from ember.convergence_history import ConvergenceHistory
from turbigen.exceptions import ConvergenceError
import subprocess
import os
from pathlib import Path
import signal
import sys
import re
from turbigen.solvers.base import BaseSolver

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


# How long to wait for the solver to create its log file before giving up, and
# how often to poll the running solver for divergence, in seconds.
_LOG_TIMEOUT_S = 60
_POLL_INTERVAL_S = 10


def _execute(ts3_config):
    """Using a given configuration, execute TS3."""

    # Store old working directory and change to this config's. The finally below
    # restores it on every exit path (timeout, divergence, non-zero return code),
    # so a failed run does not leave the process stranded in the work directory.
    old_workdir = os.getcwd()
    os.chdir(ts3_config.workdir)

    try:
        if not os.path.exists(ts3_config.environment_script):
            raise Exception(
                f"""Could not locate TS3 env script {ts3_config.environment_script}
Are you on a HPC compute node gpu-q-* (not a login node)?
If you have recently been added to the turbostream user group, log out
and then back in to refresh your access permissions.
"""
            )

        # Open a subshell, source the environment and run the solver (serial, 1 GPU)
        logger.info("Using 1 GPU.")
        cmd_str = (
            f". {ts3_config.environment_script};"
            f" mpirun -npernode 1 -np 1 turbostream"
            f" input.hdf5 output 1 > log.txt"
        )

        # Remove old probe data
        probe_dat = glob("output_probe_*.dat")
        for fname in probe_dat:
            os.remove(fname)

        # Start the Turbostream process
        with subprocess.Popen(
            cmd_str, shell=True, stderr=subprocess.PIPE, preexec_fn=os.setsid
        ) as proc:
            # Wait for the log to appear, then watch it for divergence until the
            # solver exits. proc.wait(timeout=...) returns the instant the solver
            # finishes, so a fast run is not held up by the polling interval, and
            # the divergence check happens before each wait rather than after a
            # blind sleep.
            try:
                start = timer()
                while proc.poll() is None:
                    have_log = os.path.isfile("log.txt")
                    if not have_log and (timer() - start) > _LOG_TIMEOUT_S:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        raise Exception(
                            f"Timed out after {_LOG_TIMEOUT_S}s waiting for TS3 "
                            "log file to appear"
                        )
                    if have_log and (istep_nan := _check_nan("log.txt")):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        raise ConvergenceError(
                            f"TS3 diverged at step {istep_nan}"
                        ) from None
                    try:
                        proc.wait(timeout=_POLL_INTERVAL_S)
                    except subprocess.TimeoutExpired:
                        pass
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
    finally:
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

    # Keep old log file if it exists (e.g. after a soft start)
    log_path = os.path.join(ts3_conf.workdir, "log.txt")
    if os.path.exists(log_path):
        os.rename(log_path, log_path.replace("log.txt", "log_old.txt"))

    _run(grid, ts3_conf)

    # Build the convergence history from the log. ember.ts3 owns the parsing;
    # the reference scales (V_ref, T_ref, areas) are derived from the solved
    # grid, which the log itself does not record.
    try:
        conv = ConvergenceHistory.from_ts3(log_path, grid)
        conv.write_cnv(os.path.join(ts3_conf.workdir.parent, "conv.cnv"))
    except Exception as e:
        logger.warning(f"Failed to parse log file {log_path}")
        logger.warning(f"Exception: {e}")
        conv = None

    return conv


# Live-divergence detection during a run reads only these two patterns;
# full log parsing for the convergence history lives in ember.ts3.
re_nan = re.compile(r"NAN")
re_current_step = re.compile(r"^O?U?T?E?R? ?STEP No\.\s*(\d*)", flags=re.MULTILINE)


def _check_nan(fname):
    """Return step number of divergence from TS3 log, or zero if no NANs found.

    The whole log is read and searched, so detection does not depend on where in
    the file the NAN lands (the previous chunked, anchored scan missed NANs that
    were not at the start of a 2048-byte block or straddled a block boundary). The
    log is small relative to the polling interval, so reading it whole is cheap.
    When a NAN is present the most recently logged step number is returned, or -1
    if no step has been logged yet.
    """
    with open(fname, "r") as f:
        content = f.read()
    if not re_nan.search(content):
        return 0
    steps = re_current_step.findall(content)
    return int(steps[-1]) if steps else -1

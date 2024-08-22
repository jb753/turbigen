"""Entry point for running turbigen from the shell."""

import logging
import subprocess
import turbigen.util
import turbigen.yaml
import turbigen.run
import socket
import shutil
import sys
import os
import turbigen.config
import datetime
import argparse

logger = turbigen.util.make_logger()


# Record all exceptions in the logger
def my_excepthook(excType, excValue, traceback):
    logger.error(
        "Error encountered, quitting...", exc_info=(excType, excValue, traceback)
    )


# Replace default exception handling with our hook
sys.excepthook = my_excepthook


def _make_argparser():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description=(
            "turbigen is a general turbomachinery design system. When "
            "called from the command line, the program performs mean-line design, "
            "creates annulus and blade geometry, then meshes and runs a "
            "computational fluid dynamics simulation. Most input data are specified "
            "in a configuration file; the command-line options below override some "
            "of that configuration data."
        ),
        usage="%(prog)s [FLAGS] CONFIG_YAML",
        add_help="False",
    )
    parser.add_argument(
        "CONFIG_YAML", help="filename of configuration data in yaml format"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help=(
            "output more debugging information "
            "(can also enable by setting $TURBIGEN_VERBOSE)"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-V",
        "--version",
        help="print version number and exit",
        action="version",
        version=f"%(prog)s {turbigen.__version__}",
    )
    parser.add_argument(
        "-J",
        "--no-job",
        help="disable submission of cluster job (when run on login node)",
        action="store_true",
    )
    parser.add_argument(
        "-j",
        "--job",
        help="enable submission of cluster job (when run on compute node)",
        action="store_true",
    )
    parser.add_argument(
        "-I",
        "--no-iteration",
        help=(
            "run once only, disabling iterative incidence, deviation, "
            "mean-line correction"
        ),
        action="store_true",
    )
    parser.add_argument(
        "-S",
        "--no-solve",
        help="disable running of the CFD solver, continuing with the initial guess",
        action="store_true",
    )
    parser.add_argument(
        "-e",
        "--edit",
        help="run on an edited copy of the configuration file (using $EDITOR)",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--meanline-debug",
        help="perform the mean-line design, print out debugging information and stop",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--annulus-debug",
        help="perform the annulus design, print out debugging information and stop",
        action="store_true",
    )
    parser.add_argument(
        "-W",
        "--no-wdist",
        help="skip wall distance caluclation",
        action="store_true",
    )
    return parser


def main():
    """Parse command-line arguments and call turbigen appropriately."""

    # Run the parser on sys.argv and collect input data
    args = _make_argparser().parse_args()

    # Load input data in dictionary format
    d = turbigen.yaml.read_yaml(args.CONFIG_YAML)

    # If we are planning to use embsolve
    if d.get("solver", {}).get("type") == "embsolve":
        try:
            # Check our MPI rank
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # Jump to solver slave process if not first rank
            if rank > 0:
                from turbigen.solvers import embsolve

                embsolve.run_slave()
                sys.exit(0)

        except ImportError:
            # Just run serially if we cannot import mpi4py
            pass

    # Check that we have a workdir before making a config object
    # This is because we might want to edit the input file before loading proper
    if not (workdir := d.get("workdir")):
        raise turbigen.exceptions.ConfigError(
            f"No working directory specified in {args.CONFIG_YAML}"
        )

    # Automatically number workdir if it contains placeholder
    if "*" in workdir:
        d["workdir"] = workdir = turbigen.util.next_numbered_dir(workdir)

    workdir = os.path.abspath(workdir)

    # Make workdir if needed
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    # Write config file into the working directory
    working_config = os.path.join(workdir, "config.yaml")
    turbigen.yaml.write_yaml(d, working_config)

    # Edit the config file if requested
    if args.edit:
        editor = os.environ.get("EDITOR")
        subprocess.run([f"{editor}", f"{working_config}"])

    # Now read into a configuration object proper
    conf = turbigen.config.Config.read(working_config)

    # Apply command-line overrides to the config
    if args.no_job:
        conf.job = {}

    if args.no_iteration:
        conf.iterate = {}

    if args.no_solve:
        if conf.solver:
            conf.solver["skip"] = True

    conf.wdist &= not args.no_wdist
    conf.mean_line["debug"] = args.meanline_debug
    conf.annulus["debug"] = args.annulus_debug

    # Choose log level
    # sys.tracebacklimit = 1000 if args.verbose else 1
    if args.verbose or os.environ.get("TURBIGEN_VERBOSE"):
        log_level = logging.DEBUG
    elif conf.iterate:
        log_level = logging.ITER
    else:
        log_level = logging.INFO

    logger.setLevel(level=log_level)

    # Remove job config if we are on a compute node or cannot find sbatch
    if conf.job and not conf.hypercube:
        hostname = socket.gethostname()
        if hostname.startswith("gpu") and not args.job:
            logger.info(
                f"Running on compute node {hostname}, declining to submit job to queue."
            )
            conf.job = {}
        elif not shutil.which("sbatch"):
            logger.info("No `sbatch` on PATH, declining to submit job to queue.")
            conf.job = {}

    # No logging if we are just submitting a job
    if not conf.job:
        log_path = os.path.join(workdir, "log_turbigen.txt")
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        logger.addHandler(fh)
        logger.iter(f"TURBIGEN v{turbigen.__version__}")
        logger.iter(
            f"Starting at {datetime.datetime.now().replace(microsecond=0).isoformat()}"
        )
        logger.iter(f"Working directory: {workdir}")

    success = turbigen.run.run(conf)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

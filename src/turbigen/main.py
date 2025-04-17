"""Entry point for running turbigen from the shell."""

import logging
import numpy as np
import subprocess
from turbigen import util
import turbigen.yaml
from timeit import default_timer as timer
import turbigen.slurm
import turbigen.run2
import socket
import shutil
import sys
import os
import turbigen.config
import turbigen.config2
import datetime
import argparse

logger = util.make_logger()


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

    # Ensure that the workdir is always set
    # This is because we might want to edit the input file before loading proper
    if not (workdir := d.get("workdir")):
        raise Exception(f"No working directory specified in {args.CONFIG_YAML}")

    # Automatically number workdir if it contains placeholder
    if "*" in workdir:
        d["workdir"] = workdir = util.next_numbered_dir(workdir)

    # Make workdir if needed
    workdir = os.path.abspath(workdir)
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    # Set up loud logging initially
    log_path = os.path.join(workdir, "log_turbigen.txt")
    log_level = logging.ITER
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)
    logger.setLevel(level=log_level)
    fh.setLevel(log_level)

    # Print banner
    logger.iter(f"TURBIGEN v{turbigen.__version__}")
    logger.iter(
        f"Starting at {datetime.datetime.now().replace(microsecond=0).isoformat()}"
    )

    logger.iter(f"Working directory: {workdir}")

    # Write config file into the working directory
    working_config = os.path.join(workdir, "config.yaml")
    turbigen.yaml.write_yaml(d, working_config)

    # Edit the config file if requested
    if args.edit:
        editor = os.environ.get("EDITOR")
        subprocess.run([f"{editor}", f"{working_config}"])

    # Now read back into a configuration object proper
    conf = turbigen.yaml.read_yaml(working_config)
    conf = turbigen.config2.TurbigenConfig(**conf)

    # Apply command-line overrides to the config
    if args.no_iteration:
        conf.iterate = []
    if args.no_solve:
        conf.skip = True

    # Set up logging to file
    if args.verbose or os.environ.get("TURBIGEN_VERBOSE"):
        log_level = logging.DEBUG
    else:
        if conf.iterate:
            log_level = logging.ITER
        else:
            log_level = logging.INFO
    logger.setLevel(level=log_level)
    fh.setLevel(log_level)

    # Backup the source files for later reproduction
    util.save_source_tar_gz(conf.workdir / "src.tar.gz")

    # Iterate if requested
    if not conf.iterate:
        conf.design_and_run()
        # Write back the config with actual meanline and grid
        conf.save()
        converged = True
    else:
        logger.iter(f"Iterating for max {conf.max_iter} iterations...")
        basedir = conf.workdir
        for iiter in range(conf.max_iter):
            # Set a numbered iteration workdir
            conf.workdir = basedir / f"{iiter:03d}"

            # Ensure that the iteration directory is empty
            # Do not want to pick up old meshes etc.
            if conf.workdir.exists():
                shutil.rmtree(conf.workdir)
            conf.workdir.mkdir(parents=True)

            # If we already have a solution, don't need to
            # run CFD again on first iteration
            tic = timer()
            if conf.grid and iiter == 0:
                conf.skip = True
            elif iiter > 0:
                conf.skip = False

            # Design and run the configuration
            conf.design_and_run()

            # Write back the config with actual meanline and grid
            conf.save()

            # Update the config
            conv_all, log_data = conf.step_iterate()
            toc = timer()

            # Insert timing data into log
            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            reprint = not np.mod(iiter, 5)
            if reprint:
                logger.iter("Convergence status:")
                for k, v in conv_all.items():
                    logger.iter(f"  {k}: {v}")
            logger.iter(format_iter_log(log_data, header=reprint))

            # Disable soft start after first iteration
            conf.solver.soft_start = False

            # Check for convergence
            converged = all(conv_all.values())
            if converged:
                # Copy everything from the final iteration
                # to the working directory
                shutil.copytree(conf.workdir, basedir, dirs_exist_ok=True)
                # Delete iteration directories
                for i in range(iiter + 1):
                    shutil.rmtree(basedir / f"{i:03d}")
                break

        logger.iter(f"Finished iterating, converged={converged}.")
    logger.iter(conf.format_design_vars_table())

    if not converged:
        sys.exit(1)

    quit()

    # Hypercubes are always jobs
    if conf.hypercube:
        if not conf.job:
            raise Exception("Need job submission configured to run a hypercube.")

        basedir = conf.workdir
        conf.database["conf_path"] = os.path.join(basedir, "config_db.yaml")
        conf.database["mean_line_path"] = os.path.join(basedir, "mean_line_db.yaml")
        conf.workdir = None

        if conf.hypercube.get("N"):
            logger.iter("Running a hypercube...")
            cs = conf.sample_hypercube()
            Nrunmax = conf.hypercube.get("max_jobs", 0)
            turbigen.slurm.submit_array(cs, basedir, Nrunmax)

        if conf.hypercube.get("Nedge"):
            logger.iter("Running hypercube edges...")
            ce = conf.sample_hyperfaces()
            Nrunmax = conf.hypercube.get("max_jobs", 0)
            turbigen.slurm.submit_array(ce, basedir, Nrunmax)

        success = True

    else:
        # Determine whether to try to run job or not
        hostname = socket.gethostname()
        job_flag = True
        if not conf.job:
            job_flag = False
        elif args.no_job:
            logger.iter("No job submission forced with flag -J.")
            job_flag = False
        elif not shutil.which("sbatch"):
            logger.iter("No `sbatch` on PATH, declining to submit job to queue.")
            job_flag = False
        elif hostname.startswith("gpu"):
            if args.job:
                logger.iter("Job submission from compute node forced with flag -j.")
            else:
                logger.iter(
                    f"Running on compute node {hostname}, declining to submit job to queue."
                )
                job_flag = False

        if job_flag:
            turbigen.slurm.submit(conf)
            success = True
        else:
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


def format_iter_log(log_data, header=False):
    """Format the log data in a tabular format for printing.

    Parameters
    ----------
    log_data : dict
        Dictionary of log data, keys are the column headers, values are the data.
    """

    # Find column widths from headers, with a minimum width
    col_widths = [max(len(k), 5) for k in log_data.keys()]

    # Format header row
    header_str = " ".join(f"{k:>{w}}" for k, w in zip(log_data.keys(), col_widths))

    # Format data rows
    value_strs = [f"{util.asscalar(v):.3g}"[:5] for v in log_data.values()]
    value_strs = " ".join([f"{v:>{w}}" for v, w in zip(value_strs, col_widths)])

    if header:
        out_str = header_str + "\n" + "-" * len(header_str) + "\n" + value_strs
    else:
        out_str = value_strs

    return out_str


if __name__ == "__main__":
    main()

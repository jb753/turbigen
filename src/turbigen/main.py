"""Entry point for running turbigen from the shell."""

import logging
import numpy as np
import subprocess
from turbigen import util
import turbigen.yaml
from timeit import default_timer as timer
import shutil
import sys
import os
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
            "computational fluid dynamics simulation. Optionally, the design can be "
            "iterated in response to the simulation results. A job or a series of "
            "jobs can be submitted to a queuing system. Most input data are specified "
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
        help=("output more debugging information "),
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
        help="disable submission of cluster job (when job is already configured in INPUT_YAML)",
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
    return parser


def main():
    """Parse command-line arguments and call turbigen appropriately."""

    # Run the parser on sys.argv and collect input data
    args = _make_argparser().parse_args()

    # Load input data in dictionary format
    d = turbigen.yaml.read_yaml(args.CONFIG_YAML)

    # If we are planning to use emb
    if d.get("solver", {}).get("type") == "emb":
        try:
            # Check our MPI rank
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            # Jump to solver slave process if not first rank
            if rank > 0:
                from turbigen.solvers import emb

                emb.run_slave()
                sys.exit(0)

        except ImportError:
            # Just run serially if we cannot import mpi4py
            print('Failed to import "mpi4py", running serially.')
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
    logger.iter(f"*** TURBIGEN v{turbigen.__version__} ***")
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

    # From this point we can assume the workdir exists
    # and the config file is in the working directory

    # Now read back into a configuration object proper
    # to ensure that all the defaults are set,
    # the config is valid, and pathnames are absolute
    conf = turbigen.yaml.read_yaml(working_config)
    conf = turbigen.config2.TurbigenConfig(**conf)
    # Resave the config so that the internal state and
    # the YAML file are consistent (e.g. if submitting a job)
    conf.save()

    # Determine if we are overriding iteration
    iterate_flag = conf.iterate and not args.no_iteration

    # Set up logging to file
    if args.verbose:
        log_level = logging.DEBUG
    else:
        if iterate_flag:
            log_level = logging.ITER
        else:
            log_level = logging.INFO
    logger.setLevel(level=log_level)
    fh.setLevel(log_level)

    # Backup the source files for later reproduction
    util.save_source_tar_gz(conf.workdir / "src.tar.gz")

    # If we are sampling a design space, do that and exit
    if conf.design_space and not args.no_job:
        # Put datum in non-numbered directory
        conf.workdir = conf.workdir / "datum"
        conf.save()
        # If datum not ran yet, run it first
        if not conf.mean_line_actual:
            logger.iter("Running the datum...")
            conf.job.submit(conf.fname)
        logger.iter("Sampling the design space...")
        samples = conf.design_space.sample(conf)
        if not samples:
            logger.iter("No samples to run, exiting.")

        # Write out all the sample configs
        for s in samples:
            s.save()

        # Submit as an array
        conf.job.submit_array([s.fname for s in samples])

        sys.exit(0)

    # If we are submitting a job, do that and exit
    if conf.job and not args.no_job:
        conf.job.submit(conf.fname)
        sys.exit(0)

    # Iterate if requested
    if not iterate_flag:
        conf.design_and_run(args.no_solve)
        # Write back the config with actual meanline and grid
        conf.converged = converged = not args.no_solve
        conf.save()
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
            conf.design_and_run(args.no_solve)

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
            conf.converged = converged
            if converged:
                # Copy everything from the final iteration
                # to the working directory
                shutil.copytree(conf.workdir, basedir, dirs_exist_ok=True)
                # Delete iteration directories
                for i in range(iiter + 1):
                    shutil.rmtree(basedir / f"{i:03d}")
                # Reset the workdir to the final one
                conf.workdir = basedir
                # Save the final config
                conf.save()
                break

        logger.iter(f"Finished iterating, converged={converged}.")
    logger.iter(conf.format_design_vars_table())

    if not converged:
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

"""Entry point for running turbigen from the shell."""

import logging
from turbigen import util
import turbigen.yaml_utils
import turbigen.plugins
import subprocess
from timeit import default_timer as timer
from pathlib import Path
import shutil
import sys
import os
import turbigen.config2
import datetime
import argparse

logger = logging.getLogger("turbigen")


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


def _setup_work_dir(work_dir):
    """Validate and create the working directory for a run."""

    # Cannot continue without a working directory
    # We need to know where to save things
    if not work_dir:
        raise Exception("No working directory specified, set YAML key 'work_dir'.")

    # Automatically number work_dir if it contains placeholder
    if "*" in work_dir:
        work_dir = util.next_numbered_dir(work_dir)

    # Convert to absolute path
    # So that subsequent usage of the config does not need
    # to stay in same working directory
    work_dir = Path(work_dir).absolute()

    # Create the work_dir if needed
    if not work_dir.exists():
        work_dir.mkdir(parents=True, exist_ok=True)

    return work_dir


def _setup_logging(work_dir, log_level):
    """Initialise logging to file and stderr, add a custom level."""

    logging.ITER = 25
    logging.raiseExceptions = True
    logging.addLevelName(logging.ITER, "ITER")
    logging.basicConfig(format="%(message)s")

    def _log_iter(message, *args, **kwargs):
        logger.log(logging.ITER, message, *args, **kwargs)

    logger.iter = _log_iter

    log_path = work_dir / "log_turbigen.txt"
    fh = logging.FileHandler(log_path)
    logger.addHandler(fh)
    logger.setLevel(level=log_level)
    fh.setLevel(log_level)


def main():
    """Parse command-line arguments and call turbigen appropriately."""

    # Run the parser on sys.argv and collect input data
    args = _make_argparser().parse_args()

    # Load input data in dictionary format
    d = turbigen.yaml_utils.read_yaml(args.CONFIG_YAML)

    # Validate and create working directory if needed
    d["work_dir"] = work_dir = _setup_work_dir(d.get("work_dir"))

    # Determine if we are going to iterate or not
    iterating = d.get("iterate") and not args.no_iteration

    # Choose verbosity of output
    log_level = (
        logging.DEBUG if args.verbose else logging.ITER if iterating else logging.INFO
    )
    _setup_logging(work_dir, log_level)

    # Print banner
    time_now = datetime.datetime.now().replace(microsecond=0).isoformat()
    logger.iter(f"*** TURBIGEN v{turbigen.__version__} ***")
    logger.iter(f"Starting at {time_now}")
    logger.iter(f"Working directory: {work_dir}")

    # Write back config file into the working directory
    working_config = work_dir / "config.yaml"
    turbigen.yaml_utils.write_yaml(d, working_config)

    # Edit the config file if requested
    if args.edit:
        subprocess.run([f"{os.environ.get('EDITOR')}", f"{working_config}"])

    start_tic = timer()

    # From this point we can assume the work_dir exists
    # and the config file is in the working directory

    # Now after possible editing, read back into a configuration
    # object proper to ensure that all the defaults are set,
    # the config is valid, and pathnames are absolute
    logger.debug("Reading configuration file...")
    conf = turbigen.yaml_utils.read_yaml(working_config)

    # Load plugins from specified directory if needed
    if plug_dir := conf.get("plug_dir"):
        turbigen.plugins.load_plugins(Path(plug_dir))

    logger.debug("Parsing into configuration object...")
    conf = turbigen.config2.TurbigenConfig(**conf)

    # Resave the config so that the internal state and
    # the YAML file are consistent (e.g. if submitting a job)
    # We have not changed grid or guess yet, so do not overwrite those pickles
    logger.debug("Writing back to ensure consistency...")
    conf.save(working_config)
    logger.debug("Done.")

    # Check again if iterating (editing the config could have changed value)
    iterating = conf.iterate and not args.no_iteration

    # Backup the source files for later reproduction
    util.save_source_tar_gz(conf.work_dir / "src.tar.gz")

    conf.design_and_run(args.no_solve)

    logger.iter(conf.format_design_vars_table())

    logger.iter(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")

    logger.iter(f"Working directory was: {work_dir}")

    sys.exit(0)

    # If we are sampling a design space, do that and exit
    if conf.design_space and not args.no_job:
        # Put datum in non-numbered directory
        # conf.work_dir = conf.work_dir / "datum"
        # conf.save()
        # If datum not ran yet, run it first
        # if not conf.mean_line_actual:
        # logger.iter("Running the datum...")
        # conf.job.submit(conf.fname)
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
    if not iterating:
        conf.design_and_run(args.no_solve)
        # Write back the config with actual meanline and grid
        conf.converged = converged = not args.no_solve
        conf.save()
    else:
        basedir = conf.work_dir

        if conf.design_space and conf.design_space.configs:
            logger.info("Initialising iterators with fitted design space.")
            conf.interpolate_all_iterators()

        logger.iter(f"Iterating for max {conf.max_iter} iterations...")

        for iiter in range(conf.max_iter):
            # Set a numbered iteration work_dir
            conf.work_dir = basedir / f"{iiter:03d}"

            if conf.fac_nstep_initial != 1.0:
                if iiter == 0:
                    old_nstep = conf.solver.n_step
                    conf.solver.n_step = int(old_nstep * conf.fac_nstep_initial)
                    logger.iter(
                        f"Using initial n_step={conf.fac_nstep_initial}*{old_nstep}"
                        f"={conf.solver.n_step}"
                    )
                elif iiter == 1:
                    conf.solver.n_step = old_nstep

            # Ensure that the iteration directory is empty
            # Do not want to pick up old meshes etc.
            if conf.work_dir.exists():
                shutil.rmtree(conf.work_dir)
            conf.work_dir.mkdir(parents=True)

            # Write out the config before we begin
            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

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
            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            # Update the config
            conv_all, log_data = conf.step_iterate()
            toc = timer()

            # Insert timing data into log
            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            reprint = not np.mod(iiter, 5)
            # if reprint:
            #     logger.iter("Convergence status:")
            #     for k, v in conv_all.items():
            #         logger.iter(f"  {k}: {v}")
            logger.iter(format_iter_log(log_data, header=reprint))

            # Disable soft start after first iteration
            conf.solver.soft_start = False

            # Check for convergence
            converged = all(conv_all.values())
            conf.converged = converged
            # Do some cleanup if converged
            if converged:
                # Copy everything from the final iteration
                # up to the base directory
                shutil.copytree(conf.work_dir, basedir, dirs_exist_ok=True)

                # Move iteration configs to a subdirectory
                # But delete the solutions and postprocessing
                all_iter_dir = basedir / "iterations"
                all_iter_dir.mkdir(exist_ok=True)
                for i in range(iiter + 1):
                    iter_dir = basedir / f"{i:03d}"
                    iter_conf_dest = all_iter_dir / f"config_{i:03d}.yaml"
                    if not i == iiter:
                        # If this is the last convereged iteration, we do not want to
                        # copy the configuration file, because it will be a duplicate
                        # of the one that gets copied up to the basedir
                        shutil.move(iter_dir / conf.basename, iter_conf_dest)
                    shutil.rmtree(iter_dir)
                # Reset the work_dir to the final one
                conf.work_dir = basedir
                # Save the final config
                conf.save()
                break

        logger.iter(f"Finished iterating, converged={converged}.")
    logger.iter(conf.format_design_vars_table())

    logger.iter(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")

    logger.iter(f"Working directory was: {work_dir}")

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

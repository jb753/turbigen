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
import numpy as np
import turbigen.config
import turbigen.viewer
import datetime
import argparse

logger = logging.getLogger("turbigen")


def my_excepthook(excType, excValue, traceback):
    logger.error(
        "Error encountered, quitting...", exc_info=(excType, excValue, traceback)
    )


sys.excepthook = my_excepthook


def _make_run_parser():
    parser = argparse.ArgumentParser(
        prog="turbigen",
        description=(
            "turbigen is a general turbomachinery design system. When "
            "called from the command line, the program performs mean-line design, "
            "creates annulus and blade geometry, then meshes and runs a "
            "computational fluid dynamics simulation. Optionally, the design can be "
            "iterated in response to the simulation results. Most input data are "
            "specified in a configuration file; the command-line options below "
            "override some of that configuration data."
        ),
        usage="%(prog)s [FLAGS] CONFIG_YAML",
        add_help="False",
    )
    parser.add_argument(
        "CONFIG_YAML", help="filename of configuration data in yaml format"
    )
    parser.add_argument(
        "-v", "--verbose", help="output more debugging information", action="store_true"
    )
    parser.add_argument(
        "-V",
        "--version",
        help="print version number and exit",
        action="version",
        version=f"%(prog)s {turbigen.__version__}",
    )
    parser.add_argument(
        "-J", "--no-job", help="disable submission of cluster job", action="store_true"
    )
    parser.add_argument(
        "-I",
        "--no-iteration",
        help="run once only, disabling iterative incidence, deviation, mean-line correction",
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
        "--mesh",
        nargs="?",
        const=0.5,
        type=float,
        metavar="SPF",
        help="plot mesh at span fraction SPF and exit (default 0.5)",
    )
    return parser


def _make_sample_parser():
    parser = argparse.ArgumentParser(
        prog="turbigen sample",
        description="Generate and submit design space samples.",
        usage="%(prog)s [FLAGS] CONFIG_YAML",
        add_help="False",
    )
    parser.add_argument(
        "CONFIG_YAML", help="filename of configuration data in yaml format"
    )
    parser.add_argument(
        "-v", "--verbose", help="output more debugging information", action="store_true"
    )
    parser.add_argument(
        "-J",
        "--no-job",
        help="disable submission of cluster job (write configs only)",
        action="store_true",
    )
    parser.add_argument(
        "--purge",
        help="delete all existing run_* directories before sampling",
        action="store_true",
    )
    return parser


def _setup_work_dir(work_dir):
    """Validate and create the working directory for a run."""

    if not work_dir:
        raise Exception("No working directory specified, set YAML key 'work_dir'.")

    if "*" in work_dir:
        work_dir = util.next_numbered_dir(work_dir)

    work_dir = Path(work_dir).absolute()

    if not work_dir.exists():
        work_dir.mkdir(parents=True, exist_ok=True)

    return work_dir


def _setup_logging(log_level):
    """Initialise stderr logging."""
    logging.raiseExceptions = True
    logging.basicConfig(format="%(message)s")
    logger.setLevel(log_level)


def cmd_run(args):
    """Run mean-line design, meshing, CFD, and optional iteration."""

    d = turbigen.yaml_utils.read_yaml(args.CONFIG_YAML)
    d["work_dir"] = work_dir = _setup_work_dir(d.get("work_dir"))

    fh = logging.FileHandler(work_dir / "log_turbigen.txt")
    logger.addHandler(fh)

    iterating = d.get("iterate") and not args.no_iteration
    if iterating and not args.verbose:
        logger.setLevel(logging.WARNING)
    fh.setLevel(logger.level)

    time_now = datetime.datetime.now().replace(microsecond=0).isoformat()
    logger.warning(f"*** TURBIGEN v{turbigen.__version__} ***")
    logger.warning(f"Starting at {time_now}")
    logger.warning(f"Working directory: {work_dir}")

    working_config = work_dir / "config.yaml"
    turbigen.yaml_utils.write_yaml(d, working_config)

    if args.edit:
        subprocess.run([os.environ.get("EDITOR"), str(working_config)])

    logger.debug("Reading configuration file...")
    conf = turbigen.yaml_utils.read_yaml(working_config)

    if plug_dir := conf.get("plug_dir"):
        turbigen.plugins.load_plugins(Path(plug_dir))

    logger.debug("Parsing into configuration object...")
    conf = turbigen.config.TurbigenConfig(**conf)

    logger.debug("Writing back to ensure consistency...")
    conf.save(working_config)
    logger.debug("Done.")

    iterating = conf.iterate and not args.no_iteration

    util.save_source_tar_gz(conf.work_dir / "src.tar.gz")

    start_tic = timer()

    if not iterating:
        conf.design_and_run(args.no_solve, plot_mesh=args.mesh)
    else:
        basedir = conf.work_dir

        if conf.design_space and conf.design_space.configs:
            logger.info("Initialising iterators with fitted design space.")
            conf.interpolate_all_iterators()

        logger.warning(f"Iterating for max {conf.max_iter} iterations...")

        for iiter in range(conf.max_iter):
            conf.work_dir = basedir / f"{iiter:03d}"

            if conf.fac_nstep_initial != 1.0:
                if iiter == 0:
                    old_nstep = conf.solver.n_step
                    conf.solver.n_step = int(old_nstep * conf.fac_nstep_initial)
                    logger.warning(
                        f"Using initial n_step={conf.fac_nstep_initial}*{old_nstep}"
                        f"={conf.solver.n_step}"
                    )
                elif iiter == 1:
                    conf.solver.n_step = old_nstep

            if conf.work_dir.exists():
                shutil.rmtree(conf.work_dir)
            conf.work_dir.mkdir(parents=True)

            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            tic = timer()
            if conf.grid and iiter == 0:
                conf.skip = True
            elif iiter > 0:
                conf.skip = False

            conf.design_and_run(args.no_solve)

            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            conv_all, log_data = conf.step_iterate()
            toc = timer()

            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            reprint = not np.mod(iiter, 5)
            logger.warning(format_iter_log(log_data, header=reprint))

            conf.solver.soft_start = False

            converged = all(conv_all.values())
            conf.converged = converged
            if converged:
                shutil.copytree(conf.work_dir, basedir, dirs_exist_ok=True)

                all_iter_dir = basedir / "iterations"
                all_iter_dir.mkdir(exist_ok=True)
                for i in range(iiter + 1):
                    iter_dir = basedir / f"{i:03d}"
                    iter_conf_dest = all_iter_dir / f"config_{i:03d}.yaml"
                    if not i == iiter:
                        shutil.move(iter_dir / conf.basename, iter_conf_dest)
                    shutil.rmtree(iter_dir)
                conf.work_dir = basedir
                conf.save()
                break

        logger.warning(f"Finished iterating, converged={converged}.")

    turbigen.viewer.record_metadata(conf)

    logger.warning(conf.format_design_vars_table())
    logger.warning(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")
    logger.warning(f"Working directory was: {work_dir}")

    sys.exit(0)

    # Iterate if requested
    if not iterating:
        conf.design_and_run(args.no_solve)
        conf.converged = converged = not args.no_solve
        conf.save()
    else:
        basedir = conf.work_dir

        if conf.design_space and conf.design_space.configs:
            logger.info("Initialising iterators with fitted design space.")
            conf.interpolate_all_iterators()

        logger.warning(f"Iterating for max {conf.max_iter} iterations...")

        for iiter in range(conf.max_iter):
            conf.work_dir = basedir / f"{iiter:03d}"

            if conf.fac_nstep_initial != 1.0:
                if iiter == 0:
                    old_nstep = conf.solver.n_step
                    conf.solver.n_step = int(old_nstep * conf.fac_nstep_initial)
                    logger.warning(
                        f"Using initial n_step={conf.fac_nstep_initial}*{old_nstep}"
                        f"={conf.solver.n_step}"
                    )
                elif iiter == 1:
                    conf.solver.n_step = old_nstep

            if conf.work_dir.exists():
                shutil.rmtree(conf.work_dir)
            conf.work_dir.mkdir(parents=True)

            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            tic = timer()
            if conf.grid and iiter == 0:
                conf.skip = True
            elif iiter > 0:
                conf.skip = False

            conf.design_and_run(args.no_solve)

            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            conv_all, log_data = conf.step_iterate()
            toc = timer()

            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            reprint = not np.mod(iiter, 5)
            logger.warning(format_iter_log(log_data, header=reprint))

            conf.solver.soft_start = False

            converged = all(conv_all.values())
            conf.converged = converged
            if converged:
                shutil.copytree(conf.work_dir, basedir, dirs_exist_ok=True)

                all_iter_dir = basedir / "iterations"
                all_iter_dir.mkdir(exist_ok=True)
                for i in range(iiter + 1):
                    iter_dir = basedir / f"{i:03d}"
                    iter_conf_dest = all_iter_dir / f"config_{i:03d}.yaml"
                    if not i == iiter:
                        shutil.move(iter_dir / conf.basename, iter_conf_dest)
                    shutil.rmtree(iter_dir)
                conf.work_dir = basedir
                conf.save()
                break

        logger.warning(f"Finished iterating, converged={converged}.")

    logger.warning(conf.format_design_vars_table())
    logger.warning(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")
    logger.warning(f"Working directory was: {work_dir}")

    if not converged:
        sys.exit(1)


def cmd_sample(args):
    """Generate and submit design space samples."""

    config_path = Path(args.CONFIG_YAML).absolute()
    conf_dict = turbigen.yaml_utils.read_yaml(config_path)

    work_dir = Path(conf_dict.get("work_dir", "")).absolute()
    if work_dir != config_path.parent:
        raise Exception(
            f"For 'sample', work_dir must equal the directory containing CONFIG_YAML. "
            f"Got work_dir={work_dir}, config dir={config_path.parent}"
        )

    if args.purge:
        run_dirs = sorted(work_dir.glob("run_*"))
        if run_dirs:
            logger.warning(f"Purging {len(run_dirs)} run directories...")
            for d in run_dirs:
                shutil.rmtree(d)
        for fname in ("metaData.json", "session.json"):
            f = work_dir / fname
            if f.exists():
                f.unlink()
                logger.warning(f"Purged {fname}")

    if plug_dir := conf_dict.get("plug_dir"):
        turbigen.plugins.load_plugins(Path(plug_dir))

    conf = turbigen.config.TurbigenConfig(**conf_dict)
    conf.design_space.base_dir = work_dir

    if conf.design_space and conf.design_space.independent.mean_line:
        valid = conf.mean_line.valid_design_params["all"]
        invalid = [
            k
            for k in conf.design_space.independent.mean_line
            if conf.design_space.independent._split_meanline_key(k)[0] not in valid
        ]
        if invalid:
            raise ValueError(
                f"Invalid mean_line keys in design_space.independent: {invalid}. "
                f"Valid keys: {sorted(valid)}"
            )

    start_tic = timer()
    logger.warning("Sampling the design space...")
    samples = conf.design_space.sample(conf)

    if not samples:
        logger.warning("No samples to run, exiting.")
        sys.exit(0)

    for s in samples:
        s.save()

    if not args.no_job:
        conf.job.submit_array([s.fname for s in samples])

    logger.warning(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")
    sys.exit(0)


def main():
    """Parse command-line arguments and dispatch to the appropriate subcommand."""

    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        args = _make_sample_parser().parse_args(sys.argv[2:])
        _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
        cmd_sample(args)
    else:
        args = _make_run_parser().parse_args()
        _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
        cmd_run(args)


def format_iter_log(log_data, header=False):
    col_widths = [max(len(k), 5) for k in log_data.keys()]
    header_str = " ".join(f"{k:>{w}}" for k, w in zip(log_data.keys(), col_widths))
    value_strs = [f"{util.asscalar(v):.3g}"[:5] for v in log_data.values()]
    value_strs = " ".join([f"{v:>{w}}" for v, w in zip(value_strs, col_widths)])
    if header:
        return header_str + "\n" + "-" * len(header_str) + "\n" + value_strs
    return value_strs

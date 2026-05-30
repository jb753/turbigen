"""Entry point for running turbigen from the shell."""

import logging
import gc
from turbigen import util
import turbigen.yaml_utils
import turbigen.plugins
import subprocess
from timeit import default_timer as timer
from pathlib import Path
import shutil
import sys
import os
import turbigen.config
import turbigen.viewer
import datetime
import argparse
import resource

logger = logging.getLogger("turbigen")


def _log_ram(label):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    logger.debug(f"RAM [{label}]: {rss_gb:.2f} GB")


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

    if conf.job and not args.no_job:
        conf.job.submit(working_config)
        sys.exit(0)

    iterating = conf.iterate and not args.no_iteration

    util.save_source_tar_gz(conf.work_dir / "src.tar.gz")

    start_tic = timer()

    if not iterating:
        conf.design_and_run(args.no_solve, plot_mesh=args.mesh)
        converged = not args.no_solve

        logger.info("Post-processing...")
        conf.post_process_all()
        _log_ram("after post-processing")
        logger.info("Done post-processing.")

    else:
        basedir = conf.work_dir

        if conf.design_space and conf.design_space.configs:
            nsamp = conf.design_space.nsample
            nmin = conf.design_space.nsample_min_interp
            if nsamp >= nmin:
                logger.info(
                    f"Initialising iterators with fitted design space "
                    f"({nsamp} converged samples)."
                )
                conf.interpolate_all_iterators()
            else:
                logger.warning(
                    f"Only {nsamp} converged samples (< {nmin}); "
                    f"using initial guess from input config."
                )

        logger.warning(f"Iterating for max {conf.max_iter} iterations...")
        logger.warning("Status: ✓ = within tol, ✗ = not yet converged")

        converged = False
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

            _log_ram("after design_and_run")

            conf.save(use_gzip=False, write_grids=conf.save_iteration_grids)

            _log_ram("after save")

            conv_all, log_data, tol = conf.step_iterate()
            toc = timer()
            _log_ram("after step_iterate")

            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            logger.warning(format_iter_log(log_data, tol=tol, header=(iiter == 0)))

            conf.solver.soft_start = False

            logger.info("Post-processing...")
            conf.post_process_all()
            _log_ram("after post-processing")
            logger.info("Done post-processing.")

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

            gc.collect()

        logger.warning(f"Finished iterating, converged={converged}.")

    turbigen.viewer.record_metadata(conf)

    logger.warning(conf.format_design_vars_table())
    logger.warning(f"Total time: {(timer() - start_tic) / 60.0:.2f} min")
    logger.warning(f"Working directory was: {work_dir}")

    sys.exit(0 if converged else 1)

    # Iterate if requested
    if not iterating:
        conf.design_and_run(args.no_solve)
        conf.converged = converged = not args.no_solve
        conf.save()
    else:
        basedir = conf.work_dir

        if conf.design_space and conf.design_space.configs:
            nsamp = conf.design_space.nsample
            nmin = conf.design_space.nsample_min_interp
            if nsamp >= nmin:
                logger.info(
                    f"Initialising iterators with fitted design space "
                    f"({nsamp} converged samples)."
                )
                conf.interpolate_all_iterators()
            else:
                logger.warning(
                    f"Only {nsamp} converged samples (< {nmin}); "
                    f"using initial guess from input config."
                )

        logger.warning(f"Iterating for max {conf.max_iter} iterations...")
        logger.warning("Status: ✓ = within tol, ✗ = not yet converged")

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

            conv_all, log_data, tol = conf.step_iterate()
            toc = timer()

            elapsed = toc - tic
            log_data = dict(Min=elapsed / 60.0, **log_data)

            logger.warning(format_iter_log(log_data, tol=tol, header=(iiter == 0)))

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
        from turbigen.job_server import signal_daemon

        if signal_daemon():
            logger.warning("Sent SIGHUP to queue daemon (cancel-all).")
        else:
            logger.warning("No queue daemon running; skipping cancel-all.")

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
    elif len(sys.argv) > 1 and sys.argv[1] == "queue":
        from turbigen.job_server import _make_queue_parser, cmd_queue

        args = _make_queue_parser().parse_args(sys.argv[2:])
        _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
        cmd_queue(args)
    else:
        args = _make_run_parser().parse_args()
        _setup_logging(logging.DEBUG if args.verbose else logging.INFO)
        cmd_run(args)


def _fmt_iter_cell(k, val, signed=True):
    """Per-variable value formatter for the iteration log."""
    if k.startswith("Inc") or k.startswith("Dev"):
        return f"{val:+.1f}" if signed else f"{val:.1f}"
    if k == "Min":
        return f"{val:.0f}"
    return f"{val:.3g}"


def format_iter_log(log_data, tol=None, header=False):
    """Format one iteration row.

    Drops delta columns (D*) silently. Each tracked variable gets a value
    column with a trailing '✓'/'✗' status marker against `tol`. When
    `header=True`, the result includes a header line, a tolerance row, and a
    separator before the values.
    """
    tol = tol or {}
    # Drop delta columns (D*, except 'Dev[' which is itself a tracked value)
    # and error columns (E*) used only for status checks.
    keys = [
        k
        for k in log_data.keys()
        if not (k.startswith("D") and not k.startswith("Dev["))
        and not k.startswith("E")
    ]

    value_strs = [_fmt_iter_cell(k, util.asscalar(log_data[k])) for k in keys]
    # Status compares the iterator-reported error (E<key>) against tolerance
    # when present, since the displayed value may be the absolute CFD reading
    # rather than the convergence error. Falls back to the displayed value
    # for iterators (Inc/Dev) where the value itself is the error.

    def _status_value(k):
        ek = "E" + k
        if ek in log_data:
            return util.asscalar(log_data[ek])
        return util.asscalar(log_data[k])

    statuses = [
        ("✓" if abs(_status_value(k)) <= tol[k] else "✗") if k in tol else " "
        for k in keys
    ]
    tol_strs = [
        _fmt_iter_cell(k, tol[k], signed=False) if k in tol else "" for k in keys
    ]

    widths = [
        max(len(k), len(vs) + 1, len(ts))
        for k, vs, ts in zip(keys, value_strs, tol_strs)
    ]

    header_cells = [f"{k:>{w}}" for k, w in zip(keys, widths)]
    value_cells = [
        f"{vs:>{w - 1}}{status}" for vs, status, w in zip(value_strs, statuses, widths)
    ]
    tol_cells = [f"{ts:>{w}}" for ts, w in zip(tol_strs, widths)]

    header_str = " ".join(header_cells)
    value_str = " ".join(value_cells)
    tol_str = " ".join(tol_cells)

    if header:
        sep = "-" * len(header_str)
        return f"{header_str}\n{tol_str}\n{sep}\n{value_str}"
    return value_str

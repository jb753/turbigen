"""Functions to run turbigen on config objects."""
import os
import sys
import json
import shutil
import importlib
from turbigen import (
    fluid,
    grid,
    util,
    geometry,
    slurm,
    hmesh,
    ohmesh,
)
from turbigen.exceptions import ConvergenceError, ConfigError
import turbigen.post_process
import turbigen.plot
import turbigen.average
import turbigen.annulus
import turbigen.blade
import numpy as np
from timeit import default_timer as timer


logger = util.make_logger()

LOG_FIELDS = (
    "Min",
    "Inc",
    "DInc",
    "Dev",
    "DDev",
)


def log_line(d, fields):
    """Given a list of fields and dictionary of values, print a log line."""

    out = ""

    for v in fields:
        w = max(len(v), 4)
        if d is None:
            dout = f"{v:<{w}}" + " "
        elif isinstance(d, int):
            dout = "-" * (w + 1)
        elif d == "-":
            dout = "-" * (w + 1)
        else:
            if v in d:
                if isinstance(d[v], int):
                    dout = f"{d[v]:<{w}d}"[:w] + " "
                elif isinstance(d[v], str):
                    dout = f"{d[v]:<{w}}"[:w] + " "
                elif isinstance(d[v], list):
                    dout = f"{d[v]}"[:w] + " "
                elif isinstance(d[v], np.ndarray):
                    dout = f"{d[v]}"[:w] + " "
                elif d[v] is None:
                    dout = "None" + " "
                else:
                    dout = f"{d[v]:<{w}f}"[:w] + " "
            else:
                dout = (" " * w) + " "

        out = out + dout

    if isinstance(d, int):
        out = f"Iter {d} " + out[7:]
    logger.iter(out)
    sys.stdout.flush()


def run_single(conf, gguess=None, plot=False):
    """Run turbigen on a config object."""

    times = []
    times.append(timer())

    # Inlet state
    logger.debug("Getting inlet state...")
    So1 = conf.get_inlet()
    logger.info(f"Inlet: {So1}")

    # Dynamically load the design functions based on machine type in config
    if not conf.mean_line_type:
        raise ConfigError("No mean-line type specified; quitting.")
    logger.info(f"Designing a {conf.mean_line_type}...")


    meanline_design = util.load_mean_line(conf.mean_line_type)


    # Feed mean-line arguments to the function
    times.append(timer())
    ml = meanline_design.forward(So1=So1, **conf.mean_line)
    logger.info(ml)
    times.append(timer())
    logger.debug(f"Mean-line design took {np.diff(times)[-1]:.1f}s")
    ml.check()

    # Check inversion is consistent
    try:
        logger.info("Checking mean-line inversion...")
        params_inv = meanline_design.inverse(ml)
    except AttributeError:
        raise Exception(
            f'No mean-line inversion function for type="{conf.mean_line_type}"'
        )
    params_inv.pop("So1")
    # Compare forward and inverse params, check within a tolerance
    for v in conf.mean_line:
        if v not in params_inv:
            raise Exception(
                f"Parameter {v} not returned by inverse function for meanline type"
                f' "{conf.mean_line_type}"'
            )
        if params_inv[v] is None:
            continue

        rtol = 0.05
        atol = 0.1

        error = False
        if conf.mean_line[v] == 0.0:
            if not np.allclose(conf.mean_line[v], params_inv[v], atol=atol):
                error = True
        else:
            if not np.allclose(conf.mean_line[v], params_inv[v], rtol=rtol):
                error = True
        if error:
            raise Exception(
                f"Meanline inverted {v}={params_inv[v]} not same as forward value"
                f" {v}={conf.mean_line[v]}"
            )

    # Make a working directory
    workdir = conf.workdir
    if not os.path.exists(workdir):
        os.makedirs(workdir, exist_ok=True)

    # # Write out the config
    # conf_out = conf.copy()
    # conf_out.interpolate = {}
    # conf_out.write(os.path.join(workdir, "config.yaml"))

    conf.write(os.path.join(workdir, "config.yaml"))

    # Feed annulus arguments to the geometry function
    times.append(timer())
    annulus_type = conf.annulus.pop("type", "Smooth")
    Annulus = util.load_annulus(annulus_type)
    ann = Annulus(ml.rmid, ml.span, ml.Beta, **conf.annulus)
    conf.annulus["type"] = annulus_type
    logger.info(ann)
    times.append(timer())
    logger.debug(f"Annulus design took {np.diff(times)[-1]:.1f}s")

    cut_offset = conf.solver.pop("cut_offset", None)
    xr_cut = ann.get_cut_planes(cut_offset)
    if conf.plot:
        turbigen.plot.plot_annulus(ann, os.path.join(workdir, "annulus.pdf"), xr_cut)

    # Include deviations angles with respect to free vortex in camber
    # parameters to make q_camber
    qstar_save = []
    qcamber_save = []
    # vexpon = np.array(conf.blades.get("vortex_expon"),None)
    for irow, row in enumerate(conf.sections):
        if row:
            row["spf"] = np.array(row["spf"])
            row["q_thick"] = np.array(row["q_thick"])
            qstar_camber = np.array(row.pop("qstar_camber"))
            qstar_save.append(qstar_camber + 0.0)
            ind = (irow * 2, irow * 2 + 1)
            vexpon_row = -1
            if vexpon := conf.blades.get("vortex_expon"):
                if not vexpon[irow] is None:
                    vexpon_row = np.array(vexpon[irow])
            if chi_fix := row.get("chi"):
                Alpha_rel = chi_fix
            else:
                Alpha_rel = ml.Alpha_rel_free_vortex(row["spf"], vexpon_row)[:, ind]
            Chi = Alpha_rel + qstar_camber[:, :2]
            q_camber = qstar_camber
            q_camber[:, :2] = util.tand(Chi)
            row["q_camber"] = q_camber
            qcamber_save.append(q_camber)
        else:
            qstar_save.append(None)
            qcamber_save.append(None)

    row_rmid = 0.5 * (ml.rmid[::2] + ml.rmid[1::2])

    # Make blades parameters
    bld = []
    if conf.splitter:
        splitter = []
    else:
        splitter = None
    mstack = conf.blades.get(
        "mstack",
        [
            0.5,
        ]
        * conf.nrow,
    )
    thick_rm = conf.blades.get("thick_rm")
    thick_span = conf.blades.get("thick_span")
    thick_type = conf.blades.get(
        "thick_type",
        [
            None,
        ]
        * conf.nrow,
    )
    camber_type = conf.blades.get(
        "camber_type",
        [
            None,
        ]
        * conf.nrow,
    )
    for irow, row in enumerate(conf.sections):
        if row:
            row_now = row.copy()
            row_now.pop("chi", None)
            row_now.pop("vortex_expon", None)
            if thick_rm:
                f = thick_rm[irow] * row_rmid[irow] / ann.chords(0.5)[1:-1:2][irow]
                if thick_type == "Taylor":
                    fac_thick = np.array([f, f, 1.0, 1.0, f, 1.0])
                else:
                    fac_thick = np.array([f, 1.0, 1.0, f])
                row_now["q_thick"] = fac_thick * row_now["q_thick"]
            if thick_span:
                f = (
                    thick_span[irow]
                    * ml.span[::2][irow]
                    / ann.chords(0.5)[1:-1:2][irow]
                    / 2.0
                )
                if thick_type[irow] == "Impeller":
                    fac_thick = np.array([f, 1.0, 1.0, f])
                else:
                    fac_thick = np.array([f**2.0, f, 1.0, 1.0, f, f])
                    # fac_thick = np.array([f, f, 1.0, 1.0, f, 1.0])
                row_now["q_thick"] = fac_thick * row_now["q_thick"]
            bld.append(
                geometry.Blade(
                    streamsurface=ann.xr_row(irow),
                    mstack=mstack[irow],
                    thick_type=thick_type[irow],
                    camber_type=camber_type[irow],
                    **row_now,
                )
            )
            # Now consider if we need splitters
            if conf.splitter:
                if not (splitter_now := conf.splitter[irow]):
                    splitter.append(None)
                    continue

                logger.debug(f"Designing splitters for row {irow}")

                # Apply same scaling as for main blade
                if thick_span or thick_rm:
                    splitter_now["q_thick"] = fac_thick * splitter_now["q_thick"]

                nsect = len(splitter_now["spf"])
                qstar_camber_split_save = splitter_now.pop("qstar_camber")
                splitter_now["q_camber"] = np.copy(qstar_camber_split_save)
                tmain = np.full(nsect, np.nan)
                mref = np.full(nsect, np.nan)
                for isect in range(nsect):

                    # Get angles of main blade camber line
                    mlim_sect = splitter_now["mlim"][isect]
                    spf_sect = splitter_now["spf"][isect]
                    cam_main = bld[-1]._get_cam_thick(spf_sect)[0]
                    chi_main = cam_main.chi(mlim_sect)
                    logger.debug(f"Section {isect}, main blade angles {chi_main}")
                    logger.debug(f'q_camber {splitter_now["q_camber"][isect]}')

                    # Fill in tanChi for the splitter after recamber
                    splitter_now["q_camber"][isect][:2] = util.tand(
                        chi_main + splitter_now["q_camber"][isect][:2]
                    )
                    logger.debug(f'q_camber {splitter_now["q_camber"][isect]}')

                    # Calculate the angular offset to put splitter on the main
                    # camber line at splitter stacking location
                    mref[isect] = mstack[irow] * np.ptp(mlim_sect) + mlim_sect[0]
                    mq = np.linspace(0.0, 1.0, 101)
                    xrtc = np.mean(bld[irow].evaluate_section(spf_sect, m=mq), axis=0)
                    tmain[isect] = np.interp(mref[isect], mq, xrtc[2])

                splitter.append(
                    geometry.Blade(
                        streamsurface=ann.xr_row(irow),
                        mstack=mstack[irow],
                        thick_type=thick_type[irow],
                        camber_type=camber_type[irow],
                        theta_offset=np.mean(tmain),
                        **splitter_now,
                    )
                )

                splitter_now.pop("q_camber")
                splitter_now["qstar_camber"] = qstar_camber_split_save
        else:
            bld.append(None)

    ind_out = [True if b else False for b in bld]

    if conf.plot:
        for ib, b in enumerate(bld):
            if b:
                fname_xrt = os.path.join(workdir, "blade_%d_xrt.pdf" % ib)
                fname_rrt = os.path.join(workdir, "blade_%d_rrt.pdf" % ib)
                fname_cam = os.path.join(workdir, "camber_%d_xrt.pdf" % ib)
                fname_split = os.path.join(workdir, "splitter_%d_xrt.pdf" % ib)
                turbigen.plot.plot_blade(b, spf=[0.0, 0.5, 1.0], fname=fname_xrt)
                turbigen.plot.plot_blade(
                    b, spf=[0.0, 0.5, 1.0], fname=fname_rrt, xr=False
                )
                turbigen.plot.plot_camber_line(b, fname_cam)

    # Surface length
    ell = np.array([b.surface_length(0.5) if b else None for b in bld])

    if "Re_surf" in conf.blades:
        for irow, b in enumerate(bld):
            if not (Re_row := conf.blades["Re_surf"][irow]):
                continue

            # Set viscosity to maintain surface length reynolds
            mu = (ml.rho_ref * ml.V_ref)[irow] * ell[irow] / Re_row
            ml.mu = mu
            So1.mu = mu

            break

    ell = np.array([b.surface_length(0.5) if b else np.nan for b in bld])
    Re_surf = np.array(ell[ind_out] / ml.L_visc[ind_out]).astype(float)
    Restr = np.array2string(Re_surf / 1e5, precision=1)
    logger.info(f"Re_surf/10^5={Restr}")

    # Preallocate number of blades
    Nb = np.full_like(row_rmid, np.nan)

    # Loop over rows and choose method for number of blades
    for irow in range(len(Nb)):
        # Kaufmann circulation coefficient
        if "Co" in conf.blades and (Co := conf.blades["Co"][irow]):
            s = (ml.s_ell(Co) * ell)[irow]
            Nb[irow] = np.round(2.0 * np.pi * row_rmid[irow] / s)
        # Casey blade-to-blade loading coefficient
        elif "Cb" in conf.blades and (Cb := conf.blades["Cb"][irow]):
            c = ann.chords(0.5)[1:-1:2][irow]
            Nb[irow] = float(ml.eval_Cbtob(c, Cb)[irow])
        # Fixed number of blades
        elif "Nb" in conf.blades and (Nb_now := conf.blades["Nb"][irow]):
            Nb[irow] = float(Nb_now)
        # Lieblein diffusion factor
        elif "DFL" in conf.blades and (DFL := conf.blades["DFL"][irow]):
            logger.debug("Setting Nb using Lieblein")
            s_c = ml.set_Lieblein_DF(DFL)[irow]
            logger.debug(f"s_c={s_c}")
            cx = ann.chords(0.5)[1:-1:2][ind_out][irow]
            s = s_c * cx
            Nb[irow] = np.round(2.0 * np.pi * row_rmid[irow] / s)

    iunbladed = np.where(np.logical_not(ind_out))[0]
    Nb[iunbladed] = Nb[iunbladed - 1]
    Nb = np.round(Nb).astype(int)

    s = 2.0 * np.pi * row_rmid[ind_out] / Nb
    s_cm = s / ann.chords(0.5)[1:-1:2][ind_out]
    s_cm_str = np.array2string(s_cm, precision=2)

    # Offset splitters to mid-pitch
    if conf.splitter:
        for irow in range(len(Nb)):
            if conf.splitter[irow]:
                splitter[irow].theta_offset += (
                    2.0
                    * np.pi
                    / Nb[irow]
                    * conf.blades.get("pitch_frac_splitter", 0.5)[irow]
                )

    if conf.plot:
        for ib, b in enumerate(bld):
            if b:
                if splitter and splitter[ib]:
                    turbigen.plot.plot_splitter(b, splitter[ib], fname_split)

    ml.Nb = np.repeat(Nb, 2)
    ml.Co = conf.blades.get("Co")
    ml.Lsurf = ell
    ml.mean_line_type = conf.mean_line_type
    ml.workdir = workdir

    nom_ml_path = os.path.join(workdir, "mean_line_nominal.yaml")
    ml.write(nom_ml_path)

    # Get tip gaps and apply relative to mean height
    if "tip" not in conf.blades:
        tips = np.zeros_like(s_cm)
    else:
        tips = np.array(conf.blades["tip"])
    # Replace None with zero
    for i in range(conf.nrow):
        if tips[i] is None:
            tips[i] = 0.0
    ml.tip = tips[0]

    logger.info(f"Nblade={Nb}, s_cm={s_cm_str}, tip={tips}")

    mac = geometry.Machine(ann, bld, Nb, tips, splitter)

    # solver_type = conf.solver.get("type")
    # if not solver_type:
    #     logger.iter("No solver specified, quitting.")
    #     sys.exit(0)

    # At this point, we have the geometry and mean-line set up
    # We can now generate the mesh

    # Restore the relative camber
    for irow, row in enumerate(conf.sections):
        if row:
            row.pop("q_camber", None)
            row["qstar_camber"] = qstar_save[irow].tolist()

    # Set row, hub, casing spacings using yplus and flat-plate correlations
    yplus = np.atleast_2d(conf.mesh["yplus"]).T
    Cf = (2.0 * np.log10(Re_surf) - 0.65) ** -2.3
    tauw = Cf * 0.5 * ml.rho_ref * ml.V_ref**2.0
    Vtau = np.sqrt(tauw / ml.rho_ref)
    Lvisc = np.atleast_2d(ml.mu_ref / ml.rho_ref / Vtau)
    drow = yplus * Lvisc
    # drow has dimensions: [LE/TE, irow]
    dhub = np.mean(drow)
    dcas = np.mean(drow)
    # Indicator for unbladed rows
    # ind_out = [True if b else False for b in bld]
    unbladed = [True if not b else False for b in bld]
    # At this point, we have the geometry and mean-line set up
    # We can now generate the mesh
    mesh_type = conf.mesh["type"]

    mesh_settings = conf.mesh.copy()
    mesh_settings.pop("yplus")
    mesh_settings.pop("type")

    times.append(timer())

    if mesh_type == "h":
        # Apply settings from yaml file to the default config
        hmesh_config = hmesh.HMeshConfig(**mesh_settings)
        # Make the grid object
        g = hmesh.make_grid(mac, hmesh_config, dhub, dcas, drow, unbladed)

        if conf.plot:
            turbigen.plot.plot_hmesh(g, workdir)

    elif mesh_type == "oh":
        tips *= 0.5 * (ml.span[::2] + ml.span[1::2])[ind_out]
        # Apply settings from yaml file to the default config
        ohmesh_config = ohmesh.OHMeshConfig(**mesh_settings)
        ohmesh_config.workdir = workdir

        # Make the grid object
        g = ohmesh.make_grid(mac, ohmesh_config, dhub, dcas, drow, unbladed)

    else:
        raise Exception(f'Unrecognised mesh type "{mesh_type}"')

    times.append(timer())
    logger.debug(f"Mesh generation took {np.diff(times)[-1]:.1f}s")
    logger.info(f"Mesh Npts/10^6={g.ncell/1e6:.2f}")

    # Ready to apply boundary conditions now
    logger.info("Applying boundary conditions...")

    # Wall rotations
    Omega = ml.Omega[
        ::2,
    ]
    rot_types = []
    for Omi, tip in zip(Omega, mac.tip):
        if Omi:
            if tip:
                rot_types.append("tip_gap")
            else:
                rot_types.append("shroud")
        else:
            rot_types.append("stationary")
    g.apply_rotation(rot_types, Omega)

    # if "Beta1_override" in conf.solver:
    #     Beta1 = conf.solver.pop("Beta1_override")
    #     g.apply_inlet(So1, ml.Alpha[0], Beta1)
    # else:
    #     Beta1 = None

    # # Inlet and outlet
    g.apply_inlet(So1, ml.Alpha[0], ml.Beta[0])
    g.apply_outlet(ml.P[-1])

    # Configure throttle
    if throttle_pid := conf.operating_point.get("mdot_pid"):
        restart_fac = 0.5 if gguess else 1.0
        norm_fac = ml.P.ptp() / ml.mdot[-1]
        mass_adjust = conf.solver.pop("mass_adjust", 0.0)
        chic_flag = np.abs(mass_adjust) > 0.0
        g.apply_throttle(
            ml.mdot[-1] * (1.0 + mass_adjust),
            np.array(throttle_pid) * norm_fac * restart_fac,
        )
    else:
        chic_flag = False

    # Choose whether the blocks are real or perfect
    if isinstance(So1, fluid.PerfectState):
        g = grid.Grid([b.to_perfect() for b in g])
    elif isinstance(So1, fluid.RealState):
        g = grid.Grid([b.to_real() for b in g])
    else:
        raise Exception("Unrecognised inlet state type")

    logger.info("Setting intial guess...")
    # Initial guess
    if gguess:
        g.apply_guess_3d(gguess)
        if throttle_pid:
            g.update_outlet()
    else:
        g.apply_guess_meridional(ml.interpolate_guess(mac.ann))

    if conf.wdist:
        logger.info("Calculating wall distance...")
        times.append(timer())
        g.calculate_wall_distance()
        times.append(timer())
        logger.debug(f"Setting wall distance took {np.diff(times)[-1]:.1f}s")
    else:
        logger.info("Skipping wall distance.")
        for b in g:
            b.w[:] = 0.0

    if conf.solver:
        conf.solver["workdir"] = workdir

    # The grid is ready to run. At this point, we can 'install' it
    if install_type := conf.install.pop("type", None):
        # Dynamically load the install module
        logger.info(f"Installing a {install_type}...")
        install_module = importlib.import_module(
            f".{install_type}", package="turbigen.install"
        )
        logger.debug("Successfully imported.")
        gi = install_module.forward(g, mac, **conf.install)

        if conf.solver:
            logger.info(f'Running solver {conf.solver["type"]} on installed...')
            gi.run(conf.solver, mac)
            conf.solver.pop("workdir")
        else:
            logger.info("No solver specified, continuing with initial guess...")

        logger.info("Uninstalling...")
        g = install_module.inverse(gi)

        conf.install["type"] = install_type

    else:
        if conf.solver:
            if conf.solver.get("type"):
                logger.info(f'Running solver {conf.solver["type"]}...')
                g.run(conf.solver, mac)
                conf.solver.pop("workdir")
        else:
            logger.info("No solver specified, continuing with initial guess...")

    if cut_offset is not None:
        conf.solver["cut_offset"] = cut_offset

    logger.info("Post-processing...")

    times.append(timer())

    Cmix = []
    Amix = []
    Dsmix = []
    for icut, xrci in enumerate(xr_cut):
        try:
            Cnow, Aannnow, dsnow = turbigen.average.mix_out_unstructured(
                g.unstructured_cut_marching(xrci)
            )
            Cmix.append(Cnow)
            Amix.append(Aannnow)
            Dsmix.append(dsnow)
        except Exception as e:
            raise Exception(f"Unstructured cutting failed, station {icut}") from e
    times.append(timer())
    logger.debug(f"Taking unstructured cuts took {np.diff(times)[-1]:.1f}s")

    Call = Cmix[0].stack(Cmix)
    Call.Omega = ml.Omega
    Call.Nb = ml.Nb

    ml_out = turbigen.flowfield.make_mean_line_from_flowfield(Amix, Call)

    try:
        if conf.plot:
            for spf in (0.1, 0.5, 0.9):
                pltname = os.path.join(workdir, f"pdist_spf_{int(spf*10)}.pdf")
                turbigen.plot.plot_pressure_distribution(
                    bld[0], g, ml_out, spf, pltname
                )
    except Exception:
        pass

    if conf.post_process.get("Sdot_wall"):
        times.append(timer())
        ml_out.Sdot_wall, ml_out.Asurf = turbigen.post_process.surface_dissipation(g)
        times.append(timer())
        logger.debug(f"Surface dissipation calculation took {np.diff(times)[-1]:.1f}s")

    if conf.post_process.get("tip"):
        times.append(timer())
        ml_out.Sdot_tip = turbigen.post_process.tip(g)
        times.append(timer())
        logger.debug(f"Tip loss calculation took {np.diff(times)[-1]:.1f}s")

    ml_out.Co = conf.blades.get("Co")
    ml_out.Lsurf = ell
    ml_out.tip = tips[0]
    ml_out.Ds_mix = Dsmix
    ml_out.workdir = workdir

    end_time = timer()
    mins = (end_time - times[0]) / 60.0

    logger.info("Mixed-out CFD result:")
    logger.info(ml_out)

    log_fields = LOG_FIELDS + ()
    match_vars = conf.iterate.get("mean_line", {}).get("match_tolerance", {})
    for v in match_vars:
        log_fields += (v,)
        log_fields += ("D" + v,)

    # log_line({'Iter':0, 'Row': 1, 'Inc':5.,'Dev': 4.5},log_fields)

    pdict = {"Min": mins}

    mean_line_converged = True
    if mean_opt_conf := conf.iterate.get("mean_line"):
        rf_mean = mean_opt_conf.get("relaxation_factor", 0.5)
        out_vars = meanline_design.inverse(ml_out)

        match_vars = mean_opt_conf.get("match_tolerance", {})
        for v in match_vars:
            if conf.mean_line[v] is None:
                err = np.inf
                var_new = out_vars[v]
            else:
                err = np.abs(conf.mean_line[v] - out_vars[v])
                var_new = out_vars[v] * rf_mean + (1.0 - rf_mean) * conf.mean_line[v]
            dvar = var_new - conf.mean_line[v]
            pdict[v] = out_vars[v]
            pdict["D" + v] = dvar
            if err > match_vars[v]:
                mean_line_converged = False
            conf.mean_line[v] = util.reduce_scalar(var_new)

    inc_converged = True
    if inc_conf := conf.iterate.get("incidence"):
        spf_flow, chi_stag = turbigen.post_process.incidence(
            g, mac, ml.Beta[::2], workdir if conf.plot else False
        )
        rf_inc = inc_conf.get("relaxation_factor", 0.2)
        rtol_mdot_inc = inc_conf.get("rtol_mdot", 0.05)
        mdot_err = np.abs(ml_out.mdot / ml.mdot - 1)[0]
        for irow, row in enumerate(conf.sections):
            logger.debug(f"CORRECTING INCIDENCE, row {irow}")
            if row:
                chi_flow = np.interp(row["spf"], spf_flow[irow], chi_stag[irow])
                chi_metal = util.atand(qcamber_save[irow][:, 0])
                inc_target = inc_conf.get("target", 0.0)
                inc = chi_flow - chi_metal - inc_target
                logger.debug(f"chi_flow={chi_flow}")
                logger.debug(f"chi_metal={chi_metal}")
                logger.debug(f"inc={inc}")
                inc_tol = inc_conf["tolerance"]
                inc_clip = inc_conf.get("clip", 0.5)
                if np.isnan(chi_flow).any():
                    raise Exception(
                        f'NaN stagnation point angle, row {irow}, spf={row["spf"]},'
                        f" chi_flow={chi_flow}, chi_metal={chi_metal}"
                    )
                if (np.abs(inc) > inc_tol).any():
                    inc_converged = False
                dinc = np.clip(inc * rf_inc, -inc_clip, inc_clip)
                if mdot_err > rtol_mdot_inc:
                    dinc *= 0.0
                logger.debug(f"dinc={dinc}")
                qstar_save[irow][:, 0] += dinc

                imax = np.argmax(np.abs(inc.flat))
                inc_prev = np.abs(pdict.get("Inc", inc_target) - inc_target)
                inc_now = np.abs(inc.flat[imax])
                if inc_now > inc_prev:
                    pdict["Inc"] = inc.flat[imax] + inc_target
                    pdict["DInc"] = dinc.flat[imax]

    dev_converged = True
    if dev_conf := conf.iterate.get("deviation"):
        rf_dev = dev_conf.get("relaxation_factor", 0.5)
        dev_clip = dev_conf.get("clip", 2.0)
        for irow, row in enumerate(conf.sections):
            if row:
                yaw_actual = ml_out.Alpha_rel[irow * 2 + 1]
                yaw_target = ml.Alpha_rel[irow * 2 + 1]
                dev = yaw_actual - yaw_target
                if (np.abs(dev) > dev_conf["tolerance"]).any():
                    dev_converged = False
                ddev = -np.clip(dev * rf_dev, -dev_clip, dev_clip)
                qstar_save[irow][:, 1] += ddev
                pdict["Dev"] = np.atleast_1d(dev)[0]
                pdict["DDev"] = np.atleast_1d(ddev)[0]

    # Update qstar post-optimisation
    for irow, row in enumerate(conf.sections):
        if row:
            row.pop("q_camber", None)
            row["qstar_camber"] = qstar_save[irow].tolist()

    opt_converged = (
        dev_converged and inc_converged and mean_line_converged
    ) or conf.solver.get("skip")

    if conf.iterate:
        log_line(pdict, log_fields)

    if opt_converged and not chic_flag:
        out_vars = meanline_design.inverse(ml_out)
        out_vars.pop("So1")

        var_fields = ("Design variable", "Nom   ", "CFD   ")
        log_line(None, var_fields)
        log_line("-", var_fields)
        for v in conf.mean_line:
            log_line(
                {
                    "Design variable": v,
                    "Nom   ": conf.mean_line[v],
                    "CFD   ": out_vars[v],
                },
                var_fields,
            )

        logger.iter(f"eta_tt={ml_out.eta_tt:.3f}, eta_ts={ml_out.eta_ts:.3f}")

        inverse_path = os.path.join(workdir, "inverse.yaml")
        turbigen.util.write_yaml(out_vars, inverse_path)
        logger.debug(f"Wrote inversion to {inverse_path}")

    # Write out the nominal and actual mean lines
    actual_ml_path = os.path.join(workdir, "mean_line_actual.yaml")
    ml_out.mean_line_type = conf.mean_line_type
    ml_out.write(actual_ml_path)

    logger.info(f"Elapsed time {mins:.2f} min.")

    sys.stdout.flush()

    return ml_out, opt_converged, g


def run(conf, plot=False):
    basedir = conf.workdir

    if conf.hypercube:
        basedir = conf.workdir
        conf.database["conf_path"] = os.path.join(basedir, "config_db.yaml")
        conf.database["mean_line_path"] = os.path.join(basedir, "mean_line_db.yaml")
        conf.workdir = None

        if not conf.job:
            raise ConfigError("Need job submission configured to run a hypercube.")

        if conf.hypercube.get("N"):
            logger.iter("Running a hypercube...")
            cs = conf.sample_hypercube()
            Nrunmax = conf.hypercube.get("max_jobs", 0)
            slurm.submit_array(cs, basedir, Nrunmax)

        if conf.hypercube.get("Nedge"):
            logger.iter("Running hypercube edges...")
            ce = conf.sample_hyperfaces()
            Nrunmax = conf.hypercube.get("max_jobs", 0)
            slurm.submit_array(ce, basedir, Nrunmax)

        return True

    if conf.job:
        slurm.submit(conf)
        return True

    topt_start = timer()

    # If specified use database to fill in values
    if conf.database.get("conf_path"):
        conf.interpolate_from_database()

    if conf.iterate:
        gguess = None

        max_iter = conf.iterate.get("max_iter", 20)
        logger.iter(f"Iterating for max_iter={max_iter} iterations")

        log_fields = LOG_FIELDS + ()
        if mean_line_opt_conf := conf.iterate.get("mean_line"):
            match_vars = mean_line_opt_conf.get("match_tolerance", {})
            for v in match_vars:
                log_fields += (v,)
                log_fields += ("D" + v,)
        log_line(None, log_fields)
        log_line("-", log_fields)

        for i in range(max_iter):
            iterdir = os.path.join(basedir, "%04d" % i)
            os.makedirs(iterdir, exist_ok=True)
            conf.workdir = iterdir
            # Disable soft start once we have a good initial guess
            if i > 0 and ("soft_start" in conf.solver):
                conf.solver["soft_start"] = False
            ml_out, opt_converged, gguess = run_single(conf, gguess)
            if opt_converged:
                logger.debug("Moving converged solution up to work dir")
                for f in os.listdir(iterdir):
                    src_path = os.path.join(iterdir, f)
                    dest_path = os.path.join(basedir, f)
                    logger.debug(src_path + "->" + dest_path)
                    shutil.move(src_path, dest_path)
                logger.debug("Deleting iterations")
                for j in range(i + 1):
                    del_path = os.path.join(basedir, "%04d" % j)
                    shutil.rmtree(del_path)

                # Update the guess file loation
                if old_guess_path := conf.solver.get("guess_file"):
                    old_guess_file = os.path.basename(old_guess_path)
                    new_guess_path = os.path.join(basedir, old_guess_file)
                    conf.solver["guess_file"] = new_guess_path

                conf.workdir = basedir
                conf.write(os.path.join(basedir, "config_conv.yaml"))

                # Rename the meanline
                old_ml_path = os.path.join(basedir, "mean_line_actual.yaml")
                new_ml_path = os.path.join(basedir, "mean_line_actual_conv.yaml")
                shutil.move(old_ml_path, new_ml_path)

                topt_end = timer()
                opt_mins = (topt_end - topt_start) / 60.0
                logger.iter(f"Iteration converged in {opt_mins:.1f} min.")

                break

            elif i == max_iter // 2:
                # Reduce relaxation factors halfway through
                logger.iter("Reducing relaxation factors")
                for k in ["incidence", "mean_line"]:
                    try:
                        conf.iterate[k]["relaxation_factor"] *= 0.5
                    except KeyError:
                        pass

            elif i == (max_iter * 3) // 4:
                # Reduce relaxation factors halfway through
                logger.iter("Reducing relaxation factors")
                for k in ["incidence", "mean_line"]:
                    try:
                        conf.iterate[k]["relaxation_factor"] *= 0.5
                    except KeyError:
                        pass

    else:
        ml_out, _, gguess = run_single(conf, plot=plot)
        opt_converged = True

    if not opt_converged:
        raise Exception("Iteration did not converge to specified tolerances")

    # Now run chic if requested
    if "mass_step" in conf.operating_point:
        # Disable optimisations to fix the geometry
        conf.iterate = {}

        chic_path = os.path.join(basedir, "chic.json")

        mdot_des = ml_out.mdot[0]
        logger.iter(f"Design mass flow mdot_des={mdot_des}")

        chic = {"mass": [0.0], "eta": [ml_out.eta_tt], "PR": [ml_out.PR_tt]}

        logger.iter("Now running characteristic.")

        logger.iter("Into stall...")

        np.set_printoptions(precision=3)

        mass_low = -1.0
        mass_high = 0.0
        i = 0
        gref = gguess
        dm_min = conf.operating_point["mass_step"]

        # Relax convergence checking
        conf.solver["rtol_mdot"] = 0.1
        conf.solver["atol_eta"] = 1.0

        while True:
            mass_new = 0.5 * (mass_low + mass_high)
            if (mass_high - mass_low) < (2.0 * dm_min):
                logger.iter(f"Stalled, mhigh={mass_high}, mlow={mass_low}")
                mdot_stall = (0.5 * (mass_high + mass_low) + 1.0) * mdot_des
                logger.iter(
                    f"Dimensional stalling mass flow rate, mdot_stall={mdot_stall}"
                )
                ml_out.mdot_stall = mdot_stall
                break

            logger.iter(f"mass_now={mass_new:.2f}: ")
            iterdir = os.path.join(basedir, "chic_%04d" % i)
            i += 1
            os.makedirs(iterdir, exist_ok=True)
            conf.workdir = iterdir
            conf.solver["mass_adjust"] = mass_new
            conf.solver["soft_start"] = True
            try:
                ml, _, gguess = run_single(conf, gref)
                ml_out.mdot_stall = ml.mdot[0]
                # If we do not stall, replace high mass with current
                mass_high = mass_new
                logger.iter(
                    f"PR={ml.PR_tt:.3f}, eta={ml.eta_tt:.3f}, mdot_norm={mass_new:.3f}",
                )
                chic["mass"].append(ml.mdot[0] / mdot_des - 1.0)
                chic["eta"].append(ml.eta_tt)
                chic["PR"].append(ml.PR_tt)

            except ConvergenceError as e:
                logger.iter(e)
                # If we stall, replace low mass with current
                mass_low = mass_new

        logger.iter("Into choke...")
        gguess = gref
        mass_low = 0.0
        mass_high = 1.0
        while True:
            mass_new = 0.5 * (mass_low + mass_high)
            if (mass_high - mass_low) < (2.0 * dm_min):
                logger.iter("Choked.")
                ml_out.mdot_choke = (0.5 * (mass_high + mass_low) + 1.0) * mdot_des
                break

            logger.iter(f"mass_now={mass_new:.2f}: ")
            iterdir = os.path.join(basedir, "chic_%04d" % i)
            i += 1
            os.makedirs(iterdir, exist_ok=True)
            conf.workdir = iterdir
            conf.solver["mass_adjust"] = mass_new
            try:
                ml, _, gguess = run_single(conf, gref)
                ml_out.mdot_choke = ml.mdot[0]
                mnorm_now = ml.mdot[0] / mdot_des - 1.0
                if (mass_new - mnorm_now) > 0.05:
                    logger.iter(f"mdot error {mass_new - mnorm_now}")
                    mass_high = mass_new
                    continue
                # If we do not choke, replace low mass with current
                mass_low = mass_new
                logger.iter(
                    f"PR={ml.PR_tt:.3f}, eta={ml.eta_tt:.3f}, mdot_norm={mass_new:.3f}",
                )
                chic["mass"].append(mnorm_now)
                chic["eta"].append(ml.eta_tt)
                chic["PR"].append(ml.PR_tt)
            except ConvergenceError:
                # If we choke, replace high mass with current
                mass_high = mass_new

        # Sort by mass flow
        isort = np.argsort(chic["mass"])
        for k, v in chic.items():
            chic[k] = np.take(chic[k], isort).tolist()

        with open(chic_path, "w") as f:
            json.dump(chic, f)

        # Rewrite output mean line with mdot stall/choke added
        actual_ml_path = os.path.join(basedir, "mean_line_actual.yaml")
        ml_out.write(actual_ml_path)

    # If specified, add to a database
    if conf.database.get("conf_path") and not conf.database.get("read_only", False):
        conf.write(os.path.abspath(conf.database["conf_path"]), mode="a")

    # If specified save mean-line data
    if conf.database.get("mean_line_path"):
        ml_out.write(os.path.abspath(conf.database["mean_line_path"]), mode="a")

    return opt_converged

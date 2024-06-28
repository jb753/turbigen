"""Functions to run turbigen on config objects."""
import os
import sys
import shutil
from turbigen import (
    fluid,
    grid,
    util,
    geometry,
    slurm,
    hmesh,
    ohmesh,
)
from turbigen.exceptions import ConfigError
import turbigen.post_process
import turbigen.plot
import turbigen.average
import turbigen.annulus
import numpy as np
from timeit import default_timer as timer
from scipy.optimize import minimize
from scipy.spatial import KDTree


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
        w = max(len(v), 5)
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

    if not conf.annulus:
        raise ConfigError("No annulus configuration; quitting.")

    # Feed annulus arguments to the geometry function
    times.append(timer())
    conf._check_annulus()
    annulus_type = conf.annulus.pop("type", "Smooth")
    Annulus = util.load_annulus(annulus_type)
    ann = Annulus(ml.rmid, ml.span, ml.Beta, **conf.annulus)
    conf.annulus["type"] = annulus_type
    logger.info(ann)
    times.append(timer())
    logger.debug(f"Annulus design took {np.diff(times)[-1]:.1f}s")

    cut_offset = conf.solver.pop("cut_offset", None)
    xr_cut = ann.get_cut_planes(cut_offset)

    # Include deviations angles with respect to free vortex in camber
    # parameters to make q_camber
    qstar_save = []
    qcamber_save = []
    chi_save = []
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
                logger.debug(f"Vortex exponent irow={irow} is {vexpon_row}")
                Alpha_rel = ml.Alpha_rel_free_vortex(row["spf"], vexpon_row)[:, ind]
            chi_save.append(Alpha_rel)
            Chi = Alpha_rel + qstar_camber[:, :2]
            if np.any(np.abs(Chi) > 90.0):
                raise Exception(
                    f"Cannot set a blade angle over 90 degrees! Row {irow} Chi={Chi}"
                )
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
    fit_data = conf.blades.get("fit", None)
    theta_off = conf.blades.get("theta_offset", np.zeros((conf.nrow,)))
    fit_flag = False
    for irow, row in enumerate(conf.sections):
        if row:
            row_now = row.copy()
            row_now.pop("chi", None)
            vexpon = row_now.pop("vortex_expon", None)
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

            bld_now = geometry.Blade(
                streamsurface=ann.xr_row(irow),
                mstack=mstack[irow],
                thick_type=thick_type[irow],
                camber_type=camber_type[irow],
                theta_offset=theta_off[irow],
                **row_now,
            )

            if fit_data:
                if fit_data_path := fit_data[irow]:
                    fit_flag = True

                    # Read coordinates of all sections
                    xrrt_target_all = turbigen.util.read_sections(fit_data_path)
                    nsect_dat = len(xrrt_target_all)
                    nsect_conf = len(bld_now.spf)
                    if not nsect_dat == nsect_conf:
                        raise Exception(
                            f"Mismatching number of sections to fit, "
                            f"{nsect_conf} in the config and "
                            f"{nsect_dat} in the coordinates"
                        )

                    # Locate the span fractions at which to fit
                    m = np.linspace(0.0, 1.0)
                    spf_fit = []
                    for xrrt_target in xrrt_target_all:

                        xrfit = xrrt_target[:2]

                        def eval_spf_err(spfnow, xrfit):

                            xrref = bld_now.streamsurface(spfnow, m)
                            if np.ptp(xrfit[0]) > np.ptp(xrfit[1]):
                                xrfit = xrfit[:, np.argsort(xrfit[0])]
                                xrint = np.stack(
                                    (xrref[0], np.interp(xrref[0], *xrfit))
                                )
                            else:
                                xrfit = xrfit[:, np.argsort(xrfit[1])]
                                xrint = np.stack(
                                    (
                                        np.interp(
                                            xrref[1],
                                            *xrfit[
                                                (1, 2),
                                            ],
                                        ),
                                        xrref[1],
                                    )
                                )

                            err = np.sqrt(np.mean((xrint - xrref) ** 2.0))
                            return err

                        spf_good = minimize(eval_spf_err, 0.5, args=(xrfit,)).x[0]
                        spf_fit.append(spf_good)

                    spf_fit = np.array(spf_fit)

                    # Now assemble a KDTree to look up distances from fitted
                    # surface to nearest target coordinate
                    trees = [
                        KDTree(
                            xrrt_target_all[isect][
                                (0, 2),
                            ].T
                        )
                        for isect in range(nsect_dat)
                    ]

                    for _ in range(1):

                        for isect in range(len(spf_fit)):

                            logger.info(
                                f"Fitting row {irow} at spf={spf_fit[isect]:.3f} "
                                f"to coordinates {fit_data[irow]} ..."
                            )

                            def eval_fit_err(q, tree, spf, bldi, isect):

                                bldi.set_pvec(q, isect)

                                # Get fitted surface coords
                                xrtul = np.concatenate(
                                    bldi.evaluate_section(spf, nchord=50), axis=-1
                                )
                                xrtul[2] *= xrtul[1]
                                xrtul = xrtul[
                                    (0, 2),
                                ]

                                # Lookup shortest distances to target coords
                                dist, _ = tree.query(xrtul.T)

                                return np.sqrt(np.mean(dist**2))

                            q0 = bld_now.get_pvec(isect)
                            bnd = bld_now.get_bound(isect)
                            opts = {"maxiter": 1000, "fatol": 1e-9, "xatol": 1e-9}
                            minimize(
                                eval_fit_err,
                                q0,
                                args=(trees[isect], spf_fit[isect], bld_now, isect),
                                method="Nelder-Mead",
                                bounds=bnd,
                                options=opts,
                            )

                    # Convert the tanChi camber parameters to recamber
                    Chi = np.degrees(np.arctan(bld_now.q_camber[:, :2]))
                    qstar_save[irow][:, :2] = Chi - chi_save[irow]
                    qstar_save[irow][:, 2:] = bld_now.q_camber[:, 2:]

            bld.append(bld_now)

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
                    logger.debug(f'main q_camber {row_now["q_camber"][isect]}')
                    logger.debug(
                        f'main q_camber deg {util.atand(row_now["q_camber"][isect])}'
                    )

                    # Fill in tanChi for the splitter after recamber
                    splitter_now["q_camber"][isect][:2] = util.tand(
                        chi_main + splitter_now["q_camber"][isect][:2]
                    )
                    logger.debug(f'splitter q_camber {splitter_now["q_camber"][isect]}')
                    logger.debug(
                        "splitter q_camber deg "
                        f'{util.atand(splitter_now["q_camber"][isect])}'
                    )

                    # The relative mstack for splitter is same as for main blade.
                    # i.e. if LE for main, splitter sections stacked on splitter LE
                    # i.e. if TE for main, splitter sections stacked on splitter TE
                    # i.e. if mid-chord for main, splitter stacked on splitter mid-chord
                    mstack_splitter = mstack[irow]

                    # Calculate the angular offset to put splitter on the main
                    # camber line at splitter stacking location
                    mref[isect] = mstack_splitter * np.ptp(mlim_sect) + mlim_sect[0]

                    mq = np.linspace(0.0, 1.0, 101)
                    xrtc = np.mean(bld[irow].evaluate_section(spf_sect, m=mq), axis=0)
                    tmain[isect] = np.interp(mref[isect], mq, xrtc[2])

                splitter.append(
                    geometry.Blade(
                        streamsurface=ann.xr_row(irow),
                        mstack=np.mean(mref),
                        thick_type=thick_type[irow],
                        camber_type=camber_type[irow],
                        theta_offset=np.mean(tmain),
                        **splitter_now,
                    )
                )

                splitter_now.pop("q_camber")
                splitter_now["qstar_camber"] = qstar_camber_split_save
                if vexpon is not None:
                    row_now["vexpon"] = vexpon
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
    Re_surf = np.array(ell / ml.L_visc).astype(float)
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

    s = 2.0 * np.pi * row_rmid[ind_out] / Nb[ind_out]
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
                    turbigen.plot.plot_splitter(b, splitter[ib], Nb[ib], fname_split)

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

    # At this point, we have the geometry and mean-line set up
    # We can now generate the mesh

    # Restore the relative camber
    for irow, row in enumerate(conf.sections):
        if row:
            row.pop("q_camber", None)
            row["qstar_camber"] = qstar_save[irow].tolist()
            row["q_thick"] = bld[irow].q_thick.tolist()

    # Write out the fitted sections
    if fit_flag:
        conf.blades["theta_offset"] = [b.theta_offset for b in bld]
        conf.blades.pop("fit", None)
        conf.write(os.path.join(workdir, "config.yaml"))

    # Set row, hub, casing spacings using yplus and flat-plate correlations
    yplus = np.atleast_2d(conf.mesh["yplus"]).T
    Cf = (2.0 * np.log10(Re_surf) - 0.65) ** -2.3
    tauw = Cf * 0.5 * (ml.rho_ref * ml.V_ref**2.0)
    Vtau = np.sqrt(tauw / ml.rho_ref)
    Lvisc = np.atleast_2d((ml.mu_ref / ml.rho_ref) / Vtau)
    drow = yplus * Lvisc
    # drow has dimensions: [LE/TE, irow]
    dhub = np.nanmean(drow)
    dcas = np.nanmean(drow)
    # Indicator for unbladed rows
    # ind_out = [True if b else False for b in bld]
    unbladed = [True if not b else False for b in bld]
    # At this point, we have the geometry and mean-line set up
    # We can now generate the mesh
    mesh_type = conf.mesh["type"]

    mesh_settings = conf.mesh.copy()
    mesh_settings.pop("yplus")
    mesh_settings.pop("type")
    slip_hub_inlet = mesh_settings.pop("slip_hub_inlet", False)
    check_coords = mesh_settings.pop("check_coords", True)
    if not check_coords:
        logger.info(
            "Be careful: the mesh coordinate check is disabled in the input file"
        )

    times.append(timer())

    if mesh_type == "h":
        # Apply settings from yaml file to the default config
        hmesh_config = hmesh.HMeshConfig(**mesh_settings)
        # Make the grid object
        g = hmesh.make_grid(mac, hmesh_config, dhub, dcas, drow, unbladed)

        if conf.plot:
            turbigen.plot.plot_hmesh(g, workdir)

    elif mesh_type == "oh":
        tips *= 0.5 * (ml.span[::2] + ml.span[1::2])
        # Apply settings from yaml file to the default config
        ohmesh_config = ohmesh.OHMeshConfig(**mesh_settings)
        ohmesh_config.workdir = workdir

        Omega = ml.Omega[::2]

        # Make the grid object
        g = ohmesh.make_grid(mac, ohmesh_config, dhub, dcas, drow, unbladed, Omega)

    else:
        raise Exception(f'Unrecognised mesh type "{mesh_type}"')

    times.append(timer())
    logger.debug(f"Mesh generation took {np.diff(times)[-1]:.1f}s")
    logger.info(f"Mesh Npts/10^6={g.ncell/1e6:.2f}")

    # Make zero-radius rods inviscid
    if slip_hub_inlet:
        bi = g.inlet_patches[0].block
        drhub = np.diff(bi[:, 0, 0].r)
        inose = np.where(drhub > 1e-6)[0][0]
        bi.add_patch(grid.InviscidPatch(i=(0, inose), j=0))

    # Ready to apply boundary conditions now
    logger.info("Applying boundary conditions...")

    # Wall rotations
    rot_types = []

    rpm_adjust = conf.operating_point.get("rpm_adjust", 0.0)
    if rpm_adjust:
        logger.info(f"Running off-design: adjusted rpms by {rpm_adjust:+}")
    ml.Omega *= 1.0 + rpm_adjust

    for Omi, tip in zip(ml.Omega[::2], mac.tip):
        if Omi:
            if tip:
                rot_types.append("tip_gap")
            else:
                rot_types.append("shroud")
        else:
            rot_types.append("stationary")

    # OH meshes just skip unbladed rows, so we need to remove rotation
    # information from unbladed rows
    if mesh_type == "oh":
        Omega_trim = []
        rot_types_trim = []
        for irow, Omi in enumerate(ml.Omega[::2]):
            if ind_out[irow]:
                rot_types_trim.append(rot_types[irow])
                Omega_trim.append(Omi)
        rot_types = rot_types_trim
        Omega = Omega_trim
    else:
        Omega = ml.Omega[::2]

    g.apply_rotation(rot_types, Omega)

    if "Beta1_override" in conf.solver:
        Beta1 = conf.solver.pop("Beta1_override")
        g.apply_inlet(So1, ml.Alpha[0], Beta1)
    else:
        Beta1 = ml.Beta[0]

    # # Inlet and outlet
    g.apply_inlet(So1, ml.Alpha[0], Beta1)
    g.apply_outlet(ml.P[-1])

    # Configure throttle
    mass_adjust = conf.operating_point.get("mass_adjust", 0.0)
    throttle_pid = conf.operating_point.get("mdot_pid")
    if mass_adjust and not throttle_pid:
        raise Exception(
            "Cannot adjust mass flow rate without exit throttle PID: "
            "set `mdot_pid` in the operating point configuration."
        )

    if mass_adjust:
        logger.info(f"Running off-design: adjusted mass flow rate by {mass_adjust:+}")

    if throttle_pid:
        restart_fac = 0.5 if gguess else 1.0
        norm_fac = np.ptp(ml.P) / ml.mdot[-1]
        g.apply_throttle(
            ml.mdot[-1] * (1.0 + mass_adjust),
            np.array(throttle_pid) * norm_fac * restart_fac,
        )

    # Choose whether the blocks are real or perfect
    if isinstance(So1, fluid.PerfectState):
        g = grid.Grid([b.to_perfect() for b in g])
    elif isinstance(So1, fluid.RealState):
        g = grid.Grid([b.to_real() for b in g])
    else:
        raise Exception("Unrecognised inlet state type")

    logger.info("Setting intial guess...")

    # Crude guess (may be updated later if arg gguess is supplied
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
        conf.solver["workdir"] = solve_workdir = os.path.join(workdir, "solve")
        if not os.path.exists(solve_workdir):
            os.makedirs(solve_workdir, exist_ok=True)

    # The grid is ready to run. At this point, we can 'install' it
    if conf.install:
        install_type = conf.install.pop("type")
        # Dynamically load the install module
        logger.info(f"Installing a {install_type}...")

        install_module = turbigen.util.load_install(install_type)

        logger.debug("Successfully imported.")
        gi = install_module.forward(g, mac, ml, **conf.install)

        if check_coords:
            gi.check_coordinates()

        if gguess:
            gi.apply_guess_3d(gguess)
            if throttle_pid:
                gi.update_outlet(rf=0.5)

        if conf.solver:
            logger.info(f'Running solver {conf.solver["type"]} on installed...')
            gi.run(conf.solver, mac)
            conf.solver.pop("workdir")
        else:
            logger.info("No solver specified, continuing with initial guess...")

        logger.info("Uninstalling...")
        g, install_inverse = install_module.inverse(gi)

        gguess = gi

        conf.install["type"] = install_type

    else:

        if check_coords:
            g.check_coordinates()

        if gguess:
            g.apply_guess_3d(gguess)
            if throttle_pid:
                g.update_outlet()

        if conf.solver:
            if conf.solver.get("type"):
                logger.info(f'Running solver {conf.solver["type"]}...')
                g.run(conf.solver, mac)
                conf.solver.pop("workdir")
        else:
            logger.info("No solver specified, continuing with initial guess...")

        gguess = g

    if cut_offset is not None:
        conf.solver["cut_offset"] = cut_offset

    if not np.isclose(Beta1, ml.Beta[0]):
        conf.solver["Beta1_override"] = Beta1

    logger.info("Post-processing...")

    times.append(timer())

    Cmix = []
    Amix = []
    Dsmix = []
    for icut, xrci in enumerate(xr_cut):
        try:
            CC = g.unstructured_cut_marching(xrci)
            Cnow, Aannnow, dsnow = turbigen.average.mix_out_unstructured(CC)
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

    postdir = os.path.join(workdir, "post")
    if not os.path.exists(postdir):
        os.makedirs(postdir, exist_ok=True)

    for post_name, post_conf in conf.post_process.items():
        logger.debug(f"Running post function {post_name}")
        post_func = util.load_post(post_name).post
        if post_conf is None:
            post_conf = {}
        post_func(g, mac, ml_out, postdir, **post_conf)

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

    out_vars = meanline_design.inverse(ml_out)
    if conf.install:
        out_vars.update(install_inverse)

    mean_line_converged = True
    if mean_opt_conf := conf.iterate.get("mean_line"):
        rf_mean = mean_opt_conf.get("relaxation_factor", 0.5)

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

        # Evaluate incidence
        fac_Rle = inc_conf.get("fac_RLE", 1.0)
        data = turbigen.util.incidence(g, mac, ml, fac_Rle)

        # Extract configuration parameters
        rf_inc = inc_conf.get("relaxation_factor", 0.2)
        rtol_mdot_inc = inc_conf.get("rtol_mdot", 0.05)
        mdot_err = np.abs(ml_out.mdot / ml.mdot - 1)[0]
        inc_target = inc_conf.get("target", 0.0)
        inc_tol = inc_conf["tolerance"]
        inc_clip = inc_conf.get("clip", 0.5)

        for irow, row in enumerate(conf.sections):
            logger.debug(f"CORRECTING INCIDENCE, row {irow}")
            if row:

                spf, inc = data[irow][0][:2]

                inc -= inc_target
                inc = np.interp(row["spf"], spf, inc)

                if (np.abs(inc) > inc_tol).any():
                    inc_converged = False

                dinc = np.clip(inc * rf_inc, -inc_clip, inc_clip)

                if mdot_err > rtol_mdot_inc:
                    dinc *= 0.0
                qstar_save[irow][:, 0] += dinc

                imax = np.argmax(np.abs(inc.flat))
                inc_prev = np.abs(pdict.get("Inc", inc_target) - inc_target)
                inc_now = np.abs(inc.flat[imax])
                if inc_now > inc_prev:
                    logger.debug(f"New maximum inc={inc.flat[imax] + inc_target}")
                    pdict["Inc"] = inc.flat[imax] + inc_target
                    pdict["DInc"] = dinc.flat[imax]

                if conf.splitter:
                    if splitter_now := conf.splitter[irow]:
                        logger.debug(f"CORRECTING SPLITTER row={irow}")

                        spf, inc = data[irow][1][:2]
                        inc -= inc_target
                        inc = np.interp(splitter_now["spf"], spf, inc)

                        if (np.abs(inc) > inc_tol).any():
                            inc_converged = False

                        dinc_splitter = np.clip(inc * rf_inc, -inc_clip, inc_clip)

                        if mdot_err > rtol_mdot_inc:
                            dinc_splitter *= 0.0

                        qcam_split = np.array(splitter_now["qstar_camber"])
                        qcam_split[:, 0] += dinc_splitter - dinc
                        splitter_now["qstar_camber"] = qcam_split
                        imax = np.argmax(np.abs(inc.flat))
                        inc_prev = np.abs(pdict.get("Inc", inc_target) - inc_target)
                        inc_now = np.abs(inc.flat[imax])
                        if inc_now > inc_prev:
                            logger.debug(
                                "Splitter new maximum inc="
                                f"{inc.flat[imax] + inc_target}"
                            )
                            pdict["Inc"] = inc.flat[imax] + inc_target
                            pdict["DInc"] = dinc_splitter.flat[imax]

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

    if opt_converged:

        # out_vars = meanline_design.inverse(ml_out)
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

    return ml_out, opt_converged, gguess


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

        # Apply the nstep scaling factor
        fac_nstep_initial = conf.iterate.get("fac_nstep_initial", 1.0)
        nstep_old = conf.solver["nstep"]
        conf.solver["nstep"] = int(fac_nstep_initial * nstep_old)

        for i in range(max_iter):
            iterdir = os.path.join(basedir, "%04d" % i)
            os.makedirs(iterdir, exist_ok=True)
            conf.workdir = iterdir

            # Disable soft start once we have a good initial guess
            if i > 0 and ("soft_start" in conf.solver):
                conf.solver["soft_start"] = False
            ml_out, opt_converged, gguess = run_single(conf, gguess)

            # Reset nstep
            conf.solver["nstep"] = nstep_old

            # Check for stopit to interrupt iterations
            stopit_path = os.path.join(basedir, "stopit")
            if os.path.exists(stopit_path):
                logger.iter("stopit found, terminating iterations.")
                opt_converged = True

                meanline_design = util.load_mean_line(conf.mean_line_type)
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

                os.remove(stopit_path)

            if opt_converged:

                if not conf.solver.get("skip"):
                    logger.debug("Moving converged solution up to work dir")
                    for f in os.listdir(iterdir):
                        src_path = os.path.join(iterdir, f)
                        dest_path = os.path.join(basedir, f)
                        logger.debug(src_path + "->" + dest_path)
                        if os.path.isdir(dest_path):
                            shutil.rmtree(dest_path)
                        elif os.path.exists(dest_path):
                            os.remove(dest_path)
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

                    # Rename the meanline
                    old_ml_path = os.path.join(basedir, "mean_line_actual.yaml")
                    new_ml_path = os.path.join(basedir, "mean_line_actual_conv.yaml")
                    shutil.move(old_ml_path, new_ml_path)

                conf.workdir = basedir
                conf.write(os.path.join(basedir, "config_conv.yaml"))

                topt_end = timer()
                opt_mins = (topt_end - topt_start) / 60.0
                logger.iter(f"Iteration finished in {opt_mins:.1f} min.")

                break

    else:
        ml_out, _, gguess = run_single(conf, plot=plot)
        opt_converged = True

    if not opt_converged:
        raise Exception("Iteration did not converge to specified tolerances")

    # If specified, add to a database
    if conf.database.get("conf_path") and not conf.database.get("read_only", False):
        conf.write(os.path.abspath(conf.database["conf_path"]), mode="a")

    # If specified save mean-line data
    if conf.database.get("mean_line_path"):
        ml_out.write(os.path.abspath(conf.database["mean_line_path"]), mode="a")

    return opt_converged

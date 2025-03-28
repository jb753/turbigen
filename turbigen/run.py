"""Functions to run turbigen on config objects."""

import os
import shutil
import sys
from timeit import default_timer as timer
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

import turbigen.annulus
import turbigen.average
import turbigen.flowfield
import turbigen.iterators
import turbigen.yaml
import turbigen.nblade
from turbigen import (
    fluid,
    grid,
    util,
    util_post,
    geometry,
    hmesh,
    ohmesh,
)
from turbigen.exceptions import ConfigError

warnings.simplefilter("error", RuntimeWarning)

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
                elif isinstance(d[v], (list, tuple)):
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


def run_single(conf, gguess=None):
    """Run turbigen on a config object."""

    times = [
        timer(),
    ]

    # Inlet state
    logger.debug("Getting inlet state...")
    So1 = conf.get_inlet()
    logger.info(f"Inlet: {So1}")

    # Dynamically load the design functions based on machine type in config
    if not conf.mean_line_type:
        raise ConfigError("No mean-line type specified; quitting.")
    logger.info(f"Designing a {conf.mean_line_type} mean line...")

    meanline_design = util.load_mean_line(conf.mean_line_type)

    # Check for the meanline debug flag
    meanline_debug = conf.mean_line.pop("debug", False)

    # Feed mean-line arguments to the function
    times.append(timer())
    ml = meanline_design.forward(So1=So1, **conf.mean_line)
    times.append(timer())
    logger.debug(f"Mean-line design took {np.diff(times)[-1]:.1f}s")
    logger.info(ml)
    if not ml.check():
        ml.show_debug()
        raise Exception(
            "Mean-line conservation checks failed, have printed debugging information"
        ) from None
    elif meanline_debug:
        logger.iter("Mean-line debugging requested...")
        logger.iter("Design variables:")
        for k, v in conf.mean_line.items():
            logger.iter(f"{k}: {v}")
        logger.iter("Flow field:")
        ml.show_debug()
        sys.exit(0)

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
        logger.debug(f"Checking {v}")
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

    # Warn for very high flow angles
    if np.abs(ml.Alpha_rel).max() > 85.0:
        logger.warning(
            """WARNING: Relative flow angles are approaching 90 degrees.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
        )

    # Warn for wobbly annulus
    is_radial = np.abs(ml.Beta).max() > 10.0
    is_multirow = conf.nrow > 2
    if is_radial and is_multirow:
        if np.diff(np.sign(np.diff(ml.rrms))).any():
            logger.warning(
                """WARNING: Radii do not vary monotonically.
This suggests a physically-consistent but suboptimal mean-line design
and will cause problems with meshing and solving for the flow field."""
            )

    # Make a working directory
    workdir = conf.workdir
    if not os.path.exists(workdir):
        logger.debug("Making {workdir}...")
        os.makedirs(workdir, exist_ok=True)
    config_yaml_path = os.path.join(workdir, "config.yaml")
    logger.debug(f"Writing {config_yaml_path}...")
    conf.write(config_yaml_path)

    # Backup the source files for later reproduction
    times.append(timer())
    util.save_source_tar_gz(os.path.join(workdir, "src.tar.gz"))
    times.append(timer())
    logger.debug(f"Source backup took {np.diff(times)[-1]:.1f}s")

    postdir = os.path.join(workdir, "post")
    if not os.path.exists(postdir):
        os.makedirs(postdir, exist_ok=True)

    if not conf.annulus:
        raise ConfigError("No annulus configuration; quitting.")

    # Feed annulus arguments to the geometry function
    times.append(timer())
    logger.debug("Checking annulus config...")
    # conf._check_annulus()
    annulus_type = conf.annulus.pop("type", "Smooth")
    annulus_debug = conf.annulus.pop("debug", False)
    logger.info("Designing annulus...")
    Annulus = turbigen.annulus.load_annulus(annulus_type)
    annulus_debug = conf.annulus.pop("debug", False)
    ann = Annulus(ml.rmid, ml.span, ml.Beta, **conf.annulus)
    ann.get_interfaces()

    conf.annulus["type"] = annulus_type
    logger.info(ann)
    times.append(timer())
    logger.debug(f"Annulus design took {np.diff(times)[-1]:.1f}s")

    cut_offset = conf.solver.pop("cut_offset", 0.02)
    xr_cut = ann.get_offset_planes(cut_offset)

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
            if np.any(np.abs(Chi) > 80.0):
                logger.warning(
                    f"WARNING: High blade angles may cause meshing problems: Row {irow} Chi={Chi}"
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
    thick_span = conf.blades.pop("thick_span", None)
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
    fit_mode = conf.blades.get("fit_mode", None)
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
                            "Mismatching number of sections to fit, "
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
                                                (1, 0),
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
                    if fit_mode[irow] == "xrt":
                        xyz_tg = [
                            np.stack(
                                (
                                    xrrt_tg[0],
                                    xrrt_tg[2],
                                )
                            )
                            for xrrt_tg in xrrt_target_all
                        ]
                    elif fit_mode[irow] == "yz":
                        xyz_tg = [
                            np.stack(
                                (
                                    # xrrt_tg[0],
                                    xrrt_tg[1] * np.sin(xrrt_tg[2] / xrrt_tg[1]),
                                    xrrt_tg[1] * np.cos(xrrt_tg[2] / xrrt_tg[1]),
                                )
                            )
                            for xrrt_tg in xrrt_target_all
                        ]
                    else:
                        raise Exception(f"Unrecognised fit_mode: {fit_mode}")

                    trees = [KDTree(xyz_tg[isect].T) for isect in range(nsect_dat)]

                    for _ in range(1):
                        for isect in range(len(spf_fit)):
                            logger.info(
                                f"Fitting row {irow} at spf={spf_fit[isect]:.3f} "
                                f"to coordinates {fit_data[irow]} ..."
                            )

                            def eval_fit_err(q, tree, spf, bldi, isect, plot):
                                bldi.set_pvec(q, isect)

                                # Get fitted surface coords
                                xrtul = np.concatenate(
                                    [
                                        np.concatenate(
                                            bldi.evaluate_section(
                                                spf + dspf, nchord=65
                                            ),
                                            axis=-1,
                                        )
                                        # for dspf in [-0.01, 0., 0.01]],
                                        for dspf in [
                                            0.0,
                                        ]
                                    ],
                                    axis=-1,
                                )

                                if fit_mode[irow] == "xrt":
                                    xyz_ul = np.stack(
                                        (
                                            xrtul[0],
                                            xrtul[2],
                                        )
                                    )

                                elif fit_mode[irow] == "yz":
                                    xyz_ul = np.stack(
                                        (
                                            xrtul[1] * np.sin(xrtul[2]),
                                            xrtul[1] * np.cos(xrtul[2]),
                                        )
                                    )

                                # Lookup shortest distances to target coords
                                # dist, _ = tree.query(xrtul.T)
                                dist, _ = tree.query(xyz_ul.T)
                                dist_rms = np.sqrt(np.mean(dist**2))

                                if plot:
                                    import matplotlib.pyplot as plt

                                    fig, ax = plt.subplots()
                                    ax.axis("equal")
                                    ax.plot(*xyz_ul, "x", color="C0", ms=1)
                                    ax.plot(*xyz_tg[isect], "x", color="C1", ms=1)

                                    # fig, ax = plt.subplots()
                                    # ax.axis("equal")
                                    # rr_ul = np.sqrt((xyz_ul[1:]**2).sum(axis=0))
                                    # rr_tg = np.sqrt((xyz_tg[isect][1:]**2).sum(axis=0))
                                    # ax.plot(xyz_ul[0], rr_ul, "x", color="C0", ms=1)
                                    # ax.plot(xyz_tg[isect][0], rr_tg, "x", color="C1", ms=1)

                                    thick_now = bldi._get_cam_thick(spf)[1]
                                    fig, ax = plt.subplots()
                                    mm = util.cluster_cosine(201)
                                    ax.plot(mm, thick_now.t(mm))
                                    ax.axis("equal")

                                    plt.show()

                                return dist_rms

                            q0 = bld_now.get_pvec(isect)
                            bnd = bld_now.get_bound(isect)
                            opts = {"maxiter": 1000, "fatol": 1e-9, "xatol": 1e-9}
                            res = minimize(
                                eval_fit_err,
                                q0,
                                args=(
                                    trees[isect],
                                    spf_fit[isect],
                                    bld_now,
                                    isect,
                                    False,
                                ),
                                method="Nelder-Mead",
                                bounds=bnd,
                                options=opts,
                            )
                            q0 = res.x
                            for _ in range(3):
                                res = minimize(
                                    eval_fit_err,
                                    q0,
                                    args=(
                                        trees[isect],
                                        spf_fit[isect],
                                        bld_now,
                                        isect,
                                        False,
                                    ),
                                    method="Nelder-Mead",
                                    bounds=bnd,
                                    options=opts,
                                )
                                q0 = res.x

                            # eval_fit_err(
                            #     q0, trees[isect], spf_fit[isect], bld_now, isect, True
                            # )

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
                    logger.debug(f"main q_camber {row_now['q_camber'][isect]}")
                    logger.debug(
                        f"main q_camber deg {util.atand(row_now['q_camber'][isect])}"
                    )

                    # Fill in tanChi for the splitter after recamber
                    splitter_now["q_camber"][isect][:2] = util.tand(
                        chi_main + splitter_now["q_camber"][isect][:2]
                    )
                    logger.debug(f"splitter q_camber {splitter_now['q_camber'][isect]}")
                    logger.debug(
                        "splitter q_camber deg "
                        f"{util.atand(splitter_now['q_camber'][isect])}"
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
    Nb = np.zeros_like(row_rmid)

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
            cx = ann.chords(0.5)[1:-1:2][ind_out][irow]
            s = s_c * cx
            Nb[irow] = np.round(2.0 * np.pi * row_rmid[irow] / s)

    iunbladed = np.where(np.logical_not(ind_out))[0]
    Nb[iunbladed] = Nb[iunbladed - 1]
    if Nb[0] < 1:
        Nb[0] = Nb[1]
    Nb = np.round(Nb).astype(int)

    s = 2.0 * np.pi * row_rmid[ind_out] / Nb[ind_out]
    s_cm = s / ann.chords(0.5)[1:-1:2][ind_out]
    s_cm_min = 0.2
    s_cm_max = 4.0
    if np.any(s_cm < s_cm_min):
        logger.warning("WARNING: narrow blade spacings may cause problems with meshing")
    if np.any(s_cm > s_cm_max):
        logger.warning("WARNING: large blade spacings may cause problems with meshing")
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

    if annulus_debug:
        logger.iter("Annulus debugging requested...")
        from turbigen.post import plot_annulus

        plot_annulus.post(None, mac, None, None, postdir)
        sys.exit(0)

    # At this point, we have the geometry and mean-line set up
    # We can now generate the mesh
    if not conf.mesh:
        logger.iter("Cannot proceed further without mesh configuration, quitting.")
        sys.exit(1)

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
    yplus = np.atleast_2d(conf.mesh.get("yplus", 30.0)).T
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

    times.append(timer())

    if mesh_type == "h":
        # Apply settings from yaml file to the default config
        hmesh_config = hmesh.HMeshConfig(**mesh_settings)
        # Make the grid object
        g = hmesh.make_grid(mac, hmesh_config, dhub, dcas, drow, unbladed)

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

    if not check_coords:
        logger.info(
            "Be careful: the mesh coordinate check is disabled in the input file"
        )
    else:
        g.check_coordinates()

    times.append(timer())
    logger.debug(f"Mesh generation took {np.diff(times)[-1]:.1f}s")
    logger.info(f"Mesh Npts/10^6={g.ncell / 1e6:.2f}")

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

    PR = conf.operating_point.get("PR_ts", None)
    Pout = ml.P[-1]
    if PR is not None:
        Pout = ml.Po[0] * PR
        logger.info(
            f"Running off-design: setting total-static pressure ratio PR_ts={PR}"
        )

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

    # Set inlet pitch angle using orientation of
    # the inlet patch grid (assuming on a constant i face)
    # This allow the annulus lines to differ from mean-line pitch angle
    Ain = g.inlet_patches[0].get_cut().dAi.sum(axis=(-1, -2, -3))
    Beta1 = np.degrees(np.arctan2(Ain[1], Ain[0]))

    # # Inlet and outlet
    g.apply_inlet(So1, ml.Alpha[0], Beta1)
    g.apply_outlet(Pout)

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

    # Reduce the pid constants on restart to prevent instability
    if throttle_pid:
        restart_fac_default = [0.5, 1.0, 1.0]
        restart_fac = (
            conf.operating_point.get("restart_fac", restart_fac_default)
            if gguess
            else 1.0
        )
        norm_fac = np.ptp(ml.P) / ml.mdot[-1]
        g.apply_throttle(
            ml.mdot[-1] * (1.0 + mass_adjust),
            np.array(throttle_pid) * norm_fac * np.array(restart_fac),
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

    convergence = None
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
                gi.update_outlet()

        if conf.solver:
            logger.info(f"Running solver {conf.solver['type']} on installed...")
            convergence = gi.run(conf.solver, mac)
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
                logger.info(f"Running solver {conf.solver['type']}...")
                convergence = g.run(conf.solver, mac)
                conf.solver.pop("workdir")
        else:
            logger.info("No solver specified, continuing with initial guess...")

        gguess = g

    if cut_offset is not None:
        conf.solver["cut_offset"] = cut_offset

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

    for post_name, post_conf in conf.post_process.items():
        logger.debug(f"Running post function {post_name}")
        post_func = util.load_post(post_name).post
        if post_conf is None:
            post_conf = {}
        post_func(g, mac, ml_out, convergence, postdir, **post_conf)

    # Save some 3D geometry into the meanline for later design space fitting
    ml_out.Co = conf.blades.get("Co")
    ml_out.Lsurf = ell
    ml_out.tip = tips[0]
    ml_out.Ds_mix = np.array(Dsmix)

    # Save the workdir so we can cross-reference if the output ml is added to the database
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

    pdict = {"Min": mins}

    out_vars = meanline_design.inverse(ml_out)
    if conf.install:
        out_vars.update(install_inverse)

    # Adjust the mean-line if the config requests it
    mean_line_converged = iter_mean_line(conf, out_vars, pdict)

    inc_converged = True
    if inc_conf := conf.iterate.get("incidence"):
        # Extract configuration parameters
        rf_inc = inc_conf.get("relaxation_factor", 0.2)
        rtol_mdot_inc = inc_conf.get("rtol_mdot", 0.05)
        mdot_err = np.abs(ml_out.mdot / ml.mdot - 1)[-1]
        inc_target = inc_conf.get("target", 0.0)
        inc_tol = inc_conf["tolerance"]
        inc_clip = inc_conf.get("clip", 0.5)
        atol_chi = inc_conf.get("atol_chi", 0.1)

        inc_history = inc_conf.get("history", None)

        # Preallocate for a new step in the incidence history
        inc_history_new = []
        for irow, row in enumerate(conf.sections):
            logger.debug(f"CORRECTING INCIDENCE, row {irow}")
            if row:
                chi = turbigen.util.incidence_unstructured(g, mac, ml, irow, row["spf"])

                inc = np.atleast_1d(np.diff(chi[0], axis=0).squeeze())

                inc -= inc_target
                inc_now = np.stack((qstar_save[irow][:, 0], inc))[None, ...]

                # Drop the relaxation factor if we are very near
                # to the tolerance
                if (np.abs(inc) < inc_tol * 1.5).all():
                    fac_close = 0.5
                else:
                    fac_close = 1.0

                dinc = np.clip(inc * fac_close * rf_inc, -inc_clip, inc_clip)

                # Overwrite incidence change if we can see from history
                # that the target is bracketed
                if inc_history and False:
                    # Get history so far
                    inc_old = np.array(inc_history[irow])

                    # Append incidence now to history and record
                    inc_all = np.concatenate((inc_old, inc_now))
                    inc_history_new.append(inc_all)

                    # We want to remember the most recent positive and
                    # negative values of incidence for each section
                    nhist, _, nsect = inc_all.shape
                    inc_bracket = np.full((2, 2, nsect), np.nan)

                    nrecent = np.minimum(nhist, 4)

                    # Loop over sections
                    for j in range(nsect):
                        # Get most recent -ve
                        for k in range(nrecent):
                            inc_k = inc_all[-1 - k, :, j]
                            if inc_k[1] < 0.0:
                                inc_bracket[0, :, j] = inc_k
                                break

                        # Then get most recent +ve
                        for k in range(nrecent):
                            inc_k = inc_all[-1 - k, :, j]
                            if inc_k[1] > 0.0:
                                inc_bracket[1, :, j] = inc_k
                                break

                        # If we found a bracket, then use it to set recamber
                        if not np.isnan(inc_bracket[:, :, j]).any():
                            # print(f"sect j={j} is bracketed")
                            # print(inc_bracket[:, :, j])

                            # Binary search for next recamber
                            dchi_next = np.mean(inc_bracket[:, 0, j])
                            if (np.abs(dchi_next) > atol_chi).any():
                                inc_converged = False
                                # print("not converged")
                            dinc[j] = dchi_next - qstar_save[irow][j, 0]

                else:
                    if (np.abs(inc) > inc_tol).any():
                        inc_converged = False
                    inc_history_new.append(inc_now)

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

                        inc = np.diff(chi[1], axis=0).squeeze()
                        inc -= inc_target

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
            else:
                inc_history_new.append(None)

        inc_conf["history"] = inc_history_new

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

    DF_converged = True
    if DF_conf := conf.iterate.get("DF"):
        DF_conf = turbigen.nblade.DiffusionFactorConfig(**DF_conf)
        for irow, DF_target in DF_conf.target.items():
            # Calculate the diffusion factor from CFD
            DF_actual = turbigen.nblade.get_diffusion_factor(g, mac, ml, irow, DF_conf)

            # Calculate Nblade adjustment
            dNb_rel = -DF_conf.dNb_dDF * (1.0 - DF_actual / DF_target)
            logger.iter(
                f"DF_actual={DF_actual:.3g}, DF_target={DF_target:.3g}, dNb_rel={dNb_rel:.3g}"
            )
            pdict["DF"] = DF_actual
            pdict["DNb"] = dNb_rel

            # Adjust nblade
            if "Nb" in conf.blades:
                conf.blades["Nb"][irow] += int(dNb_rel * ml.Nb[irow])
            elif "Co" in conf.blades:
                conf.blades["Co"][irow] /= 1.0 + dNb_rel
            DF_converged = False

    xpeak_converged = True
    if xpeak_conf := conf.iterate.get("xpeak"):
        xpeak_conf = turbigen.iterators.PeakSuctionConfig(**xpeak_conf)

        # Loop over rows
        for irow, xpeak_target in xpeak_conf.target.items():
            logger.iter(f"Optimising peak suction for row {irow}...")
            # Get the pressure distribution
            zeta_norm, Cp = util_post.get_pressure_distribution(
                g,
                mac,
                ml,
                irow,
                xpeak_conf.spf,
            )

            # Get peak suction location
            xpeak = np.abs(zeta_norm[Cp.argmin()].item())

            # Calclulate change in camber line parameter
            dqcamber = xpeak_conf.K * (xpeak_target - xpeak)

            # Apply change to camber line
            qstar_save[irow][:, 2] += dqcamber

            logger.iter(
                f"xpeak_actual={xpeak:.3f}, xpeak_target={xpeak_target:.3f}, dqcamber={dqcamber:.3f}"
            )
            xpeak_converged = False

    # Update qstar post-optimisation
    for irow, row in enumerate(conf.sections):
        if row:
            row.pop("q_camber", None)
            row["qstar_camber"] = qstar_save[irow].tolist()

    opt_converged = (
        dev_converged
        and inc_converged
        and mean_line_converged
        and DF_converged
        and xpeak_converged
    ) or conf.solver.get("skip")

    if conf.iterate:
        log_line(pdict, log_fields)

    out_vars.pop("So1")
    inverse_path = os.path.join(workdir, "inverse.yaml")
    turbigen.yaml.write_yaml(out_vars, inverse_path)
    logger.debug(f"Wrote inversion to {inverse_path}")

    if opt_converged:
        # out_vars = meanline_design.inverse(ml_out)
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

    # Write out the nominal and actual mean lines
    actual_ml_path = os.path.join(workdir, "mean_line_actual.yaml")
    ml_out.mean_line_type = conf.mean_line_type
    ml_out.write(actual_ml_path)

    logger.info(f"Elapsed time {mins:.2f} min.")

    sys.stdout.flush()

    return ml_out, opt_converged, gguess


def iter_mean_line(conf, vars_cfd, pdict):
    """Compare the CFD and nominal mean-lines, adjust .

    Parameters
    ----------
    conf: Config object for the last run
    vars_cfd: dict of design variables calculated from last CFD, keyed by variable name
    pdict: dict of colums to print logging information
    """

    # If we do not have mean-line iteration configured, say we are converged
    if not (mean_iter_conf := conf.iterate.get("mean_line")):
        return True

    # Initialise flag and set False later if any discrepancies exceed tolerance
    mean_line_converged = True

    # Extract values from mean-line iterate config
    rf = mean_iter_conf.get("relaxation_factor", 0.5)
    tols_match = mean_iter_conf.get("match_tolerance", {})

    # Loop over the design variables we want to match
    for vname, vtol in tols_match.items():
        # Get the CFD value for this design variable
        var_cfd = np.atleast_1d(vars_cfd[vname])

        # If there was no nominal value for this var, then set it straight to CFD value
        var_nom = conf.mean_line.get(vname)
        if var_nom is None:
            err = np.inf
            var_new = vars_cfd

        # If there was a nominal value specified, then apply relaxation
        else:
            var_nom = np.array(var_nom)
            err = np.abs(var_nom - var_cfd).max()
            var_new = var_cfd * rf + (1.0 - rf) * var_nom

        # Calculate the change to be applied to the nominal values in config
        dvar = var_new - var_nom

        # Insert convergence log data
        imax = np.argmax(err)
        pdict[vname] = var_cfd[imax]
        pdict["D" + vname] = dvar[imax]

        # We have not converged if the err tolerance is exceeded
        if err > tols_match[vname]:
            mean_line_converged = False

        # Assign back to the configuration
        if len(var_new) == 1:
            conf.mean_line[vname] = var_new.item()
        else:
            conf.mean_line[vname] = var_new.tolist()

    return mean_line_converged


def run(conf):
    basedir = conf.workdir

    topt_start = timer()

    # If specified use database to fill in values
    if conf.database.get("conf_path"):
        conf.interpolate_from_database()

    if conf.iterate:
        if not conf.solver:
            raise Exception(
                "Cannot iterate the design without a CFD solver configured."
            )
        gguess = None

        max_iter = conf.iterate.get("max_iter", 20)
        min_iter = conf.iterate.get("min_iter", 1)
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
        if "nstep" not in conf.solver:
            nstep_key = "n_step"
        else:
            nstep_key = "nstep"
        fac_nstep_initial = conf.iterate.get("fac_nstep_initial", 1.0)
        nstep_old = conf.solver[nstep_key]
        conf.solver[nstep_key] = int(fac_nstep_initial * nstep_old)

        for i in range(max_iter):
            iterdir = os.path.join(basedir, "%04d" % i)
            os.makedirs(iterdir, exist_ok=True)
            conf.workdir = iterdir

            # Disable soft start once we have a good initial guess
            if i > 0 and ("soft_start" in conf.solver):
                conf.solver["soft_start"] = False
            ml_out, opt_converged, gguess = run_single(conf, gguess)

            # Reset nstep
            conf.solver[nstep_key] = nstep_old

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

            if opt_converged and i >= (min_iter - 1):
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
        ml_out, _, gguess = run_single(conf)
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

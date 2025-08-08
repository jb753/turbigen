import numpy as np
from turbigen import util
import turbigen.grid
import turbigen.geometry
from turbigen import clusterfunc
import turbigen.mesh
import dataclasses
import matplotlib.pyplot as plt

logger = util.make_logger()


@dataclasses.dataclass
class H(turbigen.mesh.Mesher):
    """Generate a mesh using H topology for each row."""

    recluster: bool = False
    slip_cusp: bool = False

    ER_stream: float = 1.2
    """Expansion ratio of streamwise grid from first LE to inlet boundary."""

    AR_stream: float = 2.0
    """Aspect ratio in blade-to-blade plane of cells at outlet boundary."""

    AR_passage: float = 1.0
    """Nominal aspect ratio in blade-to-blade plane of mid-passage cells."""

    AR_merid: float = 1.0
    """Aspect ratio of mid-chord cells in meridional plane."""

    AR_merid_unbladed: float = 2.0
    """Aspect ratio of mid-chord cells in meridional plane."""

    ER_span: float = 1.2
    """Expansion ratio of spanwise grid away from hub and casing."""

    dm_LE: float = 0.001
    """Streamwise grid spacing at LE, normalised by meridional chord."""

    ni_TE: int = 9
    """Number of streamwise points across trailing edge."""

    dm_TE: float = 0.05
    """Normalised meridional length over which to cluster the TE points, 0. for
    the true actual TE."""

    dspf_mid: float = 0.03
    """Spanwise grid spacing at midspan, as a fraction of span."""

    ER_pitch: float = 1.2
    """Expansion ratio away from aerofoil surfaces."""

    nchord_relax: float = 1.0
    """Number of meridional chords over which pitchwise clustering is relaxed."""

    resolution_factor: float = 1.0
    """Multiply the number of points in each direction, keeping relative spacings."""

    skew_max: float = 30.0

    slip_annulus: bool = False

    AR_cusp: float = 0.0
    ni_cusp: int = 0

    yplus: float = np.nan

    plot: bool = False

    def make_grid(self, workdir, mac, dhub, dcas, dsurf, Omega=None):
        """Generate a Grid object for a machine geometry."""

        logger.info("Making an H-mesh...")

        mesh_config = self
        unbladed = [False for _ in range(mac.Nrow)]

        dsurf = np.tile(dsurf, (2, 1))

        # Spanwise grid vector
        # From hub/casing spacings and ER
        span_ref = mac.ann.get_span(1)
        dspf_hub = dhub / span_ref
        dspf_casing = dcas / span_ref

        blocks = []

        # Loop over rows
        nrow = mac.Nrow
        assert dsurf.shape == (2, nrow)
        theta_lim = None
        for irow in range(nrow):
            logger.debug(f"irow={irow}")

            # Angular pitch
            pitch_theta = 2.0 * np.pi / float(mac.Nb[irow])

            # Evaluate xr over a uniform grid
            mrow = np.linspace(2.0 * irow + 1.0, 2.0 * irow + 2)
            xr_hub = mac.ann.evaluate_xr(mrow, 0.0)
            xr_cas = mac.ann.evaluate_xr(mrow, 1.0)
            xr_mid = mac.ann.evaluate_xr(mrow, 0.5)

            # Meridional chord lengths at midspan of gaps and aerofoils
            ist = 2 * irow
            ien = ist + 3
            chord_hub = mac.ann.chords(0.0)[ist:ien]
            chord_mid = mac.ann.chords(0.5)[ist:ien]
            chord_cas = mac.ann.chords(1.0)[ist:ien]

            # Circumferential pitches
            pitch_rtheta_hub = pitch_theta * xr_hub[1]
            pitch_rtheta_cas = pitch_theta * xr_cas[1]

            # Pitch to chord ratios at hub, mid tip
            pitch_chord_hub = pitch_rtheta_hub / chord_hub[1]
            pitch_chord_cas = pitch_rtheta_cas / chord_cas[1]

            pitch_chord_ref = pitch_theta * xr_mid[1].mean() / chord_mid

            pitch_chord_max = np.maximum(pitch_chord_hub.max(), pitch_chord_cas.max())
            pitch_rtheta_max = np.maximum(
                pitch_rtheta_hub.max(), pitch_rtheta_cas.max()
            )

            # Normalised wall distance
            drt_norm = dsurf[:, irow].min() / pitch_rtheta_max

            # Row aspect ratios
            span_row = np.mean(mac.ann.get_span(np.arange(irow * 2, irow * 2 + 2)))
            AR_row = span_row / chord_mid[1]

            # Nominal pitch fractions first
            if unbladed[irow]:
                if irow == 0:
                    nk_not_resampled = 33
                    pitch_frac_nom = np.linspace(0.0, 1.0, nk_not_resampled)
                else:
                    pitch_frac_nom = mesh_config.pitchwise_grid_unbladed(
                        AR_row, pitch_chord_ref[1]
                    )
            else:
                safety_fac = 1.01
                pitch_frac_nom = mesh_config.pitchwise_grid(
                    drt_norm, pitch_chord_max * safety_fac, AR_row
                )
                logger.debug(
                    f"Nominal pitchwise grid: {drt_norm}, {pitch_chord_max}, {AR_row}"
                )
                logger.debug("Checking we can recluster")
                pitch_frac_not_resampled = mesh_config.pitchwise_grid(
                    drt_norm, pitch_chord_max * safety_fac, AR_row, resample=False
                )
                mesh_config.pitchwise_grid_fixed_npts(
                    drt_norm, pitch_chord_max, AR_row, len(pitch_frac_not_resampled)
                )
                nk_not_resampled = len(pitch_frac_not_resampled)
            nk = len(pitch_frac_nom)
            logger.debug(f"nk={nk}, nk_not_resampled={nk_not_resampled}")

            # Spanwise grid
            tip_ref = np.max(mac.tip)
            span_frac = mesh_config.spanwise_grid(dspf_hub, dspf_casing, tip_ref)

            if mesh_config.slip_annulus:
                dspf = mesh_config.dspf_mid
                span_frac = clusterfunc.symmetric.free(
                    dspf / 2.0, dspf, mesh_config.ER_span
                )

            nj = len(span_frac)
            logger.info(f"irow={irow}, nj={nj}")

            # Streamwise grid
            # From LE/TE/bcond spacings and ER
            # Choose how long to make the inlet/exit
            if nrow == 1:
                L = (1.0, 1.0)
            elif irow == 0:
                L = (1.0, 0.5)
            elif irow == (nrow - 1):
                L = (0.5, 1.0)
            else:
                L = (0.5, 0.5)

            # Generate initial streamwise grid vector at midspan
            # This fixes number of points and roughly distributes points
            if unbladed[irow]:
                tte = None
            elif mesh_config.dm_TE:
                tte = 1.0 - mesh_config.dm_TE
            else:
                xrt_u, xrt_l = mac.bld[irow][0].evaluate_section(0.5)
                mlim_now = np.array((0, 1))
                tq = np.linspace(0.8, 1.0, 500)
                _, _, tte = _theta_limits(tq, xrt_u, xrt_l, mlim_now)

            # Streamwise grid
            stream_frac, ile, ite = mesh_config.streamwise_grid(
                pitch_chord_ref,
                nk_not_resampled,
                L,
                AR_row,
                tte,
                unbladed_row=unbladed[irow],
            )

            stream_frac_hub = stream_frac  # + delta_hub
            stream_frac_cas = stream_frac  # + delta_cas

            ni = len(stream_frac)

            # No repeated grid points
            assert len(np.unique(stream_frac)) == ni

            # Grid points should monotonically increase
            assert (np.diff(stream_frac) > 0.0).all()

            spfr = span_frac.reshape(1, -1)
            stream_frac_span = stream_frac_cas.reshape(
                -1, 1
            ) * spfr + stream_frac_hub.reshape(-1, 1) * (1.0 - spfr)
            for j in range(nj):
                mlim_now = (0, 1)
                stream_frac_span[:, j] = np.interp(
                    stream_frac_span[:, j],
                    [-1, 0, 1, 2],
                    [-1, mlim_now[0], mlim_now[1], 2],
                )

            xr = mac.ann.evaluate_xr(stream_frac_span + ist + 1.0, spfr)

            # Relax the pitchwise clustering away from LE and TE
            if unbladed[irow]:
                relax = 1.0
            else:
                relax = mesh_config.pitchwise_relaxation(
                    stream_frac, pitch_chord_ref
                ).reshape(-1, 1, 1)
            uniform = np.linspace(0.0, 1.0, nk).reshape(1, 1, -1)
            assert np.all(relax >= 0.0) and np.all(relax <= 1.0)

            pitch_frac_clust = np.zeros((ni, nj, nk))

            # Get skew angles
            if unbladed[irow]:
                pass
            else:
                Theta = mac.bld[irow][0].get_chi(0.5)

            # Loop over spans and get the angular limits from blade section
            if unbladed[irow]:
                if theta_lim is not None:
                    theta_lim_old = theta_lim.copy()
                else:
                    theta_lim_old = np.zeros((2, ni, nj))
                    Nb = mac.Nb[irow]
                    dtheta = 2.0 * np.pi / float(Nb)
                    theta_lim_old[0] = -dtheta / 2.0
                    theta_lim_old[1] = +dtheta / 2.0
                    Theta = np.zeros((2,))
                theta_lim = np.zeros((2, ni, nj))

                # Get skew angle from previous blade row
                Theta_unbladed = Theta[-1]
                if not np.isfinite(Theta_unbladed):
                    raise Exception(f"Theta unbladed {Theta_unbladed}")
                Theta_max = 30.0
                Theta_now = np.clip(Theta_unbladed, -Theta_max, Theta_max)
                tanTheta = np.tan(np.radians(Theta_now))

                # Skew the mesh upstream of LE and downstream of TE
                ind_up = stream_frac < 0.0
                ind_dn = stream_frac > 1.0
                ind_mid = np.logical_and(stream_frac >= 0.0, stream_frac <= 1.0)

                for j in range(nj):
                    for i in range(ni):
                        pitch_frac_clust[i, j, :] = pitch_frac_nom

                    if np.isfinite(tlimold_now := theta_lim_old[0, -1, j]):
                        theta_lim[:, :, j] += tlimold_now

                    dtheta_skew = np.zeros_like(stream_frac)
                    chord_fac = np.ones_like(stream_frac)
                    chord_fac[ind_up] *= chord_mid[0]
                    chord_fac[ind_mid] *= chord_mid[1]
                    chord_fac[ind_dn] *= chord_mid[2]

                    # # Retrieve exit angle from previous blade row
                    dtheta_skew = tanTheta * util.cumtrapz0(
                        chord_fac / xr[1, :, j], stream_frac
                    )
                    if not np.isfinite(dtheta_skew).all():
                        raise Exception("dtheta_skew not finite")
                    theta_lim[:, :, j] += dtheta_skew

            else:
                theta_lim = np.zeros((2, ni, nj))

                if not mesh_config.recluster:
                    pitch_frac_clust = np.tile(
                        pitch_frac_nom.reshape(1, 1, -1), (ni, nj, 1)
                    )
                else:
                    for j in range(nj):
                        for i in range(ni):
                            rt_pitch_now = xr[1, i, j] * pitch_theta
                            # Determine position along blade
                            mlim_now = mac.bld[irow]._get_mlim(span_frac[j])
                            mclip = np.interp(stream_frac_span[i, j], mlim_now, [0, 1])
                            mfrac = np.array([1.0 - mclip, mclip])
                            drt_norm_now = np.sum(dsurf[:, irow] * mfrac) / rt_pitch_now

                            try:
                                pitch_frac_clust[
                                    i, j, :
                                ] = mesh_config.pitchwise_grid_fixed_npts(
                                    drt_norm_now,
                                    pitch_chord_ref[1],
                                    AR_row,
                                    nk_not_resampled,
                                )
                            except ValueError:
                                raise Exception(
                                    f"Failed to recluster: {drt_norm_now},"
                                    f" {pitch_chord_ref[1]}, {AR_row}"
                                )

                    assert np.isfinite(pitch_frac_clust).all()

                    # Smooth the pitch fraction in i and j directions
                    for _ in range(5):
                        pitch_frac_clust[1:-1, 1:-1, :] = 0.25 * (
                            pitch_frac_clust[:-2, 1:-1, :]
                            + pitch_frac_clust[2:, 1:-1, :]
                            + pitch_frac_clust[1:-1, :-2, :]
                            + pitch_frac_clust[1:-1, 2:, :]
                        )

                    assert (pitch_frac_clust >= 0.0).all()
                    assert (pitch_frac_clust <= 1.0).all()

                for j in range(nj):
                    nchord = 10000
                    m = util.cluster_cosine(nchord)
                    xrt_u, xrt_l = mac.bld[irow][0].evaluate_section(span_frac[j], m=m)

                    assert np.all(xrt_u[2] >= xrt_l[2])

                    # Get tte of current section and warp the streamwise grid
                    # vector to locate trailing edge exactly
                    mlim_now = (0, 1)

                    stream_frac_now = stream_frac_span[:, j]
                    xr[..., j] = mac.ann.evaluate_xr(
                        stream_frac_now + ist + 1.0, span_frac[j]
                    )

                    theta_lim[..., j] = _theta_limits(
                        stream_frac_now,
                        xrt_u,
                        xrt_l,
                        mlim_now,
                        Theta,
                        chord_mid[
                            (0, -1),
                        ],
                        Theta_max=mesh_config.skew_max,
                    )[:2]

            # At this point we have xrt coords for upper and lower
            # surfaces all the way to the boundaries
            # Take the opportunity to add cusps
            xrt_ul = np.stack(
                np.broadcast_arrays(
                    xr[0, ..., None],
                    xr[1, ..., None],
                    np.moveaxis(theta_lim, 0, -1),
                )
            )
            if mesh_config.AR_cusp:
                xrt_cusped = add_cusp(
                    xrt_ul,
                    ite,
                    mesh_config.AR_cusp,
                    mesh_config.ni_cusp,
                    plot=mesh_config.plot,
                )
                xr = xrt_cusped[:2, ...].mean(axis=-1)
                theta_lim = np.moveaxis(xrt_cusped[2, ...], -1, 0)

            assert np.isfinite(xr).all()
            assert np.isfinite(pitch_frac_clust).all()
            assert np.isfinite(theta_lim).all()

            # pitch_frac_relax = (1.0 - relax) * pitch_frac + relax * uniform
            assert np.isfinite(relax).all()
            assert np.isfinite(uniform).all()
            pitch_frac_relax = (1.0 - relax) * pitch_frac_clust + relax * uniform
            assert np.isfinite(pitch_frac_relax).all()
            assert (pitch_frac_relax >= 0.0).all() and (pitch_frac_relax <= 1.0).all()

            # Pinch the tip
            if mac.tip[irow] and not unbladed[irow]:
                theta_mid = np.mean(theta_lim, axis=0, keepdims=True)
                tau = mac.tip[irow]
                pinch_frac = np.interp(
                    span_frac, [1.0 - 1.5 * tau, 1.0 - 0.6 * tau, 1.0], [0.0, 1.0, 1.0]
                ).reshape(1, 1, -1)
                theta_lim = pinch_frac * theta_mid + (1.0 - pinch_frac) * theta_lim
                njtip = np.sum(pinch_frac == 1.0)
            else:
                njtip = 0

            # Convert all matrices to 3d
            xr3 = np.tile(np.expand_dims(xr, 3), (1, 1, 1, nk))  # Add pitchwise index
            pfr3 = np.expand_dims(pitch_frac_relax, 0)  # Add coord index
            theta_lim3 = np.expand_dims(theta_lim, 3)  # Add pitchwise index
            assert (np.diff(theta_lim3, axis=0) <= 0.0).all()

            # Evaluate the angular coordinates and assemble
            theta = np.flip(
                pfr3
                * theta_lim3[
                    (0,),
                ]
                + (1.0 - pfr3)
                * (
                    theta_lim3[
                        (1,),
                    ]
                    + pitch_theta
                ),
                axis=-1,
            )

            assert np.isfinite(pitch_theta)
            assert np.isfinite(theta).all()

            xrt_now = np.concatenate([xr3, theta], axis=0)

            # Make periodic patches
            if unbladed[irow]:
                patches = [
                    turbigen.grid.PeriodicPatch(i=(0, -1), k=0, label="per_k0"),
                    turbigen.grid.PeriodicPatch(i=(0, -1), k=-1, label="per_nk"),
                ]
            else:
                if mesh_config.ni_cusp:
                    icusp = ite + mesh_config.ni_cusp - 1
                    ite -= 1
                else:
                    icusp = ite
                patches = [
                    turbigen.grid.PeriodicPatch(i=(0, ile), k=0),
                    turbigen.grid.PeriodicPatch(i=(0, ile), k=-1),
                    turbigen.grid.PeriodicPatch(i=(icusp, -1), k=0),
                    turbigen.grid.PeriodicPatch(i=(icusp, -1), k=-1),
                ]
                if mesh_config.AR_cusp:
                    logger.info("Adding cusps")
                    assert mesh_config.ni_cusp > 0
                    cusp_type = (
                        turbigen.grid.InviscidPatch
                        if mesh_config.slip_cusp
                        else turbigen.grid.CuspPatch
                    )
                    patches.extend(
                        [
                            cusp_type(i=(ite, icusp), k=0, label="cusp_k0"),
                            cusp_type(i=(ite, icusp), k=-1, label="cusp_nk"),
                        ]
                    )

            # Inlet or mixing
            if irow == 0:
                patches.append(turbigen.grid.InletPatch(i=0))
            else:
                patches.append(turbigen.grid.MixingPatch(i=0))

            # Outlet or mixing
            if irow == (nrow - 1):
                patches.append(turbigen.grid.OutletPatch(i=-1))
            else:
                patches.append(turbigen.grid.MixingPatch(i=-1))

            # Tip gap
            if njtip:
                patches.extend(
                    [
                        turbigen.grid.PeriodicPatch(i=(ile, ite), j=(-njtip, -1), k=0),
                        turbigen.grid.PeriodicPatch(i=(ile, ite), j=(-njtip, -1), k=-1),
                    ]
                )

            blocks.append(
                turbigen.grid.BaseBlock.from_coordinates(
                    xrt_now, mac.Nb[irow].astype(int), patches
                )
            )

        if mesh_config.slip_annulus:
            for b in blocks:
                b.add_patch(turbigen.grid.InviscidPatch(j=0))
                b.add_patch(turbigen.grid.InviscidPatch(j=-1))

        g = turbigen.grid.Grid(blocks)

        # Ensure xr coordinates match exactly at mixing plane
        for irow in range(0, nrow - 1):
            xr0 = g[irow].xr[:, -1, :, 0]
            xr1 = g[irow + 1].xr[:, 0, :, 0]
            xrav = 0.5 * (xr0 + xr1)
            g[irow].xr[:, -1, :, :] = xrav[..., None]
            g[irow + 1].xr[:, 0, :, :] = xrav[..., None]

        g.match_patches()

        if mesh_config.plot:
            import matplotlib.pyplot as plt

            nj = g[0].shape[1]
            jplot = nj // 2

            fig, ax = plt.subplots()
            ax.axis("equal")
            for b in g:
                ax.plot(b.x[:, :, 0], b.r[:, :, 0], "k-", lw=0.5)
                ax.plot(b.x[:, :, 0].T, b.r[:, :, 0].T, "k-", lw=0.5)

            fig, ax = plt.subplots()
            for ib, b in enumerate(g):
                ARi = b.cell_ARi[:, jplot, 0]
                ARj = b.cell_ARj[:, jplot, 0]
                ARk = b.cell_ARk[:, jplot, 0]
                xc = util.node_to_cell(b.x)[:, jplot, 0]
                x = b.x[:, jplot, 0]
                ax.plot(x, ARi, color="C0", label="i")
                ax.plot(xc, ARj, color="C1", label="j")
                ax.plot(xc, ARk, color="C2", label="k")
                logger.info(
                    f"Block {ib}: ARi={ARi.max():.3f}, ARj={ARj.max():.3f}, ARk={ARk.max():.3f}"
                )

            iplot, jplot, kplot = mesh_config.plot
            fig, ax = plt.subplots()
            ax.axis("equal")
            for ib, b in enumerate(g):
                x = b.x[:, jplot, :]
                rt = b.rt[:, jplot, :]
                ax.plot(x, rt, "k-", lw=0.5)
                ax.plot(x[iplot, kplot], rt[iplot, kplot], "b*", lw=0.5)
                ax.plot(x.T, rt.T, "k-", lw=0.5)

            plt.show()

        return g

    def spanwise_grid(self, dspf_hub, dspf_casing, tip):
        # """Evaluate a spanwise grid vector given hub and casing spacings."""
        if tip:
            Lmain = 1.0 - tip

            # We want at least 9 nodes across the tip gap
            # So the minimum grid spacing should be the smallest of:
            #   - 9 pts uniform
            #   - target shroud spacing
            njtip_min = 9
            dspf_tip = np.minimum(dspf_casing, tip / njtip_min)

            spf_main = clusterfunc.double.free(
                dspf_hub, dspf_tip, self.dspf_mid, self.ER_span, 0.0, Lmain
            )

            try:
                spf_tip = clusterfunc.double.free(
                    dspf_tip, dspf_tip, 4.0 * dspf_tip, self.ER_span, Lmain, 1.0
                )
            except turbigen.clusterfunc.exceptions.ClusteringException:
                spf_tip = clusterfunc.double.fixed(
                    dspf_tip, dspf_tip, njtip_min, Lmain, 1.0
                )

            spf_main = util.resample(spf_main, self.resolution_factor)
            spf_tip = util.resample(spf_tip, self.resolution_factor)
            spf = np.concatenate((spf_main[:-1], spf_tip))

            assert spf[0] == 0.0
            assert np.isclose(spf[-1], 1.0)
            assert (np.diff(spf) > 0.0).all()

            return spf

        else:
            return util.resample(
                clusterfunc.double.free(
                    dspf_hub, dspf_casing, self.dspf_mid, self.ER_span
                ),
                self.resolution_factor,
            )

    def pitchwise_grid(self, drt_row, pitch_chord, AR_row, resample=True):
        """Evaluate a pitchwise grid vector given surface spacing."""
        dm_mid = self.dspf_mid * AR_row / self.AR_merid
        drt_mid = dm_mid / pitch_chord * self.AR_passage
        logger.debug(
            f"Free npts: drt_row={drt_row}, drt_mid={drt_mid}, ER={self.ER_pitch}"
        )

        # x1 = 0.5 * util.cluster_new_free(drt_row * 2.0, drt_mid * 2.0, self.ER_pitch)
        # x = np.concatenate((x1[:-1], 1.0 - np.flip(x1)))

        x = clusterfunc.symmetric.free(drt_row, drt_mid, self.ER_pitch)

        dx = np.diff(x)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert np.all(dx > 0.0)
        ERout = dx[1:] / dx[:-1]
        ERout[ERout < 1.0] = 1.0 / ERout[ERout < 1.0]
        assert np.all(ERout <= self.ER_pitch)
        assert np.isfinite(x).all()

        if resample:
            x = util.resample(
                x,
                self.resolution_factor,
            )

        return x

    def pitchwise_grid_fixed_npts(self, drt_row, pitch_chord, AR_row, npts):
        """Evaluate a pitchwise grid vector given surface spacing."""

        x = clusterfunc.symmetric.fixed(drt_row, npts)

        dx = np.diff(x)
        assert np.isclose(x[0], 0.0)
        assert np.isclose(x[-1], 1.0)
        assert np.all(dx > 0.0)
        assert len(x) == npts
        assert np.isfinite(x).all()

        # return x

        return util.resample(
            x,
            self.resolution_factor,
        )

    def pitchwise_grid_unbladed(self, AR_row, pitch_chord):
        # """Evaluate a pitchwise grid vector for unbladed row."""
        dm_mid = self.dspf_mid * AR_row * self.AR_merid_unbladed
        drt_mid = dm_mid / pitch_chord * self.AR_passage
        nk = np.round(1.0 / drt_mid).astype(int)
        return np.linspace(0.0, 1.0, nk)

    def streamwise_grid(
        self,
        pitch_chord,
        nrt,
        L,
        AR_row,
        tte=None,
        unbladed_row=False,
        chord_factor=1.0,
        ni_chord=None,
    ):
        # """Evaluate streamwise grid vector for a blade row."""

        assert len(pitch_chord) == 3
        assert (pitch_chord > 0.0).all()
        assert nrt > 1

        # Normalised grid spacings at endpoints (normalised by their gap chord)
        dm_boundary = self.AR_stream * pitch_chord / nrt  # * self.resolution_factor

        dm_mid = self.dspf_mid * AR_row / self.AR_merid
        dm_mid_unbladed = np.minimum(
            self.dspf_mid * AR_row * self.AR_merid_unbladed, 0.1
        )
        # dm_upstream_LE_unbladed = dm_TE * pitch_chord[0] / pitch_chord[1]
        # dm_downstream_TE_unbladed = dm_TE * pitch_chord[-1] / pitch_chord[1]

        if unbladed_row:
            Lu = np.insert(L, 1, 1.0)
            # npts = np.round(Lu / dm_boundary / 0.25).astype(int)
            npts = Lu / dm_mid_unbladed
            npts[0] /= pitch_chord[0] / pitch_chord[1]
            npts[2] /= pitch_chord[2] / pitch_chord[1]
            npts = np.round(npts).astype(int)

            t_upstream = np.linspace(-L[0], 0.0, npts[0])
            t_chord = np.linspace(0.0, 1.0, npts[1])
            t_downstream = np.linspace(0.0, L[1], npts[2]) + 1.0

            t_upstream = util.resample(t_upstream, self.resolution_factor)
            t_downstream = util.resample(t_downstream, self.resolution_factor)
            t_chord = util.resample(t_chord, self.resolution_factor)

            t = np.concatenate([t_upstream, t_chord[1:], t_downstream[1:]])
            ile = len(t_upstream) - 1
            ite = ile + len(t_chord) - 1

        else:
            # Convert the LE/TE grid spacings from chord-normalised to gap-normalised
            dm_upstream_LE = self.dm_LE * pitch_chord[0] / pitch_chord[1]
            dm_TE = (1.0 - tte) / self.ni_TE
            dm_downstream_TE = dm_TE * pitch_chord[-1] / pitch_chord[1]

            t_upstream = 1.0 - np.flip(
                clusterfunc.single.free(
                    dm_upstream_LE, dm_boundary[0] * L[0], self.ER_stream, 0.0, L[0]
                )
            )

            # Apply chord length adjustment factor
            dm_LE_adj = self.dm_LE / chord_factor
            dm_mid_adj = dm_mid / chord_factor
            dm_TE_adj = dm_TE / chord_factor

            t_chord = clusterfunc.double.free(
                dm_LE_adj, dm_TE_adj, dm_mid_adj, self.ER_stream, 0.0, tte
            )

            t_te = np.linspace(tte, 1.0, self.ni_TE)

            # free(dmin, dmax, ERmax, x0=0.0, x1=1.0, mult=8):

            try:
                t_downstream = clusterfunc.single.free(
                    dm_downstream_TE, dm_boundary[-1] * L[1], self.ER_stream, 0.0, L[1]
                )
            except turbigen.clusterfunc.exceptions.ClusteringException:
                t_downstream = clusterfunc.single.free(
                    dm_downstream_TE,
                    dm_boundary[-1] * L[1],
                    self.ER_stream,
                    0.0,
                    L[1],
                    mult=1,
                )

            # for _ in range(20):
            #     try:
            #         t_downstream = (
            #             util.cluster_one_sided_ER(
            #                 dm_downstream_TE / L[1], dm_boundary[-1], self.ER_stream
            #             )
            #             * L[1]
            #         )
            #     except ValueError:
            #         dm_boundary[-1] *= 0.8
            #         continue

            t_upstream = util.resample(t_upstream, self.resolution_factor)
            t_downstream = util.resample(t_downstream, self.resolution_factor)
            t_te = util.resample(t_te, self.resolution_factor, mult=2)
            t_chord = util.resample(t_chord, self.resolution_factor)

            t = np.concatenate(
                [t_upstream - 1.0, t_chord[1:], t_te[1:], t_downstream[1:] + 1.0]
            )

            dt = np.diff(t)
            assert (dt > 0.0).all()

            ile = len(t_upstream) - 1
            ite = ile + len(t_chord) + len(t_te) - 2

        return t, ile, ite

    def pitchwise_relaxation(self, stream_frac, pitch_chord):
        # Relax clustering towards a uniform distribution at inlet and exit
        dt_relax = (
            np.ones((2,))
            * self.nchord_relax
            * pitch_chord[
                (0, -1),
            ]
            / pitch_chord[1]
        )
        relax_ref = np.array([1.0, 0.0, 0.0, 1.0])
        t_ref = np.array([-dt_relax[0], 0.0, 1.0, 1.0 + dt_relax[1]])
        return np.interp(stream_frac, t_ref, relax_ref)


def _theta_limits(
    tq, xrt_u, xrt_l, mlim, Theta=(0.0, 0.0), c=(1.0, 1.0), Theta_max=30.0
):
    """Evaluate pitchwise limits given upper/lower surface section coordinates."""

    # Put geometric leading edge where it should be
    # Must handle axial and radial inlets differently

    # If x varies more than r near LE, is axial, split on min x
    if np.ptp(xrt_u[0][:10]) > np.ptp(xrt_u[1][:10]):
        ind_split = 0
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    # Otherwise, is radial, split on max r
    elif xrt_u[1][0] < xrt_u[1][-1]:
        ind_split = 1
        iule = np.argmin(xrt_u[ind_split])
        ille = np.argmin(xrt_l[ind_split])
    else:
        ind_split = 1
        iule = np.argmax(xrt_u[ind_split])
        ille = np.argmax(xrt_l[ind_split])

    # If the geometric leading edge is on upper surface
    # we need to move some points from upper to lower
    if xrt_u[ind_split][iule] < xrt_l[ind_split][ille]:
        xrt_l = np.concatenate(
            (np.flip(xrt_u[:, 1 : iule + 1], axis=-1), xrt_l), axis=-1
        )
        xrt_u = xrt_u[:, iule:]
    # If the geometric leading edge is on lower surface
    # we need to move some points from lower to upper
    elif xrt_u[ind_split][iule] > xrt_l[ind_split][ille]:
        xrt_u = np.concatenate(
            (np.flip(xrt_l[:, 1 : ille + 1], axis=-1), xrt_u), axis=-1
        )
        xrt_l = xrt_l[:, ille:]

    # Join the curves together at trailing edge
    if xrt_u[ind_split].max() > xrt_l[ind_split].max():
        xrt_l = np.concatenate((xrt_l, xrt_u[:, (-1,)]), axis=-1)
    else:
        xrt_u = np.concatenate((xrt_u, xrt_l[:, (-1,)]), axis=-1)

    # Evaluate normalised meridional distances for each surface
    m_u = util.cum_arc_length(xrt_u[:2])
    m_l = util.cum_arc_length(xrt_l[:2])
    m_u /= m_u[-1]
    m_l /= m_l[-1]

    m_u = mlim[0] + np.ptp(mlim) * m_u
    m_l = mlim[0] + np.ptp(mlim) * m_l

    # Interpolate the pitchwise limits
    # Values outside unit interval constant at boundary values
    theta_u = np.interp(tq, m_u, xrt_u[2])
    theta_l = np.interp(tq, m_l, xrt_l[2])

    # Look for any turning points in last 5% chord
    # These correspond to TE corner
    dtheta_u = np.diff(theta_u, n=1)
    dtheta_l = np.diff(theta_l, n=1)

    ind_l_up, ind_l_dn = util.zero_crossings(dtheta_l)
    ind_u_up, ind_u_dn = util.zero_crossings(dtheta_u)
    ind_l_te = ind_l_up[tq[ind_l_up] > mlim[1] - 0.2]
    ind_u_te = ind_u_dn[tq[ind_u_dn] > mlim[1] - 0.2]

    # If the process for setting tte does not work, then
    # arbitrarily cluster grid over last 1.0% chord
    # tte = mlim[1] -0.005
    tte = None
    if ind_l_te.size > 0:
        # print(f'TE on lower at {ind_l_te[0]}')
        tte = tq[ind_l_te[-1]]
    elif ind_u_te.size > 0:
        # print(f'TE on upper at {ind_u_te[0]}')
        tte = tq[ind_u_te[-1]]
    else:
        tte = mlim[1] - 0.01

    if np.any(theta_u < theta_l):
        raise Exception("Blade is thicker than calculated pitch!")

    r_u = np.interp(tq, m_u, xrt_u[1])
    r_l = np.interp(tq, m_l, xrt_l[1])
    rref = 0.5 * (r_u + r_l)

    # Skew the mesh upstream of LE and downstream of TE
    dtheta_skew = np.zeros_like(theta_u)
    ind_up = tq < mlim[0]
    ind_dn = tq >= mlim[1]
    Theta_now = np.clip(Theta, -Theta_max, Theta_max)
    tanTheta = np.tan(np.radians(Theta_now))
    if ind_up.any():
        dtheta_skew[ind_up] = (
            tanTheta[0] * c[0] * util.cumtrapz0(1.0 / rref[ind_up], tq[ind_up])
        )
        dtheta_skew[ind_up] -= dtheta_skew[ind_up][-1]
    if ind_dn.any():
        dtheta_skew[ind_dn] = (
            tanTheta[1] * c[1] * util.cumtrapz0(1.0 / rref[ind_dn], tq[ind_dn])
        )
    theta_u += dtheta_skew
    theta_l += dtheta_skew

    return theta_u, theta_l, tte


def add_cusp(xrt, iTE, AR_cusp, ni_cusp, plot=True):
    """Change block coordinates from square TE to cusped TE.

    This assumes that the trailing edge is located exactly
    at ni_TE points upstream of the zero-thicknes point of the blade
    (achieve this by setting dm_TE to 0.0 and ni_TE > 0 in config.)

    """

    assert AR_cusp > 0.0

    nj = xrt.shape[2]
    jmid = nj // 2

    # Convert to xrrt
    rref = xrt[1, iTE, jmid, 0]
    xrrt = util.to_xrrt_ref(xrt, rref)

    assert np.allclose(xrt[2, iTE, :, 0], xrt[2, iTE, :, 1])

    # Plot old coords
    if plot:
        fig, ax = plt.subplots()
        ax.plot(
            xrrt[0, :, jmid, (0, -1)].T,
            xrrt[2, :, jmid, (0, -1)].T,
            "r-",
            lw=0.5,
            alpha=0.2,
        )
        ax.plot(xrrt[0, iTE, jmid, 0], xrrt[2, iTE, jmid, 0], "go")

    # Determine if the exit angle is +ve or -ve
    plus_exit = np.diff(xrrt[2, iTE : iTE + 2, jmid, 0]).item() > 0.0
    logger.debug(f"{plus_exit=}")

    is_axial = bool(
        np.ptp(xrrt[0, iTE : iTE + 2, jmid, 0])
        > np.ptp(xrrt[1, iTE : iTE + 2, jmid, 0])
    )
    logger.info(f"{is_axial=}")

    # Find both corners of the trailing edge
    istlook = iTE - 12
    if plus_exit:
        # Look for turning point on lower surface
        ilower = istlook + np.argmax(xrrt[2, istlook:iTE, jmid, 0]).item()
        # Extrapolate lower surface to TE
        if is_axial:
            dt = np.diff(xrrt[2, ilower - 1 : ilower + 1, :, 0], axis=0)
            dx = np.diff(xrrt[0, ilower - 1 : ilower + 1, :, 0], axis=0)
            grad = dt / dx
            xrrt[2, ilower + 1 : iTE + 1, :, 0] = xrrt[2, ilower, :, 0] + grad * (
                xrrt[0, ilower + 1 : iTE + 1, :, 0] - xrrt[0, ilower, :, 0]
            )

        else:
            raise NotImplementedError()
        logger.debug(f"{ilower=}, iTE={iTE}")
        xrrt_TE = np.moveaxis(xrrt[:, iTE - 10 : iTE + 1, :, :], -1, 0)
    else:
        # Look for turning point on upper surface
        ilower = istlook + np.argmin(xrrt[2, istlook:iTE, jmid, -1]).item()
        # Extrapolate lower surface to TE
        if is_axial:
            dt = np.diff(xrrt[2, ilower - 1 : ilower + 1, :, -1], axis=0)
            dx = np.diff(xrrt[0, ilower - 1 : ilower + 1, :, -1], axis=0)
            grad = dt / dx
            xrrt[2, ilower : iTE + 1, :, -1] = xrrt[2, ilower, :, -1] + grad * (
                xrrt[0, ilower : iTE + 1, :, -1] - xrrt[0, ilower, :, -1]
            )

        else:
            raise NotImplementedError()
        xrrt_TE = np.stack(
            (
                xrrt[:, iTE - 10 : iTE + 1, :, 0],
                xrrt[:, iTE - 10 : iTE + 1, :, -1],
            )
        )

    # Extrapolate the lower surface to

    if plot:
        ax.plot(xrrt_TE[:, 0, :, jmid], xrrt_TE[:, 2, :, jmid], "b-x")

    # Centre of trailing edge
    xrrt_cent = np.mean(xrrt_TE[:, :, -1], axis=0)

    # Vectors across TE and along camber line
    # xrrt_TE is indexed[side, coord, i, j]
    vec_TE = xrrt_TE[0, :, -1] - xrrt_TE[1, :, -1]
    logger.info(vec_TE.shape)
    W_TE = util.vecnorm(vec_TE)
    vec_TE /= W_TE
    vec_cam = np.mean(np.diff(xrrt_TE[:, :, -2:, :], axis=2), axis=0).squeeze()
    vec_cam /= util.vecnorm(vec_cam)
    logger.info(f"{W_TE[jmid]=}")

    # Find cusp point
    L_cusp = AR_cusp * W_TE
    logger.info(f"L_cusp={L_cusp.mean():.3g}")
    xrrt_point = xrrt_cent + L_cusp * vec_cam

    if plot:
        ax.plot(xrrt_cent[0, jmid], xrrt_cent[2, jmid], "g^")
        ax.plot(xrrt_point[0, jmid], xrrt_point[2, jmid], "r*")
        plt.axis("equal")

    # Now get the coordinates to be added
    f_cusp = np.linspace(0.0, 1.0, ni_cusp).reshape(1, -1, 1)

    xrrt_cusp = (
        f_cusp * xrrt_point[None, :, None, :]
        + (1.0 - f_cusp) * xrrt_TE[:, :, None, -1, :]
    )
    if plot:
        for xrrtci in xrrt_cusp:
            ax.plot(xrrtci[0, :, jmid], xrrtci[2, :, jmid], "g^-")

    # Now make the grid spacing at TE match
    for j in range(nj):
        m_TE = util.cum_arc_length(xrrt_TE[0, :, :, j])
        dm_TE = np.diff(m_TE, axis=0)
        dm_end = dm_TE[-2].mean()
        dm_start = dm_TE[0].mean()
        m_TE_new = clusterfunc.double.fixed(
            dm_start,
            dm_end,
            len(m_TE),
            0.0,
            m_TE[-1],
        )
        for k in range(2):
            for c in range(3):
                xrrt[c, iTE - len(m_TE_new) + 1 : iTE + 1, j, k] = np.interp(
                    m_TE_new, m_TE, xrrt_TE[k, c, :, j]
                )

    xrrt_new = np.concatenate(
        (xrrt[:, : iTE + 1, :, :], np.moveaxis(xrrt_cusp, 0, -1)[:, 1:]), axis=1
    )

    # Shift the downstream coordiates to new theta TE
    dtheta_TE = xrrt_new[2, -1, :, 0] - xrrt[2, iTE, :, 0]
    xrrt_extra = xrrt[:, iTE:, :, :].copy()
    xrrt_extra[2] += dtheta_TE.reshape(1, -1, 1)

    # Define a curvilinear meridional coordinate along xrrt_extra
    # m_extra[i, j]
    m_extra = util.cum_arc_length(xrrt_extra[:2, :, :, 0])

    # Find the value of m that corresponds to cusp point
    m_point = np.zeros((nj,))
    nidown = xrrt_extra.shape[1] - ni_cusp + 1
    xrrt_new_down = np.zeros((3, nidown, nj, 2))
    if np.abs(vec_cam[0]).mean() > np.abs(vec_cam[1]).mean():
        # Mostly axial, interpolate using x
        for j in range(nj):
            m_point = np.interp(xrrt_point[0, j], xrrt_extra[0, :, j, 0], m_extra[:, j])
            # Theta offset required to exactly align TE

            dt_point = xrrt_point[2, j] - np.interp(
                m_point, m_extra[:, j], xrrt_extra[2, :, j, 0]
            )
            dt = np.interp(m_extra[:, j], [m_point, m_extra[-1, j]], [dt_point, 0.0])

            xrrt_extra[2, :, j, :] += dt.reshape(-1, 1)

            L = m_extra[-1, j] - m_point

            dm_start = util.cum_arc_length(xrrt_new[:2, -2:, j, 0])[-1]
            dm_end = m_extra[-1, j] - m_extra[-2, j]
            clu = clusterfunc.double.fixed(
                dm_start, dm_end, nidown, m_point, m_point + L
            )

            for ii in range(3):
                xrrt_new_down[ii, :, j, :] = np.interp(
                    clu, m_extra[:, j], xrrt_extra[ii, :, j, 0]
                )[:, None]

    else:
        raise NotImplementedError()

    xrrt_new = np.concatenate((xrrt_new[:, :-1, :, :], xrrt_new_down), axis=1)

    if plot:
        ax.plot(
            xrrt_new[0, :, jmid, (0, -1)].T,
            xrrt_new[2, :, jmid, (0, -1)].T,
            "k.-",
            lw=0.5,
            ms=1.0,
        )
        ax.plot(
            xrrt_extra[0, :, jmid, (0, -1)].T,
            xrrt_extra[2, :, jmid, (0, -1)].T,
            "k-",
            lw=0.5,
        )

    # Convert back to xrt
    xrt_new = util.from_xrrt_ref(xrrt_new, rref)

    return xrt_new

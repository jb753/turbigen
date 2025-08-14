"""Generic post-processor class."""

from abc import ABC, abstractmethod
import dataclasses
import numpy as np
from turbigen import util
import turbigen.base
import turbigen.average
import warnings
import matplotlib.pyplot as plt

logger = util.make_logger()

LABELS = {
    "Mas": r"Isentropic Mach Number, $\mathit{Ma}_s$",
    "Ys": "Entropy Loss Coefficient, $Y_s$",
    "Ma_rel": r"Relative Mach Number, $\mathit{Ma}^\mathrm{rel}$",
}


@dataclasses.dataclass
class BasePost(ABC):
    """Base class for post-processing."""

    @abstractmethod
    def post(self, config, pdf):
        """Perform the post processing on a config object."""
        raise NotImplementedError()


@dataclasses.dataclass
class Convergence(BasePost):
    dn_smooth: int = 49
    """Smoothing window for the time series."""

    rtol_loss: float = 0.01

    """Smoothing window for the time series."""

    def post(self, config, pdf):
        """Make a plot of convergence history of the CFD run."""

        meanline = config.mean_line.nominal
        conv = config.solver.convergence

        if conv is None:
            logger.info("No simulation log returned, skipping convergence plot.")
            return

        # Choose type of machine
        if meanline.P[-1] > meanline.P[0]:
            # Is compressor, reference to inlet velocity
            Vref = meanline.V_rel[0]
        else:
            # Is turbine, reference to exit velocity
            Vref = meanline.V_rel[-1]
        dhref = 0.5 * Vref**2

        # Get non-dimensionals
        Texit = meanline.T[-1]
        state = conv.state
        Ys = (state.s[1] - state.s[0]) * Texit / dhref
        CWx = (state.h[1] - state.h[0]) / dhref

        # Normalise work and loss as percent
        # changes with respect to averaged value
        try:
            istart = np.where(conv.istep == conv.istep_avg)[0][0]
        except Exception:
            istart = 0
        dYs = (Ys / Ys[istart:].mean() - 1.0) * 100.0
        if meanline.U.any():
            dCWx = (CWx / CWx[istart:].mean() - 1.0) * 100.0
        else:
            # Fall back to absolute in a cascade
            dCWx = CWx * 100.0
        ylim = np.array([-10.0, 10.0])
        ytick = [-8, -4, -2, -1, 0, 1, 2, 4, 8]

        if self.dn_smooth:
            conv.resid = util.moving_average_1d(conv.resid, self.dn_smooth)
            dCWx = util.moving_average_1d(dCWx, self.dn_smooth)
            dYs = util.moving_average_1d(dYs, self.dn_smooth)

        dYs_reversed = np.flip(dYs)
        istep_conv = np.flip(conv.istep)[
            np.argmax(np.abs(dYs_reversed) > self.rtol_loss * 100.0)
        ]

        # Do the plotting
        _, ax = plt.subplots(1, 3, layout="constrained")
        ax[0].plot(conv.istep, np.log10(conv.resid), marker="")
        ax[0].set_title("log(Residual)")
        ax[1].plot(conv.istep, dCWx, marker="")
        ax[1].set_title("dWork/percent")
        ax[1].set_ylim(ylim)
        ax[1].set_yticks(ytick)
        ax[2].plot(conv.istep, dYs, marker="")
        ax[2].set_ylim(ylim)
        ax[2].set_yticks(ytick)
        ax[2].set_title("dLoss/percent")

        ax[0].annotate(
            f"istep_conv={istep_conv}",
            xy=(1.0, 1.0),
            xytext=(-5.0, -5.0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="top",
            backgroundcolor="w",
            color="C1",
        )
        ax[0].annotate(
            f"istep_avg={conv.istep_avg}",
            xy=(1.0, 1.0),
            xytext=(-5.0, -25.0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="top",
            backgroundcolor="w",
            color="C2",
        )

        for axi in ax:
            axi.set_xlabel("nstep")
            axi.set_xticks(())
            distep = conv.istep[1] - conv.istep[0]
            axi.set_xlim(conv.istep[0], conv.istep[-1] + distep)
            axi.axvline(conv.istep_avg, color="C2", linestyle="--")
            axi.axvline(istep_conv, color="C1", linestyle=":")

        pdf.savefig()
        plt.close()


@dataclasses.dataclass
class Metadata(BasePost):
    def post(self, config, pdf):
        """Make a slide with some text metadata."""

        _, ax = plt.subplots(layout="constrained")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        left = 0.05
        ax.set_title("Metadata:")
        ax.text(left, 0.95, f"workdir={str(config.workdir)}")
        pdf.savefig()
        plt.close()


def calculate_nondim(C, ml, vname):
    """Calculate a non-dimensional varaiable over a cut.

    Parameters
    ----------
    C : FlowField object
        The cut to evaluate.
    ml : MeanLine object
        A single-row meanline object used to provide reference values.
    vname: str
        String indicating which variable to calculate.

    Returns
    -------
    y : ndarray
        The non-dimensional variable.
    ylabel : str
        Label for the y-axis.

    """

    # Isentropic from inlet entropy to local static
    Cs = C.copy().set_P_s(C.P, ml.s[0])
    hs = Cs.h
    ho = C.ho_rel

    # Ensure ho > hs
    dh = ho - hs
    hs += np.min(dh)

    # Evaluate velocity and Mach
    Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))
    Mas = Vs / C.a

    is_compressor = ml.P[1] > ml.P[0]

    Poref = ml.Po_rel[0]
    if is_compressor:
        Ys = ml.T[1] * (C.s - ml.s[0]) / ml.halfVsq_rel[0]
        Pref = ml.P[0]
    else:
        Ys = ml.T[1] * (C.s - ml.s[0]) / ml.halfVsq_rel[1]
        Pref = ml.P[1]

    # Work coefficient
    Cho_rel = (C.ho_rel - ml.ho_rel[0]) / ml.halfVsq_rel[1]
    Cho = (C.ho - ml.ho[0]) / ml.halfVsq[1]

    # Velocity coefficient
    CVm = C.Vm / ml.V_rel[1]

    Cp = (C.P - Poref) / (Poref - Pref)

    if vname == "Mas":
        return Mas
    elif vname == "Ys":
        return Ys
    elif vname == "Cp":
        return Cp
    elif vname == "Ma_rel":
        return C.Ma_rel
    elif vname == "Mar":
        return C.Vr / C.a
    elif vname == "Max":
        return C.Vx / C.a
    elif vname == "Mat":
        return C.Vt / C.a
    elif vname == "Mam":
        return C.Vm / C.a
    elif vname == "Cho":
        return Cho
    elif vname == "Cho_rel":
        return Cho_rel
    elif vname == "CVm":
        return CVm
    elif vname == "Alpha":
        return C.Alpha
    elif vname == "Alpha_rel":
        return C.Alpha_rel
    elif vname == "Beta":
        return C.Beta
    else:
        raise ValueError(f"Unknown variable {vname} requested.")


@dataclasses.dataclass
class SurfaceDistribution(BasePost):
    variable: str = "Mas"
    """Which variable to plot."""

    spf: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of row index to span fraction(s) to plot."""

    offset: int = 0
    """How many points away from the wall."""

    def post(self, config, pdf):
        """Plot distribution of a quantity around blade surface."""

        # Default to plotting on the designed sections
        if self.spf is not None:
            spf = {irow: config.blades[irow][0].spf for irow in range(config.nrow)}

        # Loop over rows
        for irow, spfrow in spf.items():
            if spfrow is None:
                continue

            # Setup figure
            _, ax = plt.subplots(layout="constrained")
            ax.set_title(f"Row {irow}")
            ax.set_xlabel(r"Surface Distance, $\zeta/\zeta_\mathrm{TE}$")
            ax.set_xlim((0.0, 1.0))

            label = LABELS.get(self.variable, self.variable)
            ax.set_ylabel(label)

            # Cut the entire blade
            C = config.grid.cut_blade_surfs(self.offset)[irow][0]

            # Loop over span fractions
            for spfi in spfrow:
                # Slice at required span fractions
                xrc = config.annulus.get_span_curve(spfi)
                Ci = C.meridional_slice(xrc)

                # Get the variable
                y = calculate_nondim(
                    Ci, config.mean_line.actual.get_row(irow), self.variable
                )

                # Extract surface distance and normalise
                zeta_stag = Ci.zeta_stag
                # Shift zeta=0 to minimum Mas
                if self.variable == "Mas":
                    zeta_stag -= zeta_stag[np.argmin(y)]
                # Calculate maximum zeta only on main blade
                zeta_max = zeta_stag.max(axis=0)
                zeta_min = np.abs(zeta_stag.min(axis=0))
                zeta_norm = zeta_stag.copy()
                zeta_norm[zeta_norm < 0.0] /= zeta_min
                zeta_norm[zeta_norm > 0.0] /= zeta_max

                ax.plot(
                    np.abs(zeta_norm),
                    y,
                    label=f"spf={spfi}",
                    linestyle="-",
                    marker="",
                )

            # Finish this row
            pdf.savefig()
            plt.close()


@dataclasses.dataclass
class Contour(BasePost):
    variable: str = "Ys"
    """Which variable to plot."""

    coord: str = "spf"
    """Mapping of row index to span fraction(s) to plot."""

    value: float = 0.5
    """How many points away from the wall."""

    show_mesh: bool = False
    """Show grid lines."""

    irow_ref: int = 0
    """Which row to use for reference quantities."""

    N_passage: int = 2
    """Repeat in the circumferential direction."""

    cmap: str = "plasma"
    """matplotlib colormap to use."""

    def post(self, config, pdf):
        """Plot contours over a plane."""

        try:
            for val in self.value:
                self.contour(val, config, pdf)
        except TypeError:
            # If value is not iterable, plot a single contour
            self.contour(self.value, config, pdf)

    def contour(self, val, config, pdf):
        if self.coord == "spf":
            # Span fraction cut
            # Cut and repeat each row separately
            xrc = config.annulus.get_span_curve(val)
            Crow = config.grid.cut_span_unstructured(xrc)
            Crow = [Ci.repeat_pitchwise(self.N_passage) for Ci in Crow]

            # Combine the rows
            C = turbigen.base.concatenate(Crow)

        else:
            # Get an xr curve describing the cut plane.
            if self.coord == "x":
                xrc = np.array([[val, val], [0.1, 1.0]])
            elif self.coord == "r":
                xrc = np.array([[-1.0, 1.0], [val, val]])
            elif self.coord == "m":
                xrc = config.annulus.get_cut_plane(val)[0]
            else:
                raise Exception(f"Invalid coord={self.coord}")
            C = config.grid.unstructured_cut_marching(xrc)

            C = C.repeat_pitchwise(self.N_passage)

        # Centre theta on zero
        C.t -= 0.5 * (C.t.min() + C.t.max())

        # Matplotlib style triangulate
        C_tri, triangles = C.get_mpl_triangulation()

        # Get the coordinates to plot
        if self.coord == "x":
            c = C_tri.yz
        elif self.coord == "r":
            c = C_tri.rt, C_tri.x
        elif self.coord == "spf":
            # Now generate a mapping from xr to meridional distance
            mp_from_xr = config.annulus.get_mp_from_xr(val)
            c = mp_from_xr(C_tri.xr), C_tri.t
        elif self.coord == "m":
            if np.ptp(C_tri.r) > np.ptp(C_tri.x):
                c = C_tri.yz
            else:
                c = C_tri.rt, C_tri.r
        else:
            raise Exception(f"Invalid coord={self.coord}")

        # Extract meanline reference row
        if self.coord == "m":
            irow_ref = int(val / 2 - 1)
            row = config.mean_line.actual.get_row(irow_ref)
        else:
            row = config.mean_line.actual

        # Get the variable
        v = calculate_nondim(C_tri, row, self.variable)
        # levels = clipped_levels(v)

        # Setup figure
        _, ax = plt.subplots(layout="constrained")
        ax.set_title(f"{self.variable} at {self.coord}={val:.3g}")

        # It seems that we have to pass triangles as a kwarg to tricontour,
        # not positional, but this results in a UserWarning that contour
        # does not take it as a kwarg. So catch and hide this warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = ax.tricontourf(
                *c,
                v,
                # levels,
                triangles=triangles,
                cmap=self.cmap,
                linestyles="none",
            )
        cm.set_edgecolor("face")
        cm.set_linewidth(0.05)
        ax.set_aspect("equal")  # Ensures equal scaling
        ax.set_adjustable("box")  # Ensures equal scaling
        ax.axis("off")

        if self.show_mesh:
            # Show the mesh
            ax.triplot(*c, triangles, "k-", lw=0.05)

        # Make the colorbar
        label = LABELS.get(self.variable, self.variable)
        plt.colorbar(cm, label=label, shrink=0.8)

        # Finish this row
        pdf.savefig()
        plt.close()


@dataclasses.dataclass
class Annulus(BasePost):
    m_cut: tuple = ()
    """Meridional cut planes to plot."""

    show_axis: bool = False
    """Show the axis of rotation."""

    show_blades: bool = True
    """Show blades."""

    def post(self, config, pdf):
        """Plot an x-r view of the annulus."""

        # Setup figure
        fig, ax = plt.subplots(layout="constrained")
        ax.axis("off")
        ax.axis("equal")
        ax.grid("off")

        if self.show_blades:
            config.apply_recamber()

            grey = np.ones((3,)) * 0.4
            Npts = 10
            spf = np.linspace(0.0, 1.0, Npts)
            # Meridional coordinates as a function of spf for
            # blade LE, TE, and diagonals
            m = np.stack(
                (
                    np.zeros((Npts,)),
                    spf,
                    1.0 - spf,
                    np.ones((Npts,)),
                )
            )

            # Loop over rows
            for bld in config.blades:
                if not bld:
                    continue

                # Loop over spanwise stations
                # Get xr on camber line
                xr = np.stack(
                    [
                        np.stack(bld[0].evaluate_section(spf[j], m=m[:, j])).mean(
                            axis=0
                        )[:2]
                        for j in range(Npts)
                    ]
                ).transpose(2, 1, 0)

                # Plot each of LE/TE/diagonals
                for xri in xr:
                    ax.plot(*xri, "-", color=grey)

            config.undo_recamber()

        # Plot the cut planes
        for mi in self.m_cut:
            xrc = config.annulus.get_cut_plane(mi)[0]
            ax.plot(*xrc, "-", color="C0")

        # Plot hub and casing lines
        xr_hub, xr_cas = config.annulus.get_coords().transpose(0, 2, 1)
        ax.plot(*xr_hub, "k-")
        ax.plot(*xr_cas, "k-")

        # Show axis of revolution
        if self.show_axis:
            ax.plot(xr_hub[0, (0, -1)], np.zeros((2,)), "k-.")

        pdf.savefig()
        plt.close()


@dataclasses.dataclass
class StreamtubeLoss(BasePost):
    frac_mdot_break: tuple = (0.2, 0.8)
    """Cumulative mass flow fractions for breakdown."""

    def post(self, config, pdf):
        # Find meridional coordinates of the cut planes
        xr_cut = config.annulus.get_offset_planes(config.cut_offset)

        # Take the cuts
        cuts = [
            config.grid.unstructured_cut_marching(xri).interpolate_to_structured()
            for xri in xr_cut
        ]

        # Loop over cuts
        Sdot_tube = []
        for cut in cuts:
            xtol = np.ptp(cut.r) * 1e-4
            assert np.ptp(cut.x) < xtol

            _, nj, nk = cut.shape

            # Pitchwise-integrate mass flow to kg/s per unit span
            mdot_pitch = np.trapz(cut.Nb * cut.rhoVx * cut.r, cut.t).squeeze()

            # Spanwise-integrate mass flow to kg/s
            mdot_span = util.cumtrapz0(mdot_pitch, cut.r[0, :, 0])

            # Find indices for the mass flow fractions
            frac_mdot = mdot_span / mdot_span[-1]
            jbreak = np.interp(
                self.frac_mdot_break, frac_mdot, np.arange(len(frac_mdot))
            )

            # Insert jbreaks into the cut
            new_data = insert_jbreaks(cut._data, jbreak)
            cut_new = cut.empty(new_data.shape[1:])
            cut_new._data = new_data
            cut_new._metadata = cut._metadata.copy()
            cut = cut_new

            # Recalculate jbreaks
            mdot_pitch = np.trapz(cut.Nb * cut.rhoVx * cut.r, cut.t).squeeze()
            mdot_span = util.cumtrapz0(mdot_pitch, cut.r[0, :, 0])
            frac_mdot = mdot_span / mdot_span[-1]
            jbreak = np.interp(
                self.frac_mdot_break, frac_mdot, np.arange(len(frac_mdot))
            )
            jbreak_round = np.round(jbreak).astype(int)
            assert np.allclose(jbreak_round, jbreak, atol=0.01)

            # Split the cuts into tubes at jbreak_round
            jbreak_round = np.insert(jbreak_round, 0, 0)
            jbreak_round = np.append(jbreak_round, cut.nj - 1)
            Sdot_tube.append([])
            for j0, j1 in zip(jbreak_round[:-1], jbreak_round[1:]):
                # Get current streamtube
                tube = cut[:, j0 : j1 + 1, :]

                # Integrate the entropy flow
                Sdot_pitch = np.trapz(
                    tube.Nb * tube.rhoVx * tube.s * tube.r, tube.t
                ).squeeze()
                Sdot_span = np.trapz(Sdot_pitch, tube.r[0, :, 0])
                Sdot_tube[-1].append(Sdot_span)

        # Convert entropy flows to an array
        # indexed [station, streamtube]
        Sdot_tube = np.array(Sdot_tube)

        # Change in each row
        DSdot = np.diff(Sdot_tube, axis=0)

        # Exit mixing loss
        cuts[-1].Omega = cuts[-1].Omega.mean()
        Cexit, Aexit, Dsexit = turbigen.average.mix_out(cuts[-1].squeeze())

        mdot_exit = Cexit.rhoVx * Aexit
        DSexit = Dsexit * mdot_exit
        DStot = np.sum(DSdot) + DSexit

        # Normalise by total machine entropy rise
        DSdot_norm = DSdot / DStot
        DSexit_norm = DSexit / DStot
        logger.info("Entropy flow breakdown by streamtube[irow, itube]:")
        logger.info(DSdot_norm)
        logger.info(DSdot_norm.sum())
        logger.info("Exit mixing loss:")
        logger.info(DSexit_norm)

        config.post_3d["DS_tube_norm"] = DSdot_norm
        config.post_3d["DS_exit_norm"] = DSexit_norm


@dataclasses.dataclass
class Sections(BasePost):
    spf: dict = dataclasses.field(default_factory=lambda: ({}))
    """Mapping of row index to span fraction(s) to plot."""

    def post(self, config, pdf):
        config.apply_recamber()

        # Default to plotting on the designed sections
        if not self.spf:
            spf_all = {irow: config.blades[irow][0].spf for irow in range(config.nrow)}
        else:
            spf_all = self.spf

        logger.info(f"Plotting sections at span fractions: {spf_all}")

        # Loop over rows
        for irow, spfrow in spf_all.items():
            # Set up axes
            _, ax = plt.subplots()
            ax.axis("equal")

            # Loop over span fractions
            for ispf, spf in enumerate(spfrow):
                #
                xrt_ul = np.stack(config.blades[irow][0].evaluate_section(spf))
                xrrt_ul = xrt_ul.copy()
                xrrt_ul[:, 2] *= xrrt_ul[:, 1]  # Convert to r,rt

                logger.info(f"Plotting section row={irow} at spf={spf:.2g}")

                for xrrti in xrrt_ul:
                    ax.plot(
                        xrrti[0],
                        xrrti[2],
                        "-",
                        color=f"C{ispf}",
                        label=f"spf={spf:.2g}",
                    )

            if len(spfrow) > 1:
                ax.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

        config.undo_recamber()


@dataclasses.dataclass
class InletProfiles(BasePost):
    def post(self, config, pdf):
        # Skip if no inlet profiles are available
        if config.inlet.spf is None:
            return

        spf = config.inlet.spf
        Po, To, Alpha, Beta = config.inlet.profiles

        # Setup figure
        fig, ax = plt.subplots(1, 4, layout="constrained", sharey=True)
        fig.suptitle("Inlet Profiles")
        ax[0].set_ylabel("Span Fraction")

        ax[0].plot(Po, spf)
        ax[0].set_xlabel(r"$P_0/\overline{P_0}$")

        ax[1].plot(To, spf)
        ax[1].set_xlabel(r"$T_0/\overline{T_0}$")

        ax[2].plot(Alpha, spf)
        ax[2].set_xlabel(r"$\alpha - \overline{\alpha}$")

        ax[3].plot(Beta, spf)
        ax[3].set_xlabel(r"$\beta - \overline{\beta}$")

        pdf.savefig()


def insert_jbreaks(data, jbreak):
    """
    Insert interpolated j-lines into `data` at fractional j-indices using np.insert.

    Parameters:
        data (np.ndarray): Array of shape (8, ni, nj, nk)
        jbreak (list of float): Fractional j indices where new lines should be inserted (e.g., [1.5, 2.5])

    Returns:
        np.ndarray: New array with additional j-lines inserted.
    """
    assert data.ndim == 4, "Input must be a 4D array (8, ni, nj, nk)"
    jbreak = sorted(jbreak)

    new_slices = []
    insert_indices = []

    for jb in jbreak:
        j0 = int(np.floor(jb))
        frac = jb - j0
        # Linear interpolation between j0 and j0+1
        interpolated = (1 - frac) * data[:, :, (j0,), :] + frac * data[
            :, :, (j0 + 1,), :
        ]
        new_slices.append(interpolated)
        insert_indices.append(j0 + 1)  # insert *after* j0

    # Insert along axis=2 (j-axis)
    new_slices = np.concatenate(new_slices, axis=2)
    new_data = np.insert(data, insert_indices, new_slices, axis=2)

    assert new_data.shape[2] == data.shape[2] + len(jbreak)

    return new_data

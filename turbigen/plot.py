"""All functions to plot things."""

import numpy as np
import matplotlib.pyplot as plt
import turbigen.geometry
import turbigen.util
import os


def plot_camber_line(bld, fname=None, ax=None):
    r"""Plot this camber line.

    Parameters
    ----------
    fname:
        Save the plot to this file, default to just showing.
    ax: matplotlib.pyplot.Axes
        Plot into an existing axes, default to new figure and axes.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The figure containing the plot.

    ax: matplotlib.pyplot.Axes
        Axes containing the plot.

    """

    import matplotlib.pyplot as plt

    m = turbigen.util.cluster_cosine(50)

    if ax is None:
        fig, ax = plt.subplots()

        ax.set_xlabel(r"Normalised Meridional Distance, $m$")
        ax.set_ylabel(r"Normalised Metal Angle, $\hat{\chi}$")
        ax.set_xlim((0.0, 1.0))

        # upper_lim = np.max((1.0, Chi.max()))
        # ax.set_ylim((0.0, upper_lim))

        # title_str = (
        #     (r"$\chi_\mathrm{in}=%.1f^\circ$, " % self.chi(0.0))
        #     + (r"$\chi_\mathrm{out}=%.1f^\circ$, " % self.chi(1.0))
        #     + (r"$\hat{\chi}'(0)=%.2f$, " % self.q_camber[2])
        #     + (r"$\hat{\chi}'(1)=%.2f$, " % self.q_camber[3])
        #     + (r"$\hat{\chi}''(0.5)=%.2f$" % self.q_camber[4])
        # )
        # print(title_str)

        # mctrl = (0.0, 0.5, 1.0)
        # ax.plot(mctrl, np.polyval(self._coeff, mctrl), "ro", ms=10)

        # ax.set_title(title_str, pad=10)
        plt.tight_layout()

    else:
        fig = ax.get_figure()

    for spf in (0.2, 0.5, 0.8):
        cam, _ = bld._get_cam_thick(spf)
        ax.plot(m, cam.chi(m))

    if fname:
        plt.savefig(fname)
        plt.close()

    return fig, ax


def plot_thickness(bld, fname=None, ax=None):
    r"""Plot this thickness distribution.

    Parameters
    ----------
    fname:
        Save the plot to this file, default to just showing.
    ax: matplotlib.pyplot.Axes
        Plot into an existing axes, default to new figure and axes.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The figure containing the plot.

    ax: matplotlib.pyplot.Axes
        Axes containing the plot.

    """

    self = bld

    _, thick = bld._get_cam_thick(0.5)

    import matplotlib.pyplot as plt

    m = turbigen.util.cluster_cosine(50)

    if ax is None:
        fig, ax = plt.subplots()

        col = "C0"

        ax.set_xlabel(r"Normalised Meridional Distance, $m$")
        ax.set_ylabel(r"Normalised Thickness, $t$ or $\tau$")
        ax.set_xlim((0.0, 1.0))

        title_str = (
            (r"$R_\mathrm{LE}=%.2f$, " % self.q_thick[0])
            + (r"$t_\mathrm{max}=%.2f$, " % self.q_thick[1])
            + (r"$x_{t_\mathrm{max}}=%.2f$, " % self.q_thick[2])
            + (r"$\kappa_{t_\mathrm{max}}=%.2f$, " % self.q_thick[3])
            + (r"$t_\mathrm{TE}=%.2f$, " % self.q_thick[4])
            + (r"$\tan\zeta=%.1f^\circ$" % self.q_thick[5])
        )

        ax.set_title(title_str, pad=10)

        plt.tight_layout()

    else:
        N = len(ax.lines) // 3
        col = "C%d" % N
        fig = ax.get_figure()
        ax.set_title("", pad=10)

    ax.plot(m, thick.t(m), "-", color=col, label="Real space, $t$")
    ax.plot(m, thick._tau(m), "--", color=col, label=r"Shape space, $\tau$")

    ax.legend()

    mctrl = (0.0, thick.s_tmax, 1.0)
    ax.plot(mctrl, thick._tau(mctrl), "o", color=col, ms=10)

    if fname:
        plt.savefig(fname)
        plt.close()

    return fig, ax


def plot_meridional_line(self, fname=None, ax=None):
    """Plot this meridional line.

    Parameters
    ----------
    fname:
        Save the plot to this file, default to just showing.
    ax: matplotlib.pyplot.Axes
        Plot into an existing axes, default to new figure and axes.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The figure containing the plot.

    ax: matplotlib.pyplot.Axes
        Axes containing the plot.

    """

    import matplotlib.pyplot as plt

    m = np.linspace(0.0, 1.0, 100 * self.N)

    if ax is None:
        fig, ax = plt.subplots()

        col = "C0"
        ax.set_xlabel(r"Axial Coordinate, $x$")
        ax.set_ylabel(r"Radial Coordinate, $r$")

    else:
        N = len(ax.lines) // 2
        col = "C%d" % N
        fig = ax.get_figure()

    ax.plot(*self.xr(m), "-", color=col)

    tctrl = np.linspace(0.0, self.N - 1.0, self.N)
    ax.plot(*self._xr(tctrl), "o", color=col, ms=10, markerfacecolor="none")
    ax.axis("equal")
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
        plt.close()

    return fig, ax


def plot_annulus(self, fname=None, xr_cut=None, ax=None, blades=None):
    r"""Plot this annulus.

    Parameters
    ----------
    fname:
        Save the plot to this file, default to just showing.
    ax: matplotlib.pyplot.Axes
        Plot into an existing axes, default to new figure and axes.

    Returns
    -------
    fig: matplotlib.pyplot.Figure
        The figure containing the plot.

    ax: matplotlib.pyplot.Axes
        Axes containing the plot.

    """

    import matplotlib.pyplot as plt

    m = np.linspace(0.0, 1.0, 100)

    if ax is None:
        fig, ax = plt.subplots()

        ax.set_xlabel("Axial Coordinate, $x$")
        ax.set_ylabel("Radial Coordinate, $r$")

        plt.tight_layout()
    else:
        fig = ax.get_figure()

    ax.plot(*self.hub.xr(m), "k-")
    ax.plot(*self.cas.xr(m), "k-")

    # ax.plot(*self.xr_mid(m), "k--")

    ax.plot(*self.hub.xr(self.hub.mctrl), "ro", fillstyle="none", ms=10)
    ax.plot(*self.cas.xr(self.cas.mctrl), "ro", fillstyle="none", ms=10)

    x = self.hub.xr((0.0, 1.0))[0]
    ax.plot(x, np.zeros_like(x), "k-.")

    if xr_cut is not None:
        for xri in xr_cut:
            ax.plot(*xri, "b--")

    ax.axis("equal")
    # ax.axis("off")
    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.tight_layout()
    # plt.show()

    if fname:
        plt.savefig(fname)
        plt.close()

    return fig, ax


def plot_blade(self, spf, ax=None, fname=None, xr=True):
    """Plot this blade."""

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
        plt.tight_layout()
    else:
        fig = ax.get_figure()

    spf = np.atleast_1d(spf)

    mq = np.linspace(0.0, 1.0, 5)

    for i in range(len(spf)):
        xrtu, xrtl = self.evaluate_section(spf[i])
        xrtuq, xrtlq = self.evaluate_section(spf[i], m=mq)
        xrrtu = np.stack((*xrtu[:2],) + (xrtu[1] * xrtu[2],))
        xrrtl = np.stack((*xrtl[:2],) + (xrtl[1] * xrtl[2],))
        xrrtuq = np.stack((*xrtuq[:2],) + (xrtuq[1] * xrtuq[2],))
        xrrtlq = np.stack((*xrtlq[:2],) + (xrtlq[1] * xrtlq[2],))
        xrrtc = 0.5 * (xrrtuq + xrrtlq)
        if xr:
            ax.plot(
                *xrrtu[
                    (0, 2),
                ],
                color="C%d" % i,
            )
            ax.plot(
                *xrrtl[
                    (0, 2),
                ],
                color="C%d" % i,
            )
            ax.plot(
                *xrrtc[
                    (0, 2),
                ],
                "x--",
                color="C%d" % i,
            )
        else:
            ax.plot(
                *xrrtu[
                    (1, 2),
                ],
                color="C%d" % i,
            )
            ax.plot(
                *xrrtl[
                    (1, 2),
                ],
                color="C%d" % i,
            )
            ax.plot(
                *xrrtc[
                    (1, 2),
                ],
                "x--",
                color="C%d" % i,
            )
        ax.axis("equal")
        # plt.tight_layout()

    if fname:
        plt.savefig(fname)
        plt.close()

    return fig, ax


def plot_splitter(main, split, Nb, fname=None):
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    plt.tight_layout()
    plt.tight_layout()
    spf = np.atleast_1d(0.5)

    mq = np.linspace(0.0, 1.0, 11)
    pitch = np.pi * 2.0 / Nb

    for i in range(len(spf)):
        for self in (main, split):
            for dt in (0.0, pitch):
                xrtu, xrtl = self.evaluate_section(spf[i])
                xrtuq, xrtlq = self.evaluate_section(spf[i], m=mq)
                xrrtu = np.stack((*xrtu[:2],) + (xrtu[1] * (xrtu[2] + dt),))
                xrrtl = np.stack((*xrtl[:2],) + (xrtl[1] * (xrtl[2] + dt),))
                xrrtuq = np.stack((*xrtuq[:2],) + (xrtuq[1] * (xrtuq[2] + dt),))
                xrrtlq = np.stack((*xrtlq[:2],) + (xrtlq[1] * (xrtlq[2] + dt),))
                xrrtc = 0.5 * (xrrtuq + xrrtlq)
                ax1.plot(
                    *xrrtu[
                        (0, 2),
                    ],
                    color="C%d" % i,
                )
                ax1.plot(
                    *xrrtl[
                        (0, 2),
                    ],
                    color="C%d" % i,
                )
                ax1.plot(
                    *xrrtc[
                        (0, 2),
                    ],
                    "x--",
                    color="C%d" % i,
                )
                ax1.axis("equal")
                ax2.plot(
                    *xrrtu[
                        (1, 2),
                    ],
                    color="C%d" % i,
                )
                ax2.plot(
                    *xrrtl[
                        (1, 2),
                    ],
                    color="C%d" % i,
                )
                ax2.plot(
                    *xrrtc[
                        (1, 2),
                    ],
                    "x--",
                    color="C%d" % i,
                )
            ax2.axis("equal")
            # plt.tight_layout()

    if fname:
        fig1.savefig(fname)
        fig2.savefig(fname.replace(".pdf", "rrt.pdf"))
        plt.close("all")

    return fig1, ax1


def plot_grid_b2b(g, spf, axial, fname=None):
    C = g.cut_span(spf)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")
    for b in C:
        bs = b.squeeze()
        bsr = bs.copy()
        bsr.t += 2 * np.pi / bsr.Nb
        for bb in (bs, bsr):
            if axial:
                ax.plot(bb.x, bb.rt, "k-", lw=0.1)
                ax.plot(bb.x.T, bb.rt.T, "k-", lw=0.1)
            else:
                ax.plot(bb.y, bb.z, "k-", lw=0.1)
                ax.plot(bb.y.T, bb.z.T, "k-", lw=0.1)

    plt.tight_layout()
    plt.savefig(fname)

    if fname:
        plt.savefig(fname)
        plt.close()


def plot_hmesh(g, workdir):
    plot_grid_meridional(g, os.path.join(workdir, "mesh_xr.pdf"))
    plot_grid_b2b(g, 0.5, True, fname=os.path.join(workdir, "mesh_b2b.pdf"))


def plot_grid_meridional(g, fname=None):
    fig, ax = plt.subplots()
    ax.axis("equal")
    for b in g:
        C = b[:, :, 0].squeeze()
        ax.plot(C.x, C.r, "k-", lw=0.2)
        ax.plot(C.x.T, C.r.T, "k-", lw=0.2)

    if fname:
        plt.savefig(fname)
        plt.close()

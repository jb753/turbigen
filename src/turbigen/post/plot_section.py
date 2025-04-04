import os
import turbigen.util
import matplotlib.pyplot as plt
import numpy as np

logger = turbigen.util.make_logger()


def post(
    grid,
    machine,
    meanline,
    _,
    postdir,
    row_spf,
    coord_sys="mpt",
    compare=None,
    K_offset=0.0,
    t_offset=0.0,
    x_cut=None,
):
    """plot_section(row_spf, coord_sys="mpt", compare=None, K_offset=0.0)
    Plot views of the blade sections.

    Parameters
    ----------
    row_spf: list
        For each row of the machine,  a nested list of span fractions to plot at. For example, in a three-row machine,
        to plot the first row at mid-span, the second row at three locations, and omit the third row, use
        `[[0.5,], [0.1, 0.5, 0.9,], []]`.
    coord_sys: str
        Which coordinate system to use, `mpt` for unwrapped radial or `xrt` for axial machines.
    compare: str
        Path to an xrt data file with coordinates to compare to current sections.
    K_offset: float
        Tangential offset factor of the sections. Use to prevent sections from overlapping.

    """

    # Loop over rows
    for irow, spfrow in enumerate(row_spf):
        if not spfrow:
            continue

        logger.info(f"Plotting section row={irow} at spf={spfrow}")

        # Store section coordinates
        sect_grid = []
        sect_compare = []

        fig, ax = plt.subplots()

        # Loop over span fractions
        for ispf, spf in enumerate(spfrow):
            jspf = grid.spf_index(spf)

            if x_cut:
                xrnow = np.stack(
                    (
                        np.full((2,), x_cut[ispf]),
                        (1e-6, 1e3),
                    )
                )
                surf = grid.cut_blade_surfs()[irow][0].meridional_slice(xrnow)
            else:
                surf = grid.cut_blade_surfs()[irow][0][:, jspf, :].squeeze()

            # if ispf == 0:
            #     tavg = 0.5 * (surf.t.min() + surf.t.max()) - np.pi / 2.0

            mlim = (-0.1, 1.1)
            mp_from_xr, spf_actual = turbigen.util.get_mp_from_xr(
                grid, machine, irow, spf, mlim
            )

            mps = mp_from_xr(surf.xr)

            if coord_sys == "mpt":
                x1 = mps
                x2 = surf.t
            elif coord_sys == "xrt":
                x1 = surf.x
                x2 = surf.rt
            elif coord_sys == "rrt":
                x1 = surf.r
                x2 = surf.rt
            elif coord_sys == "yz":
                x1 = -surf.y
                x2 = surf.z

                # # Extract coordinates
                # yz = np.stack((surf.y,surf.z))

                # # Rotate so that r is horizontal
                # cost = np.cos(tavg)
                # sint = np.sin(tavg)
                # Rot = np.array([[cost, -sint], [sint, cost]])
                # yz = Rot @ yz

                # if ispf==0:
                #     for tnow in (surf.t.min(), surf.t.max()):
                #         rref = np.linspace(surf.r.min(), surf.r.max())
                #         tref = np.ones_like(rref)*tnow
                #         yzref = np.stack((rref * np.sin(tref),rref * np.cos(tref)))
                #         yzref = Rot @ yzref
                #         ax.plot(*yzref,'k--')

                # ax.plot(*yz, "-", label=f"spf={spf}")

            xoff = K_offset * (spf - 0.5) * np.ptp(x2)
            # ax.plot(x1, x2 + xoff, "-", label=f"spf={spf}")
            sect_grid.append((x1, x2 + xoff))

            if compare:
                if compare_dat := compare[irow]:
                    xrrt = turbigen.util.read_sections(compare_dat)[ispf]
                    if coord_sys == "xrt":
                        x1c = xrrt[0]
                        x2c = xrrt[2]
                    elif coord_sys == "yz":
                        tc = -xrrt[2] / xrrt[1]

                        x1c = xrrt[1] * np.sin(tc)
                        x2c = xrrt[1] * np.cos(tc)

                    elif coord_sys == "rrt":
                        x1c = xrrt[1]
                        tc = xrrt[2] / xrrt[1]
                        tc -= tc.min()
                        tc *= -1
                        x2c = x1c * tc
                    else:
                        pass

                    # Shuffle and only select 100 points
                    ishuf = list(range(len(x1c)))
                    np.random.shuffle(ishuf)
                    ishuf = ishuf[:1000]
                    x1c = x1c[ishuf]
                    x2c = x2c[ishuf]

                sect_compare.append((x1c, x2c + xoff))

            # if coord_sys == "yz":
            #     N = 50
            #     rln = surf.r.min() * np.ones((N,))
            #     tln = -np.linspace(surf.t.min(), surf.t.max(), N)
            #     yln = rln * np.sin(tln)
            #     zln = rln * np.cos(tln)
            #     ax.plot(yln, zln, "k--")

        # Now plot the compare sections first
        if compare:
            for ispf in range(len(spfrow)):
                ax.plot(*sect_compare[ispf], "kx", ms=2)  # , color=f"C{ispf}", ms=2)
        for ispf, spf in enumerate(spfrow):
            ax.plot(*sect_grid[ispf], "-", color=f"C{ispf}", label=f"spf={spf}")

        ax.legend()
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        plotname = os.path.join(postdir, f"section_row_{irow}.pdf")
        plt.tight_layout(pad=0)
        plt.savefig(plotname)
        plt.close()

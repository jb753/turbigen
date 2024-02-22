import numpy as np
import turbigen.geometry
import turbigen.grid
import os
import turbigen.util

logger = turbigen.util.make_logger()


def _calc_blade_inc(irow, current_surf, current_blade, current_Beta, plot, plot_prefix):

    if plot_prefix:
        import matplotlib.pyplot as plt

    surf = current_surf[..., 0]

    # Preallocate and loop over spanwise points
    nj = current_surf.shape[1]

    jplot = np.round(
        np.interp(
            [0.2, 0.5, 0.8], current_surf.spf[0, :, 0], np.linspace(0.0, nj - 1, nj)
        )
    ).astype(int)

    spf_row = np.ones(nj) * np.nan
    chi_stag_row = np.ones(nj) * np.nan
    for j in range(nj):
        # Find current span fraction
        spf_row[j] = spf_now = current_surf.spf[0, j, 0]

        xrrt_le_cent = current_blade.get_LE_cent(spf_now)

        x = surf.x[:, j]
        r = surf.r[:, j]
        rt = surf.rt[:, j]
        Vi = surf.Vi_rel[:, j]

        # Take surface distance and normalise to (-1, 1)
        zeta = surf.zeta[:, j]
        zeta /= 0.5 * zeta.ptp()
        zeta -= 1.0

        # Find a zero crossing near LE
        iz_all = np.where(
                (np.abs(np.diff(np.sign(Vi))) > 0) & (np.abs(zeta) < 0.3)[:-1]
        )[0]
        iiz = np.argmin(np.abs(zeta[iz_all]))
        iz = iz_all[iiz]

        zeta_stag = np.interp(0, Vi[iz : iz + 2], zeta[iz : iz + 2])

        if plot and j == 25 and plot_prefix:
            fig, ax = plt.subplots()
            ax.plot(zeta, Vi, "k")
            ax.plot(zeta[iz], Vi[iz], "b*")
            ax.plot(zeta_stag, np.interp(zeta_stag, zeta, Vi), "ro")
            plt.savefig(f"zeta_Vi_j_25_{plot_prefix}_irow_{irow}.pdf")

        x_stag = np.interp(zeta_stag, zeta, x)
        rt_stag = np.interp(zeta_stag, zeta, rt)
        r_stag = np.interp(zeta_stag, zeta, r)
        xrrt_le_stag = np.array((x_stag, r_stag, rt_stag)).reshape(-1, 1)

        # Get angle between stagnation point and centre of LE
        dxrrt = xrrt_le_cent - xrrt_le_stag

        # Small pitch angle => the leading edge is oriented axially
        denom = np.sqrt(dxrrt[0]**2 + dxrrt[1]**2)
        chi_stag_row[j] = np.degrees(np.arctan2(dxrrt[2],denom)).item()

        # If we are going radially inwards then flip sign
        if current_Beta < 0.0:
            chi_stag_row[j] *= -1.0

        # Choose how to calculate angle
        if np.abs(current_Beta) < 45.0:

            if plot and j in jplot and plot_prefix:
                fig, ax = plt.subplots()
                ax.plot(surf.x[:, j], surf.rt[:, j], "-")
                ax.plot(
                    *xrrt_le_stag[
                        (0, 2),
                    ],
                    "r*",
                )
                ax.plot(
                    *xrrt_le_cent[
                        (0, 2),
                    ],
                    "b*",
                )
                ax.axis("equal")
                Lref = turbigen.util.vecnorm(dxrrt) * 2.0
                ax.set_ylim(np.array((-Lref, Lref)) + xrrt_le_cent[2])
                ax.set_xlim(np.array((-Lref, Lref)) + xrrt_le_cent[0])
                plt.savefig(
                    os.path.join(plot, f"inc_xr_{plot_prefix}_row_{irow}_j_{j}.pdf")
                )
                plt.close()

        else:
            # Large pitch angle => the leading edge is oriented radially

            if plot and j in jplot and plot_prefix:
                fig, ax = plt.subplots()
                ax.plot(surf.r[:, j], surf.rt[:, j], "-")
                ax.plot(
                    *xrrt_le_stag[
                        (1, 2),
                    ],
                    "r*",
                )
                ax.plot(
                    *xrrt_le_cent[
                        (1, 2),
                    ],
                    "b*",
                )
                ax.axis("equal")
                Lref = turbigen.util.vecnorm(dxrrt) * 2.0
                ax.set_ylim(np.array((-Lref, Lref)) + xrrt_le_cent[2])
                ax.set_xlim(np.array((-Lref, Lref)) + xrrt_le_cent[1])

                ax.plot(
                    [xrrt_le_cent[1], xrrt_le_cent[1] + Lref],
                    [xrrt_le_cent[2], xrrt_le_cent[2]],
                    "k--",
                )
                ax.set_title(str(chi_stag_row[j]) + ", " + str(spf_now))
                plt.savefig(
                    os.path.join(plot, f"inc_rrt_{plot_prefix}_row_{irow}_j_{j}.pdf")
                )
                plt.close()

    return spf_row, chi_stag_row


def incidence(g, machine, Beta_in, plot=False):
    """Spanwise profile of incidence for each blade row.

    Note that this assumes small pitch angles, i.e. will not work for radial turbines.
    But will work for axial machines and radial compressors."""

    if plot:
        import matplotlib.pyplot as plt

    surfs = g.cut_blade_surfs()

    # Preallocate and loop over rows
    spf = []
    chi_stag = []
    chi_stag_splitter = []
    for irow, blocks in enumerate(g.row_blocks):

        if surfs[irow] is None:
            spf.append(None)
            chi_stag.append(None)
            continue

        current_surf = surfs[irow][0]
        current_Beta = Beta_in[irow]
        current_blade = machine.bld[irow]

        spf_row, chi_stag_row = _calc_blade_inc(
            irow, current_surf, current_blade, current_Beta, plot, "main"
        )

        spf.append(spf_row)
        chi_stag.append(chi_stag_row)

        if len(surfs[irow]) > 1:
            chi_stag_splitter.append(
                _calc_blade_inc(
                    irow,
                    surfs[irow][1],
                    machine.split[irow],
                    current_Beta,
                    plot,
                    "splitter",
                )[1]
            )
        else:
            chi_stag_splitter.append(None)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(chi_stag[-1], spf[-1], "-x")

        # Smooth to remove sharp discontinuities
        inan = np.logical_not(np.isnan(chi_stag[-1]))
        nsmooth = 1
        chi_stag_smooth = chi_stag[-1][inan]
        for _ in range(nsmooth):
            chi_stag_smooth[1:-1] = 0.5 * (chi_stag_smooth[2:] + chi_stag_smooth[:-2])
        chi_stag[-1][inan] = chi_stag_smooth

        if plot:
            ax.plot(chi_stag[-1], spf[-1], "--o")
            plt.savefig(os.path.join(plot, f"inc_spf_row_{irow}.pdf"))
            plt.close()

    return spf, chi_stag, chi_stag_splitter


def check_phase(g):
    for ib, b in enumerate(g):
        n = (b.is_two_phase).sum()
        if n:
            logger.info(f"Block {ib}: {n}/{b.size} cells not in gas phase")


def surface_dissipation(g, Cd=0.002):
    """Integrate surface dissipation."""

    # Get inlet entropy
    Cin = g.inlet_patches[0].get_cut().mix_out()[0]
    sin = Cin.s

    def _integrate_cut(C, rel, w=None):
        C = C.squeeze()
        # Pull out the face-centered properties we need
        dA = np.linalg.norm(C.surface_area, axis=-1, ord=2)
        rho = turbigen.util.node_to_face(C.rho)
        T = turbigen.util.node_to_face(C.T)

        # Isentropic to local static pressure
        Cs = C.copy().set_P_s(C.P, sin)
        hs = turbigen.util.node_to_face(Cs.h)

        # Subtract isentropic static enthalpy from real relative stagnation
        # enthalpy to get relative isenstopic exit dynamic head
        if rel:
            ho_rel = turbigen.util.node_to_face(C.ho_rel)
            Vs = np.sqrt(2.0 * np.maximum(ho_rel - hs, 0.0))
        else:
            ho = turbigen.util.node_to_face(C.ho)
            Vs = np.sqrt(2.0 * np.maximum(ho - hs, 0.0))

        # Apply weight for cuts that are not all wall
        if w is None:
            wf = np.ones_like(dA)
        else:
            wf = turbigen.util.node_to_face(w.squeeze())

        # Multiply by the wall indicator to zero out non-walls
        # Perform the integration and accumulate total
        return Cd * np.sum(wf * rho * Vs**3.0 / T * dA) * C.Nb, np.sum(wf * dA) * C.Nb

    Sdot = np.zeros((g.nrow, 2))
    Asurf = np.zeros((g.nrow, 2))
    for irow, row_block in enumerate(g.row_blocks):
        for block in row_block:
            # Preallocate wall indicator
            is_wall = np.ones(block.shape)
            is_wall[1:-1, 1:-1, 1:-1] = 0.0
            is_rot = np.zeros(block.shape)

            # Loop over patches
            for patch in block.patches:
                # Unset wall indicator if patch is not wall
                if isinstance(patch, turbigen.grid.RotatingPatch):
                    is_rot[patch.get_slice()] = 1.0
                if type(patch) in turbigen.grid.NOT_WALL_PATCHES:
                    is_wall[patch.get_slice()] = 0.0

            # Hub and casing
            for ind in (0, -1):
                if is_rot[:, ind, :].all():
                    Sdot_now, A_now = _integrate_cut(block[:, ind, :], True)
                else:
                    Sdot_now, A_now = _integrate_cut(block[:, ind, :], False)
                Sdot[irow, 0] += Sdot_now
                Asurf[irow, 0] += A_now

            # Blade surfaces
            for ind in (0, -1):
                if is_rot[:, :, ind].all():
                    Sdot_now, A_now = _integrate_cut(
                        block[:, :, ind], True, is_wall[:, :, ind]
                    )
                else:
                    Sdot_now, A_now = _integrate_cut(
                        block[:, :, ind], False, is_wall[:, :, ind]
                    )
                Sdot[irow, 1] += Sdot_now
                Asurf[irow, 1] += A_now

    return Sdot, Asurf


def blockage(g, C):
    """Evaluate blockage over a cut."""

    C = C.squeeze()

    # Pull out the face-centered properties we need
    dA = np.linalg.norm(C.surface_area, axis=-1, ord=2).reshape(-1)
    A = np.sum(dA)
    rhoVr = turbigen.util.node_to_face(C.rhoVr).reshape(-1)

    # Sort in order of descending rhoVr
    isort = np.argsort(-rhoVr)
    dA = dA[isort]
    rhoVr = rhoVr[isort]

    # Define free-stream velocity by integrating of the 20% of area with highest rhoVr
    iinf = np.argmax(np.cumsum(dA) > 0.2 * A)
    Ainf = np.sum(dA[:iinf])
    rhoVr_inf = np.sum((rhoVr * dA)[:iinf]) / Ainf

    return np.sum((1.0 - rhoVr / rhoVr_inf) * dA) / A


def tip(g):
    # Get inlet entropy
    Cin = g.inlet_patches[0].get_cut().mix_out()[0]
    sin = Cin.s

    C = g.cut_blade_sides()[0]

    Vs = np.full((2,) + C[0].shape, np.nan)
    for i in range(2):
        # Isentropic to local static pressure
        Cs = C[i].copy().set_P_s(C[i].P, sin)

        # Subtract isentropic static enthalpy from real relative stagnation
        # enthalpy to get relative isenstopic exit dynamic head
        Vs[i] = np.sqrt(2.0 * np.maximum(C[i].ho_rel - Cs.h, 0.0))

    jref = np.argmin(np.abs(C[0].spf[0, :] - 0.9))
    imid = C[0].shape[0] // 2

    dtheta = np.abs(C[0].t[imid, :, 0] - C[1].t[imid, :, 0])
    jtip = np.argmax(dtheta < 1e-5)

    # Now cut the tip
    Ctip = C[0][:, jtip:, 0]
    Vtreltip = turbigen.util.node_to_face(Ctip.Vt_rel)
    rhotip = turbigen.util.node_to_face(Ctip.rho)

    dm = np.abs(Ctip.dAt * rhotip * Vtreltip)

    if C[0].P.mean() > C[1].P.mean():
        Cps, Css = C
        Vps = turbigen.util.node_to_face(Vs[0, :, jref : (jref + 2), 0])
        Vss = turbigen.util.node_to_face(Vs[1, :, jref : (jref + 2), 0])
        Tss = turbigen.util.node_to_face(C[1].T[:, jref : (jref + 2), 0])
    else:
        Css, Cps = C
        Vps = turbigen.util.node_to_face(Vs[1, :, jref : (jref + 2), 0])
        Vss = turbigen.util.node_to_face(Vs[0, :, jref : (jref + 2), 0])
        Tss = turbigen.util.node_to_face(C[0].T[:, jref : (jref + 2), 0])

    # mdot_tip = np.sum(dm) * C[0].Nb
    Sdot_tip = np.sum(Vss**2 / Tss * (1.0 - Vps / Vss) * dm) * C[0].Nb

    return Sdot_tip


def ske(C):
    # Get bulk flow direction
    Cm = C.mix_out()[0]
    norm = (Cm.Vxrt / Cm.V).reshape(1, 1, 1, -1)

    # Numpy cross function assumes that the components are in last axis
    Vxrt = np.moveaxis(C.Vxrt, 0, -1).astype(np.float64)

    # The secondary flow vector is perp to bulk flow dirn
    Vsec = np.cross(Vxrt, norm).squeeze()

    dA = C.squeeze().surface_area_xrrt
    # dA[...,(0,1),] = 0.
    rho_cell = turbigen.util.node_to_face(C.squeeze().rho)
    Vxrt_cell = np.stack(
        [turbigen.util.node_to_face(C.squeeze().Vxrt[i]) for i in range(3)], axis=-1
    )
    dm = np.full_like(dA[..., 0], np.nan)
    Vsec_cell = np.stack(
        [turbigen.util.node_to_face(Vsec[..., i]) for i in range(3)], axis=-1
    )
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            dm[i, j] = np.sum(dA[i, j, :] * Vxrt_cell[i, j, :]) * rho_cell[i, j]

    ske = np.abs(np.sum(dm[..., None] * Vsec_cell**2)) * C.Nb

    # Cske = ske / np.sum(dm) / Cm.V**2
    return ske


def Ms(g):
    # Inlet entropy
    Cin = g.inlet_patches[0].get_cut().mix_out()[0]
    sin = Cin.s

    # Isentropic Mach
    C = g.cut_blade_sides()[0]
    Cs = [Ci.copy().set_P_s(Ci.P, sin) for Ci in C]
    Ms = np.stack(
        [
            np.sqrt(2.0 * np.maximum(Ci.ho_rel - Csi.h, 0.0)) / Ci.a
            for Ci, Csi in zip(C, Cs)
        ]
    )
    return Ms


#     # Extract on some js
#     fig, ax = plt.subplots()
#     ax.axis("equal")
#     for b in g:
#         xrnow = b[:, j, :].xr
#         Lnow = turbigen.geometry.DiscreteMeridionalLine(xrnow)
#         C = b[:, j, :].squeeze()
#         mp = Lnow.mp_from_xr(C.xr)

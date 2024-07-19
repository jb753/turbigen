import numpy as np
import turbigen.geometry
import turbigen.grid
import turbigen.util

logger = turbigen.util.make_logger()


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

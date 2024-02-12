from turbigen.solvers.base import BaseSolver
import numpy as np
import turbigen.util
import turbigen.laplacian
from turbigen.smooth import smooth


# class NativeConfig(BaseSolver):
#     """Settings with default values for the TS4 solver."""

#     _name = "Native"


def get_timestep(b):
    Vref = node_to_vol(b.V)
    aref = node_to_vol(b.a)
    dli = turbigen.util.vecnorm(b.dli)
    dlj = turbigen.util.vecnorm(b.dlj)
    dlk = turbigen.util.vecnorm(b.dlk)
    dli = 0.25 * (
        dli[:, :-1, :-1] + dli[:, :-1, 1:] + dli[:, 1:, :-1] + dli[:, :-1, :-1]
    )
    dlj = 0.25 * (
        dlj[:-1, :, :-1] + dlj[:-1, :, 1:] + dlj[1:, :, :-1] + dlj[:-1, :, :-1]
    )
    dlk = 0.25 * (
        dlk[:-1, :-1, :] + dlk[:-1, 1:, :] + dlk[1:, :-1, :] + dlk[:-1, :-1, :]
    )
    dlmin = np.minimum(dli, dlj, dlk)
    dt = CFL * dlmin / (aref + Vref)
    return dt


def node_to_face(x):
    # x has shape [?,ni,nj,nk]
    # return averaged values on const i, const j, const k faces
    # xi [?,ni,nj-1, nk-1]
    # xj [?,ni-1,nj, nk-1]
    # xk [?,ni-1,nj-1, nk]

    xi = np.stack(
        (
            x[..., :, :-1, :-1],
            x[..., :, 1:, :-1],
            x[..., :, 1:, 1:],
            x[..., :, :-1, 1:],
        ),
    ).mean(axis=0)

    xj = np.stack(
        (
            x[..., :-1, :, :-1],
            x[..., 1:, :, :-1],
            x[..., 1:, :, 1:],
            x[..., :-1, :, 1:],
        ),
    ).mean(axis=0)

    xk = np.stack(
        (
            x[..., :-1, :-1, :],
            x[..., 1:, :-1, :],
            x[..., 1:, 1:, :],
            x[..., :-1, 1:, :],
        ),
    ).mean(axis=0)

    return xi, xj, xk


def node_to_vol(x):
    # x has shape [?,ni,nj,nk]
    # return averaged values for each cell
    # xi [?,ni-1,nj-1, nk-1]
    return np.stack(
        (
            x[..., :-1, :-1, :-1],  # i, j, k
            x[..., 1:, :-1, :-1],  # i+1, j, k
            x[..., 1:, 1:, :-1],  # i+1, j+1, k
            x[..., :-1, 1:, :-1],  # i, j+1, k
            x[..., :-1, :-1, 1:],  # i, j, k+1
            x[..., 1:, :-1, 1:],  # i+1, j, k+1
            x[..., 1:, 1:, 1:],  # i+1, j+1, k+1
            x[..., :-1, 1:, 1:],  # i, j+1, k+1
        ),
    ).mean(axis=0)


def cell_to_node(x):
    # x has shape [?,ni-1,nj-1,nk-1]
    # return values for each node
    # xi [?,ni,nj, nk]
    *other, nim1, njm1, nkm1 = x.shape
    ni = nim1 + 1
    nj = njm1 + 1
    nk = nkm1 + 1

    xn = np.full(tuple(other) + (ni, nj, nk), np.nan)

    # Interior nodes take 1/8th from each of i
    xn[..., 1:-1, 1:-1, 1:-1] = node_to_vol(x)

    # i=(0,-1) takes 1/4 from j, j+1, k, k+1
    for i in (0, -1):
        xn[..., i, 1:-1, 1:-1] = np.stack(
            (
                x[..., i, :-1, :-1],  # j, k
                x[..., i, 1:, :-1],  # j+1, k
                x[..., i, :-1, 1:],  # j, k+1
                x[..., i, 1:, 1:],  # j+1, k+1
            )
        ).mean(axis=0)

    # j=(0,-1) takes 1/4 from i, i+1, k, k+1
    for j in (0, -1):
        xn[..., 1:-1, j, 1:-1] = np.stack(
            (
                x[..., :-1, j, :-1],
                x[..., 1:, j, :-1],
                x[..., :-1, j, 1:],
                x[..., 1:, j, 1:],
            )
        ).mean(axis=0)

    # k=(0,-1) takes 1/4 from i, i+1, k, k+1
    for k in (0, -1):
        xn[..., 1:-1, 1:-1, k] = np.stack(
            (
                x[..., :-1, :-1, k],
                x[..., 1:, :-1, k],
                x[..., :-1, 1:, k],
                x[..., 1:, 1:, k],
            )
        ).mean(axis=0)

    # Edges take half from nearest two cells

    # Along i lines
    for j in (0, -1):
        for k in (0, -1):
            xn[..., 1:-1, j, k] = 0.5 * (x[..., :-1, j, k] + x[..., 1:, j, k])

    # Along j lines
    for i in (0, -1):
        for k in (0, -1):
            xn[..., i, 1:-1, k] = 0.5 * (x[..., i, :-1, k] + x[..., i, 1:, k])

    # Along k lines
    for i in (0, -1):
        for j in (0, -1):
            xn[..., i, j, 1:-1] = 0.5 * (x[..., i, j, :-1] + x[..., i, j, 1:])

    # Corners take entire change from nearest cell
    for i in (0, -1):
        for j in (0, -1):
            for k in (0, -1):
                xn[..., i, j, k] = x[..., i, j, k]

    return xn


# def convective_fluxes(U_node, ho_node, r_face, wall):
#     """Calculate the convective fluxes from nodal conserved vars."""

#     U_face = node_to_face(U_node)
#     ho_face = node_to_face(ho_node)

#     # Loop over i/j/k faces
#     flux = []
#     for n in range(3):

#         rho, rhoVx, rhoVr, rhorVt, rhoe  = U_face[n]
#         ho = ho_face[n]

#         rVt = rhorVt / rho
#         rhoVt = rhorVt / r_face[n]
#         Vx = rhoVx / rho
#         Vr = rhoVr / rho

#         flux_mass = np.expand_dims(np.stack((rhoVx, rhoVr, rhoVt)),axis=1)

#         flux_all = flux_mass * np.stack(
#             (
#                 np.ones_like(rho),  # mass
#                 Vx,  # x-mom
#                 Vr, # r-mom
#                 rVt, # rt-mom
#                 ho, # energy
#             )
#         )

#         # Sum over coordinate directions
#         flux_sum = np.sum(flux_all, axis=0)

#         flux_sum[:,wall[n]] = 0.

#         flux.append(flux_sum)

#     return flux

# def pressure_fluxes(P_node, r_face, Omega):

#     P_face = node_to_face(P_node)

#     # Loop over i/j/k faces
#     flux = []
#     for n in range(3):

#         P = P_face[n]
#         Z = np.zeros_like(P)
#         r = r_face[n]

#         flux_all = np.array(
#             (
#                 (Z, Z, Z), # mass
#                 (P, Z, Z), # x-mom
#                 (Z, P, Z), # r-mom
#                 (Z, Z, r*P), # rt-mom
#                 (Z, Z, Z), # energy TODO put Omega term in here
#             )
#         )

#         flux.append(np.stack(
#             (
#                 np.zeros_like(P),  # mass
#                 Vx,  # x-mom
#                 Vr, # r-mom
#                 rVt, # rt-mom
#                 ho, # energy
#             )
#         )

#     )

#     return flux

# def get_fluxes(vars_node, Omega):

#     vars_face = node_to_face(vars_node)

#     # vars_node = np.stack((*conserved, P, ho, b.r))
#     flux = []
#     for n in range(3):

#         rho, rhoVx, rhoVr, rhorVt, rhoe, P, ho, r  = vars_face[n]

#         rVt = rhorVt / rho
#         rhoVt = rhorVt / r
#         Vx = rhoVx / rho
#         Vr = rhoVr / rho
#         Vt = rhoVt / rho

#         flux.append(np.array(
#             (
#                 (rhoVx, rhoVr, rhoVt),  # mass
#                 (rhoVx * Vx + P, rhoVr * Vx, rhoVt * Vx),  # x-mom
#                 (rhoVx * Vr, rhoVr * Vr + P, rhoVt * Vr),  # r-mom
#                 (rhoVx * rVt, rhoVr * rVt, rhoVt * Vt + r * P),  # rt-mom
#                 (rhoVx * ho, rhoVr * ho, rhoVt * ho + Omega * r * P),  # energy
#             )
#         ))

#     return flux


sfin = 0.5
CFL = 0.4
sf = CFL * sfin


# def smooth(x):
#     xs = x.copy()
#     for i in range(x.shape[0]):
#         xs[i] += turbigen.laplacian.laplacian2(x[i], sf / 6.0)
#     return xs


def get_wall(b):
    # Find logical indices that zero the fluxes on wall faces
    thresh = 0.99  # To allow for floating point error
    return [w > thresh for w in node_to_face(b.get_wall())]


def apply_bconds(b):
    """Return properties needed to time march after in/out boundaries applied"""

    P = b.P.copy()
    ho = b.ho.copy()
    conserved = b.conserved.copy()

    rho, rhoVx, rhoVr, rhorVt, rhoe = conserved

    # Adjust static pressure for exit boundary conditions
    for patch in b.outlet_patches:
        P[patch.get_slice()] = patch.Pout * 0.1 + 0.9 * (patch.get_cut().P)

    # Change inlet patches
    for patch in b.inlet_patches:

        ipatch = patch.get_slice()
        inlet = b[ipatch]
        rfin = patch.rfin

        if patch.store:
            # Relax changes in density if we have a stored state
            rho_now = rfin * inlet.rho + (1.0 - rfin) * patch.store.rho
            # Check for flow reversal
            rho_now[rho_now > patch.state.rho] = patch.state.rho * 0.9999
            # Isentropic expansion from stagnation state
            inlet.set_rho_s(rho_now, patch.state.s)
            assert np.allclose(inlet.rho, rho_now)

        patch.store = inlet

        # Get the velocity
        dhin = patch.state.h - inlet.h
        if (dhin <= 0.0).any():
            assert False
        Vin = np.sqrt(2 * dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin * np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
        Vrin = Vxin * tanBeta
        Vmin = np.sqrt(Vxin**2 + Vrin**2)
        Vtin = Vmin * tanAlpha

        # Reset conserved vars on inlet
        # rho[ipatch] = inlet.rho
        rhoVx[ipatch] = inlet.rho * Vxin
        rhoVr[ipatch] = inlet.rho * Vrin
        rhorVt[ipatch] = inlet.rho * inlet.r * Vtin
        rhoe[ipatch] = inlet.rho * (inlet.u + 0.5 * Vin**2)

        # Reset pressure and hstag on inlet
        ho[ipatch] = inlet.h + 0.5 * Vin**2
        P[ipatch] = inlet.P

    Pref = 1e5
    P -= Pref

    # All nodal variables are ready
    return np.stack((*conserved, P, ho, b.r))


def step(b, dt, wall):

    assert b.Omega.ptp() < 1e-6
    Omega = b.Omega.mean()

    conservedPhor = apply_bconds(b)

    # Flux vectors
    fi, fj, fk = get_fluxes(conservedPhor, Omega, wall)

    # ff = fi, fj, fk
    # print('**Flux vectors')
    # for n, lab in enumerate(('i', 'j', 'k')):
    #     print(f'  {lab} faces')
    #     for m, kind in enumerate(('mass', 'xmom', 'rmom', 'rtmom', 'ho')):
    #         for p, c in enumerate(('x','r','t')):
    #             print(f'    flux of {kind} in {c}-dirn: {ff[n][m][p].mean()}')

    # Dot with areas and sum over cells
    Fi = np.sum(fi * b.dAi, axis=1)
    Fj = np.sum(fj * b.dAj, axis=1)
    Fk = np.sum(fk * b.dAk, axis=1)

    # ff = [Fi, Fj, Fk]
    # print('**Total fluxes')
    # for n, lab in enumerate(('i', 'j', 'k')):
    #     print(f'  {lab} faces')
    #     for m, kind in enumerate(('mass', 'xmom', 'rmom', 'rtmom', 'ho')):
    #         print(f'    flux of {kind}: {ff[n][m].mean()}')

    Fnet = [
        -np.diff(Fi, axis=-3),
        -np.diff(Fj, axis=-2),
        -np.diff(Fk, axis=-1),
    ]

    vol_r = (
        - np.diff(b.dAi[1], axis=0)
        - np.diff(b.dAj[1], axis=1)
        - np.diff(b.dAk[1], axis=2)
    )
    rvol = node_to_vol(b.r)
    print(vol_r.flat[0], (b.vol/rvol).flat[0])



    S = get_source(conservedPhor)
    Svol = np.mean(S * b.vol, axis=(-1, -2, -3))

    # print('**Net fluxes')
    # for n, lab in enumerate(('i', 'j', 'k')):
    #     print(f'  {lab} faces')
    #     for m, kind in enumerate(('mass', 'xmom', 'rmom', 'rtmom', 'ho')):
    #         Fnow = Fnet[n][m]
    #         print(f'    flux of {kind}: {Fnow.mean(), Fnow.min(), Fnow.max()}')

    fsum = (
        -np.diff(Fi, axis=-3)  # i faces
        - np.diff(Fj, axis=-2)  # j faces
        - np.diff(Fk, axis=-1)  # k faces
    )

    # print('**After source')
    # for m, kind in enumerate(('mass', 'xmom', 'rmom', 'rtmom', 'ho')):
    #     print(f'    {kind}: {fsum[m].mean() + Svol[m].mean()}')
    # quit()

    dU = (fsum / b.vol + S) * dt

    # dUabs = np.abs(dU)
    # dUav = dUabs.mean(axis=(-3,-2,-1))
    # dUav[dUav<=0.] = 1e-9
    # dUav = np.expand_dims(dUav, (1,2,3))
    # damp = 10.
    # dU /= (1.+dUabs/dUav/damp)

    # dU[2] = 0.
    # dU[-1,...] = 0.

    Unew = b.conserved + cell_to_node(dU)

    Unew = smooth(Unew, sf)
    # Unew[:, 0, :, :] = 0.5 * (Unew[:, 0, :, :] + Unew[:, 1, :, :])
    # Unew[:, -1, :] = 0.5 * (Unew[:, -1, :, :] + Unew[:, -2, :, :])

    # # # Extra smoothing at inlet and exit
    # sff = 0.05
    # sf1 = 1.-sff
    # Unew[:,0] = sf1*Unew[:,0]+sff*Unew[:,1]
    # Unew[:,-1] = sf1*Unew[:,-1]+sff*Unew[:,-2]

    b.set_conserved(Unew)
    assert np.allclose(b.conserved, Unew)

    return dU


def get_source(conservedPhor_node):

    rho, rhoVx, rhoVr, rhorVt, rhoe, P, ho, r = conservedPhor_node
    Vt = rhorVt / rho / r
    Z = np.zeros_like(rho)
    S_node = np.stack(
        (
            Z,  # mass
            Z,  # xmom
            (P + rho * Vt**2) / r,  # rmom
            Z,  # rtmom
            Z,  # energy
        )
    )

    S_vol = node_to_vol(S_node)

    return S_vol


def get_fluxes(conservedPhor_node, Omega, wall):

    # Convert all properties to face-averaged
    conservedPhor_face = node_to_face(conservedPhor_node)

    # We treat i/j/k independently: loop over them
    flux = []
    for n in range(3):

        rho, rhoVx, rhoVr, rhorVt, rhoe, P, ho, r = conservedPhor_face[n]

        rVt = rhorVt / rho
        rhoVt = rhorVt / r
        Vx = rhoVx / rho
        Vr = rhoVr / rho
        Vt = rhoVt / rho

        flux_mass = np.expand_dims(np.stack((rhoVx, rhoVr, rhoVt)), 0)

        # Convective flux vectors
        flux_conv = flux_mass * np.expand_dims(
            np.array(
                (
                    np.ones_like(rho),  # mass
                    Vx,  # x-mom
                    Vr,  # r-mom
                    rVt,  # rt-mom
                    ho,  # energy
                )
            ),
            axis=1,
        )

        flux_conv[..., wall[n]] = 0.0

        # Pressure flux vectors
        Z = np.zeros_like(P)
        flux_pressure = np.array(
            (
                (Z, Z, Z),  # mass
                (P, Z, Z),  # x-mom
                (Z, P, Z),  # r-mom
                (Z, Z, r * P),  # rt-mom
                (Z, Z, Omega * r * P),  # energy
            )
        )

        flux.append(flux_conv + flux_pressure)

    return flux

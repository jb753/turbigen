from turbigen.solvers.base import BaseSolver
from scipy.signal import convolve
import numpy as np
import turbigen.util


class NativeConfig(BaseSolver):
    """Settings with default values for the TS4 solver."""

    _name = "Native"


def get_timestep(b, CFL):
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


def get_fluxes(conserved, P, ho, r, Omega):

    rho, rhoVx, rhoVr, rhorVt, rhoe = conserved

    rVt = rhorVt / rho
    rhoVt = rhorVt / r
    Vx = rhoVx / rho
    Vr = rhoVr / rho
    Vt = rhoVt / rho

    flux = np.array(
        (
            (rhoVx, rhoVr, rhoVt),  # mass
            (rhoVx * Vx + P, rhoVr * Vx, rhoVt * Vx),  # x-mom
            (rhoVx * Vr, rhoVr * Vr + P, rhoVt * Vr),  # r-mom
            (rhoVx * rVt, rhoVr * rVt, rhoVt * Vt + r * P),  # rt-mom
            (rhoVx * ho, rhoVr * ho, rhoVt * ho + Omega * r * P),  # energy
        )
    )

    return flux


# Coefficients for central difference second derivative order 2, 4, 6 accurate
D2_CENTRAL = np.array(
    [
        [0, 0, 1, -2, 1, 0, 0],
        [0, -1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12, 0],
        [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90],
    ]
)

# Coefficients for boundary difference second derivative order 1, 3, 5 accurate
D2_BOUNDARY = np.array(
    [
        [1, -2, 1, 0, 0, 0, 0, 0],
        [35 / 12, -26 / 3, 19 / 2, -14 / 3, 11 / 12, 0, 0, 0],
        [203 / 45, -87 / 5, 117 / 4, -254 / 9, 33 / 2, -27 / 5, 137 / 180, 0],
    ]
)

sf = 0.5


def make_stencil(ityp, jtyp, ktyp, order):

    # Get 1D coefficients
    iord = order // 2 - 1
    cent = D2_CENTRAL[iord]
    bound = D2_BOUNDARY[iord]

    # Preallocate stencil size
    iz = np.nonzero(cent)[0]
    N = len(iz)
    N2 = N // 2
    cent = cent[iz]
    bound = bound[:N]
    D = np.zeros((N, N, N))

    def get_origin(typ):
        if typ == 1:
            o = 0
        elif typ == 0:
            o = N2
        elif typ == -1:
            o = -1
        else:
            raise Exception("typ should be in (0,1,-1)")
        return o

    # Select the 'origin' of the stencil
    io = get_origin(ityp)
    jo = get_origin(jtyp)
    ko = get_origin(ktyp)

    # Set the coefficients for i/j/k in turn

    for i in range(N):
        ii = i - N2 + order // 2
        if ityp:
            D[i, jo, ko] += bound[ii]
        else:
            D[i, jo, ko] += cent[ii]

    for j in range(N):
        jj = j - N2 + order // 2
        if jtyp:
            D[io, j, ko] += bound[jj]
        else:
            D[io, j, ko] += cent[jj]

    for k in range(N):
        kk = k - N2 + order // 2
        if ktyp:
            D[io, jo, k] += bound[kk]
        else:
            D[io, jo, k] += cent[kk]

    return D / 6.0 * sf


# Assemble all the stencils we need
KERN2 = {
    # Interior
    "ijk": make_stencil(0, 0, 0, 2),
    # 8 Faces
    "i0": make_stencil(1, 0, 0, 2),
    "ni": make_stencil(-1, 0, 0, 2),
    "j0": make_stencil(0, 1, 0, 2),
    "nj": make_stencil(0, -1, 0, 2),
    "k0": make_stencil(0, 0, 1, 2),
    "nk": make_stencil(0, 0, -1, 2),
    # 12 Edges
    # i=0
    "i0j0": make_stencil(1, 1, 0, 2),
    "i0nj": make_stencil(1, -1, 0, 2),
    "i0k0": make_stencil(1, 0, 1, 2),
    "i0nk": make_stencil(1, 0, -1, 2),
    # i=ni
    "nij0": make_stencil(-1, 1, 0, 2),
    "ninj": make_stencil(-1, -1, 0, 2),
    "nik0": make_stencil(-1, 0, 1, 2),
    "nink": make_stencil(-1, 0, -1, 2),
    # j
    "j0k0": make_stencil(0, 1, 1, 2),
    "j0nk": make_stencil(0, 1, -1, 2),
    "njk0": make_stencil(0, -1, 1, 2),
    "njnk": make_stencil(0, -1, -1, 2),
    # 8 Vertices
    "i0j0k0": make_stencil(1, 1, 1, 2),
    "i0njk0": make_stencil(1, -1, 1, 2),
    "i0j0nk": make_stencil(1, 1, -1, 2),
    "i0njnk": make_stencil(1, -1, -1, 2),
    "nij0k0": make_stencil(-1, 1, 1, 2),
    "ninjk0": make_stencil(-1, -1, 1, 2),
    "nij0nk": make_stencil(-1, 1, -1, 2),
    "ninjnk": make_stencil(-1, -1, -1, 2),
}


def conv(y, sten):
    return convolve(y, KERN2[sten], mode="valid").squeeze()


def smooth(x):
    # x has shape [?,ni,nj,nk]
    # return smoothed nodal values

    xs = x.copy()

    for i in range(x.shape[0]):

        # Interior
        xs[i, 1:-1, 1:-1, 1:-1] += conv(x[i], "ijk")

        # 6 Faces
        xs[i, 0, 1:-1, 1:-1] += conv(x[i, :3, :, :], "i0")
        xs[i, -1, 1:-1, 1:-1] += conv(x[i, -3:, :, :], "ni")
        xs[i, 1:-1, 0, 1:-1] += conv(x[i, :, :3, :], "j0")
        xs[i, 1:-1, -1, 1:-1] += conv(x[i, :, -3:, :], "nj")
        xs[i, 1:-1, 1:-1, 0] += conv(x[i, :, :, :3], "k0")
        xs[i, 1:-1, 1:-1, -1] += conv(x[i, :, :, -3:], "nk")

        # 12 Edges
        # i0
        xs[i, 0, 0, 1:-1] += conv(x[i, :3, :3, :], "i0j0")
        xs[i, 0, -1, 1:-1] += conv(x[i, :3, -3:, :], "i0nj")
        xs[i, 0, 1:-1, 0] += conv(x[i, :3, :, :3], "i0k0")
        xs[i, 0, 1:-1, -1] += conv(x[i, :3, :, -3:], "i0nk")
        # ni
        xs[i, -1, 0, 1:-1] += conv(x[i, -3:, :3, :], "nij0")
        xs[i, -1, -1, 1:-1] += conv(x[i, -3:, -3:, :], "ninj")
        xs[i, -1, 1:-1, 0] += conv(x[i, -3:, :, :3], "nik0")
        xs[i, -1, 1:-1, -1] += conv(x[i, -3:, :, -3:], "nink")
        # j
        xs[i, 1:-1, 0, 0] += conv(x[i, :, :3, :3], "j0k0")
        xs[i, 1:-1, 0, -1] += conv(x[i, :, :3, -3:], "j0nk")
        xs[i, 1:-1, -1, 0] += conv(x[i, :, -3:, :3], "njk0")
        xs[i, 1:-1, -1, -1] += conv(x[i, :, -3:, -3:], "njnk")

        # 8 Corners
        # i=0
        xs[i, 0, 0, 0] += conv(x[i, :3, :3, :3], "i0j0k0")
        xs[i, 0, -1, 0] += conv(x[i, :3, -3:, :3], "i0njk0")
        xs[i, 0, 0, -1] += conv(x[i, :3, :3, -3:], "i0j0nk")
        xs[i, 0, -1, -1] += conv(x[i, :3, -3:, -3:], "i0njnk")
        # ni
        xs[i, -1, 0, 0] += conv(x[i, -3:, :3, :3], "nij0k0")
        xs[i, -1, -1, 0] += conv(x[i, -3:, -3:, :3], "ninjk0")
        xs[i, -1, 0, -1] += conv(x[i, -3:, :3, -3:], "nij0nk")
        xs[i, -1, -1, -1] += conv(x[i, -3:, -3:, -3:], "ninjnk")

    return xs


def step(b, dt):

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
        Cin = b[ipatch]
        rfin = patch.rfin
        print(Cin.Po.mean(), Cin.To.mean(), Cin.V.mean())

        if patch.store:
            # Relax changes in density if we have a stored state
            rho_now = rfin * Cin.rho + (1.0 - rfin) * patch.store.rho
            # Isentropic expansion from stagnation state
            Cin.set_rho_s(rho_now, patch.state.s)
            assert np.allclose(Cin.rho, rho_now)

        patch.store = Cin

        # Get the velocity
        dhin = patch.state.h - Cin.h
        dhin[dhin <= 0.0] = 1e-9
        Vin = np.sqrt(2 * dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin * np.sqrt((1.0 + tanAlpha**2) * (1.0 + tanBeta**2))
        Vrin = Vxin * tanBeta
        Vmin = np.sqrt(Vxin**2 + Vrin**2)
        Vtin = Vmin * tanAlpha

        # Reset conserved vars on inlet
        rhoVx[ipatch] = Cin.rho * Vxin
        rhoVr[ipatch] = Cin.rho * Vrin
        rhorVt[ipatch] = Cin.rho * Cin.r * Vtin
        rhoe[ipatch] = Cin.rho * (Cin.u + 0.5 * Vin**2)

        # Reset pressure and hstag on inlet
        ho[ipatch] = Cin.h + 0.5 * Vin**2
        P[ipatch] = Cin.P

    flux = get_fluxes(conserved, P, ho, b.r, b.Omega)

    fi, fj, fk = node_to_face(flux)
    sumf = (
        -np.diff(fi * b.dAi, axis=-3)  # i faces
        - np.diff(fj * b.dAj, axis=-2)  # j faces
        - np.diff(fk * b.dAk, axis=-1)  # k faces
    ).sum(axis=1)

    S = node_to_vol(b.source_all)
    dU = (sumf / b.vol + S) * dt

    b.set_conserved(smooth(b.conserved + cell_to_node(dU)))

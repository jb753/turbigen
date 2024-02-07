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
    dli = 0.25*(dli[:,:-1,:-1] + dli[:,:-1,1:]+ dli[:,1:,:-1]+ dli[:,:-1,:-1])
    dlj = 0.25*(dlj[:-1,:,:-1] + dlj[:-1,:,1:]+ dlj[1:,:,:-1]+ dlj[:-1,:,:-1])
    dlk = 0.25*(dlk[:-1,:-1,:] + dlk[:-1,1:,:]+ dlk[1:,:-1,:]+ dlk[:-1,:-1,:])
    dlmin = np.minimum(dli, dlj, dlk)
    dt = CFL*dlmin/(aref+Vref)
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

    rVt = rhorVt/rho
    rhoVt = rhorVt/r
    Vx = rhoVx/rho
    Vr = rhoVr/rho
    Vt = rhoVt/rho

    flux = np.array(
        (
            (rhoVx, rhoVr, rhoVt),  # mass
            (rhoVx * Vx + P, rhoVr * Vx, rhoVt * Vx),  # x-mom
            (rhoVx * Vr, rhoVr * Vr + P, rhoVt * Vr),  # r-mom
            (rhoVx * rVt, rhoVr * rVt, rhoVt * Vt + r*P),  # rt-mom
            (rhoVx * ho, rhoVr * ho, rhoVt * ho + Omega * r * P),  # energy
        )
    )

    return flux

def smooth(x):
    # x has shape [?,ni,nj,nk]
    # return smoothed nodal values

    xs = np.full_like(x, np.nan)
    xs = x.copy()

    kern0 = np.array([[0, 0, 0],[0,1,0],[0,0,0]])
    kern1 = np.array([[0, 1, 0],[1,-6,1],[0,1,0]])
    kern2 = np.array([[0, 0, 0],[0,1,0],[0,0,0]])
    sf = 0.5
    kern = np.stack((kern0, kern1, kern2))/6.*sf

    kern_i0 = np.stack(
            [
                [[0,1,0],[1,-3,1],[0,1,0]],
                [[0,0,0],[0,-2,0],[0,0,0]],
                [[0,0,0],[0,1,0],[0,0,0]],
            ]
    )/6.*sf
    kern_j0 = np.moveaxis(kern_i0,0,1)
    kern_k0 = np.moveaxis(kern_i0,0,2)

    kern_ni = np.stack(
            [
                [[0,0,0],[0, 1, 0],[0,0,0]],
                [[0,0,0],[0,-2,0],[0,0,0]],
                [[0,1,0],[1,-3,1],[0,1,0]],
            ]
    )/6.*sf
    kern_nj = np.moveaxis(kern_ni,0,1)
    kern_nk = np.moveaxis(kern_ni,0,2)

    for i in range(x.shape[0]):
        xs[i,1:-1,1:-1,1:-1] += convolve(x[i], kern, mode='valid')

        xs[i, -1, 1:-1, 1:-1] += convolve(x[i,-3:,:,:], kern_ni, mode='valid').squeeze()
        xs[i, 1:-1, -1, 1:-1] += convolve(x[i,:,-3:,:], kern_nj, mode='valid').squeeze()
        xs[i, 1:-1, 1:-1, -1] += convolve(x[i,:,:,-3:], kern_nk, mode='valid').squeeze()

        xs[i, 0, 1:-1, 1:-1] += convolve(x[i,:3,:,:], kern_i0, mode='valid').squeeze()
        xs[i, 1:-1, 0, 1:-1] += convolve(x[i,:,:3,:], kern_j0, mode='valid').squeeze()
        xs[i, 1:-1, 1:-1, 0] += convolve(x[i,:,:,:3], kern_k0, mode='valid').squeeze()

        # xs[i, 0, 0, 1:-1] += convolve(x[i,:3,:3,:], kern_i0j0, mode='valid').squeeze()
        # xs[i, 0, 1:-1, 0] += convolve(x[i,:3,:3,:], kern_i0k0, mode='valid').squeeze()
        # xs[i, 1:-1, 0, 0] += convolve(x[i,:3,:3,:], kern_j0k0, mode='valid').squeeze()
        # xs[i, -1, :, :] += (x[i, 0, :, :] - 2.*x[i, 1, :, :] + x[i, 2, :, :])*sf/6.
        # xs[i, 0, :, :] += (x[i, -3, :, :] - 2.*x[i, -2, :, :] + x[i, -1, :, :])*sf/6.

    return xs

def get_wall(b):
    # Find logical indices that zero the fluxes on wall faces
    thresh = 0.99  # To allow for floating point error
    return [w>thresh for w in node_to_face(b.get_wall())]

def step(b, dt, wall):

    P = b.P.copy()
    ho = b.ho.copy()
    conserved = b.conserved.copy()

    rho, rhoVx, rhoVr, rhorVt, rhoe = conserved
    r = b.r

    # Adjust static pressure for exit boundary conditions
    for patch in b.outlet_patches:
        P[patch.get_slice()] = patch.Pout * 0.1 + 0.9*(patch.get_cut().P)

    # Change inlet patches
    for patch in b.inlet_patches:

        ipatch = patch.get_slice()
        Cin = b[ipatch]
        rfin = 0.1#patch.rfin
        print(Cin.Po.mean(), Cin.To.mean(), Cin.V.mean())

        if patch.store:
            # Relax changes in density if we have a stored state
            rho_now = rfin*Cin.rho + (1.-rfin)*patch.store.rho
            # Isentropic expansion from stagnation state
            Cin.set_rho_s(rho_now, patch.state.s)
            assert np.allclose(Cin.rho, rho_now)

        patch.store = Cin

        # Get the velocity
        dhin = patch.state.h - Cin.h
        dhin[dhin<=0.] = 1e-9
        Vin = np.sqrt(2*dhin)

        tanAlpha = turbigen.util.tand(patch.Alpha)
        tanBeta = turbigen.util.tand(patch.Beta)

        # Resolve velocity components
        Vxin = Vin * np.sqrt((1.+tanAlpha**2)*(1.+tanBeta**2))
        Vrin = Vxin * tanBeta
        Vmin = np.sqrt(Vxin**2 + Vrin**2)
        Vtin = Vmin * tanAlpha

        # Reset conserved vars on inlet
        rhoVx[ipatch] = Cin.rho * Vxin
        rhoVr[ipatch] = Cin.rho * Vrin
        rhorVt[ipatch] = Cin.rho * Cin.r * Vtin
        rhoe[ipatch] = Cin.rho*(Cin.u + 0.5*Vin**2)

        # Reset pressure and hstag on inlet
        ho[ipatch] = Cin.h + 0.5*Vin**2
        P[ipatch] = Cin.P

    flux = get_fluxes(conserved, P, ho, b.r, b.Omega)


    fi, fj, fk = node_to_face(flux)

    # # Zeros fluxes on walls
    # wi, wj, wk = wall
    # fi[...,wi] = 0.
    # fj[...,wj] = 0.
    # fk[...,wk] = 0.

    sumf = (
        -np.diff(fi * b.dAi, axis=-3)  # i faces
        - np.diff(fj * b.dAj, axis=-2)  # j faces
        - np.diff(fk * b.dAk, axis=-1)  # k faces
    ).sum(axis=1)

    S = node_to_vol(b.source_all)
    dU = (sumf / b.vol + S) * dt

    b.set_conserved(smooth(b.conserved + cell_to_node(dU)))

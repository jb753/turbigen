from turbigen.solvers.base import BaseSolver
import numpy as np


class NativeConfig(BaseSolver):
    """Settings with default values for the TS4 solver."""

    _name = "Native"


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
            x[..., :-1, :-1, :-1], # i, j, k
            x[..., 1:, :-1, :-1], # i+1, j, k
            x[..., 1:, 1:, :-1], # i+1, j+1, k
            x[..., :-1, 1:, :-1], # i, j+1, k
            x[..., :-1, :-1, 1:], # i, j, k+1
            x[..., 1:, :-1, 1:], # i+1, j, k+1
            x[..., 1:, 1:, 1:], # i+1, j+1, k+1
            x[..., :-1, 1:, 1:], # i, j+1, k+1
        ),
    ).mean(axis=0)

def cell_to_node(x):
    # x has shape [?,ni-1,nj-1,nk-1]
    # return values for each node
    # xi [?,ni,nj, nk]
    *other, nim1, njm1, nkm1 = x.shape
    ni = nim1+1
    nj = njm1+1
    nk = nkm1+1

    xn = np.full(tuple(other) + (ni, nj, nk),np.nan)

    # Interior nodes take 1/8th from each of i
    xn[...,1:-1,1:-1,1:-1] = node_to_vol(x)

    # i=(0,-1) takes 1/4 from j, j+1, k, k+1
    for i in (0,-1):
        xn[...,i,1:-1,1:-1] = np.stack(
                (
                    x[...,i,:-1,:-1],  # j, k
                    x[...,i,1:,:-1],  # j+1, k
                    x[...,i,:-1,1:],  # j, k+1
                    x[...,i,1:,1:],  # j+1, k+1
                )
        ).mean(axis=0)

    # j=(0,-1) takes 1/4 from i, i+1, k, k+1
    for j in (0,-1):
        xn[...,1:-1,j,1:-1] = np.stack(
                (
                    x[...,:-1,j,:-1],
                    x[...,1:,j,:-1],
                    x[...,:-1,j,1:],
                    x[...,1:,j,1:],
                )
        ).mean(axis=0)

    # k=(0,-1) takes 1/4 from i, i+1, k, k+1
    for k in (0,-1):
        xn[...,1:-1,1:-1,k] = np.stack(
                (
                    x[...,:-1,:-1,k],
                    x[...,1:,:-1,k],
                    x[...,:-1,1:,k],
                    x[...,1:,1:,k],
                )
        ).mean(axis=0)

    # Edges take half from nearest two cells


    # Corners take entire change from nearest cell
    for i in (0,-1):
        for j in (0,-1):
            for k in (0,-1):
                xn[...,i,j,k] = x[...,i,j,k]

    print(xn[0,:3,:3,0])
    assert not np.isnan(xn).any()

def step(g, dt):

    fi, fj, fk = node_to_face(g.flux_all)
    sumf = (
            (fi[...,:-1,:,:] - fi[...,1:,:,:]) * g.dAi  # i faces
            + (fj[...,:,:-1,:] - fj[...,:,1:,:]) * g.dAj  # j faces
            + (fk[...,:,:,:-1] - fk[...,:,:,1:]) * g.dAk  # k faces
    ).sum(axis=1)
    S = node_to_vol(g.source_all)
    dU = (sumf/g.vol + S)*dt


    cell_to_node(dU)

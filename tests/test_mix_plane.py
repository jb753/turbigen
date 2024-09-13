"""Run a quasi-1D nozzle in the native solver."""

import turbigen.solvers.embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.util
import numpy as np
from timeit import default_timer as timer
from copy import copy
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

def make_grid(
    L_h=2.0,
    htr=0.95,
    Ma1=0.3,
    rpm=350.
):
    """Generate the grid."""

    # Geometry
    h = 0.1
    L = h * L_h
    rm = 0.5 * h * (1.0 + htr) / (1.0 - htr)
    rh = rm - 0.5 * h
    rt = rm + 0.5 * h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-3
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0
    To1a = To1
    TR = 2.
    To1b = To1*TR

    # Rotating reference frame
    Omega = rpm / 60. * np.pi * 2.
    U = Omega*rm

    # Set inlet Ma to get inlet static state
    rgas = cp * (ga - 1.0) / ga
    V = cf.V_cpTo_from_Ma(Ma1, ga) * np.sqrt(cp * To1)
    P1 = Po1 / cf.Po_P_from_Ma(Ma1, ga)
    T1 = To1 / cf.To_T_from_Ma(Ma1, ga)


    # Relative flow angle
    Alpha = 0.
    Vt = V * np.sin(np.radians(Alpha))
    Vx = V * np.cos(np.radians(Alpha))
    Vt_rel = Vt - U
    Alpha_rel = np.degrees(np.arctan2(Vt_rel, V))

    # Numbers of grid points
    AR_pitch = 1.
    AR_merid = 2.
    nj = 33+ 8
    nk = 33
    ni = int(nj * L_h/ AR_merid)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1) * AR_pitch
    Nb = int(2.0 * np.pi * rm / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    tv = np.linspace(0.0, dt, nk)
    xv = np.linspace(0.0, L, ni)
    rv = np.linspace(rh, rt, nj)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij"))

    jmid = nj//2

    # Split into blocks
    blocks = []
    nblock = 2
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    patches = [
        [
            turbigen.grid.InletPatch(i=0),
            turbigen.grid.MixingPatch(i=-1),
            # turbigen.grid.PeriodicPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            turbigen.grid.InviscidPatch(j=0),
            turbigen.grid.InviscidPatch(j=-1),
        ],
        [
            turbigen.grid.MixingPatch(i=0),
            # turbigen.grid.PeriodicPatch(i=0),
            turbigen.grid.OutletPatch(i=-1),
            turbigen.grid.PeriodicPatch(k=0),
            turbigen.grid.PeriodicPatch(k=-1),
            turbigen.grid.InviscidPatch(j=0),
            turbigen.grid.InviscidPatch(j=-1),
        ],
    ]

    for iblock in range(nblock):
        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches[iblock]
        )
        block.label = f"b{iblock}"
        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()

    # Boundary conditions
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    g.apply_inlet(So1, 0., 0.)
    g.calculate_wall_distance()
    g.apply_outlet(P1)

    # Fluid props
    for b in g:
        b.cp = cp
        b.gamma = ga
        b.mu = mu

    # Initial guess
    for ib, b in enumerate(g):
        b.Vx = Vx
        b.Vr = 0.#Vx*0.01
        b.Vt = Vt
        b.set_P_T(P1, T1)
        b.Omega = 0.
    g[1].Vx += Vx*0.01
    g[1].Vr += Vx*0.01
    g[1].Vt += Vx*0.01

    g.match_patches()

    return g


settings = {
    "n_step": 2000,
    "n_step_avg": 1,
    "n_step_log": 100,
    "nstep_damp": -1,
    "damping_factor": 3.,
    "fmgrid": 0.,
    "CFL": 0.1,
    "i_loss": 0,
    # "plot_conv": True,
}
conf = turbigen.solvers.embsolve.Config(**settings)

def test_CFL_0():
    """Without any update from the interior, mixing plane should be stable."""

    settings = {
        "n_step": 2000,
        "n_step_avg": 1,
        "n_step_log": 100,
        "nstep_damp": -1,
        "CFL": 0.0,
    }
    g = make_grid()
    conf = turbigen.solvers.embsolve.Config(**settings)
    np.set_printoptions(precision=2)
    turbigen.solvers.embsolve.run(g, conf)


def test_mix_plane():
    """"""

    g = make_grid()
    np.set_printoptions(precision=2)
    turbigen.solvers.embsolve.run(g, conf)

    fig, ax = plt.subplots()
    lev_To= np.linspace(290., 310, 11)
    ax.set_title('To')
    for b in g:
        C = b[:,b.nj//2,:]
        ax.contourf(C.x, C.rt, C.To, lev_To)

    fig, ax = plt.subplots()
    lev_Vx= np.linspace(97., 99., 11)
    for b in g:
        C = b[:,b.nj//2,:]
        cm = ax.contourf(C.x, C.rt, C.Vx, lev_Vx)
    plt.colorbar(cm)

    fig, ax = plt.subplots()
    ax.axis('equal')
    C = g[0][-1,:,:]
    cm = ax.contourf(C.y, C.z, C.Vx/C.Vx.mean())
    plt.colorbar(cm)

    fig, ax = plt.subplots()
    ax.axis('equal')
    C = g[0][-1,:,:]
    cm = ax.contourf(C.y, C.z, C.To/C.To.mean())
    plt.colorbar(cm)



    plt.show()

    # # Check To conservation
    # C1 = g[0][0,:,:]
    # C2 = g[0][-1,:,:]
    # C1m, A, _ = C1.mix_out()
    # C2m, _, _ = C2.mix_out()
    # print(C1m.To, C2m.To)

    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # b = g[0]
    # C = b[:,:,b.nk//2]
    # cm = ax.contourf(C.x, C.r, C.To)
    # plt.colorbar(cm)
    # plt.show()


if __name__ == "__main__":
    # test_CFL_0()
    test_mix_plane()

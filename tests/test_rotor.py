"""Run a rotor."""

import turbigen.solvers.embsolve
import turbigen.compflow_native as cf
import turbigen.grid
import turbigen.util
import turbigen.base
import numpy as np
from timeit import default_timer as timer
import sys
from scipy.interpolate import pchip_interpolate
import matplotlib.pyplot as plt
import pytest

def make_rotor(
    Ma1_rel,
    phi1,
):
    """Generate the grid."""

    # Geometry
    h = 0.1
    r1 = 0.6
    L_h = 4
    r2 = r1 + L_h*h

    # Boundary conditions
    ga = 1.4
    cp = 1005.0
    mu = 1.8e-5
    Beta = 0.0
    Po1 = 1e5
    To1 = 300.0

    # Guess static speed of sound and iter
    So1 = turbigen.fluid.PerfectState.from_properties(cp, ga, mu)
    So1.set_P_T(Po1, To1)
    S1 = So1.copy()
    MAXITER=50
    for _ in range(MAXITER):
        Vr1 = S1.a * Ma1_rel
        U1 = Vr1 / phi1
        V1sq = Vr1**2 + U1**2
        h1 = So1.h - 0.5*V1sq
        S1.set_h_s(h1, So1.s)

    Omega = -U1/r1
    U2 = Omega * r2
    Vxrt1 = np.array([0., Vr1, U1])
    Vxrt2 = np.array([0., Vr1*r1/r2, U2])
    V2sq = np.sum(Vxrt2**2)

    # Euler work eqn to outlet
    # Vtrel=0 so Vt=U
    ho2 = So1.h + (U2**2-U1**2)
    h2 = ho2 - V2sq
    S2 = S1.copy().set_h_s(h2, S1.s)

    # Numbers of grid points
    nj = 17
    nk = 17
    ni = int(np.round(nj * L_h/8)*8 + 1)

    # Use pitchwise aspect ratio to find cell spacing, pitch and Nb
    pitch = h / (nj - 1) * (nk - 1)
    Nb = int(2.0 * np.pi * (r1+r2)/2. / pitch)
    dt = 2.0 * np.pi / float(Nb)

    # Make the coordinates
    xv = np.linspace(h, 0.0, nj)
    rv = np.linspace(r1, r2, ni)
    tv = np.linspace(-dt/2., dt/2., nk)

    xrt = np.stack(np.meshgrid(xv, rv, tv, indexing="ij")).transpose(0, 2, 1, 3)

    # Split into blocks
    blocks = []
    nblock = 1
    istb = [ni // nblock * iblock for iblock in range(nblock)]
    ienb = [ni // nblock * (iblock + 1) + 1 for iblock in range(nblock)]
    ienb[-1] = ni

    for iblock in range(nblock):

        # Special case for only one block
        if nblock == 1:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # First block has an inlet
        elif iblock == 0:
            patches = [
                turbigen.grid.InletPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

        # Last block has outlet
        elif iblock == (nblock - 1):
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.OutletPatch(i=-1),
            ]

        # Middle blocks are both periodic
        else:
            patches = [
                turbigen.grid.PeriodicPatch(i=0),
                turbigen.grid.PeriodicPatch(i=-1),
            ]

        patches.extend(
            [
                turbigen.grid.InviscidPatch(j=0),
                turbigen.grid.InviscidPatch(j=-1),
                turbigen.grid.InviscidPatch(k=0),
                turbigen.grid.InviscidPatch(k=-1),
            ]
        )

        block = turbigen.grid.PerfectBlock.from_coordinates(
            xrt[:, istb[iblock] : ienb[iblock], :, :], Nb, patches
        )
        block.label = f"b{iblock}"

        blocks.append(block)

    # Make the grid object
    g = turbigen.grid.Grid(blocks)
    g.check_coordinates()
    g.match_patches()

    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # for b in g:
    #     C = b[:,b.nj//2, :]
    #     ax.plot(C.y, C.z, 'k-', lw=0.5)
    #     ax.plot(C.y.T, C.z.T, 'k-', lw=0.5)
    # fig, ax = plt.subplots()
    # ax.axis('equal')
    # for b in g:
    #     C = b[:,:, b.nk//2]
    #     ax.plot(C.x, C.r, 'k-', lw=0.5)
    #     ax.plot(C.x.T, C.r.T, 'k-', lw=0.5)
    # plt.show()

    # Fluid props
    for b in g:
        b.cp = cp
        b.gamma = ga
        b.mu = mu
        b.Omega = Omega

    # Evaulate 1D analytical
    U = Omega*rv
    Vr = Vr1*r1/rv

    F = g[0].empty(shape=(ni,))
    F.r = rv
    F.x = 0.
    F.t = 0.
    F.Omega = Omega
    F.Vx = 0.0
    F.Vr = Vr
    F.Vt = U
    I1 = So1.h - U1**2
    h = I1 - 0.5*(F.V_rel**2 - U**2)
    F.set_h_s(h, So1.s)

    Vre = np.expand_dims(Vr, (1,2))
    Ue = np.expand_dims(U, (1,2))
    he = np.expand_dims(h, (1,2))

    # Initial guess
    for ib, b in enumerate(g):
        b.Vx = 0.0
        b.Vr = Vre[istb[ib]:ienb[ib]]
        b.Vt = Ue[istb[ib]:ienb[ib]]
        b.set_h_s(he[istb[ib]:ienb[ib]], So1.s)

    # Boundary conditions
    g.apply_inlet(So1, F.Alpha[0], F.Beta[0])
    g.calculate_wall_distance()
    g.apply_outlet(S2.P)

    return g, F


settings = {
    "n_step": 400,
    "n_step_avg": 5,
    "n_step_log": 50,
    "i_loss": 0,
    "nstep_damp": -1,
    "plot_conv": True,
}

settings = {
    "n_step": 353,
    "n_step_avg": 1,
    "n_step_log": 1,
    "i_loss": 0,
    # "i_exit": 0,
    "i_inlet": 0,
    "i_scheme": 0,
    "CFL": 0.4,
    "nstep_damp": -1,
    "damping_factor": 3.,
    "plot_conv": True,
}

conf = turbigen.solvers.embsolve.Config(**settings)

def plot_rotor(g, F):

    r1 = F.r[0]
    r2 = F.r[-1]
    rn = (F.r - r1)/(r2-r1)
    To1 = F.To[0]
    U1 = F.U[0]
    CTo = (F.To-To1)/U1**2
    print(U1)

    fig, ax = plt.subplots()
    ax.plot(F.To, rn, 'k-')
    for b in g:
        C = b[:,b.nj//2, b.nk//2]
        rnC = (C.r - r1)/(r2-r1)
        CToC = (C.To-To1)/U1**2
        ax.plot(C.To, rnC, '-x')

    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_title('Vr')
    for b in g:
        C = b[:,b.nj//2, :].squeeze()
        cm = ax.contourf(C.y, C.z, C.Vr)
    plt.colorbar(cm)
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_title('Vt_rel')
    for b in g:
        C = b[:,b.nj//2, :].squeeze()
        cm = ax.contourf(C.y, C.z, C.Vt_rel)
    plt.colorbar(cm)




    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.set_title('To')
    for b in g:
        C = b[:,b.nj//2, :].squeeze()
        cm = ax.contourf(C.y, C.z, C.To)
    plt.colorbar(cm)
    plt.show()



    plt.show()


def test_static():
    g, F = make_rotor(Ma1_rel=0.4, phi1=1e6)
    print(F[0].Alpha, F[0].Alpha_rel)
    print(F[-1].Alpha, F[-1].Alpha_rel)
    np.set_printoptions(precision=2)
    turbigen.solvers.embsolve.run(g, conf)
    plot_rotor(g, F)

def test_rotating(phi1):
    g, F = make_rotor(Ma1_rel=0.4, phi1=phi1)
    np.set_printoptions(precision=2)
    print(F[0].Alpha, F[0].Alpha_rel)
    print(F[0].Vt)
    turbigen.solvers.embsolve.run(g, conf)
    plot_rotor(g, F)



if __name__ == "__main__":

    test_rotating(phi1=0.5)
    # test_rpm(500.)

    # print('testing exit, aligned grid')
    # test_patch_A_avg()
    # test_exit(0.)

    # print('testing uniform, vary Ma')
    # test_Ma(0.9)

    # print('testing uniform, aligned grid')
    # test_uniform(-30.)

    # print('testing uniform, skewed grid')
    # test_skew(-0.)

    # print('testing radius change, aligned grid')
    # test_radius(30.)

    # print('testing con-di nozzles')
    # print(Alpha, g[0].dlmin[0,0,0], g[0].vol[0,0,0], g[)
    # test_condi('t')
